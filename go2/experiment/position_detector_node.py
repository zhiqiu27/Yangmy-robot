#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
position_detector_node.py
ROS node for position detection using ZED camera.
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl
import time
import threading
import queue
import logging
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String, Float64
from std_srvs.srv import Trigger, TriggerResponse

# Configure basic logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class PositionDetectorNode:
    def __init__(self, camera_fps=24, camera_resolution_str="VGA"):
        rospy.init_node('position_detector', anonymous=False)
        logger.info("Initializing PositionDetectorNode...")
        
        self.camera_fps = camera_fps
        
        if camera_resolution_str.upper() == "VGA":
            self.camera_resolution = sl.RESOLUTION.VGA
        elif camera_resolution_str.upper() == "HD720":
            self.camera_resolution = sl.RESOLUTION.HD720
        else:
            logger.warning(f"Unsupported camera resolution string: {camera_resolution_str}. Defaulting to VGA.")
            self.camera_resolution = sl.RESOLUTION.VGA

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        self.init_params.camera_fps = self.camera_fps
        self.init_params.camera_resolution = self.camera_resolution
        
        self.runtime_params = sl.RuntimeParameters()
        self.img_mat = sl.Mat()
        self.xyz_mat = sl.Mat()

        self.bbox_lock = threading.Lock()
        self.last_known_bbox_shared = {'bbox': None}  # bbox will be (x1,y1,x2,y2) tuple
        
        self.latest_object_xy_lock = threading.Lock()
        self.latest_object_xy_shared = {'xy': None}  # Stores (x, y)

        self.stop_event = threading.Event()
        self.processing_thread = None
        
        # ROS Publishers
        self.object_position_pub = rospy.Publisher('/vision/object_position', Point, queue_size=10)
        self.detection_status_pub = rospy.Publisher('/vision/detection_status', Bool, queue_size=10)
        
        # 新增发布校准结果的话题
        self.calibration_result_pub = rospy.Publisher('/vision/calibration_result', String, queue_size=10)
        self.calibration_x_pub = rospy.Publisher('/vision/calibration_x', Float64, queue_size=10)
        self.calibration_y_pub = rospy.Publisher('/vision/calibration_y', Float64, queue_size=10)
        
        # ROS Services
        # 将自定义服务替换为标准Trigger服务
        self.calibration_srv = rospy.Service('/vision/calibrate_object', Trigger, self.handle_calibration)
        self.reset_srv = rospy.Service('/vision/reset', Trigger, self.handle_reset)
        
        # Parameters
        self.calibration_samples = rospy.get_param('~calibration_samples', 5)
        self.calibration_delay = rospy.get_param('~calibration_delay', 0.5)
        
        # For test mode - simulated bbox
        self.test_mode = rospy.get_param('~test_mode', False)
        if self.test_mode:
            logger.warning("Running in TEST MODE - using simulated detections")
            self.test_bbox = [320, 240, 380, 300]  # Default test bbox
            
        logger.info("PositionDetectorNode initialized.")

    def _open_camera(self):
        if self.test_mode:
            logger.info("Test mode: Skipping camera open")
            return
            
        logger.info("Opening ZED camera...")
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED Camera open failed: {status}")
            raise RuntimeError(f"ZED Camera open failed: {status}")
        logger.info("ZED camera opened successfully.")

    def _bbox_center_xyz(self, bbox, xyz_measure):
        if self.test_mode:
            # Return a simulated position in test mode
            return (0.6, 0.0, 0.5)  # Simulated X,Y,Z in meters
            
        x1, y1, x2, y2 = map(int, bbox)
        
        img_h, img_w = xyz_measure.get_height(), xyz_measure.get_width()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Bounding box {bbox} became invalid after clamping to image dimensions ({img_w}x{img_h}).")
            return None

        xyz_data = xyz_measure.get_data() 
        patch = xyz_data[y1:y2 + 1, x1:x2 + 1, :3]
        
        mask = np.isfinite(patch[..., 0]) & np.isfinite(patch[..., 1]) & np.isfinite(patch[..., 2])
        if not np.any(mask):
            return None
        
        median_coord = tuple(np.median(patch[mask], axis=0))
        return median_coord

    def _processing_loop(self):
        logger.info("Vision processing loop started.")
        
        # For test mode - generate fake detections periodically
        if self.test_mode:
            test_counter = 0
            
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                if self.test_mode:
                    # Simulate camera input in test mode
                    test_counter += 1
                    if test_counter % 100 == 0:
                        logger.info("Test mode: Generating simulated detection")
                    
                    # Simulate object detection
                    if test_counter % 50 < 40:  # 80% of the time have a detection
                        with self.bbox_lock:
                            self.last_known_bbox_shared['bbox'] = self.test_bbox
                        
                        # Simulate position calculation (slightly randomized)
                        x = 0.6 + 0.02 * np.sin(test_counter/10.0)
                        y = 0.05 * np.cos(test_counter/10.0)
                        
                        with self.latest_object_xy_lock:
                            self.latest_object_xy_shared['xy'] = (x, y)
                            
                        # Publish position
                        pos_msg = Point()
                        pos_msg.x = x
                        pos_msg.y = y
                        pos_msg.z = 0.5  # Constant z in test mode
                        self.object_position_pub.publish(pos_msg)
                        
                        # Publish detection status
                        self.detection_status_pub.publish(Bool(True))
                    else:
                        # Simulate no detection for 20% of the time
                        with self.bbox_lock:
                            self.last_known_bbox_shared['bbox'] = None
                        with self.latest_object_xy_lock:
                            self.latest_object_xy_shared['xy'] = None
                        self.detection_status_pub.publish(Bool(False))
                    
                    # Simulate camera frame rate
                    time.sleep(1.0/self.camera_fps)
                    continue
                    
                # Regular mode with real camera
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(self.img_mat, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZ)

                    current_bbox_local = None
                    with self.bbox_lock:
                        current_bbox_local = self.last_known_bbox_shared['bbox']
                    
                    if current_bbox_local:
                        depth_coords = self._bbox_center_xyz(current_bbox_local, self.xyz_mat)
                        if depth_coords:
                            with self.latest_object_xy_lock:
                                self.latest_object_xy_shared['xy'] = (depth_coords[0], depth_coords[1])
                                
                            # Publish the position as a ROS message
                            pos_msg = Point()
                            pos_msg.x = depth_coords[0]
                            pos_msg.y = depth_coords[1]
                            pos_msg.z = depth_coords[2]
                            self.object_position_pub.publish(pos_msg)
                            
                            # Publish detection status
                            self.detection_status_pub.publish(Bool(True))
                        else:
                            with self.latest_object_xy_lock:
                                self.latest_object_xy_shared['xy'] = None
                            self.detection_status_pub.publish(Bool(False))
                    else:
                        with self.latest_object_xy_lock:
                            self.latest_object_xy_shared['xy'] = None
                        self.detection_status_pub.publish(Bool(False))
                
                # Check for incoming ROS messages/service calls
                rospy.sleep(0.001)  # Short sleep to allow ROS callbacks to process
            
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                rospy.sleep(1.0)  # Sleep longer after an error
        
        logger.info("Vision processing loop ended.")

    def start(self):
        """Start the position detector node"""
        logger.info("Starting PositionDetectorNode...")
        try:
            self._open_camera()
            
            self.stop_event.clear()
            
            # Start the processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, name="VisionProcessingThread")
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("PositionDetectorNode started successfully.")
            
            # Spin to receive ROS callbacks
            rospy.spin()
            
        except Exception as e:
            logger.error(f"Failed to start PositionDetectorNode: {e}", exc_info=True)
            self.shutdown()
            raise

    def get_current_object_xy(self):
        """Returns the latest calculated (X, Y) coordinates of the object or None."""
        with self.latest_object_xy_lock:
            xy = self.latest_object_xy_shared['xy']
        return xy

    def handle_calibration(self, req):
        """Handle calibration service request"""
        logger.info(f"Received calibration request for {self.calibration_samples} samples")
        
        # 使用标准TriggerResponse而不是自定义响应
        response = TriggerResponse()
        collected_coords = []
        
        try:
            for i in range(self.calibration_samples):
                xy = self.get_current_object_xy()
                if xy:
                    logger.info(f"Calibration sample {i+1}/{self.calibration_samples}: {xy}")
                    collected_coords.append(xy)
                else:
                    logger.warning(f"Calibration sample {i+1}/{self.calibration_samples}: No coordinates detected.")
                time.sleep(self.calibration_delay)
            
            if not collected_coords:
                logger.error("Calibration failed: No valid coordinates collected.")
                response.success = False
                response.message = "No valid coordinates detected during calibration"
                return response
            
            avg_x = np.mean([coord[0] for coord in collected_coords])
            avg_y = np.mean([coord[1] for coord in collected_coords])
            
            response.success = True
            response.message = f"Calibration successful. Average position: [{avg_x:.3f}, {avg_y:.3f}]"
            
            # 通过话题发布校准结果，而不是直接在服务响应中返回
            self.calibration_result_pub.publish(String(response.message))
            self.calibration_x_pub.publish(Float64(avg_x))
            self.calibration_y_pub.publish(Float64(avg_y))
            
            logger.info(f"Calibration completed. Average position: [{avg_x:.3f}, {avg_y:.3f}]")
            return response
            
        except Exception as e:
            logger.error(f"Error during calibration: {e}", exc_info=True)
            response.success = False
            response.message = f"Error during calibration: {str(e)}"
            return response

    def handle_reset(self, req):
        """Reset the detection system"""
        try:
            with self.bbox_lock:
                self.last_known_bbox_shared['bbox'] = None
            with self.latest_object_xy_lock:
                self.latest_object_xy_shared['xy'] = None
                
            response = TriggerResponse()
            response.success = True
            response.message = "Detection system reset successfully"
            return response
        except Exception as e:
            logger.error(f"Error resetting detection system: {e}")
            response = TriggerResponse()
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

    def shutdown(self):
        """Shutdown the node"""
        logger.info("Shutting down PositionDetectorNode...")
        self.stop_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not terminate cleanly")
        
        if not self.test_mode and self.zed and self.zed.is_opened():
            logger.info("Closing ZED camera...")
            self.zed.close()
            logger.info("ZED camera closed.")
        
        logger.info("PositionDetectorNode shutdown complete.")

if __name__ == "__main__":
    node = None
    try:
        # Parse command line arguments if needed
        camera_fps = rospy.get_param('~camera_fps', 24)
        camera_resolution = rospy.get_param('~camera_resolution', "VGA")
        
        node = PositionDetectorNode(camera_fps=camera_fps, camera_resolution=camera_resolution)
        node.start()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logger.critical(f"Critical error in PositionDetectorNode: {e}", exc_info=True)
    finally:
        if node:
            node.shutdown() 