#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zed_bbox_to_3d.py
实时读取 ZED-Mini，配合 2D 目标检测，输出目标在
RIGHT_HANDED_Z_UP_X_FWD 相机坐标系下的 (前 X, 左 Y, 上 Z) 位置。
按  q  键退出。
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl
import socket
import struct # For packing/unpacking data size
import time   # For small delays
import threading
import queue
import logging

# Configure basic logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Default port numbers
DEFAULT_IMAGE_PORT = 12345
DEFAULT_BBOX_PORT = 12346

# New: Timeout configurations
DEFAULT_MAX_FRAME_PROCESSING_TIME_SECONDS = 0.5 # Skip frame if ZED processing + encoding takes longer
DEFAULT_SEND_OPERATION_TIMEOUT_SECONDS = 0.5    # Timeout for individual socket sendall operations in _send_all

class PositionDetector:
    def __init__(self, image_port=DEFAULT_IMAGE_PORT, bbox_port=DEFAULT_BBOX_PORT, camera_fps=24, camera_resolution_str="VGA",
                 max_frame_processing_time=DEFAULT_MAX_FRAME_PROCESSING_TIME_SECONDS,
                 send_operation_timeout=DEFAULT_SEND_OPERATION_TIMEOUT_SECONDS):
        logger.info("Initializing PositionDetector...")
        self.image_port = image_port
        self.bbox_port = bbox_port
        self.camera_fps = camera_fps
        self.max_frame_processing_time = max_frame_processing_time
        self.send_operation_timeout = send_operation_timeout
        
        if camera_resolution_str.upper() == "VGA":
            self.camera_resolution = sl.RESOLUTION.VGA
        elif camera_resolution_str.upper() == "HD720":
            self.camera_resolution = sl.RESOLUTION.HD720
        # Add more resolutions as needed
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

        self.image_queue = queue.Queue(maxsize=50)
        self.bbox_lock = threading.Lock()
        self.last_known_bbox_shared = {'bbox': None} # bbox will be (x1,y1,x2,y2) tuple
        
        self.latest_object_xy_lock = threading.Lock()
        self.latest_object_xy_shared = {'xy': None} # Stores (x, y)

        # New: For visual servoing based on pixel coordinates
        self.latest_visual_info_lock = threading.Lock()
        self.latest_visual_info_shared = {'pixel_cx': None, 'image_width': None, 'bbox_available': False}

        self.stop_event = threading.Event()

        self.image_server_socket = None
        self.bbox_server_socket = None
        self.image_conn = None
        self.bbox_conn = None
        
        self.image_sender_thread = None
        self.bbox_receiver_thread = None
        self.zed_processing_thread = None

        self.image_sender_stop_event = threading.Event()
        self.bbox_receiver_stop_event = threading.Event()
        logger.info("PositionDetector initialized.")

    def _open_camera(self):
        logger.info("Opening ZED camera...")
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED Camera open failed: {status}")
            raise RuntimeError(f"ZED Camera open failed: {status}")
        logger.info("ZED camera opened successfully.")

    def _setup_sockets(self):
        logger.info(f"Setting up image server on port {self.image_port}...")
        self.image_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.image_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.image_server_socket.bind(('0.0.0.0', self.image_port))
        self.image_server_socket.listen(1)
        logger.info(f"Image server listening on port {self.image_port}.")

        logger.info(f"Setting up bbox server on port {self.bbox_port}...")
        self.bbox_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bbox_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bbox_server_socket.bind(('0.0.0.0', self.bbox_port))
        self.bbox_server_socket.listen(1)
        logger.info(f"Bbox server listening on port {self.bbox_port}.")

    def _bbox_center_xyz(self, bbox, xyz_measure):
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
            # logger.debug(f"No valid depth points in bounding box {bbox}") # Can be too verbose
            return None
        
        median_coord = tuple(np.median(patch[mask], axis=0))
        return median_coord

    def _send_all(self, sock, data):
        original_timeout = sock.gettimeout()
        try:
            sock.settimeout(self.send_operation_timeout) # Use configured timeout
            sock.sendall(data)
            return True
        except socket.timeout:
            logger.warning(f"Socket timeout during sendall after {self.send_operation_timeout}s. Client may be unresponsive or network issue.")
            return False
        except socket.error as e:
            if e.errno == socket.errno.EPIPE or e.errno == socket.errno.ECONNRESET:
                logger.warning(f"Socket error during send (client disconnected?): {e}")
            else:
                logger.error(f"Socket error during send: {e}")
            return False
        finally:
            sock.settimeout(original_timeout) # Restore original timeout

    def _receive_all(self, sock, n):
        data_buffer = bytearray()
        # Using sock.gettimeout() to respect existing timeout if set by accept loop
        # For individual recv calls, a specific timeout might be needed if the global one is too long
        original_timeout = sock.gettimeout()
        sock.settimeout(1.0) # Short timeout for individual recv calls
        
        try:
            start_time_recv_all = time.time()
            while len(data_buffer) < n:
                if time.time() - start_time_recv_all > 5.0: # Overall timeout for receiving n bytes
                    logger.warning(f"Overall timeout receiving {n} bytes.")
                    return None
                if self.stop_event.is_set(): # Check global stop event
                    logger.info("Stop event set, aborting receive_all.")
                    return None
                try:
                    packet = sock.recv(n - len(data_buffer))
                    if not packet:
                        logger.warning("Connection broken while receiving (recv returned empty).")
                        return None
                    data_buffer.extend(packet)
                except socket.timeout: 
                    # logger.debug(f"Socket.recv timed out, retrying for {n} bytes...") # Too verbose
                    if self.stop_event.is_set():
                        logger.info("Stop event set during recv timeout, aborting.")
                        return None
                    continue 
            return data_buffer
        except socket.error as e:
            if e.errno == socket.errno.EPIPE or e.errno == socket.errno.ECONNRESET:
                logger.warning(f"Socket error during receive (client disconnected?): {e}")
            else:
                logger.error(f"Unexpected socket error during receive: {e}")
            return None
        finally:
            sock.settimeout(original_timeout) # Restore original timeout

    def _image_sender_worker(self):
        logger.info("Image sender worker started.")
        temp_image_conn = self.image_conn # Use a local reference
        while not self.image_sender_stop_event.is_set() and not self.stop_event.is_set():
            try:
                jpeg_size, frame_bytes = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if temp_image_conn is None: 
                logger.warning("Image sender: No connection. Stopping.")
                break
            try:
                if not self._send_all(temp_image_conn, struct.pack('>I', jpeg_size)):
                    logger.error("Image sender: Failed to send image size. Closing connection.")
                    break
                if jpeg_size > 0 and frame_bytes:
                    if not self._send_all(temp_image_conn, frame_bytes):
                        logger.error("Image sender: Failed to send image data. Closing connection.")
                        break
                self.image_queue.task_done()
            except Exception as e:
                logger.error(f"Image sender: Error: {e}. Exiting.")
                break
        
        logger.info("Image sender worker finished.")
        if temp_image_conn:
            try: temp_image_conn.close()
            except Exception as e_close: logger.debug(f"Exception closing image_conn in sender: {e_close}")
        # No need to set self.image_conn to None here, main loop handles that

    def _bbox_receiver_worker(self):
        logger.info("Bbox receiver worker started.")
        temp_bbox_conn = self.bbox_conn # Use a local reference
        while not self.bbox_receiver_stop_event.is_set() and not self.stop_event.is_set():
            if temp_bbox_conn is None:
                logger.warning("Bbox receiver: No connection. Stopping.")
                break
            try:
                # Set a short timeout for this specific operation to allow periodic checks of stop_event
                original_timeout = temp_bbox_conn.gettimeout()
                temp_bbox_conn.settimeout(0.1) 
                
                bbox_size_data = self._receive_all(temp_bbox_conn, 4)
                temp_bbox_conn.settimeout(original_timeout) # Restore

                if bbox_size_data is None:
                    if self.bbox_receiver_stop_event.is_set() or self.stop_event.is_set(): break
                    time.sleep(0.05) 
                    continue

                bbox_size = struct.unpack('>I', bbox_size_data)[0]
                if bbox_size == 0: 
                    with self.bbox_lock:
                        self.last_known_bbox_shared['bbox'] = None
                    with self.latest_object_xy_lock: # Also clear XY if no bbox
                        self.latest_object_xy_shared['xy'] = None
                    continue
                
                if 0 < bbox_size < 100: 
                    original_timeout = temp_bbox_conn.gettimeout()
                    temp_bbox_conn.settimeout(1.0) # Timeout for bbox data
                    bbox_data_bytes = self._receive_all(temp_bbox_conn, bbox_size)
                    temp_bbox_conn.settimeout(original_timeout)

                    if bbox_data_bytes:
                        bbox_str = bbox_data_bytes.decode('utf-8')
                        try:
                            coords_str = bbox_str.split(',')
                            if len(coords_str) == 4:
                                received_bbox_tuple = tuple(map(int, coords_str))
                                with self.bbox_lock:
                                    self.last_known_bbox_shared['bbox'] = received_bbox_tuple
                            else:
                                logger.warning(f"Bbox receiver: Malformed bbox string: {bbox_str}")
                        except ValueError:
                            logger.warning(f"Bbox receiver: Cannot parse bbox string: {bbox_str}")
                    else:
                        if self.bbox_receiver_stop_event.is_set() or self.stop_event.is_set(): break
                        logger.warning("Bbox receiver: Failed to receive bbox data after size or client disconnected.")
                        time.sleep(0.05)
                        continue
                else:
                    logger.warning(f"Bbox receiver: Suspicious bbox_size: {bbox_size}. Ignoring.")
            
            except socket.timeout: 
                continue # Loop again to check stop_event or try receiving again
            except Exception as e:
                logger.error(f"Bbox receiver: Unhandled error: {e}. Exiting.", exc_info=True)
                break
        
        logger.info("Bbox receiver worker finished.")
        if temp_bbox_conn:
            try: temp_bbox_conn.close()
            except Exception as e_close: logger.debug(f"Exception closing bbox_conn in receiver: {e_close}")
        # No need to set self.bbox_conn to None here, main loop handles that

    def _manage_connections(self):
        # --- Manage Image Client Connection ---
        if self.image_conn is None or (self.image_sender_thread and not self.image_sender_thread.is_alive()):
            if self.image_sender_thread and not self.image_sender_thread.is_alive():
                logger.warning("Image sender thread died or finished. Attempting to re-establish...")
                self.image_sender_stop_event.set() # Ensure it's told to stop if stuck
                if self.image_sender_thread.is_alive(): self.image_sender_thread.join(timeout=0.5)
            if self.image_conn:
                try: self.image_conn.close()
                except Exception: pass
                self.image_conn = None
            
            logger.info("Waiting for IMAGE client connection...")
            self.image_server_socket.settimeout(1.0) # Timeout for accept
            try:
                self.image_conn, img_addr = self.image_server_socket.accept()
                logger.info(f"IMAGE client connected from {img_addr}")
                self.image_sender_stop_event.clear() # Clear before starting new thread
                self.image_sender_thread = threading.Thread(target=self._image_sender_worker, name="ImageSenderThread")
                self.image_sender_thread.daemon = True
                self.image_sender_thread.start()
            except socket.timeout:
                pass # Will retry in the next loop iteration
            except Exception as e:
                logger.error(f"Error accepting IMAGE client: {e}")
                time.sleep(0.5) # Wait a bit before retrying accept

        # --- Manage Bbox Client Connection ---
        if self.bbox_conn is None or (self.bbox_receiver_thread and not self.bbox_receiver_thread.is_alive()):
            if self.bbox_receiver_thread and not self.bbox_receiver_thread.is_alive():
                logger.warning("Bbox receiver thread died or finished. Attempting to re-establish...")
                self.bbox_receiver_stop_event.set()
                if self.bbox_receiver_thread.is_alive(): self.bbox_receiver_thread.join(timeout=0.5)
            if self.bbox_conn:
                try: self.bbox_conn.close()
                except Exception: pass
                self.bbox_conn = None

            logger.info("Waiting for BBOX client connection...")
            self.bbox_server_socket.settimeout(1.0) # Timeout for accept
            try:
                self.bbox_conn, bbox_addr = self.bbox_server_socket.accept()
                logger.info(f"BBOX client connected from {bbox_addr}")
                self.bbox_receiver_stop_event.clear()
                self.bbox_receiver_thread = threading.Thread(target=self._bbox_receiver_worker, name="BboxReceiverThread")
                self.bbox_receiver_thread.daemon = True
                self.bbox_receiver_thread.start()
            except socket.timeout:
                pass
            except Exception as e:
                logger.error(f"Error accepting BBOX client: {e}")
                time.sleep(0.5)

    def _run_zed_loop(self):
        logger.info("ZED processing loop started.")
        fps_reporting_interval = self.camera_fps * 2 # Report every 2 seconds approx
        frame_count = 0
        start_time_fps = time.time()

        while not self.stop_event.is_set():
            self._manage_connections()

            if not (self.image_conn and self.image_sender_thread and self.image_sender_thread.is_alive()):
                time.sleep(0.1) # Wait if no image client to send to, or thread not running
                continue

            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                frame_process_start_time = time.time() # Start timing for frame processing

                self.zed.retrieve_image(self.img_mat, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZ)

                current_bbox_local = None
                with self.bbox_lock:
                    current_bbox_local = self.last_known_bbox_shared['bbox']
                
                if current_bbox_local:
                    depth_coords = self._bbox_center_xyz(current_bbox_local, self.xyz_mat)
                    if depth_coords:
                        # logger.info(f"Depth coordinates: {depth_coords}") # Logged if needed for debug
                        with self.latest_object_xy_lock:
                            self.latest_object_xy_shared['xy'] = (depth_coords[0], depth_coords[1])
                        # New: Store pixel cx and image width if bbox was processed
                        pixel_center_x = (current_bbox_local[0] + current_bbox_local[2]) / 2.0
                        img_width = self.img_mat.get_width() # Get current image width
                        with self.latest_visual_info_lock:
                            self.latest_visual_info_shared['pixel_cx'] = pixel_center_x
                            self.latest_visual_info_shared['image_width'] = img_width
                            self.latest_visual_info_shared['bbox_available'] = True
                    else: # No valid depth, clear stored XY and visual info if it relied on this bbox
                        with self.latest_object_xy_lock:
                            self.latest_object_xy_shared['xy'] = None
                        with self.latest_visual_info_lock:
                            self.latest_visual_info_shared['bbox_available'] = False
                            self.latest_visual_info_shared['pixel_cx'] = None 
                            # image_width could be kept if img_mat is valid, but None if dependent on bbox success
                            # For simplicity, let's clear it too, or keep last valid one?
                            # Let's keep it if available from self.img_mat if we assume img_mat is always fresh.
                            if self.img_mat and self.img_mat.is_init():
                                self.latest_visual_info_shared['image_width'] = self.img_mat.get_width()
                            else:
                                self.latest_visual_info_shared['image_width'] = None
                else: # No bbox, clear stored XY and visual info
                    with self.latest_object_xy_lock:
                        self.latest_object_xy_shared['xy'] = None
                    with self.latest_visual_info_lock:
                        self.latest_visual_info_shared['bbox_available'] = False
                        self.latest_visual_info_shared['pixel_cx'] = None
                        # Keep image_width if img_mat is valid
                        if self.img_mat and self.img_mat.is_init():
                             self.latest_visual_info_shared['image_width'] = self.img_mat.get_width()
                        else:
                            self.latest_visual_info_shared['image_width'] = None


                bgr_image_cpu = self.img_mat.get_data()
                rgb_image_cpu = cv2.cvtColor(bgr_image_cpu, cv2.COLOR_BGRA2BGR)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40] # Quality 40 for less bandwidth
                result, frame_jpeg = cv2.imencode('.jpg', rgb_image_cpu, encode_param)

                frame_process_duration = time.time() - frame_process_start_time

                if frame_process_duration > self.max_frame_processing_time:
                    logger.warning(f"Frame processing ({frame_process_duration:.3f}s) exceeded max time ({self.max_frame_processing_time}s). Skipping frame.")
                elif result:
                    try:
                        self.image_queue.put_nowait((len(frame_jpeg), frame_jpeg.tobytes()))
                    except queue.Full:
                        try:
                            self.image_queue.get_nowait() # Clear one old item
                            self.image_queue.put_nowait((len(frame_jpeg), frame_jpeg.tobytes()))
                        except queue.Empty:
                            pass
                        except queue.Full:
                            logger.warning("IMAGE_QUEUE still full after clearing one item, frame dropped.")
                elif not result: # Explicitly check for result being False if not skipped
                    logger.error("JPEG encoding failed.")
                    try:
                        self.image_queue.put_nowait((0, b'')) # Signal encoding error
                    except queue.Full:
                         logger.warning("IMAGE_QUEUE full, failed to send JPEG encoding error signal.")
                
                frame_count += 1
                if frame_count >= fps_reporting_interval:
                    elapsed_time = time.time() - start_time_fps
                    if elapsed_time > 0:
                        fps = frame_count / elapsed_time
                        logger.debug(f"ZED loop frame production rate: {fps:.2f} FPS (to IMAGE_QUEUE)")
                    frame_count = 0
                    start_time_fps = time.time()
            else:
                # logger.warning("ZED grab failed.") # Can be too verbose if temporary
                time.sleep(0.01) # Brief pause if grab fails
        
        logger.info("ZED processing loop finished.")

    def start(self):
        logger.info("Starting PositionDetector...")
        try:
            self._open_camera()
            self._setup_sockets()
            
            self.stop_event.clear() # Ensure stop event is clear before starting threads
            self.image_sender_stop_event.clear()
            self.bbox_receiver_stop_event.clear()

            # The ZED processing loop (which includes connection management) runs in its own thread
            self.zed_processing_thread = threading.Thread(target=self._run_zed_loop, name="ZEDLoopThread")
            self.zed_processing_thread.daemon = True
            self.zed_processing_thread.start()
            logger.info("PositionDetector started successfully. ZED loop and server setups initiated.")
        except Exception as e:
            logger.error(f"Failed to start PositionDetector: {e}", exc_info=True)
            self.shutdown() # Attempt to clean up if start fails
            raise # Re-raise the exception

    def get_current_object_xy(self):
        """Returns the latest calculated (X, Y) coordinates of the object or None."""
        with self.latest_object_xy_lock:
            xy = self.latest_object_xy_shared['xy']
        # logger.debug(f"get_current_object_xy returning: {xy}") # For debugging if needed
        return xy

    def get_current_visual_info(self):
        """Returns the latest calculated (pixel_cx, image_width, bbox_available) or None for values if not available."""
        with self.latest_visual_info_lock:
            info = self.latest_visual_info_shared.copy() # Return a copy
        # logger.debug(f"get_current_visual_info returning: {info}")
        return info

    def shutdown(self):
        logger.info("Shutting down PositionDetector...")
        self.stop_event.set() # Signal all loops and threads to stop

        # Signal individual worker threads to stop as well
        self.image_sender_stop_event.set()
        self.bbox_receiver_stop_event.set()

        threads_to_join = []
        if self.zed_processing_thread and self.zed_processing_thread.is_alive():
            threads_to_join.append(self.zed_processing_thread)
        # The sender/receiver threads are joined by _manage_connections or implicitly when their conn is None
        # However, if they were started and _manage_connections isn't running due to shutdown,
        # or if they are stuck, explicitly joining them here is safer.
        if self.image_sender_thread and self.image_sender_thread.is_alive():
             # logger.debug("Attempting to join image sender thread from shutdown...")
             threads_to_join.append(self.image_sender_thread)
        if self.bbox_receiver_thread and self.bbox_receiver_thread.is_alive():
             # logger.debug("Attempting to join bbox receiver thread from shutdown...")
             threads_to_join.append(self.bbox_receiver_thread)

        for thread in threads_to_join:
            try:
                logger.info(f"Joining thread: {thread.name}")
                thread.join(timeout=1.0) # Increased timeout slightly
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not terminate in time.")
            except Exception as e:
                logger.error(f"Error joining thread {thread.name}: {e}")
        
        # Close client connections
        if self.image_conn:
            try: self.image_conn.close()
            except Exception as e: logger.debug(f"Exception closing image_conn: {e}")
            self.image_conn = None
        if self.bbox_conn:
            try: self.bbox_conn.close()
            except Exception as e: logger.debug(f"Exception closing bbox_conn: {e}")
            self.bbox_conn = None
        
        # Close server sockets
        if self.image_server_socket:
            try: self.image_server_socket.close()
            except Exception as e: logger.debug(f"Exception closing image_server_socket: {e}")
            self.image_server_socket = None
        if self.bbox_server_socket:
            try: self.bbox_server_socket.close()
            except Exception as e: logger.debug(f"Exception closing bbox_server_socket: {e}")
            self.bbox_server_socket = None
        
        if self.zed and self.zed.is_opened():
            logger.info("Closing ZED camera...")
            self.zed.close()
            logger.info("ZED camera closed.")
        
        logger.info("PositionDetector shutdown complete.")

# Main execution block for running pos_detect.py standalone
if __name__ == "__main__":
    logger.info("Starting pos_detect.py as standalone script.")
    detector = None
    try:
        # Example: Pass port numbers from command line if needed, or use defaults
        img_port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PORT
        box_port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BBOX_PORT
        
        detector = PositionDetector(image_port=img_port, bbox_port=box_port)
        detector.start()
        
        # Keep the main thread alive, periodically checking for external stop or KeyboardInterrupt
        # And also print detected XY for standalone testing
        while True:
            time.sleep(1)
            xy = detector.get_current_object_xy()
            if xy:
                logger.info(f"Standalone Mode - Detected Object XY: ({xy[0]:.3f}, {xy[1]:.3f}) meters")
            if detector.stop_event.is_set(): # If something internal signaled stop
                logger.info("Stop event detected in main, exiting standalone.")
                break
                
    except KeyboardInterrupt:
        logger.info("\nUser interrupted (KeyboardInterrupt). Shutting down...")
    except Exception as e:
        logger.error(f"Unhandled error in standalone main: {e}", exc_info=True)
    finally:
        if detector:
            logger.info("Initiating shutdown sequence from standalone finally block...")
            detector.shutdown()
        logger.info("Standalone script finished.")
