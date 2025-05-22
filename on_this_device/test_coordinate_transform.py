#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_coordinate_transform.py

Tests the conversion of relative object coordinates (from PositionDetector)
to global coordinates using the robot's current pose.

Instructions:
1. Ensure the ZED camera is connected.
2. This script will start PositionDetector.
3. An external object detection client MUST connect to PositionDetector:
   - Image server: default port 12345
   - Bbox server: default port 12346
   The client should send bounding boxes of the target object.
4. Run this script with the network interface as an argument:
   python3 test_coordinate_transform.py <network_interface>
   (e.g., python3 test_coordinate_transform.py eth0)
5. Type "start" in the terminal to begin printing coordinates.
6. Press Ctrl+C to stop.
"""

import sys
import time
import math
import logging
import threading

# Assuming pos_detect.py is in the same directory or Python path
from pos_detect import PositionDetector, DEFAULT_IMAGE_PORT, DEFAULT_BBOX_PORT

# Unitree SDK imports
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    # SportClient might not be strictly needed if we only read state,
    # but often state subscription is tied to its lifecycle or general SDK init.
    from unitree_sdk2py.go2.sport.sport_client import SportClient 
except ImportError as e:
    print(f"Error importing Unitree SDK components: {e}")
    print("Please ensure the Unitree SDK for Python is installed and accessible.")
    sys.exit(1)


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store robot state
robot_state_data = {"state": None, "lock": threading.Lock()}

def HighStateHandler(msg: SportModeState_):
    """Callback function to update global robot state."""
    with robot_state_data["lock"]:
        robot_state_data["state"] = msg
    # logger.debug(f"Received SportModeState: Pos(x:{msg.position[0]:.2f}, y:{msg.position[1]:.2f}), Yaw:{msg.imu_state.rpy[2]:.2f}")

def convert_relative_to_global(relative_coord_robot_frame, current_robot_px, current_robot_py, current_robot_yaw):
    """
    Converts coordinates relative to the robot's current position AND ORIENTATION 
    to global coordinates.
    relative_coord_robot_frame[0] is distance forward along robot's current heading (X_cam).
    relative_coord_robot_frame[1] is distance to the left of the robot's current heading (Y_cam).
    """
    relative_forward = relative_coord_robot_frame[0]
    relative_left = relative_coord_robot_frame[1]
    
    # Rotate the relative coordinates to align with the global frame
    offset_x_global = relative_forward * math.cos(current_robot_yaw) - relative_left * math.sin(current_robot_yaw)
    offset_y_global = relative_forward * math.sin(current_robot_yaw) + relative_left * math.cos(current_robot_yaw)
    
    # Add the global offset to the robot's current global position
    global_x = current_robot_px + offset_x_global
    global_y = current_robot_py + offset_y_global
    
    return global_x, global_y

def main():
    if len(sys.argv) < 2:
        logger.error(f"Usage: python3 {sys.argv[0]} <network_interface>")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    logger.info(f"Using network interface: {network_interface}")

    pos_detector = None
    sport_client = None # Keep a reference for potential cleanup
    state_sub = None

    try:
        # Initialize Unitree SDK ChannelFactory
        logger.info("Initializing Unitree SDK ChannelFactory...")
        ChannelFactoryInitialize(0, network_interface)
        logger.info("ChannelFactory initialized.")

        # Initialize PositionDetector
        logger.info("Initializing PositionDetector...")
        # Using default ports from pos_detect.py
        pos_detector = PositionDetector(image_port=DEFAULT_IMAGE_PORT, bbox_port=DEFAULT_BBOX_PORT)
        pos_detector.start() # This starts the ZED camera and socket servers
        logger.info("PositionDetector started.")

        # Initialize SportClient (primarily for state subscription context)
        # Some SDK patterns might require SportClient to be active for state messages.
        try:
            sport_client = SportClient()
            sport_client.Init() # Minimal init
            logger.info("SportClient initialized (for state subscription context).")
        except Exception as e:
            logger.warning(f"Could not initialize SportClient: {e}. State updates might not work.")
            # Depending on SDK, state might still work if channel factory is up.

        # Subscribe to robot state
        logger.info("Subscribing to robot sport mode state...")
        state_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        state_sub.Init(HighStateHandler, 10) # Adjust queue size as needed
        logger.info("Subscribed to robot state. Waiting for first state message...")

        # Wait a bit for the first state message to arrive
        time.sleep(2) 
        with robot_state_data["lock"]:
            if robot_state_data["state"] is None:
                logger.warning("Did not receive robot state after 2 seconds. Global coordinate conversion will be impacted.")
            else:
                logger.info("Initial robot state received.")

        print() # Print a newline first
        print("="*50)
        print("Test Coordinate Transformation Script")
        print("Ensure an external client is sending bounding boxes to PositionDetector.")
        print(f"Image server on port: {pos_detector.image_port}, Bbox server on port: {pos_detector.bbox_port}")
        print("Type 'start' and press Enter to begin printing coordinates.")
        print("Press Ctrl+C to exit.")
        print("="*50)
        print() # Print a newline at the end

        while True:
            user_input = input("> ").strip().lower()
            if user_input == "start":
                break
            else:
                logger.info("Type 'start' to begin.")
        
        logger.info("Starting continuous coordinate reporting...")
        running = True
        while running:
            try:
                relative_xy = pos_detector.get_current_object_xy()
                
                current_px, current_py, current_yaw = None, None, None
                with robot_state_data["lock"]:
                    if robot_state_data["state"]:
                        current_px = robot_state_data["state"].position[0]
                        current_py = robot_state_data["state"].position[1]
                        current_yaw = robot_state_data["state"].imu_state.rpy[2] # Yaw in radians

                if relative_xy:
                    if current_px is not None: # Implies current_py and current_yaw are also set
                        global_x, global_y = convert_relative_to_global(relative_xy, current_px, current_py, current_yaw)
                        
                        print(f"Relative Coords (Cam X_FWD, Y_LEFT): ({relative_xy[0]:.3f}, {relative_xy[1]:.3f}) | "
                              f"Robot Pose (Global X, Y, Yaw_rad): ({current_px:.3f}, {current_py:.3f}, {current_yaw:.3f}) | "
                              f"==> Calculated Object Global Coords: ({global_x:.3f}, {global_y:.3f})")
                    else:
                        print(f"Relative Coords (Cam X_FWD, Y_LEFT): ({relative_xy[0]:.3f}, {relative_xy[1]:.3f}) | "
                              f"Robot Pose: Not yet available. Cannot calculate global coordinates.")
                else:
                    # This will be frequent if no bbox client is connected or no object detected
                    # print("No relative coordinates detected by PositionDetector.") 
                    pass 

                time.sleep(0.2) # Adjust reporting frequency as needed

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Stopping...")
                running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                running = False # Stop on other errors as well

    except KeyboardInterrupt:
        logger.info("User interrupted program start-up.")
    except Exception as e:
        logger.error(f"An error occurred during initialization: {e}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        if pos_detector:
            logger.info("Shutting down PositionDetector...")
            pos_detector.shutdown()
            logger.info("PositionDetector shut down.")
        
        # Note: SportClient and ChannelSubscriber in unitree_sdk2py
        # might not have explicit shutdown/close methods, or rely on
        # garbage collection or ChannelFactory deinitialization (if any).
        # If SportClient was used for commands, a StopMove() might be relevant here.
        if sport_client:
             logger.info("SportClient instance will be cleaned up.")
        if state_sub:
            logger.info("ChannelSubscriber instance will be cleaned up.")

        logger.info("Test script finished.")

if __name__ == "__main__":
    main() 