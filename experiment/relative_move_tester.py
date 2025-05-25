import time
import math
import sys
import logging
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# --- Global Robot State ---
robot_state = None

def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg
    # Optional: print basic state info for debugging
    # logger.debug(f"Robot State Updated: Pos=({msg.position[0]:.2f}, {msg.position[1]:.2f}), Yaw={msg.imu_state.rpy[2]:.2f}")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (RelMoveTest) %(message)s')
logger = logging.getLogger(__name__)

def normalize_angle(angle_rad):
    """Normalize an angle in radians to [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad

class RelativeMoveTester:
    def __init__(self, sport_client):
        self.sport_client = sport_client

    def move_relative(self, rel_x_robot, rel_y_robot, rel_yaw_degrees):
        """
        Moves the robot to a position relative to its current pose.
        rel_x_robot: desired forward/backward movement in robot's frame (meters).
        rel_y_robot: desired left/right movement in robot's frame (meters, positive is left).
        rel_yaw_degrees: desired change in yaw relative to current yaw (degrees).
        """
        if robot_state is None:
            logger.error("Robot state not available. Cannot execute relative move.")
            return False

        current_global_x = robot_state.position[0]
        current_global_y = robot_state.position[1]
        current_global_yaw_rad = robot_state.imu_state.rpy[2]
        
        rel_yaw_rad = math.radians(rel_yaw_degrees)

        logger.info(f"Current Global Pose: X={current_global_x:.3f}m, Y={current_global_y:.3f}m, Yaw={math.degrees(current_global_yaw_rad):.2f}°")
        logger.info(f"Relative Input: dX_robot={rel_x_robot:.3f}m, dY_robot={rel_y_robot:.3f}m, dYaw={rel_yaw_degrees:.2f}°")

        # Calculate global offset based on relative movement in robot's frame
        # rel_x_robot is along the robot's current heading (robot's local X)
        # rel_y_robot is to the robot's left (robot's local Y)
        delta_global_x = rel_x_robot * math.cos(current_global_yaw_rad) - rel_y_robot * math.sin(current_global_yaw_rad)
        delta_global_y = rel_x_robot * math.sin(current_global_yaw_rad) + rel_y_robot * math.cos(current_global_yaw_rad)

        # Calculate absolute global target position
        target_global_x = current_global_x + delta_global_x
        target_global_y = current_global_y + delta_global_y

        # Calculate absolute global target yaw
        target_global_yaw_rad = normalize_angle(current_global_yaw_rad + rel_yaw_rad)

        logger.info(f"Calculated Global Target: X={target_global_x:.3f}m, Y={target_global_y:.3f}m, Yaw={math.degrees(target_global_yaw_rad):.2f}° ({target_global_yaw_rad:.3f} rad)")

        try:
            logger.info(f"Executing MoveToPos...")
            ret = self.sport_client.MoveToPos(target_global_x, target_global_y, target_global_yaw_rad)
            if ret == 0:
                logger.info("MoveToPos command sent successfully.")
                # Note: MoveToPos is non-blocking. The robot will start moving.
                # Add a delay here if you want to wait for the movement to roughly complete.
                # The duration depends on the magnitude of the relative move.
                time.sleep(3.0) # Example: wait 3 seconds
                logger.info("Assumed movement initiated/completed after delay.")
                return True
            else:
                logger.error(f"MoveToPos command failed with error code: {ret}")
                return False
        except Exception as e:
            logger.error(f"Error during MoveToPos execution: {e}")
            return False

def main():
    if len(sys.argv) < 2:
        logger.error(f"Usage: python3 {sys.argv[0]} networkInterface")
        print(f"Example: python3 {sys.argv[0]} eth0")
        sys.exit(-1)

    ChannelFactoryInitialize(0, sys.argv[1])
    sport_client = SportClient()
    sport_client.Init()

    # Subscribe to robot state
    # Adjust topic name and message type if different for your SDK version or robot
    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_) 
    sub.Init(HighStateHandler, 10) # Handler, interval_ms
    
    logger.info("Waiting for initial robot state (approx 2 seconds)...")
    time.sleep(2) # Give some time for the subscriber to receive the first message
    if robot_state is None:
        logger.warning("Failed to receive initial robot state after 2s. Proceeding, but moves may fail if state remains unknown.")

    tester = RelativeMoveTester(sport_client)

    try:
        while True:
            if robot_state is None:
                logger.warning("Robot state is still None. Waiting...")
                time.sleep(1)
                continue
            
            print("\n------------------------------------")
            print(f"Current Est. Global Pose: X={robot_state.position[0]:.2f}m, Y={robot_state.position[1]:.2f}m, Yaw={math.degrees(robot_state.imu_state.rpy[2]):.1f}°")
            print("Enter relative movement commands (or 'q' to quit any input):")
            
            try:
                rel_x_str = input("  Relative X (forward/backward, meters): ").strip()
                if rel_x_str.lower() == 'q': break
                rel_x = float(rel_x_str)

                rel_y_str = input("  Relative Y (left/right, meters, +ve left): ").strip()
                if rel_y_str.lower() == 'q': break
                rel_y = float(rel_y_str)
                
                rel_yaw_str = input("  Relative Yaw (degrees, +ve counter-clockwise/left): ").strip()
                if rel_yaw_str.lower() == 'q': break
                rel_yaw_degrees = float(rel_yaw_str)

                if not tester.move_relative(rel_x, rel_y, rel_yaw_degrees):
                    logger.warning("Relative move execution failed or was NOP.")

            except ValueError:
                logger.warning("Invalid input. Please enter numeric values or 'q'.")
            except Exception as e:
                logger.error(f"An error occurred in the input loop: {e}")

    except KeyboardInterrupt:
        logger.info("\nUser interrupted. Exiting program.")
    finally:
        logger.info("Cleaning up: Sending StopMove command...")
        if 'sport_client' in locals() and sport_client:
            sport_client.StopMove()
        # Consider if subscriber needs explicit closing: sub.Close()
        logger.info("Program terminated.")

if __name__ == "__main__":
    main() 