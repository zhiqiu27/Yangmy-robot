import time
import math
import enum
import sys
import logging
import numpy as np
import serial # Added import for serial communication
from simple_pid import PID # Added import for simple-pid
import rospy  # Added for ROS
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from robot_vision_msgs.srv import CalibrationSrv  # Custom service
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from pitch import PitchController
import threading

# 配置日志（输出到终端和文件）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/home/unitree/robot_controller.log'),
        logging.StreamHandler()
    ]
)

# 模拟目标列表（全局坐标）
# "coord" for "shirt" will be updated after calibration
target_list = [
    {"name": "shirt", "coord": [0.0, 0.0]}, # Initial placeholder, will be calibrated
    {"name": "person", "coord": [-1.2, 0]}  # Assuming this is relative for now, as per original logic
]

# 状态定义
class State(enum.Enum):
    IDLE = 0
    CALIBRATE_SHIRT_POSITION = 8 # New state for calibration
    NAVIGATE_TO_OBJECT = 1
    LOWER_HEAD = 2
    GRASP_OBJECT = 3
    SEND_NEXT_COMMAND = 4
    NAVIGATE_TO_PERSON = 5
    RELEASE_OBJECT = 6
    DONE = 7
    WAIT = 9  # Renumbering WAIT to avoid conflict (was 8)


# 机器人状态（全局变量，初始为 None）
robot_state = None

def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg

# Helper function to send hex data to serial port
def send_hex_to_serial(hex_data, port='/dev/wheeltec_mic', baudrate=115200):
    logger = logging.getLogger(__name__)
    try:
        # Configure serial port
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1
        )

        # Ensure serial port is open
        if ser.is_open:
            logger.info(f"Connected to {ser.name}")

            # Send hex data
            ser.write(hex_data)
            logger.info(f"Sent hex data: {hex_data.hex(' ')}")

            # Wait for a moment to ensure data is sent
            time.sleep(0.1)

        # Close serial port
        ser.close()
        return True
    except serial.SerialException as e:
        logger.error(f"Serial port error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error sending hex data: {e}")
        return False

def transform_global_xy_to_robot_xy(global_xy, robot_xy, yaw):
    robot_x = robot_xy[0]
    robot_y = robot_xy[1]
    global_x = global_xy[0]
    global_y = global_xy[1]
    target_from_go1_xyz = [global_x - robot_x, global_y - robot_y]
    global_x_in_robot = target_from_go1_xyz[0] * np.cos(yaw) + target_from_go1_xyz[1] * np.sin(yaw)
    global_y_in_robot = - target_from_go1_xyz[0] * np.sin(yaw) + target_from_go1_xyz[1] * np.cos(yaw)
    return np.array([global_x_in_robot, global_y_in_robot])

class RobotController:
    CLAW_OFFSET_FORWARD = 0.8 # Placeholder: distance from robot center to claw tip along robot's forward axis (meters). PLEASE MEASURE AND ADJUST.
    CALIBRATION_SAMPLES = 5 # Number of samples to average for calibration
    CALIBRATION_DELAY = 0.5 # Seconds between calibration samples

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化 RobotController")
        
        # Initialize ROS node
        try:
            rospy.init_node('robot_controller', anonymous=False)
            self.logger.info("ROS node initialized")
            
            # Subscribe to position detection topics
            self.object_position = None
            self.object_detected = False
            self.position_lock = threading.Lock()
            
            rospy.Subscriber('/vision/object_position', Point, self.position_callback)
            rospy.Subscriber('/vision/detection_status', Bool, self.detection_callback)
            
            # Wait for services
            self.logger.info("Waiting for vision services...")
            try:
                rospy.wait_for_service('/vision/calibrate_object', timeout=5.0)
                self.calibrate_service = rospy.ServiceProxy('/vision/calibrate_object', CalibrationSrv)
                rospy.wait_for_service('/vision/reset', timeout=5.0)
                self.reset_service = rospy.ServiceProxy('/vision/reset', Trigger)
                self.logger.info("Vision services found")
            except rospy.ROSException:
                self.logger.warning("Timeout waiting for vision services. Will attempt to continue.")
            except Exception as e:
                self.logger.error(f"Error connecting to vision services: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS: {e}")
            # Allow to continue, as ROS might not be critical for all operations
        
        # 初始化 SportClient
        try:
            self.sport_client = SportClient()
            self.sport_client.Init()
            self.logger.info("SportClient 初始化成功")
        except Exception as e:
            self.logger.error(f"SportClient 初始化失败: {e}")
            raise RuntimeError(f"SportClient 初始化失败: {e}")
        
        # 初始化状态订阅
        try:
            self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self.sub.Init(HighStateHandler, 20)
            time.sleep(2) # Allow time for first message
            if robot_state is None:
                self.logger.error("未能接收到机器人状态")
                # Allow to continue for now, some operations might not need immediate state
                # raise RuntimeError("状态订阅失败：无数据") 
            self.logger.info("状态订阅初始化成功")
        except Exception as e:
            self.logger.error(f"状态订阅初始化失败: {e}")
            raise RuntimeError(f"状态订阅初始化失败: {e}")

        self.state = State.IDLE
        self.target_index = 0 # This might need adjustment based on how targets are handled now
        self.current_coord = [0.0, 0.0] # Robot's own global position, updated after nav

        # PID 控制器（优化参数） using simple-pid
        self.yaw_pid = PID(Kp=0.5, Ki=0.02, Kd=0.3, output_limits=(-0.6, 0.6)) # For larger initial turns (Phase 1)
        self.yaw_pid_fine_tune = PID(Kp=0.25, Ki=0.01, Kd=0.15, output_limits=(-0.3, 0.3)) # For gentle adjustments (Phase 2)
        # Setpoint for yaw_pid(s) will be set dynamically before use.
        
        self.dist_pid = PID(Kp=0.25, Ki=0.0, Kd=0.25, setpoint=0, output_limits=(0.0, 0.15))
        # For dist_pid, setpoint is 0 (target distance is 0). Input will be -distance.

        # 初始化 PitchController
        self.pitch_controller = PitchController(self.sport_client, interpolation_duration_s=2.0)
        self.logger.info("PitchController initialized.")
        # 关键点：启动 PitchController 的后台线程
        try:
            self.pitch_controller.start_control()
            self.logger.info("PitchController control started.")
        except Exception as e:
            self.logger.error(f"Failed to start PitchController: {e}")
            # 根据您的错误处理策略，这里可能需要 raise e 或者采取其他措施
            
    # ROS Callbacks
    def position_callback(self, msg):
        """Callback for position updates from vision system"""
        with self.position_lock:
            self.object_position = (msg.x, msg.y)
            # self.logger.debug(f"Position update: {self.object_position}")
    
    def detection_callback(self, msg):
        """Callback for detection status updates"""
        with self.position_lock:
            self.object_detected = msg.data
            if not self.object_detected:
                self.object_position = None
                
    def get_current_object_xy(self):
        """Returns the latest detected object position (X, Y) or None"""
        with self.position_lock:
            return self.object_position if self.object_detected else None

    def convert_relative_to_global(self, relative_coord_robot_frame):
        """Converts coordinates relative to the robot's current position AND ORIENTATION to global coordinates.
        relative_coord_robot_frame[0] is distance forward along robot's current heading.
        relative_coord_robot_frame[1] is distance to the left of the robot's current heading.
        """
        if robot_state is None:
            self.logger.error("Robot state not available for coordinate conversion.")
            raise RuntimeError("Robot state not available for coordinate conversion.")
        
        current_robot_px = robot_state.position[0]
        current_robot_py = robot_state.position[1]
        current_robot_yaw = robot_state.imu_state.rpy[2] # Yaw in radians
        
        relative_forward = relative_coord_robot_frame[0]
        relative_left = relative_coord_robot_frame[1]
        
        # Rotate the relative coordinates to align with the global frame
        offset_x_global = relative_forward * math.cos(current_robot_yaw) - relative_left * math.sin(current_robot_yaw)
        offset_y_global = relative_forward * math.sin(current_robot_yaw) + relative_left * math.cos(current_robot_yaw)
        
        # Add the global offset to the robot's current global position
        global_x = current_robot_px + offset_x_global
        global_y = current_robot_py + offset_y_global
        
        self.logger.info(f"Coordinate conversion (robot-oriented): Robot at [{current_robot_px:.2f}, {current_robot_py:.2f}, yaw:{current_robot_yaw:.2f}rad]. Relative (Fwd,Left) {relative_coord_robot_frame} -> Global Offset [{offset_x_global:.2f}, {offset_y_global:.2f}] -> Global Goal [{global_x:.2f}, {global_y:.2f}]")
        return [global_x, global_y] 

    def try_move(self, vx, vy, vyaw, identifier: int):
        #print(f"Calling Move - ID: {identifier}, vx: {vx}, vy: {vy}, vyaw: {vyaw}")
        self.sport_client.Move(vx, vy, vyaw)

    def stop_moving(self, identifier: int = 6):
        """Stops the robot's movement using try_move."""
        self.logger.info(f"Issuing stop command via try_move with ID: {identifier}")
        self.sport_client.StopMove()

    def normalize_angle(self, angle):
        """将角度归一化到 [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _phase1_rotate_to_target(self, global_goal, initial_yaw, yaw_tolerance, max_rot_time, control_period):
        """第一阶段：只旋转朝向目标"""
        self.logger.info(f"导航阶段1: 开始旋转朝向目标 {global_goal}")
        if robot_state is None:
            raise RuntimeError("机器人状态不可用 (阶段1)")
        px = robot_state.position[0]
        py = robot_state.position[1]
        
        goal_body = transform_global_xy_to_robot_xy(global_goal, [px, py], initial_yaw) # Use initial_yaw for consistent target angle
        robot_x, robot_y = goal_body[0], goal_body[1]
        theta = math.atan2(robot_y, robot_x) # Angle in robot's initial frame to the target
        
        # Target yaw is the robot's initial yaw + the angle to the target in the robot's frame
        target_yaw = self.normalize_angle(initial_yaw + theta)
        self.logger.info(f"阶段1 - 当前Yaw: {initial_yaw:.2f}, 目标点相对角度: {theta:.2f}, 计算目标全局Yaw: {target_yaw:.2f} rad")

        start_time = time.time()
        while time.time() - start_time < max_rot_time:
            if robot_state is None:
                raise RuntimeError("机器人状态不可用 (阶段1 循环中)")
            current_yaw = robot_state.imu_state.rpy[2]
            yaw_error = self.normalize_angle(target_yaw - current_yaw)

            if abs(yaw_error) < yaw_tolerance:
                self.logger.info("阶段1: 旋转完成")
                self.sport_client.StopMove() # Stop rotation before proceeding
                return True
            
            self.yaw_pid.setpoint = target_yaw 
            vyaw = self.yaw_pid(current_yaw) # simple-pid computes output based on (setpoint - input)
            
            self.logger.debug(f"阶段1 - 当前Yaw: {current_yaw:.2f}, 目标Yaw: {target_yaw:.2f}, 误差: {yaw_error:.2f}, vyaw: {vyaw:.2f}")
            self.try_move(0.0, 0.0, vyaw, 11) # ID 11 for phase 1 rotation
            time.sleep(control_period)
            
        self.logger.error("阶段1: 旋转超时")
        self.sport_client.StopMove()
        raise RuntimeError("旋转未完成 (阶段1超时)")

    def _phase2_move_and_adjust_to_target(self, global_goal, dist_tolerance, yaw_tolerance, max_nav_time, control_period):
        """第二阶段：边旋转边前进到目标"""
        self.logger.info(f"导航阶段2: 开始前进并调整朝向目标 {global_goal}")
        start_time = time.time()
        while time.time() - start_time < max_nav_time:
            if robot_state is None:
                raise RuntimeError("机器人状态不可用 (阶段2)")
            px = robot_state.position[0]
            py = robot_state.position[1]
            current_yaw = robot_state.imu_state.rpy[2]
            
            goal_body = transform_global_xy_to_robot_xy(global_goal, [px, py], current_yaw)
            robot_x, robot_y = goal_body[0], goal_body[1]
            distance = math.sqrt(robot_x**2 + robot_y**2)
            
            if distance < dist_tolerance:
                self.logger.info("阶段2: 前进完成 (已达目标点)")
                self.sport_client.StopMove() # Stop movement
                return True
            
            # Theta is the angle to the target in the current robot's body frame.
            # We want to drive this angle to 0.
            theta_to_target_in_body = math.atan2(robot_y, robot_x)
            
            # PID for yaw: setpoint is 0 for theta_to_target_in_body.
            # The PID input is theta_to_target_in_body.
            # If theta_to_target_in_body is positive (target to the left), we need positive vyaw.
            # simple-pid output = Kp*(setpoint - input) + ... = Kp*(0 - theta_to_target_in_body)
            # So, to get corrective vyaw, we need vyaw = -pid_output if pid input is theta.
            self.yaw_pid_fine_tune.setpoint = 0 
            vyaw_from_pid = self.yaw_pid_fine_tune(theta_to_target_in_body) # Use fine-tune PID
            vyaw = -vyaw_from_pid if abs(theta_to_target_in_body) > yaw_tolerance else 0.0

            # PID for distance: setpoint is 0. Input is current distance.
            # We want to reduce distance, so vx should be positive.
            # simple-pid output = Kp*(setpoint - input) + ... = Kp*(0 - distance) = -Kp*distance
            # So, to get positive vx, we need vx = -pid_output if pid input is distance.
            # Or, pid_input = -distance, then vx = pid_output.
            self.dist_pid.setpoint = 0
            vx = self.dist_pid(-distance) # Input -distance to get positive output for forward motion
            
            # Limit speed, especially when close, to prevent overshoot
            # This is a simple proportional scaling based on distance, could be more sophisticated
            vx = min(vx, 0.05 + 0.1 * (distance / (dist_tolerance * 5))) # Slower when closer
            vx = max(0.01, vx) # Ensure some minimal movement if not at target

            self.logger.debug(f"阶段2 - 距离: {distance:.2f}m, 目标角度(体): {theta_to_target_in_body:.2f}rad, vx: {vx:.3f}, vyaw: {vyaw:.3f}")
            self.try_move(vx, 0.0, vyaw, 12) # ID 12 for phase 2 navigation
            time.sleep(control_period)
            
        self.logger.error("阶段2: 前进超时")
        self.sport_client.StopMove()
        raise RuntimeError("前进未完成 (阶段2超时)")

    def navigate_to(self, global_goal):
        """导航到全局坐标目标点 [x, y]""" 
        if robot_state is None:
            self.logger.error("Cannot record initial state: Robot state unavailable at start of navigate_to.")
            initial_robot_px, initial_robot_py, initial_robot_yaw = 0.0, 0.0, 0.0
            self.logger.warning("Robot state was None. Using (0,0,0) as initial state, which might be incorrect.")
        else:
            initial_robot_px = robot_state.position[0]
            initial_robot_py = robot_state.position[1]
            initial_robot_yaw = robot_state.imu_state.rpy[2]
        
        self.logger.info(f"Navigate_to started. From: Global [{initial_robot_px:.2f}, {initial_robot_py:.2f}, Yaw:{initial_robot_yaw:.2f}rad] To: Global {global_goal}")

        control_period = 0.02 
        yaw_tolerance = 0.08  # Increased slightly for smoother transition
        dist_tolerance = 0.08 # Reduced slightly for better accuracy
        max_rot_time = 15.0   # Max time for initial rotation
        max_nav_time = 90.0   # Max time for moving to target
        
        try:
            # Reset PIDs before starting a new navigation sequence
            self.yaw_pid.reset()
            self.yaw_pid_fine_tune.reset() # Reset the fine-tune PID as well
            self.dist_pid.reset()

            # 第一阶段：只旋转
            self.logger.info("调用导航阶段1: 旋转...")
            self._phase1_rotate_to_target(global_goal, initial_robot_yaw, yaw_tolerance, max_rot_time, control_period)
            self.logger.info("导航阶段1完成.")
            
            # Wait a bit after rotation before moving
            time.sleep(0.5)

            # 第二阶段：边旋转边前进
            self.logger.info("调用导航阶段2: 前进和调整...")
            self._phase2_move_and_adjust_to_target(global_goal, dist_tolerance, yaw_tolerance, max_nav_time, control_period)
            self.logger.info("导航阶段2完成.")
            
            # Ensure robot is stopped at the end of navigation
            self.sport_client.StopMove()
            self.logger.info("导航成功完成，机器人已停止。")

            # Update current coordinate based on the goal, assuming success
            self.current_coord = global_goal 
            # For more accuracy, could use robot_state.position here, but goal is what we aimed for.
            # If robot_state is available:
            # if robot_state:
            #    self.current_coord = [robot_state.position[0], robot_state.position[1]]
            #    self.logger.info(f"实际到达位置: ({self.current_coord[0]:.2f}, {self.current_coord[1]:.2f})")
            # else:
            #    self.logger.warning("Robot state not available to confirm final position, assuming goal was reached.")
            self.logger.info(f"已到达全局目标 (或导航序列结束) {self.current_coord}")

            achieved_delta_x_global = self.current_coord[0] - initial_robot_px
            achieved_delta_y_global = self.current_coord[1] - initial_robot_py
            achieved_relative_forward = achieved_delta_x_global * math.cos(initial_robot_yaw) + \
                                        achieved_delta_y_global * math.sin(initial_robot_yaw)
            achieved_relative_left = -achieved_delta_x_global * math.sin(initial_robot_yaw) + \
                                       achieved_delta_y_global * math.cos(initial_robot_yaw)
            self.logger.info(f"相对于导航出发点的移动: 前进 {achieved_relative_forward:.2f} 米, 向左 {achieved_relative_left:.2f} 米.")
        
        except RuntimeError as e: # Catch specific RuntimeErrors from phases
            self.logger.error(f"导航失败 (RuntimeError): {e}")
            self.sport_client.StopMove() # Ensure robot is stopped on failure
            raise # Re-raise the exception to be caught by the main loop if necessary
        except Exception as e:
            self.logger.error(f"导航中发生未知错误: {e}")
            self.sport_client.StopMove() # Ensure robot is stopped
            raise RuntimeError(f"导航中发生未知错误: {e}") 

    def lower_head(self, desired_pitch_angle: float = 20.0):
        self.logger.info(f"降低头部，目标俯仰角: {desired_pitch_angle}°")

        if self.pitch_controller:
            self.pitch_controller.set_pitch(desired_pitch_angle)
        else:
            self.logger.warning("PitchController 未在状态机中初始化，无法调整头部俯仰角。")


    def grasp(self):
        self.logger.info("Grabbing object by sending hex command...")
        # Hex data for grasping (assuming data1 from claw.py is for grasping)
        grasp_data = bytes([0x7b, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf8, 0x7d])
        success = send_hex_to_serial(grasp_data)
        if success:
            self.logger.info("Grasp command sent successfully.")
        else:
            self.logger.error("Failed to send grasp command.")
        time.sleep(1.5) # Adding a delay to allow physical grasping
        return success

    def release(self):
        self.logger.info("Releasing object by sending hex command...")
        # Hex data for releasing (assuming data2 from claw.py is for releasing)
        # Using the actual data2 variable content: bytes([0x7b, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf9, 0x7d])
        release_data = bytes([0x7b, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf9, 0x7d])
        success = send_hex_to_serial(release_data)
        if success:
            self.logger.info("Release command sent successfully.")
        else:
            self.logger.error("Failed to send release command.")
        time.sleep(1.0) # Adding a delay to allow physical releasing
        # No return value needed for release as per original, but good practice to return success
        return success

    def send_next_to_vlm(self):
        self.logger.info("向 VLM 发送 'next' (模拟)")
        time.sleep(0.3)

    def get_next_target_from_vlm(self):
        self.logger.info("获取下一个目标坐标")
        time.sleep(0.3)
        return target_list[1]["coord"] 

    def run(self):
        # Start in calibration state if ROS calibration service is available
        try:
            if hasattr(self, 'calibrate_service'):
                self.state = State.CALIBRATE_SHIRT_POSITION
            else:
                self.logger.warning("Vision calibration service not available. Skipping calibration. Using default coordinates.")
                target_list[0]["coord"] = [0.6, 0.0]  # Fallback default
                self.state = State.NAVIGATE_TO_OBJECT
        except Exception as e:
            self.logger.error(f"Error checking ROS services: {e}")
            self.state = State.NAVIGATE_TO_OBJECT  # Fallback to direct navigation

        while self.state != State.DONE and not rospy.is_shutdown():
            try:
                if self.state == State.WAIT:
                    ret2 = self.sport_client.StopMove()  # Ensures robot is stopped in WAIT
                    self.logger.info(f"进入WAIT状态，运动已停止。下一个状态: {getattr(self, '_next_state', 'None')}")
                    time.sleep(1)  # General wait time
                    
                    # General transition from WAIT to _next_state
                    if hasattr(self, '_next_state') and self._next_state is not None:
                        # Only transition if not waiting for specific conditions
                        if not (hasattr(self, '_wait_for_pitch') and self._wait_for_pitch) and \
                           not (hasattr(self, '_wait_for_user_ack_grasp') and self._wait_for_user_ack_grasp):
                            self.state = self._next_state
                            self.logger.info(f"从WAIT状态转换到: {self.state}")
                            self._next_state = None  # Clear after transition
                    continue  # Loop back to re-evaluate WAIT or new state
                
                elif self.state == State.CALIBRATE_SHIRT_POSITION:
                    self.logger.info("进入 CALIBRATE_SHIRT_POSITION 状态")
                    
                    try:
                        # Call the ROS calibration service
                        self.logger.info("调用视觉系统校准服务...")
                        response = self.calibrate_service()
                        
                        if response.success:
                            self.logger.info(f"校准成功: {response.message}")
                            calibrated_shirt_coord_camera_frame = [response.x, response.y]
                            
                            user_input = ""
                            while user_input.lower() != 'ok':
                                user_input = input(f"校准完成。计算得到的'shirt'坐标(相机X,Y): [{response.x:.3f}, {response.y:.3f}]。"
                                                   f"输入 'ok' 确认并使用此坐标，或输入 'retry' 重新校准: ")
                                if user_input.lower() == 'retry':
                                    self.logger.info("用户选择重试校准。")
                                    break  # Break from input loop to restart calibration
                                elif user_input.lower() != 'ok':
                                    self.logger.info("输入无效。")
                            
                            if user_input.lower() == 'ok':
                                # Store the calibrated coordinates
                                target_list[0]["coord"] = calibrated_shirt_coord_camera_frame
                                self.logger.info(f"用户已确认。'shirt' 目标更新为 (机器人相对坐标系): {target_list[0]['coord']}")
                                self._next_state = State.NAVIGATE_TO_OBJECT
                                self.state = State.WAIT
                            elif user_input.lower() == 'retry':
                                continue  # Stay in calibration state
                        else:
                            self.logger.error(f"校准失败: {response.message}")
                            retry = input("校准失败。输入 'retry' 重试，或输入 'skip' 使用默认坐标: ")
                            if retry.lower() == 'retry':
                                continue  # Stay in calibration state
                            else:
                                # Use default coordinates
                                target_list[0]["coord"] = [0.6, 0.0]  # Default fallback
                                self.logger.warning(f"使用默认坐标: {target_list[0]['coord']}")
                                self._next_state = State.NAVIGATE_TO_OBJECT
                                self.state = State.WAIT
                    
                    except rospy.ServiceException as e:
                        self.logger.error(f"调用校准服务失败: {e}")
                        retry = input("校准服务调用失败。输入 'retry' 重试，或输入 'skip' 使用默认坐标: ")
                        if retry.lower() == 'retry':
                            continue  # Stay in calibration state
                        else:
                            # Use default coordinates
                            target_list[0]["coord"] = [0.6, 0.0]  # Default fallback
                            self.logger.warning(f"使用默认坐标: {target_list[0]['coord']}")
                            self._next_state = State.NAVIGATE_TO_OBJECT
                            self.state = State.WAIT
                
                elif self.state == State.NAVIGATE_TO_OBJECT:
                    if robot_state is None:
                        self.logger.error("Robot state not available for offset navigation goal calculation. Waiting.")
                        time.sleep(1)  # Wait for robot state to become available
                        continue  # Re-evaluate state in next loop iteration

                    # target_list[0]["coord"] should now be the calibrated (or default)
                    # relative X, Y for the shirt in the robot's forward/left frame.
                    object_relative_coord_robot_frame = target_list[0]["coord"]
                    
                    self.logger.info(f"NAVIGATE_TO_OBJECT: Current robot relative target for 'shirt': {object_relative_coord_robot_frame}")

                    # Convert this robot-relative coordinate to a global coordinate for navigation
                    object_global_coord = self.convert_relative_to_global(object_relative_coord_robot_frame)
                    self.logger.info(f"Object's target global coordinate (from relative): {object_global_coord}")

                    # Calculate the vector from robot's current position to the object's global position
                    current_robot_px = robot_state.position[0]
                    current_robot_py = robot_state.position[1]
                    
                    delta_x_to_object_global = object_global_coord[0] - current_robot_px
                    delta_y_to_object_global = object_global_coord[1] - current_robot_py
                    
                    approach_yaw_global = math.atan2(delta_y_to_object_global, delta_x_to_object_global)
                    
                    robot_center_target_x = object_global_coord[0] - self.CLAW_OFFSET_FORWARD * math.cos(approach_yaw_global)
                    robot_center_target_y = object_global_coord[1] - self.CLAW_OFFSET_FORWARD * math.sin(approach_yaw_global)
                    
                    adjusted_global_nav_goal_for_robot_center = [robot_center_target_x, robot_center_target_y]
                    
                    self.logger.info(f"NAVIGATE_TO_OBJECT: Original object global (derived from relative): [{object_global_coord[0]:.2f}, {object_global_coord[1]:.2f}]. "
                                     f"Approach Yaw (global): {approach_yaw_global:.2f} rad. "
                                     f"Claw offset: {self.CLAW_OFFSET_FORWARD}m. "
                                     f"Adjusted robot center global target: [{adjusted_global_nav_goal_for_robot_center[0]:.2f}, {adjusted_global_nav_goal_for_robot_center[1]:.2f}]")
                                     
                    self.navigate_to(adjusted_global_nav_goal_for_robot_center)
                    self._next_state = State.LOWER_HEAD
                    self.state = State.WAIT
                
                elif self.state == State.LOWER_HEAD:
                    self.lower_head()  # Target pitch is default 20.0 degrees
                    interpolation_duration = self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0)
                    self.logger.info(f"等待 {interpolation_duration:.1f}s 以完成头部俯仰角调整...")
                    self.logger.info("降低头部命令已发送。")
                    self._next_state = State.GRASP_OBJECT
                    self.state = State.WAIT
                
                elif self.state == State.GRASP_OBJECT:
                    success = self.grasp()
                    if success:
                        self.logger.info("抓取成功")
                        user_ack = ""
                        while user_ack.lower() != 'ok':
                            user_ack = input("物体已抓取。输入 'ok' 使机器人头部恢复至0度并继续: ")
                            if user_ack.lower() != 'ok':
                                self.logger.info("输入无效，请输入 'ok'。")
                        self.logger.info("用户已确认。机器人头部恢复至0度俯仰角...")
                        if self.pitch_controller:
                            self.pitch_controller.reset_pitch()
                            interpolation_duration = self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0)
                            self.logger.info(f"等待 {interpolation_duration:.1f}s 以完成头部归位...")
                            self.logger.info("头部归位完成。")
                        else:
                            self.logger.warning("PitchController 不可用，无法恢复头部姿态。")
                        self._next_state = State.SEND_NEXT_COMMAND  # This will be the state after ack and pitch reset
                        self.state = State.WAIT
                    else:
                        self.logger.warning("抓取失败，重试降低头部")
                        self._next_state = State.LOWER_HEAD
                        self.state = State.WAIT
                
                elif self.state == State.SEND_NEXT_COMMAND:
                    self.send_next_to_vlm()
                    next_coord_relative = self.get_next_target_from_vlm()  # This is relative
                    target_list[1]["coord"] = next_coord_relative  # Update target_list with new relative coord
                    self.logger.info(f"SEND_NEXT_COMMAND: Updated person target (relative): {next_coord_relative}")
                    self._next_state = State.NAVIGATE_TO_PERSON
                    self.state = State.WAIT
                
                elif self.state == State.NAVIGATE_TO_PERSON:
                    # Assuming target_list[1]["coord"] is already in robot's relative X_FWD, Y_LEFT frame
                    relative_coord_person_robot_frame = target_list[1]["coord"] 
                    global_nav_goal_person = self.convert_relative_to_global(relative_coord_person_robot_frame)
                    
                    self.logger.info(f"NAVIGATE_TO_PERSON: Targeting global goal {global_nav_goal_person} (from relative {relative_coord_person_robot_frame})")
                    
                    self.navigate_to(global_nav_goal_person)  # Directly navigate to person's (converted) global position
                    
                    self._next_state = State.RELEASE_OBJECT
                    self.state = State.WAIT
                
                elif self.state == State.RELEASE_OBJECT:
                    self.release()
                    self.logger.info("任务完成")
                    self._next_state = State.DONE
                    self.state = State.WAIT
                
                # Short sleep to prevent tight loop
                if self.state != State.WAIT:
                    time.sleep(0.05)  # Small delay for non-WAIT states

            except rospy.ROSInterruptException:
                self.logger.info("ROS shutdown detected. Stopping robot controller.")
                break
            
            except RuntimeError as e:  # Catch RuntimeErrors from navigation or other critical parts
                self.logger.error(f"运行循环中发生 Runtime错误: {e}", exc_info=True)
                if hasattr(self, 'sport_client'):
                    self.sport_client.StopMove()
                self.logger.info("机器人已停止。程序将终止。")
                break 
            
            except Exception as e:
                self.logger.error(f"运行循环中发生未知错误: {e}", exc_info=True)
                if hasattr(self, 'sport_client'):
                    self.sport_client.StopMove()
                self.logger.info("机器人已停止。程序将终止。")
                break  # Exit on other unhandled exceptions

        # Loop finished (State.DONE or break due to error)
        self.logger.info("RobotController run loop finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)
    
    print("警告：请确保机器人周围没有障碍物。")
    
    controller = None  # Initialize controller to None for finally block
    try:
        ChannelFactoryInitialize(0, sys.argv[1])
        controller = RobotController()
        # User ack for starting can be here if desired
        input("RobotController 初始化完毕。按 Enter 键开始执行状态机...")
        controller.run()
    except rospy.ROSInterruptException:
        logging.info("ROS shutdown detected. Exiting gracefully.")
    except Exception as e:
        logging.critical(f"主程序发生严重错误，程序终止: {e}", exc_info=True)
    finally:
        logging.info("程序开始清理...")
        if controller:
            # Stop robot movement first
            try:
                if hasattr(controller, 'sport_client') and controller.sport_client:
                    logging.info("主清理：正在停止机器人移动...")
                    controller.sport_client.StopMove() 
                    logging.info("主清理：机器人移动已停止。")
            except Exception as e_stop:
                logging.error(f"主清理：停止机器人移动时发生错误: {e_stop}")

            # Shutdown PitchController
            try:
                if hasattr(controller, 'pitch_controller') and controller.pitch_controller:
                    logging.info("主清理：正在停止 PitchController...")
                    controller.pitch_controller.stop_control(transition_to_zero_pitch=True) 
                    logging.info("主清理：PitchController 已停止。")
            except Exception as e_pitch:
                logging.error(f"主清理：停止 PitchController 时发生错误: {e_pitch}")
        
        logging.info("清理完成。程序退出。") 