import time
import math
import enum
import sys
import logging
import numpy as np
import serial # Added import for serial communication
import socket # Added for image server communication
import json   # Added for image server communication
from simple_pid import PID # Added import for simple-pid
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from pitch import PitchController
# from pos_detect import PositionDetector # Import the new PositionDetector # Removed
import collections # Added for deque

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
    {"name": "person", "coord": [-0.6, 0]}  # Assuming this is relative for now, as per original logic
]

# Image Server Configuration
IMAGE_SERVER_HOST = 'localhost' # Should match image_server.py
IMAGE_SERVER_PORT = 50001       # Should match image_server.py

# 状态定义
class State(enum.Enum):
    IDLE = 0
    CALIBRATE_SHIRT_POSITION = 8 # New state for calibration
    NAVIGATE_TO_OBJECT = 1
    FINE_ADJUST_AND_APPROACH = 10 # 新状态
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

MIN_EFFECTIVE_VYAW = 0.08 # rad/s, minimum yaw speed that has an effect on the robot

class RobotController:
    CLAW_OFFSET_FORWARD = 0.8 # Placeholder: distance from robot center to claw tip along robot's forward axis (meters). PLEASE MEASURE AND ADJUST.
    # CALIBRATION_SAMPLES and CALIBRATION_DELAY are now managed by image_server.py
    # CALIBRATION_SAMPLES = 5
    # CALIBRATION_DELAY = 0.5

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化 RobotController")
        
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

        # 初始化 PositionDetector - REMOVED
        # try:
        #     self.pos_detector = PositionDetector() # Using default ports
        #     self.pos_detector.start()
        #     self.logger.info("PositionDetector 初始化并启动成功")
        # except Exception as e:
        #     self.logger.error(f"PositionDetector 初始化失败: {e}")
        #     # Decide if this is critical. For now, let's allow proceeding without it,
        #     # but calibration will fail.
        #     self.pos_detector = None 
        #     # raise RuntimeError(f"PositionDetector 初始化失败: {e}")


        self.state = State.IDLE
        self.target_index = 0 # This might need adjustment based on how targets are handled now
        self.current_coord = [0.0, 0.0] # Robot's own global position, updated after nav

        # PID 控制器（优化参数） using simple-pid
        self.yaw_pid = PID(Kp=0.5, Ki=0.02, Kd=0.3, output_limits=(-0.6, 0.6)) # For larger initial turns (Phase 1)
        self.yaw_pid_fine_tune = PID(Kp=0.25, Ki=0.01, Kd=0.15, output_limits=(-0.3, 0.3)) # For gentle adjustments (Phase 2)
        # Setpoint for yaw_pid(s) will be set dynamically before use.
        
        self.dist_pid = PID(Kp=0.25, Ki=0.0, Kd=0.25, setpoint=0, output_limits=(-0.3, 0.3))
        # For dist_pid, setpoint is 0 (target distance is 0). Input will be -distance.

        # New PID for visual yaw alignment
        self.yaw_pid_visual = PID(Kp=0.001, Ki=0.00005, Kd=0.0005, output_limits=(-0.3, 0.3), setpoint=0) 
        # Setpoint for yaw_pid_visual is 0 (target pixel error is 0)

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
        
        # 高度调整相关常量
        self.BODY_HEIGHT_REL_MIN = -0.18
        self.FOOT_RAISE_REL_MIN = -0.06
        self.DEFAULT_BODY_HEIGHT_ABS = 0.33
        self.DEFAULT_FOOT_RAISE_HEIGHT_ABS = 0.09
        self.current_body_height_abs = self.DEFAULT_BODY_HEIGHT_ABS
        self.current_foot_raise_height_abs = self.DEFAULT_FOOT_RAISE_HEIGHT_ABS
        self.current_relative_body_height = 0.0 # Added to track current relative body height
        self.stage_c_pixel_error_buffer = collections.deque(maxlen=5) # Buffer for averaging pixel error in Stage C

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

    def set_body_height_relative(self, relative_height: float):
        self.logger.info(f"调整机身相对高度为: {relative_height:.3f} m (插值时间1秒)")
        if not (self.BODY_HEIGHT_REL_MIN <= relative_height <= 0.03): # 0.03 is BODY_HEIGHT_REL_MAX
            self.logger.error(f"机身相对高度 {relative_height:.3f} 超出允许范围 [{self.BODY_HEIGHT_REL_MIN}, 0.03]。")
            return False

        interpolation_duration = 1.0  # 1 second
        control_period = 0.02  # 20ms control period, same as navigation
        num_steps = int(interpolation_duration / control_period)
        
        start_relative_height = self.current_relative_body_height
        height_difference = relative_height - start_relative_height

        if num_steps <= 0: # If duration is too short, go directly
            num_steps = 1 
        
        last_sdk_call_successful = True

        for i in range(1, num_steps + 1):
            interpolated_fraction = i / num_steps
            current_target_relative_height = start_relative_height + height_difference * interpolated_fraction
            
            self.logger.debug(f"机身高度插值步骤 {i}/{num_steps}: 目标相对高度 {current_target_relative_height:.4f} m")
            try:
                # We send the command but don't check return code for intermediate steps
                # to avoid flooding logs with success messages or premature failure.
                # The final call outside the loop will determine overall success.
                self.sport_client.BodyHeight(current_target_relative_height) 
            except Exception as e:
                self.logger.error(f"执行 BodyHeight({current_target_relative_height:.4f}) 时发生错误 (插值中): {e}")
                # If an error occurs during interpolation, we might want to stop or handle it.
                # For now, let's log and continue, the final call will be the definitive one.
                last_sdk_call_successful = False # Mark that an error occurred

            time.sleep(control_period)

        # Final call with the exact target height to ensure it's reached
        # and to get the definitive success/failure status.
        self.logger.info(f"机身高度插值完成。发送最终目标相对高度: {relative_height:.3f} m")
        try:
            ret = self.sport_client.BodyHeight(relative_height)
            if ret == 0:
                self.logger.info(f"BodyHeight({relative_height:.3f}) 最终调用成功。")
                self.current_relative_body_height = relative_height # Update current relative height
                self.current_body_height_abs = self.DEFAULT_BODY_HEIGHT_ABS + relative_height
                self.logger.info(f"机器人新的近似绝对机身高度约为: {self.current_body_height_abs:.3f} m")
                return True and last_sdk_call_successful # Return overall success
            else:
                self.logger.error(f"BodyHeight({relative_height:.3f}) 最终调用失败，错误码: {ret}")
                return False
        except Exception as e:
            self.logger.error(f"执行 BodyHeight({relative_height:.3f}) (最终调用) 时发生错误: {e}")
            return False

    def set_foot_raise_height_relative(self, relative_height: float):
        self.logger.info(f"调整抬足相对高度为: {relative_height:.3f} m")
        if not (self.FOOT_RAISE_REL_MIN <= relative_height <= 0.03): # 0.03 is FOOT_RAISE_REL_MAX
            self.logger.error(f"抬足相对高度 {relative_height:.3f} 超出允许范围 [{self.FOOT_RAISE_REL_MIN}, 0.03]。")
            return False
        try:
            ret = self.sport_client.FootRaiseHeight(relative_height)
            if ret == 0:
                self.logger.info(f"FootRaiseHeight({relative_height:.3f}) 调用成功。")
                self.current_foot_raise_height_abs = self.DEFAULT_FOOT_RAISE_HEIGHT_ABS + relative_height
                self.logger.info(f"机器人新的近似绝对抬足高度约为: {self.current_foot_raise_height_abs:.3f} m")
                return True
            else:
                self.logger.error(f"FootRaiseHeight({relative_height:.3f}) 调用失败，错误码: {ret}")
                return False
        except Exception as e:
            self.logger.error(f"执行 FootRaiseHeight 时发生错误: {e}")
            return False

    def reset_body_and_foot_height(self):
        """恢复机身和抬足到默认（相对0）高度"""
        self.logger.info("正在恢复机身和抬足到默认高度...")
        body_reset_ok = self.set_body_height_relative(0.0) # 相对0即为默认绝对高度
        time.sleep(0.5) # 给机器人一点时间调整
        foot_reset_ok = self.set_foot_raise_height_relative(0.0)
        if body_reset_ok and foot_reset_ok:
            self.logger.info("机身和抬足高度已成功恢复到默认。")
            return True
        else:
            self.logger.error("恢复机身或抬足高度失败。")
            return False
            
    # def _final_approach_to_grasp_depth(self, target_grasp_depth=0.3, max_approach_time=150.0, control_period=0.05):
    #     self.logger.info(f"最终接近阶段：前进至目标深度 {target_grasp_depth:.2f}m")
    #     start_time = time.time()
        
    #     final_approach_vx = 0.02  # 非常慢的前进速度 (例如 2 cm/s)
        
    #     self.yaw_pid_visual.reset()

    #     while time.time() - start_time < max_approach_time:
    #         vyaw = 0.0
    #         vx = 0.0
    #         current_depth_x = None
    #         visual_info = self._request_visual_info_from_server()

    #         if visual_info and visual_info.get('bbox_available'):
    #             pixel_cx = visual_info['pixel_cx']
    #             image_width = visual_info['image_width']
    #             current_depth_x = visual_info.get('depth_x')

    #             if current_depth_x is not None:
    #                 self.logger.info(f"最终接近 - 实时深度X: {current_depth_x:.3f} m")
    #                 if current_depth_x < target_grasp_depth:
    #                     self.logger.info(f"最终接近 - 目标抓取深度 {target_grasp_depth:.2f}m 已达到。当前深度: {current_depth_x:.2f}m。机器人停止。")
    #                     self.sport_client.StopMove()
    #                     return True 
    #             else:
    #                 self.logger.info("最终接近 - 实时深度X: 未获取到 (None)。机器人将不前进。")
    #                 self.sport_client.StopMove() 
    #                 # return False # Or retry a few times then return False

    #             target_pixel_x = image_width / 2.0
    #             pixel_error = pixel_cx - target_pixel_x
    #             yaw_pixel_tolerance_final = 3 
    #             if abs(pixel_error) > yaw_pixel_tolerance_final:
    #                 vyaw = self.yaw_pid_visual(pixel_error)
    #                 if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
    #                     vyaw = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                
    #             if current_depth_x is not None and current_depth_x >= target_grasp_depth:
    #                 vx = final_approach_vx 
    #             else:
    #                 vx = 0.0 

    #         elif visual_info and not visual_info.get('bbox_available'):
    #             self.logger.warning("最终接近: 未检测到边界框。机器人将停止移动。")
    #             self.sport_client.StopMove()
    #             return False 
    #         else: 
    #             self.logger.error("最终接近: 从服务器获取视觉信息失败。机器人将停止移动。")
    #             self.sport_client.StopMove()
    #             return False

    #         self.logger.info(f"最终接近 - vx: {vx:.3f}, 视觉vyaw: {vyaw:.3f}, CurDepth: {current_depth_x}, TrgGraspDepth: {target_grasp_depth}")
    #         self.try_move(vx, 0.0, vyaw, 14) 
    #         time.sleep(control_period)

    #     self.logger.error(f"最终接近: 前进至抓取深度 {target_grasp_depth:.2f}m 超时。")
    #     self.sport_client.StopMove()
    #     return False

    # 新方法1：前进到确认深度并进行偏航对准
    def _approach_and_confirm_alignment(self, target_confirm_depth=0.2, target_grasp_depth=0.25, yaw_pixel_tolerance_confirm=1, max_approach_time=120.0, max_align_time_at_depth=100.0, max_crawl_time=100.0, control_period=0.05):
        self.logger.info(f"精细接近与对准流程开始：阶段A目标深度 {target_confirm_depth:.2f}m, 阶段B精确偏航对准 (阶段C已注释)")
        start_process_time = time.time() # Overall time for the entire 3-stage process
        
        approach_vx = 0.02  # 接近时的慢速 (阶段A)
        # crawl_vx = 0.015    # 最后爬行时的极慢速 (阶段C) # No longer used
        
        self.yaw_pid_visual.reset() # 重置视觉偏航PID一次，供整个过程使用
        # self.stage_c_pixel_error_buffer.clear() # Clear the buffer at the start of the process (Stage C removed)

        # 阶段 A: 前进到 target_confirm_depth
        self.logger.info(f"  --- 阶段 A: 前进至深度 {target_confirm_depth:.2f}m 开始 ---")
        current_depth_x_for_logging = None 
        start_stage_A_time = time.time()
        while time.time() - start_stage_A_time < max_approach_time:
            vyaw = 0.0
            vx = 0.0
            current_depth_x = None
            visual_info = self._request_visual_info_from_server()

            if visual_info and visual_info.get('bbox_available'):
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                current_depth_x = visual_info.get('depth_x')
                current_depth_x_for_logging = current_depth_x

                if current_depth_x is not None:
                    self.logger.info(f"  阶段A - 实时深度X: {current_depth_x:.3f} m")
                    if current_depth_x <= target_confirm_depth: 
                        self.logger.info(f"  阶段A: 已到达或小于目标确认深度 {target_confirm_depth:.2f}m (当前: {current_depth_x:.2f}m)。结束阶段A前进。")
                        #self.sport_client.StopMove()
                        break 
                else: 
                    self.logger.info("  阶段A - 实时深度X: 未获取到 (None)。机器人将不前进。")
                    self.sport_client.StopMove()
                    # Consider if this should be an immediate failure of the whole process

                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x
                # Use a slightly more relaxed tolerance for yaw during initial approach to 0.3m
                if abs(pixel_error) > (yaw_pixel_tolerance_confirm + 2): # e.g. 4px if confirm is 2px
                    vyaw = self.yaw_pid_visual(pixel_error)
                    if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
                        vyaw = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                
                if current_depth_x is not None and current_depth_x > target_confirm_depth:
                    vx = approach_vx
                else: 
                    vx = 0.0
            
            elif visual_info and not visual_info.get('bbox_available'):
                self.logger.warning("  阶段A: 未检测到边界框。机器人停止移动。流程失败。")
                self.sport_client.StopMove()
                return False 
            else: # visual_info is None
                self.logger.error("  阶段A: 从服务器获取视觉信息失败。机器人停止移动。流程失败。")
                self.sport_client.StopMove()
                return False

            self.logger.info(f"  阶段A - vx: {vx:.3f}, vyaw: {vyaw:.3f}, CurDepth: {current_depth_x}")
            self.try_move(vx, 0.0, vyaw, 15) 
            time.sleep(control_period)
            
            if time.time() - start_stage_A_time >= max_approach_time: # Check for stage A timeout specifically
                 if current_depth_x is None or current_depth_x > target_confirm_depth:
                     self.logger.error(f"  --- 阶段 A: 前进至确认深度 {target_confirm_depth:.2f}m 超时。流程失败。 ---")
                     self.sport_client.StopMove()
                     return False
        self.logger.info(f"  --- 阶段 A: 前进至深度 {target_confirm_depth:.2f}m 完成。 ---")
        #self.sport_client.StopMove() # 停止机器人移动
        # self.logger.info(f"阶段A成功完成，已到达目标确认深度 {target_confirm_depth:.2f}m。流程结束。") # Removed this, proceeding to B
        # return True # Removed this to proceed to B

        # 阶段 B: 在 target_confirm_depth (~0.3m) 进行精确偏航对准
        self.logger.info(f"  --- 阶段 B: 在当前深度 (~{current_depth_x_for_logging if current_depth_x_for_logging is not None else '未知'}m) 进行精确偏航对准 (容忍度: {yaw_pixel_tolerance_confirm}px) 开始 ---")
        start_stage_B_time = time.time()
        # self.yaw_pid_visual.reset() # Already reset at the beginning of the method

        while time.time() - start_stage_B_time < max_align_time_at_depth:
            vyaw = 0.0
            visual_info = self._request_visual_info_from_server()

            if visual_info and visual_info.get('bbox_available'):
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                # depth_at_align = visual_info.get('depth_x') # For logging if needed

                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x
                self.logger.info(f"  阶段B - 精确对准中 - PxErr: {pixel_error:.1f}")

                if abs(pixel_error) < yaw_pixel_tolerance_confirm:
                    self.logger.info(f"  阶段B: 精确偏航对准完成。像素误差: {pixel_error:.1f}")
                    self.sport_client.StopMove() # Stop robot
                    self.logger.info(f"  --- 阶段 B: 精确偏航对准完成。流程成功。 ---")
                    return True # Alignment successful, end of active phases

                # Set vyaw to a fixed value based on the direction of the pixel error
                if pixel_error > 0:
                    vyaw = -0.1  # Turn right
                else:
                    vyaw = 0.1   # Turn left

                self.try_move(0.0, 0.0, vyaw, 16)
            
            elif visual_info and not visual_info.get('bbox_available'):
                self.logger.warning("  阶段B: 精确对准中未检测到边界框。对准失败。流程失败。")
                self.sport_client.StopMove()
                return False
            else: # visual_info is None
                self.logger.error("  阶段B: 精确对准中从服务器获取视觉信息失败。对准失败。流程失败。")
                self.sport_client.StopMove()
                return False
            
            time.sleep(control_period)
            
            if time.time() - start_stage_B_time >= max_align_time_at_depth:
                self.logger.error(f"  --- 阶段 B: 精确偏航对准超时。流程失败。 ---")
                self.sport_client.StopMove()
                return False
        # This part should ideally not be reached if loop completes or times out with returns
        self.logger.info(f"  --- 阶段 B: 逻辑意外结束（可能未对准或未超时）。流程失败。 ---")
        self.sport_client.StopMove()
        return False

        # 阶段 C: 最后一段爬行至抓取深度 target_grasp_depth (~0.15m)
        # self.logger.info(f"  --- 阶段 C: 最后爬行至抓取深度 {target_grasp_depth:.2f}m 开始 ---")
        # start_stage_C_time = time.time()
        # # self.yaw_pid_visual.reset() # Usually not needed here if stage B aligned well
        # # self.stage_c_pixel_error_buffer.clear() # Moved to the beginning of the method

        # while time.time() - start_stage_C_time < max_crawl_time:
        #     vyaw = 0.0 # Default to no yaw adjustment unless needed
        #     vx = 0.0
        #     current_depth_x = None # Reset for fresh read
        #     visual_info = self._request_visual_info_from_server()

        #     if visual_info and visual_info.get('bbox_available'):
        #         pixel_cx = visual_info['pixel_cx']
        #         image_width = visual_info['image_width']
        #         current_depth_x = visual_info.get('depth_x')

        #         if current_depth_x is not None:
        #             self.logger.info(f"  阶段C - 最终爬行 - 实时深度X: {current_depth_x:.3f} m")
        #             if current_depth_x < target_grasp_depth:
        #                 self.logger.info(f"  阶段C - 目标抓取深度 {target_grasp_depth:.2f}m 已达到 (当前: {current_depth_x:.2f}m)。机器人停止。流程成功！")
        #                 self.sport_client.StopMove()
        #                 return True # Entire 3-stage process successful
        #         else:
        #             self.logger.info("  阶段C - 最终爬行 - 实时深度X: 未获取到 (None)。机器人将不前进。流程失败。")
        #             self.sport_client.StopMove()
        #             return False # Losing depth at this stage is critical

        #         # 微弱的偏航调整 (可选, 如果需要保持在爬行时对准)
        #         target_pixel_x = image_width / 2.0
        #         pixel_error = pixel_cx - target_pixel_x
        #         self.stage_c_pixel_error_buffer.append(pixel_error) # Add current error to buffer
                
        #         averaged_pixel_error = pixel_error # Default to current if buffer not full enough
        #         if len(self.stage_c_pixel_error_buffer) == self.stage_c_pixel_error_buffer.maxlen:
        #             averaged_pixel_error = sum(self.stage_c_pixel_error_buffer) / len(self.stage_c_pixel_error_buffer)
        #             self.logger.debug(f"  阶段C - PxErr瞬时: {pixel_error:.1f}, PxErr平均: {averaged_pixel_error:.1f}")
        #         else:
        #             self.logger.debug(f"  阶段C - PxErr瞬时: {pixel_error:.1f} (缓冲区未满, 使用瞬时值)")

        #         yaw_pixel_tolerance_crawl = 4 # 爬行时容忍度可以稍大，主要靠直线
        #         if abs(averaged_pixel_error) > yaw_pixel_tolerance_crawl:
        #             vyaw_raw = self.yaw_pid_visual(averaged_pixel_error) 
        #             vyaw = vyaw_raw * 0.3 # 大幅减弱偏航调整效果
        #             if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW * 0.3: # 更小的有效vyaw (可调)
        #                 vyaw = math.copysign(MIN_EFFECTIVE_VYAW * 0.3, vyaw)
        #             self.logger.debug(f"  阶段C - 微弱偏航调整: AvgPxErr {averaged_pixel_error:.1f}, vyaw_raw {vyaw_raw:.3f}, vyaw_adj {vyaw:.3f}")
            
        #         if current_depth_x is not None and current_depth_x >= target_grasp_depth:
        #             vx = crawl_vx
        #         else:
        #             vx = 0.0 #已到达或深度无效

        #     elif visual_info and not visual_info.get('bbox_available'):
        #         self.logger.warning("  阶段C - 最终爬行: 未检测到边界框。机器人停止移动。流程失败。")
        #         self.sport_client.StopMove()
        #         return False 
        #     else: # visual_info is None
        #         self.logger.error("  阶段C - 最终爬行: 从服务器获取视觉信息失败。机器人停止移动。流程失败。")
        #         self.sport_client.StopMove()
        #         return False

        #     self.logger.info(f"  阶段C - vx: {vx:.3f}, vyaw: {vyaw:.3f}, CurDepth: {current_depth_x}")
        #     self.try_move(vx, 0.0, vyaw, 17) 
        #     time.sleep(control_period)

        # self.logger.error(f"  --- 阶段 C: 最终爬行至抓取深度 {target_grasp_depth:.2f}m 超时。流程失败。 ---")
        # self.sport_client.StopMove()
        # return False

    # Remove or comment out the old _final_crawl_to_grasp method as its logic is now in stage C
    # def _final_crawl_to_grasp(self, target_grasp_depth=0.15, max_crawl_time=10.0, control_period=0.05):
    #     ...

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
        """第二阶段：边旋转边前进到目标，使用视觉进行偏航调整，并基于实时深度停止。"""
        self.logger.info(f"导航阶段2: 开始前进并调整朝向目标 (基于深度停止)。原始全局目标(参考): {global_goal}")
        start_time = time.time()
        
        # Configure dist_pid for depth control
        TARGET_DEPTH = 0.8  # meters
        self.dist_pid.setpoint = TARGET_DEPTH 
        # For dist_pid with depth: input is current_depth. Output = Kp*(TARGET_DEPTH - current_depth) + ...
        # If current_depth > TARGET_DEPTH (too far), output is negative. We need positive vx.
        # So, vx = -PID_output, or we invert Kp, Ki, Kd signs in PID init, or use input as -(current_depth - TARGET_DEPTH)
        # Let's try vx = -PID_output for now. Kp,Ki,Kd are currently positive.
        # Kp=0.25, Ki=0.0, Kd=0.25, output_limits=(0.0, 0.15)
        # Max output is 0.15. If Kp=0.25, and error is (0.5 - 2.0) = -1.5m, output is 0.25 * -1.5 = -0.375. So -output = 0.375, capped at 0.15.
        # If error is (0.5 - 0.6) = -0.1, output = 0.25 * -0.1 = -0.025. So -output = 0.025. This seems plausible.

        # Ensure visual PID is reset if it's shared and might have state from phase 1
        self.yaw_pid_visual.reset() # Good practice, though it might be reset before navigate_to too.

        consecutive_no_depth_for_stop_check = 0
        MAX_CONSECUTIVE_NO_DEPTH_STOP = 5 # If depth is None for this many checks near supposed stopping point, stop anyway as a failsafe.

        while time.time() - start_time < max_nav_time:
            # Get current robot state (primarily for logging or if other parts still use it)
            # px = robot_state.position[0] if robot_state else 0
            # py = robot_state.position[1] if robot_state else 0
            # current_yaw = robot_state.imu_state.rpy[2] if robot_state and robot_state.imu_state else 0
            
            # --- Visual Info Acquisition and Yaw Adjustment (remains similar) ---
            vyaw = 0.0
            vx = 0.0 # Default to no forward movement if no depth
            current_depth_x = None

            visual_info = self._request_visual_info_from_server()

            if visual_info and visual_info.get('bbox_available'):
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                current_depth_x = visual_info.get('depth_x') # This can be None if depth measurement failed for the bbox

                # 新增的INFO级别日志，专门打印深度
                if current_depth_x is not None:
                    self.logger.info(f"阶段2 实时深度X: {current_depth_x:.3f} m")
                else:
                    self.logger.info("阶段2 实时深度X: 未获取到 (None)")
                
                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x
                
                self.logger.debug(f"阶段2 - ImgW: {image_width}, PxCX: {pixel_cx:.1f}, TrgPxX: {target_pixel_x:.1f}, PxErr: {pixel_error:.1f}, DepthX: {current_depth_x}")

                yaw_pixel_tolerance_phase2 = 3
                if abs(pixel_error) > yaw_pixel_tolerance_phase2:
                    vyaw = self.yaw_pid_visual(pixel_error)
                    if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
                        vyaw = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                
                # --- Depth-based Stopping Logic ---
                if current_depth_x is not None:
                    consecutive_no_depth_for_stop_check = 0 # Reset counter
                    if current_depth_x < TARGET_DEPTH:
                        self.logger.info(f"阶段2: 目标深度 {TARGET_DEPTH:.2f}m 已达到或小于。当前深度: {current_depth_x:.2f}m。机器人停止。")
                        self.sport_client.StopMove()
                        return True # Navigation successful based on depth
                    
                    # --- Depth-based Forward Velocity Control ---
                    # dist_pid setpoint is TARGET_DEPTH. Input is current_depth_x.
                    # Output = Kp*(TARGET_DEPTH - current_depth_x)
                    # If current_depth_x > TARGET_DEPTH (too far), error is negative, output is negative.
                    # We want positive vx, so vx = -pid_output.
                    pid_output_vx = self.dist_pid(current_depth_x) 
                    vx = -pid_output_vx
                    # Ensure vx is not negative (i.e., robot doesn't back up if it overshoots and PID tries to correct by making vx negative)
                    vx = max(0.0, vx) 
                    # Apply a minimum speed if moving forward, and a cap from PID output_limits
                    if vx > 0.001: # If intending to move forward
                         vx = max(0.01, vx) # Min effective speed
                    vx = min(vx, self.dist_pid.output_limits[1]) # Adhere to max speed from PID config

                else: # current_depth_x is None, but bbox is available
                    self.logger.warning("阶段2: 有边界框但无深度信息。机器人将不前进 (vx=0)。仅调整偏航。")
                    vx = 0.0 # No forward movement if depth is unknown
                    consecutive_no_depth_for_stop_check += 1
                    if consecutive_no_depth_for_stop_check > MAX_CONSECUTIVE_NO_DEPTH_STOP and vx < 0.01 : # If close to stopping and repeatedly no depth
                         self.logger.warning(f"阶段2: 连续 {MAX_CONSECUTIVE_NO_DEPTH_STOP} 次无深度信息，且机器人接近停止，强制停止以防万一。")
                         self.sport_client.StopMove()
                         return False # Stopped due to lack of depth as a failsafe

            elif visual_info and not visual_info.get('bbox_available'):
                self.logger.warning("阶段2: 前进中未检测到边界框。机器人将停止移动。")
                vx = 0.0
                vyaw = 0.0 
                # Option: Stop completely if no bbox for a certain duration
                self.sport_client.StopMove() # More conservative: stop if bbox lost
                # Potentially return False or raise error if bbox is lost for too long
                # For now, just stopping and hoping it reappears or timeout occurs.

            else: # visual_info is None (communication failed)
                self.logger.error("阶段2: 前进中从服务器获取视觉信息失败。机器人将停止移动。")
                vx = 0.0
                vyaw = 0.0
                self.sport_client.StopMove() # Stop if comms fail
            
            # self.logger.debug(f"阶段2 - 距离(to global_goal): {distance:.2f}m, vx: {vx:.3f}, 视觉vyaw: {vyaw:.3f}, DepthX: {current_depth_x}") 
            # The old distance to global_goal is no longer the primary driver for stopping or vx.
            self.logger.info(f"阶段2 - vx: {vx:.3f}, 视觉vyaw: {vyaw:.3f}, CurDepth: {current_depth_x}, TrgDepth: {TARGET_DEPTH}")
            self.try_move(vx, 0.0, vyaw, 12)
            time.sleep(control_period)
            
        self.logger.error("阶段2: 前进超时")
        self.sport_client.StopMove()
        raise RuntimeError("前进未完成 (阶段2超时，基于深度或时间)")

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
            self.yaw_pid_visual.reset() # Reset visual PID too

            # 第一阶段 (新): 视觉对准
            self.logger.info("调用导航阶段1 (新): 视觉Yaw对准...")
            self._phase1_visual_align_yaw() # Uses pixel info to align yaw
            self.logger.info("导航阶段1 (新) 视觉Yaw对准完成.")

            # 对准后，获取精确的世界坐标作为导航目标
            self.logger.info("视觉对准后，重新请求精确的校准(世界)坐标...")
            # Re-use CALIBRATE_SHIRT_POSITION's logic for getting user confirmation for shirt
            # For now, directly use the server data for the refined global goal
            # This assumes the object being visually aligned is the one we want world coords for.
            
            # IMPORTANT: The calibration (world coord) server request should happen AFTER visual alignment
            # The calibrated_shirt_coord_camera_frame will be relative to the *newly aligned* robot orientation.
            # We then convert this relative coord to global for phase 2.
            
            # Let's assume NAVIGATE_TO_OBJECT is the only state using this visual alignment for now.
            # The target_list[0]["coord"] should be updated with the new, more accurate relative coord.

            # --- This logic block for getting world coordinates after visual alignment needs refinement --- 
            # --- It's currently mixing with the main state machine's calibration flow --- 
            # --- For a generic navigate_to, we need a way to specify *what* we are aligning to get coords for, 
            # --- or assume the target_list[0] (shirt) is always the visually aligned object.
            
            # This call will get the X,Y in a camera frame that is *now* pointing straight at the target.
            refined_relative_coord_camera_frame = self._request_calibration_from_server()
            
            if not refined_relative_coord_camera_frame:
                self.logger.error("视觉对准后未能从服务器获取精确的校准坐标。将使用初始目标或中止导航。")
                # Decide on fallback: use original global_goal, or raise error?
                # For now, let's attempt to proceed with the original global_goal if refinement fails.
                # This might not be ideal if visual alignment significantly changed robot's position/yaw from initial state.
                self.logger.warning(f"将尝试使用视觉对准前的目标 {global_goal} 进行阶段2导航。")
                # Or, better, re-calculate global_goal based on current robot pose and original *relative* target if that was how global_goal was first derived.
                # This part needs careful thought based on how global_goal is first computed before navigate_to is called.
                # Let's assume for now navigate_to was called with a global_goal and visual alignment is a refinement step.
                # If visual alignment succeeded, the robot is facing the target. 
                # The original global_goal might still be okay if the robot hasn't moved much, only rotated.
                pass # Fall through to use existing global_goal for phase 2
            else:
                # The refined_relative_coord_camera_frame is [X_fwd, Y_left] relative to current robot pose.
                # USER MODIFICATION: We will now primarily use X_fwd as the distance.
                # Y_left will be mostly ignored for global coordinate calculation as robot is visually aligned.
                self.logger.info(f"视觉对准后获得的精确相对坐标 (相机/机器人): {refined_relative_coord_camera_frame}")
                refined_X_distance_to_object = refined_relative_coord_camera_frame[0]
                # refined_Y_offset_at_object = refined_relative_coord_camera_frame[1] # We will ignore this for global calculation

                # Convert this new relative coordinate to a new global coordinate for the object
                # This uses the robot's *current* state after visual alignment.
                if robot_state is None:
                    self.logger.error("视觉对准后机器人状态不可用，无法计算精确的全局目标。")
                    raise RuntimeError("视觉对准后机器人状态不可用。")
                
                current_aligned_robot_px = robot_state.position[0]
                current_aligned_robot_py = robot_state.position[1]
                current_aligned_robot_yaw = robot_state.imu_state.rpy[2]

                # Calculate precise_object_global_coord based on current robot pose and refined_X_distance.
                # Since robot is visually aligned, the object is considered directly in front.
                precise_object_global_coord_x = current_aligned_robot_px + refined_X_distance_to_object * math.cos(current_aligned_robot_yaw)
                precise_object_global_coord_y = current_aligned_robot_py + refined_X_distance_to_object * math.sin(current_aligned_robot_yaw)
                precise_object_global_coord = [precise_object_global_coord_x, precise_object_global_coord_y]
                
                self.logger.info(f"视觉对准后计算出的精确物体全局坐标 (基于X距离和当前朝向): {precise_object_global_coord}")

                # Now, re-calculate the robot's stopping point goal using this precise_object_global_coord
                # The robot is already facing the object (due to visual alignment), so its current yaw IS the approach_yaw_global.
                approach_yaw_global_refined = current_aligned_robot_yaw 
                
                robot_center_target_x_refined = precise_object_global_coord[0] - self.CLAW_OFFSET_FORWARD * math.cos(approach_yaw_global_refined)
                robot_center_target_y_refined = precise_object_global_coord[1] - self.CLAW_OFFSET_FORWARD * math.sin(approach_yaw_global_refined)
                
                global_goal = [robot_center_target_x_refined, robot_center_target_y_refined]
                self.logger.info(f"导航至物体：视觉对准后更新的机器人中心全局目标: [{global_goal[0]:.2f}, {global_goal[1]:.2f}]")

            # --- End of refinement block --- 

            # Wait a bit after visual rotation before moving for phase 2
            time.sleep(0.5)

            # 第二阶段：边旋转边前进 (使用可能已优化的 global_goal)
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

    def _request_visual_info_from_server(self):
        """Requests visual information (pixel_cx, image_width) from the image server."""
        #self.logger.info(f"向图像服务器 ({IMAGE_SERVER_HOST}:{IMAGE_SERVER_PORT}) 请求视觉信息...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3) # Shorter timeout for potentially faster visual feedback loop
                s.connect((IMAGE_SERVER_HOST, IMAGE_SERVER_PORT))
                self.logger.debug("已连接到图像服务器 (请求视觉信息)。")
                s.sendall(b"REQUEST_VISUAL_INFO")
                self.logger.debug("已发送 'REQUEST_VISUAL_INFO' 命令。")
                
                response_data = s.recv(1024)
                if not response_data:
                    self.logger.error("从图像服务器收到空响应 (视觉信息)。")
                    return None
                
                response = json.loads(response_data.decode())
                self.logger.debug(f"从图像服务器收到响应 (视觉信息): {response}")

                if response.get("status") == "success":
                    if response.get("bbox_available") and 'pixel_cx' in response and 'image_width' in response:
                        return {
                            'pixel_cx': float(response['pixel_cx']),
                            'image_width': int(response['image_width']),
                            'bbox_available': True,
                            'depth_x': response.get('depth_x') # Will be None if not present or None in response
                        }
                    elif not response.get("bbox_available"):
                        self.logger.info("视觉信息：图像服务器报告当前无可用边界框。")
                        return {'bbox_available': False, 'depth_x': None}
                    else:
                        self.logger.error(f"图像服务器成功响应但视觉信息格式不正确: {response}")
                        return None # Or {'bbox_available': False} depending on desired handling
                else:
                    self.logger.error(f"图像服务器报告错误 (视觉信息): {response.get('message', '未知错误')}")
                    return None
        except socket.timeout:
            self.logger.warning(f"连接图像服务器 ({IMAGE_SERVER_HOST}:{IMAGE_SERVER_PORT}) 或接收视觉信息超时。")
            return None
        except socket.error as e:
            self.logger.error(f"与图像服务器通信时发生套接字错误 (视觉信息): {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"解析来自图像服务器的JSON响应时出错 (视觉信息): {e}")
            return None
        except Exception as e:
            self.logger.error(f"请求视觉信息时发生意外错误: {e}")
            return None

    def _request_calibration_from_server(self):
        """Requests calibration data from the image server."""
        self.logger.info(f"向图像服务器 ({IMAGE_SERVER_HOST}:{IMAGE_SERVER_PORT}) 请求校准数据...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10) # Set a timeout for connection and operations
                s.connect((IMAGE_SERVER_HOST, IMAGE_SERVER_PORT))
                self.logger.info("已连接到图像服务器。")
                s.sendall(b"REQUEST_CALIBRATION")
                self.logger.info("已发送 'REQUEST_CALIBRATION' 命令。")
                
                response_data = s.recv(1024)
                if not response_data:
                    self.logger.error("从图像服务器收到空响应。")
                    return None
                
                response = json.loads(response_data.decode())
                self.logger.info(f"从图像服务器收到响应: {response}")

                if response.get("status") == "success" and "calibrated_coord" in response:
                    calibrated_coord = response["calibrated_coord"]
                    # Ensure coordinates are floats, if they come as string or int from json
                    calibrated_coord = [float(c) for c in calibrated_coord]
                    self.logger.info(f"成功从图像服务器获取校准坐标: {calibrated_coord}")
                    return calibrated_coord
                else:
                    self.logger.error(f"图像服务器报告错误或响应格式不正确: {response.get('message', '未知错误')}")
                    return None
        except socket.timeout:
            self.logger.error(f"连接图像服务器 ({IMAGE_SERVER_HOST}:{IMAGE_SERVER_PORT}) 或接收数据超时。")
            return None
        except socket.error as e:
            self.logger.error(f"与图像服务器通信时发生套接字错误: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"解析来自图像服务器的JSON响应时出错: {e}")
            return None
        except Exception as e:
            self.logger.error(f"请求校准数据时发生意外错误: {e}")
            return None

    def _phase1_visual_align_yaw(self, yaw_pixel_tolerance=3, max_align_time=20.0, control_period=0.05):
        """第一阶段（新）：使用视觉反馈调整Yaw，使目标边界框中心对准图像中心。"""
        self.logger.info(f"导航阶段1 (视觉对准): 开始使用像素信息调整Yaw。 Tolerance: +/-{yaw_pixel_tolerance}px")
        start_time = time.time()
        self.yaw_pid_visual.reset()

        consecutive_no_bbox_count = 0
        max_consecutive_no_bbox = 10 # e.g., 10 * 0.05s = 0.5s without bbox before failing

        while time.time() - start_time < max_align_time:
            visual_info = self._request_visual_info_from_server()

            if visual_info and visual_info.get('bbox_available'):
                consecutive_no_bbox_count = 0 # Reset counter
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x

                self.logger.debug(f"视觉对准 - ImgWidth: {image_width}, PxCX: {pixel_cx:.1f}, TargetPxX: {target_pixel_x:.1f}, PxError: {pixel_error:.1f}")

                if abs(pixel_error) < yaw_pixel_tolerance:
                    self.logger.info(f"阶段1 (视觉对准): Yaw对准完成。像素误差: {pixel_error:.1f}")
                    self.sport_client.StopMove()
                    return True # Alignment successful
                
                # PID input is the error, setpoint is 0 (already configured in PID init)
                # simple_pid output is Kp * (setpoint - input) + ...
                # Here, setpoint = 0. So, output is Kp * (-input) + ...
                # We define pixel_error = pixel_cx - target_pixel_x.
                # If pixel_error > 0 (object to the right), we want vyaw < 0 (turn right).
                # If pixel_error < 0 (object to the left), we want vyaw > 0 (turn left).
                # So, vyaw should be proportional to -pixel_error.
                # If we set PID input to pixel_error, then PID output is ~ -Kp * pixel_error.
                # This is what we want for vyaw.
                vyaw = self.yaw_pid_visual(pixel_error) 

                # Apply minimum effective vyaw if the calculated vyaw is too small but not zero
                if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
                    vyaw_adjusted = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                    self.logger.debug(f"视觉对准 - vyaw original: {vyaw:.4f}, adjusted to: {vyaw_adjusted:.4f} (due to min effective speed)")
                    vyaw = vyaw_adjusted
                
                self.logger.debug(f"视觉对准 - vyaw final: {vyaw:.3f}")
                self.try_move(0.0, 0.0, vyaw, 13) # ID 13 for visual alignment rotation
            
            elif visual_info and not visual_info.get('bbox_available'):
                self.logger.warning("阶段1 (视觉对准): 未检测到边界框，机器人将暂停并等待。")
                self.sport_client.StopMove() # Stop if no bbox
                consecutive_no_bbox_count += 1
                if consecutive_no_bbox_count > max_consecutive_no_bbox:
                    self.logger.error("阶段1 (视觉对准): 连续多次未检测到边界框，对准失败。")
                    raise RuntimeError("视觉对准失败：长时间无边界框")
            else:
                self.logger.error("阶段1 (视觉对准): 从服务器获取视觉信息失败。机器人将暂停。")
                self.sport_client.StopMove() # Stop if comms fail
                # Potentially add a counter here too before failing
                # For now, let's assume it might recover and rely on max_align_time
                # Or, more robustly, also raise an error after a few failed attempts:
                # consecutive_comm_fail_count +=1 ...
            
            time.sleep(control_period)

        self.logger.error("阶段1 (视觉对准): Yaw对准超时。")
        self.sport_client.StopMove()
        raise RuntimeError("视觉对准未完成 (超时)")

    def receive_direction_from_viewer(self):
        """接收来自image_server.py转发的方向命令（监听本地端口50002）"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', 50002))
                s.listen(1)
                s.settimeout(500)  # 500秒超时
                conn, addr = s.accept()
                data = conn.recv(1024)
                if data:
                    direction = data.decode('utf-8').strip()
                    self.logger.info(f"收到方向命令: {direction}")
                    conn.sendall(b"ACK")  # 发送确认
                    return direction
                return None
        except socket.timeout:
            self.logger.warning("等待方向命令超时")
            return None
        except Exception as e:
            self.logger.error(f"接收方向命令失败: {e}")
            return None

    def _execute_direction_rotation(self, direction):
        """根据方向命令执行旋转"""
        if robot_state is None:
            self.logger.error("机器人状态不可用，无法执行方向旋转")
            return False
        
        # 获取当前位置和朝向
        current_x = robot_state.position[0]
        current_y = robot_state.position[1]
        current_yaw = robot_state.imu_state.rpy[2]
        
        # 定义方向对应的角度增量（弧度）
        direction_angles = {
            "forward": 0.0,
            "backward": math.pi,
            "left": math.pi/2,
            "right": -math.pi/2,
            "front-left": math.pi/4,
            "front-right": -math.pi/4,
            "back-left": 3*math.pi/4,
            "back-right": -3*math.pi/4
        }
        
        if direction not in direction_angles:
            self.logger.error(f"未知的方向命令: {direction}")
            return False
        
        # 计算新的yaw角度
        angle_increment = direction_angles[direction]
        new_yaw = self.normalize_angle(current_yaw + angle_increment)
        
        self.logger.info(f"执行方向旋转: {direction}")
        self.logger.info(f"当前位置: ({current_x:.3f}, {current_y:.3f}, yaw: {current_yaw:.3f})")
        self.logger.info(f"目标位置: ({current_x:.3f}, {current_y:.3f}, yaw: {new_yaw:.3f})")
        
        try:
            # 使用MoveToPos执行旋转（保持x,y不变，只改变yaw）
            ret = self.sport_client.MoveToPos(current_x, current_y, new_yaw)
            if ret == 0:
                self.logger.info(f"方向旋转命令发送成功: {direction}")
                # 等待旋转完成
                time.sleep(3.0)  # 给机器人时间完成旋转
                return True
            else:
                self.logger.error(f"MoveToPos调用失败，错误码: {ret}")
                return False
        except Exception as e:
            self.logger.error(f"执行方向旋转时发生错误: {e}")
            return False

    def run(self):
        # Start in calibration state if pos_detector is available - CHANGED
        # Now, we always attempt calibration via the server.
        # The CALIBRATE_SHIRT_POSITION state will handle failure.
        self.state = State.CALIBRATE_SHIRT_POSITION
        # else:
        #     self.logger.warning("PositionDetector not available. Skipping calibration. Shirt coordinates will be default.")
        #     self.state = State.NAVIGATE_TO_OBJECT # Or some other default starting state

        while self.state != State.DONE:
            try:
                if self.state == State.WAIT:
                    # self.stop_moving() # Original comment
                    # ret1 = self.sport_client.SwitchGait(0) # Original comment
                    ret2 = self.sport_client.StopMove() # Ensures robot is stopped in WAIT - USER REQUESTED REMOVAL
                    self.logger.info(f"进入WAIT状态 (StopMove 已移除)。下一个状态: {getattr(self, '_next_state', 'None')}")
                    time.sleep(1) # General wait time
                    
                    # General transition from WAIT to _next_state
                    if hasattr(self, '_next_state') and self._next_state is not None:
                        # Only transition if not waiting for specific conditions above
                        if not (hasattr(self, '_wait_for_pitch') and self._wait_for_pitch) and \
                           not (hasattr(self, '_wait_for_user_ack_grasp') and self._wait_for_user_ack_grasp):
                            self.state = self._next_state
                            self.logger.info(f"从WAIT状态转换到: {self.state}")
                            self._next_state = None # Clear after transition
                    continue # Loop back to re-evaluate WAIT or new state
                
                elif self.state == State.CALIBRATE_SHIRT_POSITION:
                    self.logger.info("进入 CALIBRATE_SHIRT_POSITION 状态")
                    
                    # Request calibration data from the image server
                    calibrated_shirt_coord_camera_frame = self._request_calibration_from_server()

                    if not calibrated_shirt_coord_camera_frame:
                        self.logger.error("无法从图像服务器获取校准数据。将使用默认/旧的 'shirt' 坐标。")
                        target_list[0]["coord"] = [0.6, 0.0] # Fallback to original default
                        self.logger.warning(f"Shirt target set to default: {target_list[0]['coord']}")
                        # Ask user if they want to proceed with default or retry server connection
                        user_choice = ""
                        while user_choice.lower() not in ['proceed', 'retry']:
                            user_choice = input("无法连接到图像服务器或校准失败。输入 'proceed' 使用默认坐标，或 'retry' 重新尝试连接服务器: ")
                        if user_choice.lower() == 'retry':
                            self.state = State.CALIBRATE_SHIRT_POSITION # Stay to retry connection
                            time.sleep(1) # Brief pause before retry
                            continue
                        else: # proceed
                            self._next_state = State.NAVIGATE_TO_OBJECT
                            self.state = State.WAIT
                            continue # Go to WAIT state

                    # If we get here, calibrated_shirt_coord_camera_frame is valid
                    avg_x, avg_y = calibrated_shirt_coord_camera_frame[0], calibrated_shirt_coord_camera_frame[1]
                    self.logger.info(f"从图像服务器获取的 'shirt' 坐标 (相机坐标系 RIGHT_HANDED_Z_UP_X_FWD): [{avg_x:.3f}, {avg_y:.3f}]")
                    
                    user_input = ""
                    while user_input.lower() != 'ok':
                        user_input = input(f"校准完成 (通过服务器)。计算得到的'shirt'坐标(相机X,Y): [{avg_x:.3f}, {avg_y:.3f}]。 输入 'ok' 确认并使用此坐标，或输入 'retry' 重新从服务器请求校准: ")
                        if user_input.lower() == 'retry':
                            self.logger.info("用户选择重试从服务器校准。")
                            self.state = State.CALIBRATE_SHIRT_POSITION # Stay in this state to retry
                            break # Break from input loop to restart calibration request
                        elif user_input.lower() != 'ok':
                            self.logger.info("输入无效。")
                    
                    if user_input.lower() == 'ok':
                        target_list[0]["coord"] = calibrated_shirt_coord_camera_frame
                        self.logger.info(f"用户已确认。'shirt' 目标更新为 (机器人相对坐标系): {target_list[0]['coord']}")
                        self._next_state = State.NAVIGATE_TO_OBJECT
                        self.state = State.WAIT
                    elif user_input.lower() == 'retry':
                        continue # Go to next iteration of main while loop, will re-enter CALIBRATE

                elif self.state == State.NAVIGATE_TO_OBJECT:
                    if robot_state is None:
                        self.logger.error("Robot state not available for offset navigation goal calculation. Waiting.")
                        time.sleep(1) # Wait for robot state to become available
                        continue # Re-evaluate state in next loop iteration

                    object_relative_coord_robot_frame = target_list[0]["coord"]
                    self.logger.info(f"NAVIGATE_TO_OBJECT: Current robot relative target for 'shirt': {object_relative_coord_robot_frame}")
                    object_global_coord_estimate = self.convert_relative_to_global(object_relative_coord_robot_frame)
                    self.logger.info(f"Object's estimated global coordinate (from relative): {object_global_coord_estimate}")

                    current_robot_px = robot_state.position[0]
                    current_robot_py = robot_state.position[1]
                    delta_x_to_object_global = object_global_coord_estimate[0] - current_robot_px
                    delta_y_to_object_global = object_global_coord_estimate[1] - current_robot_py
                    approach_yaw_global_estimate = math.atan2(delta_y_to_object_global, delta_x_to_object_global)
                    
                    robot_center_target_x_estimate = object_global_coord_estimate[0] - self.CLAW_OFFSET_FORWARD * math.cos(approach_yaw_global_estimate)
                    robot_center_target_y_estimate = object_global_coord_estimate[1] - self.CLAW_OFFSET_FORWARD * math.sin(approach_yaw_global_estimate)
                    adjusted_global_nav_goal_for_robot_center_estimate = [robot_center_target_x_estimate, robot_center_target_y_estimate]
                    
                    self.logger.info(f"NAVIGATE_TO_OBJECT: Estimated object global: [{object_global_coord_estimate[0]:.2f}, {object_global_coord_estimate[1]:.2f}]. "
                                     f"Estimated Approach Yaw (global): {approach_yaw_global_estimate:.2f} rad. "
                                     f"Claw offset: {self.CLAW_OFFSET_FORWARD}m. "
                                     f"Estimated robot center global target: [{adjusted_global_nav_goal_for_robot_center_estimate[0]:.2f}, {adjusted_global_nav_goal_for_robot_center_estimate[1]:.2f}]")
                                     
                    self.navigate_to(adjusted_global_nav_goal_for_robot_center_estimate)
                    self._next_state = State.FINE_ADJUST_AND_APPROACH # Transition to new state
                    self.state = State.WAIT

                elif self.state == State.FINE_ADJUST_AND_APPROACH:
                    self.logger.info("进入 FINE_ADJUST_AND_APPROACH 状态")
                    pitch_adjustment_done = False
                    height_adjustment_done = False
                    # final_approach_succeeded = False # This variable is set by the calls below

                    # 1. 降低机身高度到最小值
                    self.logger.info(f"精细调整：步骤1 - 设置机身相对高度为最小值 {self.BODY_HEIGHT_REL_MIN}m")
                    if self.set_body_height_relative(self.BODY_HEIGHT_REL_MIN):
                        time.sleep(2) # 给机器人足够的时间降低身体
                        self.logger.info("精细调整：机身高度调整完成。")
                        # 2. 降低抬足高度到最小值 (只有在机身高度调整成功后才进行)
                        self.logger.info(f"精细调整：步骤2 - 设置抬足相对高度为最小值 {self.FOOT_RAISE_REL_MIN}m")
                        if self.set_foot_raise_height_relative(self.FOOT_RAISE_REL_MIN):
                            time.sleep(1.5) # 给机器人时间调整
                            self.logger.info("精细调整：抬足高度调整完成。")
                            height_adjustment_done = True
                        else:
                            self.logger.error("精细调整：设置抬足高度失败。")
                    else:
                        self.logger.error("精细调整：设置机身高度失败。")

                    if not height_adjustment_done:
                        self.logger.error("高度调整失败，恢复默认姿态并中止操作。")
                        self.reset_body_and_foot_height() # 尝试恢复高度
                        # Pitch has not been touched yet, so no need to reset it here.
                        time.sleep(2.0) # Wait for height reset attempt
                        self.state = State.DONE # Or an error state
                        continue
                    
                    # 3. 调整俯仰角到10度 (在高度调整之后)
                    desired_pitch = 10.0
                    self.logger.info(f"精细调整：步骤3 - 设置俯仰角为 {desired_pitch}°")
                    if self.pitch_controller:
                        self.pitch_controller.set_pitch(desired_pitch)
                        time.sleep(self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0) + 0.5)
                        self.logger.info("精细调整：俯仰角调整完成。")
                        pitch_adjustment_done = True
                    else:
                        self.logger.error("精细调整：PitchController 不可用，无法调整俯仰角。")
                        # If pitch is critical and controller missing, perhaps abort.
                        # For now, allow proceeding, but log it as a significant issue.
                        pitch_adjustment_done = True # Mark as done to proceed, but it's an issue

                    if not pitch_adjustment_done and self.pitch_controller: # Should only fail if controller was present and set_pitch failed implicitly
                        self.logger.error("俯仰角调整失败，恢复默认姿态并中止操作。")
                        self.reset_body_and_foot_height() # 恢复高度
                        # Pitch controller was available but failed, so don't try to reset pitch with it.
                        self.state = State.DONE # Or an error state
                        continue
                        
                    # 4. 执行完整精细接近、对准和最终爬行流程
                    self.logger.info("精细调整：步骤4 - 开始完整精细接近、对准和最终爬行流程")
                    # 调用统一的方法，它内部处理0.3m对准和0.15m抓取深度
                    overall_fine_approach_succeeded = self._approach_and_confirm_alignment(
                        target_confirm_depth=0.15, 
                        target_grasp_depth=0.15, 
                        yaw_pixel_tolerance_confirm=1
                        # max_approach_time, max_align_time_at_depth, max_crawl_time, control_period 沿用方法中的默认值或按需传递
                    )

                    if overall_fine_approach_succeeded:
                        self.logger.info("精细调整：完整流程成功完成（已到达~0.15m并对准）。")
                        self._next_state = State.GRASP_OBJECT 
                        self.state = State.WAIT
                    else:
                        self.logger.error("精细调整：完整精细接近、对准或最终爬行流程失败。恢复默认姿态并中止操作。")
                        self.reset_body_and_foot_height()
                        if self.pitch_controller: self.pitch_controller.reset_pitch()
                        time.sleep(self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0) + 1.0) # 等待姿态恢复
                        self.state = State.DONE # Or an error state

                elif self.state == State.LOWER_HEAD:
                    self.logger.warning("LOWER_HEAD 状态被意外调用，俯仰角应已在 FINE_ADJUST_AND_APPROACH 中设置。检查状态转换逻辑。")
                    # Defaulting to GRASP_OBJECT if this state is somehow reached after fine adjust.
                    # Or, if it's a valid path from elsewhere, original logic can be kept.
                    # self.lower_head() 
                    # interpolation_duration = self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0)
                    # self.logger.info(f"等待 {interpolation_duration:.1f}s 以完成头部俯仰角调整...")
                    # self.logger.info("降低头部命令已发送。")
                    self._next_state = State.GRASP_OBJECT # Directly go to grasp
                    self.state = State.WAIT

                elif self.state == State.GRASP_OBJECT:
                    self.logger.info("GRASP_OBJECT: Re-setting body height, foot height, and pitch before grasping.")

                    # 1. Re-set body height
                    self.logger.info(f"GRASP_OBJECT: Setting body relative height to {self.BODY_HEIGHT_REL_MIN}m")
                    body_set_success = self.set_body_height_relative(self.BODY_HEIGHT_REL_MIN)
                    if body_set_success:
                        time.sleep(0.5) # Allow time for adjustment
                        self.logger.info("GRASP_OBJECT: Body height re-set.")
                    else:
                        self.logger.error("GRASP_OBJECT: Failed to re-set body height. Attempting grasp anyway.")

                    # 2. Re-set foot raise height
                    self.logger.info(f"GRASP_OBJECT: Setting foot relative height to {self.FOOT_RAISE_REL_MIN}m")
                    foot_set_success = self.set_foot_raise_height_relative(self.FOOT_RAISE_REL_MIN)
                    if foot_set_success:
                        time.sleep(1.5) # Allow time for adjustment
                        self.logger.info("GRASP_OBJECT: Foot raise height re-set.")
                    else:
                        self.logger.error("GRASP_OBJECT: Failed to re-set foot raise height. Attempting grasp anyway.")

                    # 3. Re-set pitch
                    grasp_pitch_angle = 10.0 # Consistent with FINE_ADJUST_AND_APPROACH
                    self.logger.info(f"GRASP_OBJECT: Setting pitch to {grasp_pitch_angle}°")
                    if self.pitch_controller:
                        self.pitch_controller.set_pitch(grasp_pitch_angle)
                        pitch_wait_time = self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0) + 0.5
                        self.logger.info(f"GRASP_OBJECT: Waiting {pitch_wait_time:.1f}s for pitch adjustment.")
                        time.sleep(pitch_wait_time)
                        self.logger.info("GRASP_OBJECT: Pitch re-set.")
                    else:
                        self.logger.warning("GRASP_OBJECT: PitchController not available, cannot re-set pitch.")

                    # Now, proceed with the grasp
                    success = self.grasp()
                    if success:
                        self.logger.info("抓取成功")
                        user_ack = ""
                        # Optionally skip user ack for faster testing, but it's a good safety check
                        while user_ack.lower() != 'ok':
                            user_ack = input("物体已抓取。输入 \'ok\' 使机器人头部恢复至0度并继续: ")
                            if user_ack.lower() != 'ok':
                                self.logger.info("输入无效，请输入 \'ok\'。")
                        self.logger.info("用户已确认。机器人姿态将恢复...")
                        
                        if self.pitch_controller: self.pitch_controller.reset_pitch()
                        self.reset_body_and_foot_height() # Restore body/foot height
                        
                        # Wait for restorations to complete
                        # Total time might be max of pitch reset and height reset + buffer
                        time.sleep(max(self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0), 2.5) + 0.5)
                        self.logger.info("机器人姿态已恢复。")
                        self._next_state = State.SEND_NEXT_COMMAND 
                        self.state = State.WAIT
                    else:
                        self.logger.warning("抓取失败，恢复默认姿态。")
                        if self.pitch_controller: self.pitch_controller.reset_pitch()
                        self.reset_body_and_foot_height()
                        time.sleep(max(self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0), 2.5) + 0.5)
                        # Decide what to do on grasp failure - e.g., retry fine approach or end
                        self.logger.error("抓取失败，任务中止。")
                        self.state = State.DONE 

                elif self.state == State.SEND_NEXT_COMMAND:
                    self.send_next_to_vlm()
                    next_coord_relative = self.get_next_target_from_vlm() # This is relative
                    target_list[1]["coord"] = next_coord_relative # Update target_list with new relative coord
                    self.logger.info(f"SEND_NEXT_COMMAND: Updated person target (relative): {next_coord_relative}")
                    
                    # 用户确认是否需要接收方向命令
                    user_choice = ""
                    while user_choice.lower() not in ['y', 'n', 'yes', 'no']:
                        user_choice = input("是否需要等待接收方向命令？(y/n): ")
                        if user_choice.lower() not in ['y', 'n', 'yes', 'no']:
                            self.logger.info("输入无效，请输入 y/n 或 yes/no。")
                    
                    if user_choice.lower() in ['y', 'yes']:
                        self.logger.info("等待接收方向命令...")
                        direction = self.receive_direction_from_viewer()
                        if direction:
                            self.logger.info(f"收到方向命令: {direction}")
                            # 根据方向命令计算新的yaw角度并执行旋转
                            self._execute_direction_rotation(direction)
                        else:
                            self.logger.warning("未收到有效的方向命令")
                    else:
                        self.logger.info("跳过方向命令接收，直接继续")
                    
                    self._next_state = State.NAVIGATE_TO_PERSON
                    self.state = State.WAIT
                elif self.state == State.NAVIGATE_TO_PERSON:
                    # Assuming target_list[1]["coord"] is already in robot's relative X_FWD, Y_LEFT frame
                    relative_coord_person_robot_frame = target_list[1]["coord"] 
                    global_nav_goal_person = self.convert_relative_to_global(relative_coord_person_robot_frame)
                    
                    self.logger.info(f"NAVIGATE_TO_PERSON: Targeting global goal {global_nav_goal_person} (from relative {relative_coord_person_robot_frame})")
                    
                    # For NAVIGATE_TO_PERSON, we might not need the claw offset logic,
                    # or it might be different. For now, navigate directly to the converted global goal.
                    # If an offset is needed (e.g., stop *in front* of the person), that logic would go here,
                    # similar to NAVIGATE_TO_OBJECT but potentially with a different offset value or strategy.
                    self.navigate_to(global_nav_goal_person) # Directly navigate to person's (converted) global position
                    
                    self._next_state = State.RELEASE_OBJECT
                    self.state = State.WAIT
                elif self.state == State.RELEASE_OBJECT:
                    self.release()
                    self.logger.info("任务完成")
                    self._next_state = State.DONE
                    self.state = State.WAIT
                
                # Short sleep to prevent tight loop if a state doesn't transition immediately
                # and doesn't have its own internal sleep/wait.
                # The WAIT state handles its own longer sleeps.
                if self.state != State.WAIT : time.sleep(0.05) # Small delay for non-WAIT states

            except RuntimeError as e: # Catch RuntimeErrors from navigation or other critical parts
                self.logger.error(f"运行循环中发生 Runtime错误: {e}", exc_info=True)
                if hasattr(self, 'sport_client'): self.sport_client.StopMove()
                # Optionally, transition to a specific ERROR state or try to recover
                self.logger.info("机器人已停止。程序将终止。")
                break 
            except Exception as e:
                self.logger.error(f"运行循环中发生未知错误: {e}", exc_info=True)
                if hasattr(self, 'sport_client'): self.sport_client.StopMove()
                self.logger.info("机器人已停止。程序将终止。")
                break # Exit on other unhandled exceptions

        # Loop finished (State.DONE or break due to error)
        self.logger.info("RobotController run loop finished.")
        # Final cleanup is handled in the main finally block

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)
    
    print("警告：请确保机器人周围没有障碍物。")
    # input("按 Enter 键继续...") # Temporarily comment out for faster testing if needed
    
    controller = None # Initialize controller to None for finally block
    try:
        ChannelFactoryInitialize(0, sys.argv[1])
        controller = RobotController()
        # User ack for starting can be here if desired
        input("RobotController 初始化完毕。按 Enter 键开始执行状态机...")
        controller.run()
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
                    # controller.stop_moving(identifier=99) # stop_moving calls StopMove
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
            
            # Shutdown PositionDetector - REMOVED
            # try:
            #     if hasattr(controller, 'pos_detector') and controller.pos_detector:
            #         logging.info("主清理：正在关闭 PositionDetector...")
            #         controller.pos_detector.shutdown()
            #         logging.info("主清理：PositionDetector 已关闭。")
            # except Exception as e_pos:
            #     logging.error(f"主清理：关闭 PositionDetector 时发生错误: {e_pos}")
        
        # Final safety net, ensure robot is told to stop if sport_client was ever initialized
        # This is somewhat redundant if controller.sport_client.StopMove() above worked,
        # but good as a fallback if controller object itself had issues during init.
        # Consider if ChannelFactory needs explicit deinitialization if it's used globally.

        logging.info("清理完成。程序退出。")