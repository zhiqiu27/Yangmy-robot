# robot_utils.py
"""
机器人控制器工具函数
包含数学计算、坐标转换、串口通信等辅助功能
"""

import math
import time
import logging
import serial
import numpy as np
from robot_config import SERIAL_PORT, SERIAL_BAUDRATE

logger = logging.getLogger(__name__)

def normalize_angle(angle):
    """将角度归一化到 [-π, π]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def transform_global_xy_to_robot_xy(global_xy, robot_xy, yaw):
    """将全局坐标转换为机器人坐标系"""
    robot_x = robot_xy[0]
    robot_y = robot_xy[1]
    global_x = global_xy[0]
    global_y = global_xy[1]
    target_from_go1_xyz = [global_x - robot_x, global_y - robot_y]
    global_x_in_robot = target_from_go1_xyz[0] * np.cos(yaw) + target_from_go1_xyz[1] * np.sin(yaw)
    global_y_in_robot = - target_from_go1_xyz[0] * np.sin(yaw) + target_from_go1_xyz[1] * np.cos(yaw)
    return np.array([global_x_in_robot, global_y_in_robot])

def send_hex_to_serial(hex_data, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
    """发送十六进制数据到串口"""
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        
        if ser.is_open:
            logger.info(f"连接到 {ser.name}")
            ser.write(hex_data)
            logger.info(f"发送十六进制数据: {hex_data.hex(' ')}")
            time.sleep(0.1)
        
        ser.close()
        return True
    except serial.SerialException as e:
        logger.error(f"串口错误: {e}")
        return False
    except Exception as e:
        logger.error(f"发送十六进制数据时出错: {e}")
        return False

def convert_relative_to_global(relative_coord_robot_frame, robot_state):
    """将相对于机器人的坐标转换为全局坐标"""
    if robot_state is None:
        logger.error("机器人状态不可用，无法进行坐标转换")
        raise RuntimeError("机器人状态不可用，无法进行坐标转换")
    
    current_robot_px = robot_state.position[0]
    current_robot_py = robot_state.position[1]
    current_robot_yaw = robot_state.imu_state.rpy[2]
    
    relative_forward = relative_coord_robot_frame[0]
    relative_left = relative_coord_robot_frame[1]
    
    # 将相对坐标旋转到全局坐标系
    offset_x_global = relative_forward * math.cos(current_robot_yaw) - relative_left * math.sin(current_robot_yaw)
    offset_y_global = relative_forward * math.sin(current_robot_yaw) + relative_left * math.cos(current_robot_yaw)
    
    # 添加全局偏移到机器人当前全局位置
    global_x = current_robot_px + offset_x_global
    global_y = current_robot_py + offset_y_global
    
    logger.info(f"坐标转换: 机器人位置 [{current_robot_px:.2f}, {current_robot_py:.2f}, yaw:{current_robot_yaw:.2f}rad]")
    logger.info(f"相对坐标 (前进,左) {relative_coord_robot_frame} -> 全局目标 [{global_x:.2f}, {global_y:.2f}]")
    
    return [global_x, global_y]

def calculate_robot_center_target(object_global_coord, claw_offset_forward):
    """计算机器人中心应该到达的目标位置"""
    current_robot_px = object_global_coord[0]  # 这里需要实际的机器人状态
    current_robot_py = object_global_coord[1]
    
    # 计算接近角度
    delta_x = object_global_coord[0] - current_robot_px
    delta_y = object_global_coord[1] - current_robot_py
    approach_yaw = math.atan2(delta_y, delta_x)
    
    # 计算机器人中心目标位置
    robot_center_x = object_global_coord[0] - claw_offset_forward * math.cos(approach_yaw)
    robot_center_y = object_global_coord[1] - claw_offset_forward * math.sin(approach_yaw)
    
    return [robot_center_x, robot_center_y], approach_yaw

def setup_logging(log_level, log_file, log_format):
    """设置日志配置"""
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    ) 