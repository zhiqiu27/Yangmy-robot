# robot_config.py
"""
机器人控制器配置文件
集中管理所有常量和配置参数
"""

import enum
import logging

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FILE = '/home/unitree/robot_controller.log'
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'

# 图像服务器配置
IMAGE_SERVER_HOST = 'localhost'
IMAGE_SERVER_PORT = 50001
DIRECTION_COMMAND_PORT = 50002
NEXT_TARGET_PORT = 12347  # PC端用于接收NEXT_TARGET命令的端口

# 机器人物理参数
CLAW_OFFSET_FORWARD = 0.08  # 机器人中心到爪子前端的距离(米)

# 运动控制参数
MIN_EFFECTIVE_VYAW = 0.08  # 最小有效偏航速度(rad/s)

# 高度控制参数
BODY_HEIGHT_REL_MIN = -0.18
BODY_HEIGHT_REL_MAX = 0.03
FOOT_RAISE_REL_MIN = -0.06
FOOT_RAISE_REL_MAX = 0.03
DEFAULT_BODY_HEIGHT_ABS = 0.33
DEFAULT_FOOT_RAISE_HEIGHT_ABS = 0.09

# PID控制器参数
YAW_PID_PARAMS = {'Kp': 0.5, 'Ki': 0.02, 'Kd': 0.3, 'output_limits': (-0.6, 0.6)}
YAW_PID_FINE_PARAMS = {'Kp': 0.25, 'Ki': 0.01, 'Kd': 0.15, 'output_limits': (-0.3, 0.3)}
DIST_PID_PARAMS = {'Kp': 0.25, 'Ki': 0.0, 'Kd': 0.25, 'setpoint': 0, 'output_limits': (-0.3, 0.3)}
YAW_PID_VISUAL_PARAMS = {'Kp': 0.001, 'Ki': 0.00005, 'Kd': 0.0005, 'output_limits': (-0.3, 0.3), 'setpoint': 0}

# 方向旋转PID参数
DIRECTION_ROTATION_PID_PARAMS = {
    'Kp': 0.4,           # 比例系数
    'Ki': 0.01,          # 积分系数
    'Kd': 0.1,           # 微分系数
    'max_integral': 1.0, # 积分饱和保护
    'tolerance': 0.087,  # 容差（约5度）
    'timeout': 18.0,      # 超时时间（秒）
    'max_speed_base': 0.4,    # 基础最大速度
    'min_speed': 0.1,         # 最小速度
    'speed_factor': 0.6       # 速度因子
}

# 导航参数
NAVIGATION_CONTROL_PERIOD = 0.02
YAW_TOLERANCE = 0.08
DIST_TOLERANCE = 0.08
MAX_ROTATION_TIME = 15.0
MAX_NAVIGATION_TIME = 90.0
TARGET_DEPTH = 0.8  # 目标深度(米)

# 视觉对准参数
YAW_PIXEL_TOLERANCE = 3
YAW_PIXEL_TOLERANCE_PHASE2 = 3
YAW_PIXEL_TOLERANCE_CONFIRM = 3
YAW_PIXEL_TOLERANCE_CRAWL = 4
MAX_VISUAL_ALIGN_TIME = 20.0
MAX_CONSECUTIVE_NO_BBOX = 10

# 精细接近参数
TARGET_CONFIRM_DEPTH = 0.2
TARGET_GRASP_DEPTH = 0.15
MAX_APPROACH_TIME = 120.0
MAX_ALIGN_TIME_AT_DEPTH = 100.0
MAX_CRAWL_TIME = 100.0
APPROACH_VX = 0.04
CRAWL_VX = 0.015

# 俯仰角参数
DESIRED_PITCH_ANGLE = 20.0
GRASP_PITCH_ANGLE = 10

# 串口参数
SERIAL_PORT = '/dev/wheeltec_mic'
SERIAL_BAUDRATE = 115200

# 爪子控制命令
GRASP_COMMAND = bytes([0x7b, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf8, 0x7d])
RELEASE_COMMAND = bytes([0x7b, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf9, 0x7d])

# 状态定义
class State(enum.Enum):
    IDLE = 0
    NAVIGATE_TO_OBJECT = 1
    LOWER_HEAD = 2
    GRASP_OBJECT = 3
    SEND_NEXT_COMMAND = 4
    NAVIGATE_TO_PERSON = 5
    RELEASE_OBJECT = 6
    DONE = 7
    CALIBRATE_SHIRT_POSITION = 8
    WAIT = 9
    FINE_ADJUST_AND_APPROACH = 10

# 方向命令映射
DIRECTION_ANGLES = {
    "forward": 0.0,
    "backward": 3.14159,  # math.pi
    "left": 1.5708,       # math.pi/2
    "right": -1.5708,     # -math.pi/2
    "front-left": 0.7854,  # math.pi/4
    "front-right": -0.7854, # -math.pi/4
    "back-left": 2.3562,   # 3*math.pi/4
    "back-right": -2.3562  # -3*math.pi/4
}

# 目标列表
TARGET_LIST = [
    {"name": "shirt", "coord": [0.0, 0.0]},  # 初始占位符，将在校准后更新
    {"name": "person", "coord": [-0.6, 0]}   # 相对坐标
] 