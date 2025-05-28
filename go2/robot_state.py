"""
机器人状态全局变量模块
避免循环导入问题
"""

import threading
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# 全局机器人状态和锁
robot_state = None
_state_lock = threading.Lock()

def HighStateHandler(msg: SportModeState_):
    """机器人状态处理器"""
    global robot_state
    with _state_lock:
        robot_state = msg

def get_robot_state():
    """线程安全地获取机器人状态的副本"""
    with _state_lock:
        return robot_state

def get_robot_position():
    """线程安全地获取机器人位置信息"""
    with _state_lock:
        if robot_state is None:
            return None
        return {
            'x': robot_state.position[0],
            'y': robot_state.position[1],
            'yaw': robot_state.imu_state.rpy[2]
        } 