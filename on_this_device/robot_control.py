# robot_control.py
"""
机器人控制模块
包含高度控制、爪子控制、俯仰角控制等功能
"""

import time
import math
import logging
from robot_config import *
from robot_utils import send_hex_to_serial, normalize_angle

logger = logging.getLogger(__name__)

class RobotControl:
    """机器人基础控制功能"""
    
    def __init__(self, sport_client, pitch_controller):
        self.sport_client = sport_client
        self.pitch_controller = pitch_controller
        
        # 高度状态跟踪
        self.current_body_height_abs = DEFAULT_BODY_HEIGHT_ABS
        self.current_foot_raise_height_abs = DEFAULT_FOOT_RAISE_HEIGHT_ABS
        self.current_relative_body_height = 0.0
    
    def set_body_height_relative(self, relative_height: float):
        """设置机身相对高度"""
        logger.info(f"调整机身相对高度为: {relative_height:.3f}m")
        
        if not (BODY_HEIGHT_REL_MIN <= relative_height <= BODY_HEIGHT_REL_MAX):
            logger.error(f"机身相对高度 {relative_height:.3f} 超出允许范围")
            return False
        
        interpolation_duration = 1.0
        control_period = 0.02
        num_steps = max(1, int(interpolation_duration / control_period))
        
        start_relative_height = self.current_relative_body_height
        height_difference = relative_height - start_relative_height
        
        last_sdk_call_successful = True
        
        # 插值调整
        for i in range(1, num_steps + 1):
            interpolated_fraction = i / num_steps
            current_target_relative_height = start_relative_height + height_difference * interpolated_fraction
            
            try:
                self.sport_client.BodyHeight(current_target_relative_height)
            except Exception as e:
                logger.error(f"执行 BodyHeight 时发生错误: {e}")
                last_sdk_call_successful = False
            
            time.sleep(control_period)
        
        # 最终调用
        try:
            ret = self.sport_client.BodyHeight(relative_height)
            if ret == 0:
                logger.info(f"BodyHeight 调用成功")
                self.current_relative_body_height = relative_height
                self.current_body_height_abs = DEFAULT_BODY_HEIGHT_ABS + relative_height
                return True and last_sdk_call_successful
            else:
                logger.error(f"BodyHeight 调用失败，错误码: {ret}")
                return False
        except Exception as e:
            logger.error(f"执行 BodyHeight 时发生错误: {e}")
            return False
    
    def set_foot_raise_height_relative(self, relative_height: float):
        """设置抬足相对高度"""
        logger.info(f"调整抬足相对高度为: {relative_height:.3f}m")
        
        if not (FOOT_RAISE_REL_MIN <= relative_height <= FOOT_RAISE_REL_MAX):
            logger.error(f"抬足相对高度 {relative_height:.3f} 超出允许范围")
            return False
        
        try:
            ret = self.sport_client.FootRaiseHeight(relative_height)
            if ret == 0:
                logger.info(f"FootRaiseHeight 调用成功")
                self.current_foot_raise_height_abs = DEFAULT_FOOT_RAISE_HEIGHT_ABS + relative_height
                return True
            else:
                logger.error(f"FootRaiseHeight 调用失败，错误码: {ret}")
                return False
        except Exception as e:
            logger.error(f"执行 FootRaiseHeight 时发生错误: {e}")
            return False
    
    def reset_body_and_foot_height(self):
        """恢复机身和抬足到默认高度"""
        logger.info("正在恢复机身和抬足到默认高度...")
        
        body_reset_ok = self.set_body_height_relative(0.0)
        time.sleep(0.5)
        foot_reset_ok = self.set_foot_raise_height_relative(0.0)
        
        if body_reset_ok and foot_reset_ok:
            logger.info("机身和抬足高度已成功恢复到默认")
            return True
        else:
            logger.error("恢复机身或抬足高度失败")
            return False
    
    def set_pitch_angle(self, angle: float):
        """设置俯仰角"""
        logger.info(f"设置俯仰角为: {angle}°")
        
        if self.pitch_controller:
            self.pitch_controller.set_pitch(angle)
            return True
        else:
            logger.warning("PitchController 未初始化，无法调整俯仰角")
            return False
    
    def reset_pitch(self):
        """重置俯仰角到0度"""
        logger.info("重置俯仰角到0度")
        
        if self.pitch_controller:
            self.pitch_controller.reset_pitch()
            return True
        else:
            logger.warning("PitchController 未初始化，无法重置俯仰角")
            return False
    
    def grasp_object(self):
        """抓取物体"""
        logger.info("执行抓取动作")
        success = send_hex_to_serial(GRASP_COMMAND)
        
        if success:
            logger.info("抓取命令发送成功")
        else:
            logger.error("抓取命令发送失败")
        
        time.sleep(1.5)  # 等待物理抓取完成
        return success
    
    def release_object(self):
        """释放物体"""
        logger.info("执行释放动作")
        success = send_hex_to_serial(RELEASE_COMMAND)
        
        if success:
            logger.info("释放命令发送成功")
        else:
            logger.error("释放命令发送失败")
        
        time.sleep(1.0)  # 等待物理释放完成
        return success
    
    def stop_movement(self):
        """停止机器人移动"""
        logger.info("停止机器人移动")
        self.sport_client.StopMove()
    
    def execute_direction_rotation(self, direction, robot_state):
        """根据方向命令执行旋转"""
        if robot_state is None:
            logger.error("机器人状态不可用，无法执行方向旋转")
            return False
        
        if direction not in DIRECTION_ANGLES:
            logger.error(f"未知的方向命令: {direction}")
            return False
        
        # 获取当前状态
        current_x = robot_state.position[0]
        current_y = robot_state.position[1]
        current_yaw = robot_state.imu_state.rpy[2]
        
        # 计算新的yaw角度
        angle_increment = DIRECTION_ANGLES[direction]
        new_yaw = normalize_angle(current_yaw + angle_increment)
        
        logger.info(f"执行方向旋转: {direction}")
        logger.info(f"当前位置: ({current_x:.3f}, {current_y:.3f}, yaw: {current_yaw:.3f})")
        logger.info(f"目标位置: ({current_x:.3f}, {current_y:.3f}, yaw: {new_yaw:.3f})")
        
        try:
            ret = self.sport_client.MoveToPos(current_x, current_y, new_yaw)
            if ret == 0:
                logger.info(f"方向旋转命令发送成功: {direction}")
                time.sleep(3.0)  # 等待旋转完成
                return True
            else:
                logger.error(f"MoveToPos调用失败，错误码: {ret}")
                return False
        except Exception as e:
            logger.error(f"执行方向旋转时发生错误: {e}")
            return False
    
    def prepare_for_grasp(self):
        """准备抓取姿态"""
        logger.info("准备抓取姿态")
        
        # 设置机身和抬足高度
        body_ok = self.set_body_height_relative(BODY_HEIGHT_REL_MIN)
        if body_ok:
            time.sleep(0.5)
            foot_ok = self.set_foot_raise_height_relative(FOOT_RAISE_REL_MIN)
            if foot_ok:
                time.sleep(1.5)
                # 设置俯仰角
                pitch_ok = self.set_pitch_angle(GRASP_PITCH_ANGLE)
                if pitch_ok:
                    time.sleep(2.5)  # 等待俯仰角调整完成
                    logger.info("抓取姿态准备完成")
                    return True
        
        logger.error("准备抓取姿态失败")
        return False
    
    def restore_default_posture(self):
        """恢复默认姿态"""
        logger.info("恢复默认姿态")
        
        # 重置俯仰角
        self.reset_pitch()
        
        # 恢复高度
        height_ok = self.reset_body_and_foot_height()
        
        # 等待恢复完成
        if self.pitch_controller:
            pitch_duration = self.pitch_controller._shared_state.get('interpolation_duration_s', 2.0)
            time.sleep(max(pitch_duration, 2.5) + 0.5)
        else:
            time.sleep(3.0)
        
        if height_ok:
            logger.info("默认姿态恢复完成")
            return True
        else:
            logger.error("恢复默认姿态失败")
            return False 