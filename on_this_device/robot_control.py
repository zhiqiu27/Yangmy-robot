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
from robot_state import robot_state, get_robot_state, get_robot_position

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
        # 导入机器人状态获取函数

        
        # 获取最新的机器人位置信息
        current_position = get_robot_position()
        if current_position is None:
            logger.error("机器人状态不可用，无法执行方向旋转")
            return False
        
        if direction not in DIRECTION_ANGLES:
            logger.error(f"未知的方向命令: {direction}")
            return False
        
        # 获取初始状态
        current_x = current_position['x']
        current_y = current_position['y']
        initial_yaw = current_position['yaw']  # 记录初始角度
        
        # 计算新的yaw角度
        angle_increment = DIRECTION_ANGLES[direction]
        new_yaw = normalize_angle(initial_yaw + angle_increment)
        
        logger.info(f"执行方向旋转: {direction}")
        logger.info(f"当前位置: ({current_x:.3f}, {current_y:.3f}, yaw: {initial_yaw:.3f})")
        logger.info(f"目标位置: ({current_x:.3f}, {current_y:.3f}, yaw: {new_yaw:.3f})")
        logger.info(f"需要旋转角度: {angle_increment:.3f} 弧度 ({math.degrees(angle_increment):.1f}°)")
        
        # 从配置文件获取PID参数
        pid_params = DIRECTION_ROTATION_PID_PARAMS
        
        try:
            # PID参数
            Kp = pid_params['Kp']
            Ki = pid_params['Ki']
            Kd = pid_params['Kd']
            max_integral = pid_params['max_integral']
            tolerance = pid_params['tolerance']
            timeout = pid_params['timeout']
            max_speed_base = pid_params['max_speed_base']
            min_speed = pid_params['min_speed']
            speed_factor = pid_params['speed_factor']
            
            integral = 0
            prev_error = 0
            start_time = time.time()
            last_time = start_time
            
            while time.time() - start_time < timeout:
                # 重新获取最新的机器人位置信息
                current_position = get_robot_position()
                if current_position is None:
                    logger.error("机器人状态丢失，停止旋转")
                    break
                
                # 获取当前yaw角
                current_yaw = current_position['yaw']
                current_time = time.time()
                dt = current_time - last_time
                
                # 计算误差和已旋转角度
                error = normalize_angle(new_yaw - current_yaw)
                rotated_angle = normalize_angle(current_yaw - initial_yaw)
                remaining_angle = normalize_angle(new_yaw - current_yaw)
                
                # 实时打印旋转进度
                logger.info(f"旋转进度 - 需要: {math.degrees(angle_increment):.1f}°, "
                           f"已转: {math.degrees(rotated_angle):.1f}°, "
                           f"剩余: {math.degrees(remaining_angle):.1f}°, "
                           f"误差: {math.degrees(error):.1f}°")
                
                # 更新积分和微分项（使用实际dt）
                if dt > 0:
                    integral += error * dt
                    # 积分饱和保护
                    integral = max(min(integral, max_integral), -max_integral)
                    derivative = (error - prev_error) / dt
                else:
                    derivative = 0
                
                # PID控制器输出
                vyaw = Kp * error + Ki * integral + Kd * derivative
                
                # 动态速度限制
                max_speed = min(max_speed_base, max(min_speed, abs(error) * speed_factor))
                vyaw = max(min(vyaw, max_speed), -max_speed)
                
                # 执行旋转
                self.sport_client.Move(0.0, 0.0, vyaw)
                
                # 检查是否达到目标
                if abs(error) < tolerance:
                    self.sport_client.StopMove()
                    time.sleep(0.1)  # 确保停止命令执行
                    logger.info(f"方向旋转完成: {direction}")
                    logger.info(f"最终旋转角度: {math.degrees(rotated_angle):.1f}°")
                    return True
                    
                prev_error = error
                last_time = current_time
                time.sleep(0.1)
            
            # 超时处理
            self.sport_client.StopMove()
            time.sleep(0.1)
            final_position = get_robot_position()
            if final_position:
                final_yaw = final_position['yaw']
                final_rotated = normalize_angle(final_yaw - initial_yaw)
                logger.warning(f"方向旋转超时 - 目标: {math.degrees(angle_increment):.1f}°, "
                              f"实际: {math.degrees(final_rotated):.1f}°")
            return False
            
        except Exception as e:
            logger.error(f"执行方向旋转时发生错误: {e}")
            try:
                self.sport_client.StopMove()
            except:
                pass
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