#!/usr/bin/env python3
"""
导航第二阶段调试文件
提取自state_machine.py的_phase2_move_and_adjust_to_target方法
支持相对坐标导航调试
"""

import time
import math
import sys
import logging
import numpy as np
from simple_pid import PID
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# 从原文件导入必要的常量
from state_machine import (
    MIN_EFFECTIVE_VYAW,
    IMAGE_SERVER_HOST,
    IMAGE_SERVER_PORT
)
import socket
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/home/unitree/navigation_debug.log'),
        logging.StreamHandler()
    ]
)

class NavigationDebugger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化 NavigationDebugger")
        
        # 本地机器人状态变量
        self.robot_state = None
        
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
            self.sub.Init(self._state_handler, 20)
            time.sleep(2) # Allow time for first message
            if self.robot_state is None:
                self.logger.error("未能接收到机器人状态")
            else:
                self.logger.info("状态订阅初始化成功")
        except Exception as e:
            self.logger.error(f"状态订阅初始化失败: {e}")
            raise RuntimeError(f"状态订阅初始化失败: {e}")

        # 初始化PID控制器
        self.dist_pid = PID(Kp=0.5, Ki=0.1, Kd=0.05, setpoint=0, output_limits=(-0.5, 0.5))
        self.yaw_pid_visual = PID(Kp=0.001, Ki=0.00005, Kd=0.0005, output_limits=(-0.3, 0.3), setpoint=0)
        
        self.logger.info("NavigationDebugger 初始化完成")

    def _state_handler(self, msg: SportModeState_):
        """本地状态处理器"""
        self.robot_state = msg
        self.logger.debug(f"机器人状态更新: 位置({msg.position[0]:.2f}, {msg.position[1]:.2f}), 偏航角{msg.imu_state.rpy[2]:.2f}")

    def convert_relative_to_global(self, relative_coord_robot_frame):
        """将相对于机器人当前位置和方向的坐标转换为全局坐标"""
        if self.robot_state is None:
            self.logger.error("Robot state not available for coordinate conversion.")
            raise RuntimeError("Robot state not available for coordinate conversion.")
        
        current_robot_px = self.robot_state.position[0]
        current_robot_py = self.robot_state.position[1]
        current_robot_yaw = self.robot_state.imu_state.rpy[2]
        
        relative_forward = relative_coord_robot_frame[0]
        relative_left = relative_coord_robot_frame[1]
        
        # 旋转相对坐标以对齐全局坐标系
        offset_x_global = relative_forward * math.cos(current_robot_yaw) - relative_left * math.sin(current_robot_yaw)
        offset_y_global = relative_forward * math.sin(current_robot_yaw) + relative_left * math.cos(current_robot_yaw)
        
        # 将全局偏移量添加到机器人当前全局位置
        global_x = current_robot_px + offset_x_global
        global_y = current_robot_py + offset_y_global
        
        self.logger.info(f"坐标转换: 机器人位置 [{current_robot_px:.2f}, {current_robot_py:.2f}, yaw:{current_robot_yaw:.2f}rad]. "
                        f"相对坐标 (前进,左) {relative_coord_robot_frame} -> 全局目标 [{global_x:.2f}, {global_y:.2f}]")
        return [global_x, global_y]

    def _request_visual_info_from_server(self):
        """从图像服务器请求视觉信息"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((IMAGE_SERVER_HOST, IMAGE_SERVER_PORT))
                self.logger.debug("已连接到图像服务器 (请求视觉信息)。")
                s.sendall(b"REQUEST_VISUAL_INFO")
                
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
                            'depth_x': response.get('depth_x')
                        }
                    elif not response.get("bbox_available"):
                        self.logger.info("视觉信息：图像服务器报告当前无可用边界框。")
                        return {'bbox_available': False, 'depth_x': None}
                    else:
                        self.logger.error(f"图像服务器成功响应但视觉信息格式不正确: {response}")
                        return None
                else:
                    self.logger.error(f"图像服务器报告错误 (视觉信息): {response.get('message', '未知错误')}")
                    return None
        except socket.timeout:
            self.logger.warning(f"连接图像服务器超时。")
            return None
        except Exception as e:
            self.logger.error(f"请求视觉信息时发生错误: {e}")
            return None

    def try_move(self, vx, vy, vyaw, identifier: int):
        """执行机器人移动命令"""
        self.logger.debug(f"Move命令 - ID: {identifier}, vx: {vx:.3f}, vy: {vy:.3f}, vyaw: {vyaw:.3f}")
        self.sport_client.Move(vx, vy, vyaw)

    def phase2_navigate_with_distance_control(self, global_target, distance_tolerance=0.1, max_nav_time=90.0, control_period=0.02):
        """
        基于全局位置距离的导航控制：离目标越近速度越慢
        
        Args:
            global_target: 全局目标坐标 [x, y]
            distance_tolerance: 距离容差 (米)
            max_nav_time: 最大导航时间 (秒)
            control_period: 控制周期 (秒)
        
        Returns:
            bool: 导航是否成功
        """
        self.logger.info(f"开始基于距离的导航控制: 目标 {global_target}, 距离容差 {distance_tolerance:.2f}m")
        start_time = time.time()
        
        # 配置距离PID控制器 - 设定点为0（目标距离）
        self.dist_pid.setpoint = 0
        self.dist_pid.reset()
        
        # 速度参数
        max_speed = 0.3  # 最大前进速度 m/s
        min_speed = 0.02  # 最小有效速度 m/s
        
        while time.time() - start_time < max_nav_time:
            if self.robot_state is None:
                self.logger.error("机器人状态不可用，停止导航")
                self.sport_client.StopMove()
                return False
            
            # 获取当前机器人位置
            current_x = self.robot_state.position[0]
            current_y = self.robot_state.position[1]
            current_yaw = self.robot_state.imu_state.rpy[2]
            
            # 计算到目标的距离和方向
            delta_x = global_target[0] - current_x
            delta_y = global_target[1] - current_y
            distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
            
            self.logger.info(f"当前位置: [{current_x:.3f}, {current_y:.3f}], 目标: {global_target}, 距离: {distance_to_target:.3f}m")
            
            # 检查是否到达目标
            if distance_to_target <= distance_tolerance:
                self.logger.info(f"已到达目标！距离: {distance_to_target:.3f}m <= 容差: {distance_tolerance:.3f}m")
                #self.sport_client.StopMove()
                return True
            
            # 计算目标方向（全局坐标系）
            target_yaw_global = math.atan2(delta_y, delta_x)
            
            # 计算偏航误差
            yaw_error = target_yaw_global - current_yaw
            # 将偏航误差归一化到 [-π, π]
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi
            
            # 偏航控制
            vyaw = 0.0
            yaw_tolerance = 0.1  # 偏航容差 (弧度)
            if abs(yaw_error) > yaw_tolerance:
                # 使用简单比例控制
                vyaw = yaw_error * 0.5  # 比例系数
                vyaw = max(-0.3, min(0.3, vyaw))  # 限制偏航速度
            
            # 基于距离的前进速度控制
            vx = 0.0
            if abs(yaw_error) < 0.3:  # 只有当方向基本正确时才前进
                # 直接基于距离计算速度，距离越远速度越快，距离越近速度越慢
                # 使用简单的比例控制而不是PID
                if distance_to_target > distance_tolerance:
                    # 基础速度与距离成正比
                    vx = distance_to_target * 0.2  # 比例系数0.2
                    
                    # 应用速度限制
                    vx = min(vx, max_speed)  # 最大速度限制
                    vx = max(min_speed, vx)  # 最小有效速度
                    
                    # 在接近目标时进一步减速
                    if distance_to_target < 0.5:  # 在0.5米内开始额外减速
                        slow_factor = distance_to_target / 0.5
                        vx *= slow_factor
                        vx = max(min_speed, vx)  # 保持最小速度
                else:
                    vx = 0.0  # 已到达目标范围
            
            # 执行移动命令
            self.logger.info(f"vx: {vx:.3f}, vyaw: {vyaw:.3f}, 距离: {distance_to_target:.3f}m, 偏航误差: {yaw_error:.3f}rad")
            self.try_move(vx, 0.0, vyaw, 13)
            time.sleep(control_period)
        
        # 超时
        self.logger.error("基于距离的导航控制超时")
        self.sport_client.StopMove()
        return False

    def navigate_to_relative_target(self, relative_target, distance_tolerance=0.1):
        """
        使用相对坐标导航到目标
        
        Args:
            relative_target: [前进距离, 左移距离] 相对于机器人当前位置和方向
            distance_tolerance: 距离容差 (米)
        
        Returns:
            bool: 导航是否成功
        """
        self.logger.info(f"开始相对坐标导航: 目标 {relative_target}, 距离容差 {distance_tolerance}m")
        
        # 将相对坐标转换为全局坐标
        try:
            global_target = self.convert_relative_to_global(relative_target)
            self.logger.info(f"转换后的全局目标: {global_target}")
        except Exception as e:
            self.logger.error(f"坐标转换失败: {e}")
            return False
        
        # 重置PID控制器
        self.dist_pid.reset()
        
        # 执行基于距离的导航
        success = self.phase2_navigate_with_distance_control(
            global_target=global_target,
            distance_tolerance=distance_tolerance,
            max_nav_time=90.0,
            control_period=0.02
        )
        
        if success:
            self.logger.info("相对坐标导航成功完成")
        else:
            self.logger.error("相对坐标导航失败")
        
        return success

    def test_navigation(self):
        """测试导航功能"""
        self.logger.info("开始导航测试")
        
        # 等待机器人状态稳定
        time.sleep(1)
        
        if self.robot_state is None:
            self.logger.error("机器人状态不可用，无法进行测试")
            return False
        
        # 测试相对坐标导航
        # 示例：前进1.5米，左移0米，距离容差0.1米
        relative_target = [0.5, 0.0]
        distance_tolerance = 0.1
        
        user_input = input(f"准备测试导航到相对目标 {relative_target}，距离容差 {distance_tolerance}m。按Enter开始，或输入'q'退出: ")
        if user_input.lower() == 'q':
            return False
        
        success = self.navigate_to_relative_target(relative_target, distance_tolerance)
        
        if success:
            time.sleep(8)
            self.logger.info("导航测试成功完成")
        else:
            self.logger.error("导航测试失败")
        
        return success

    def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理...")
        try:
            if hasattr(self, 'sport_client') and self.sport_client:
                self.sport_client.StopMove()
                self.logger.info("机器人已停止")
        except Exception as e:
            self.logger.error(f"清理时发生错误: {e}")

def main():
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)
    
    print("导航调试器 - 警告：请确保机器人周围没有障碍物。")
    
    debugger = None
    try:
        ChannelFactoryInitialize(0, sys.argv[1])
        debugger = NavigationDebugger()
        
        input("NavigationDebugger 初始化完毕。按 Enter 键开始测试...")
        debugger.test_navigation()
        
    except Exception as e:
        logging.critical(f"程序发生严重错误: {e}", exc_info=True)
    finally:
        if debugger:
            debugger.cleanup()
        logging.info("程序退出")

if __name__ == "__main__":
    main() 