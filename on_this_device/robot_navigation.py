# robot_navigation.py
"""
机器人导航模块
包含导航、视觉对准、精细接近等功能
"""

import time
import math
import logging
import collections
import numpy as np
from simple_pid import PID
from robot_config import *
from robot_utils import normalize_angle
from robot_communication import RobotCommunication

# 导入全局robot_state
from robot_state import robot_state, get_robot_state, get_robot_position

logger = logging.getLogger(__name__)

class RobotNavigation:
    """机器人导航控制器"""
    
    def __init__(self, sport_client):
        self.sport_client = sport_client
        self.communication = RobotCommunication()
        
        # 初始化PID控制器
        self.yaw_pid = PID(**YAW_PID_PARAMS)
        self.yaw_pid_fine_tune = PID(**YAW_PID_FINE_PARAMS)
        self.dist_pid = PID(**DIST_PID_PARAMS)
        self.yaw_pid_visual = PID(**YAW_PID_VISUAL_PARAMS)
        
        # 像素误差缓冲区
        self.stage_c_pixel_error_buffer = collections.deque(maxlen=5)
    
    def navigate_to(self, global_goal, robot_state):
        """导航到全局坐标目标点"""
        if robot_state is None:
            logger.error("机器人状态不可用")
            initial_robot_px, initial_robot_py, initial_robot_yaw = 0.0, 0.0, 0.0
            logger.warning("使用默认初始状态 (0,0,0)")
        else:
            initial_robot_px = robot_state.position[0]
            initial_robot_py = robot_state.position[1]
            initial_robot_yaw = robot_state.imu_state.rpy[2]
        
        logger.info(f"开始导航: 从 [{initial_robot_px:.2f}, {initial_robot_py:.2f}, Yaw:{initial_robot_yaw:.2f}rad] 到 {global_goal}")
        
        try:
            # 重置所有PID控制器
            self._reset_all_pids()
            
            # 第一阶段：视觉对准
            logger.info("阶段1: 视觉Yaw对准...")
            self._phase1_visual_align_yaw()
            logger.info("阶段1: 视觉Yaw对准完成")
            
            # 获取精确的世界坐标
            refined_global_goal = self._refine_global_goal_after_alignment(global_goal, robot_state)
            
            time.sleep(0.5)  # 等待视觉旋转完成
            
            # 第二阶段：前进和调整
            logger.info("阶段2: 前进和调整...")
            self._phase2_move_and_adjust_to_target(refined_global_goal, robot_state)
            logger.info("阶段2: 完成")
            
            self.sport_client.StopMove()
            logger.info("导航成功完成，机器人已停止")
            
            return refined_global_goal
            
        except RuntimeError as e:
            logger.error(f"导航失败: {e}")
            self.sport_client.StopMove()
            raise
        except Exception as e:
            logger.error(f"导航中发生未知错误: {e}")
            self.sport_client.StopMove()
            raise RuntimeError(f"导航中发生未知错误: {e}")
    
    def approach_and_confirm_alignment(self, robot_state):
        """精细接近与对准流程"""
        logger.info("开始精细接近与对准流程")
        
        self.yaw_pid_visual.reset()
        
        # 获取平均深度值
        avg_depth = self._get_average_depth()
        if avg_depth is None:
            logger.error("无法获取有效深度值")
            return False

        # 阶段A：前进到确认深度
        if not self._stage_a_approach_to_depth(avg_depth):
            return False
        
        # 阶段B：精确偏航对准
        #if not self._stage_b_precise_yaw_alignment():
           # return False
        
        logger.info("精细接近与对准流程成功完成")
        return True
    
    def _reset_all_pids(self):
        """重置所有PID控制器"""
        self.yaw_pid.reset()
        self.yaw_pid_fine_tune.reset()
        self.dist_pid.reset()
        self.yaw_pid_visual.reset()
    
    def _phase1_visual_align_yaw(self):
        """第一阶段：使用视觉反馈调整Yaw"""
        logger.info("开始视觉对准")
        start_time = time.time()
        self.yaw_pid_visual.reset()
        
        consecutive_no_bbox_count = 0
        
        while time.time() - start_time < MAX_VISUAL_ALIGN_TIME:
            visual_info = self.communication.request_visual_info()
            
            if visual_info and visual_info.get('bbox_available'):
                consecutive_no_bbox_count = 0
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x
                
                logger.debug(f"视觉对准 - 像素误差: {pixel_error:.1f}")
                
                if abs(pixel_error) < YAW_PIXEL_TOLERANCE:
                    logger.info(f"视觉对准完成，像素误差: {pixel_error:.1f}")
                    self.sport_client.StopMove()
                    return True
                
                vyaw = self.yaw_pid_visual(pixel_error)
                
                # 应用最小有效偏航速度
                if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
                    vyaw = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                
                self.sport_client.Move(0.0, 0.0, vyaw)
                
            elif visual_info and not visual_info.get('bbox_available'):
                logger.warning("视觉对准: 未检测到边界框")
                self.sport_client.StopMove()
                consecutive_no_bbox_count += 1
                if consecutive_no_bbox_count > MAX_CONSECUTIVE_NO_BBOX:
                    logger.error("视觉对准失败：长时间无边界框")
                    raise RuntimeError("视觉对准失败：长时间无边界框")
            else:
                logger.error("视觉对准: 获取视觉信息失败")
                self.sport_client.StopMove()
            
            time.sleep(NAVIGATION_CONTROL_PERIOD)
        
        logger.error("视觉对准超时")
        self.sport_client.StopMove()
        raise RuntimeError("视觉对准超时")
    
    def _refine_global_goal_after_alignment(self, original_global_goal, robot_state_param):
        """视觉对准后优化全局目标"""
        refined_relative_coord = self.communication.request_calibration()
        
        if not refined_relative_coord:
            logger.warning("未能获取精确校准坐标，使用原始目标")
            return original_global_goal
        
        logger.info(f"获得精确相对坐标: {refined_relative_coord}")
        
        # 获取视觉对准后的最新机器人状态（使用全局robot_state）
        if robot_state is None:
            logger.error("视觉对准后无法获取当前机器人状态")
            if robot_state_param is None:
                raise RuntimeError("视觉对准后机器人状态不可用")
            else:
                logger.warning("使用传入的robot_state作为备用")
                current_robot_state = robot_state_param
        else:
            current_robot_state = robot_state
        
        # 计算精确的物体全局坐标
        current_px = current_robot_state.position[0]
        current_py = current_robot_state.position[1]
        current_yaw = current_robot_state.imu_state.rpy[2]
        
        logger.info(f"视觉对准后机器人位置: [{current_px:.3f}, {current_py:.3f}, yaw:{current_yaw:.3f}rad]")
        
        refined_distance = refined_relative_coord[0]
        precise_object_x = current_px + refined_distance * math.cos(current_yaw)
        precise_object_y = current_py + refined_distance * math.sin(current_yaw)
        
        # 计算机器人中心目标位置
        robot_center_x = precise_object_x - CLAW_OFFSET_FORWARD * math.cos(current_yaw)
        robot_center_y = precise_object_y - CLAW_OFFSET_FORWARD * math.sin(current_yaw)
        
        refined_goal = [robot_center_x, robot_center_y]
        logger.info(f"优化后的机器人中心目标: {refined_goal}")
        
        return refined_goal
    
    def _phase2_move_and_adjust_to_target(self, global_goal, robot_state):
        """第二阶段：前进并调整到目标"""
        logger.info("开始前进并调整到目标")
        start_time = time.time()
        
        self.dist_pid.setpoint = TARGET_DEPTH
        consecutive_no_depth_count = 0
        
        while time.time() - start_time < MAX_NAVIGATION_TIME:
            vyaw = 0.0
            vx = 0.0
            
            visual_info = self.communication.request_visual_info()
            
            if visual_info and visual_info.get('bbox_available'):
                current_depth_x = visual_info.get('depth_x')
                
                if current_depth_x is not None:
                    logger.info(f"当前深度: {current_depth_x:.3f}m")
                    consecutive_no_depth_count = 0
                    
                    if current_depth_x < TARGET_DEPTH:
                        logger.info(f"到达目标深度 {TARGET_DEPTH:.2f}m")
                        self.sport_client.StopMove()
                        return True
                    
                    # 基于深度的前进速度控制
                    pid_output = self.dist_pid(current_depth_x)
                    vx = max(0.0, -pid_output)
                    if vx > 0.001:
                        vx = max(0.01, vx)
                    vx = min(vx, self.dist_pid.output_limits[1])
                else:
                    logger.warning("有边界框但无深度信息")
                    vx = 0.0
                    consecutive_no_depth_count += 1
                
                # 视觉偏航调整
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x
                
                if abs(pixel_error) > YAW_PIXEL_TOLERANCE_PHASE2:
                    vyaw = self.yaw_pid_visual(pixel_error)
                    if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
                        vyaw = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                
            elif visual_info and not visual_info.get('bbox_available'):
                logger.warning("前进中未检测到边界框")
                self.sport_client.StopMove()
            else:
                logger.error("前进中获取视觉信息失败")
                self.sport_client.StopMove()
            
            logger.info(f"vx: {vx:.3f}, vyaw: {vyaw:.3f}")
            self.sport_client.Move(vx, 0.0, vyaw)
            time.sleep(NAVIGATION_CONTROL_PERIOD)
        
        logger.error("前进阶段超时")
        self.sport_client.StopMove()
        raise RuntimeError("前进阶段超时")
    
    def _stage_a_approach_to_depth(self, avg_depth):
        """阶段A：前进到确认深度"""
        logger.info(f"阶段A: 前进到深度 {TARGET_CONFIRM_DEPTH:.2f}m")
        start_time = time.time()
        
        # 记录起始位置（使用线程安全的状态获取）
        start_position = get_robot_position()
        if start_position is None:
            logger.error("阶段A: 无法获取机器人状态")
            return False
            
        # 记录起始位置
        self.px0 = start_position['x']
        self.py0 = start_position['y']
        
        estimated_move_distance = avg_depth - CLAW_OFFSET_FORWARD
        logger.info(f"估算需要前进距离: {estimated_move_distance:.3f}m (基于平均深度: {avg_depth:.3f}m)")
        logger.info(f"起始位置: [{self.px0:.3f}, {self.py0:.3f}]")
        
        while time.time() - start_time < MAX_APPROACH_TIME:
            visual_info = self.communication.request_visual_info()
            
            # 实时获取当前机器人位置（使用线程安全的状态获取）
            current_position = get_robot_position()
            if current_position is None:
                logger.warning("阶段A: 无法获取当前机器人状态")
                # 使用备用方案：继续前进固定时间
                distance = 0.0
            else:
                # 更新当前位置
                self.px1 = current_position['x']
                self.py1 = current_position['y']
                
                # 计算已移动距离
                distance = np.sqrt((self.px1 - self.px0)**2 + (self.py1 - self.py0)**2)
            
            if not visual_info:
                logger.error("阶段A: 获取视觉信息失败")
                self.sport_client.StopMove()
                return False
            
            if not visual_info.get('bbox_available'):
                logger.error("阶段A: 无边界框")
                self.sport_client.StopMove()
                return False
            
            logger.info(f"阶段A - 已移动距离: {distance:.3f}m")
            
            # 获取当前深度信息
            current_depth_x = visual_info.get('depth_x')
            if current_depth_x is None:
                logger.warning("阶段A: 无深度信息")
                current_depth_x = float('inf')
            
            logger.info(f"阶段A - 当前深度: {current_depth_x:.3f}m, 已移动距离: {distance:.3f}m")
            
            # 判断是否到达目标（优先使用深度信息，备用距离判断）
            if distance >= estimated_move_distance:
                logger.info(f"阶段A: 到达确认深度")
                self.sport_client.Move(0.0, 0.0, 0.0)
                break
            
            # 像素对齐控制
            pixel_cx = visual_info['pixel_cx']
            image_width = visual_info['image_width']
            pixel_error = pixel_cx - image_width / 2.0
            
            vyaw = 0.0
            if abs(pixel_error) > (YAW_PIXEL_TOLERANCE_CONFIRM + 2):
                vyaw = -0.1 if pixel_error > 0 else 0.1
            
            # 前进控制
            vx = APPROACH_VX
            self.sport_client.Move(0.1, 0.0, vyaw)
            time.sleep(NAVIGATION_CONTROL_PERIOD)
        
        logger.info("阶段A: 完成")
        return True
    
    def _stage_b_precise_yaw_alignment(self):
        """阶段B：精确偏航对准"""
        logger.info("阶段B: 精确偏航对准")
        start_time = time.time()
        
        while time.time() - start_time < MAX_ALIGN_TIME_AT_DEPTH:
            visual_info = self.communication.request_visual_info()
            
            if not visual_info or not visual_info.get('bbox_available'):
                logger.error("阶段B: 无有效视觉信息")
                self.sport_client.StopMove()
                return False
            
            pixel_cx = visual_info['pixel_cx']
            image_width = visual_info['image_width']
            pixel_error = pixel_cx - image_width / 2.0
            
            logger.info(f"阶段B - 像素误差: {pixel_error:.1f}")
            
            if abs(pixel_error) < YAW_PIXEL_TOLERANCE_CONFIRM:
                # 获取平均深度值
                avg_depth = self._get_average_depth()
                
                if avg_depth:
                    logger.info(f"阶段B: 对准完成，平均深度: {avg_depth:.3f}m")
                else:
                    logger.info("阶段B: 对准完成，未获取有效深度")
                
                return True
            
            # 固定偏航调整
            vyaw = -0.1 if pixel_error > 0 else 0.1
            self.sport_client.Move(0.0, 0.0, vyaw)
            time.sleep(NAVIGATION_CONTROL_PERIOD)
        
        logger.error("阶段B: 精确对准超时")
        self.sport_client.StopMove()
        return False
    
    def approach_person_to_distance(self, target_distance=1.5):
        """像素对齐并接近人员到指定距离"""
        logger.info(f"开始接近人员到距离 {target_distance:.1f}m")
        
        # 重置PID控制器
        self.yaw_pid_visual.reset()
        self.dist_pid.setpoint = target_distance
        
        start_time = time.time()
        consecutive_no_bbox_count = 0
        
        while time.time() - start_time < MAX_NAVIGATION_TIME:
            visual_info = self.communication.request_visual_info()
            
            if visual_info and visual_info.get('bbox_available'):
                consecutive_no_bbox_count = 0
                
                # 获取当前深度
                current_depth_x = visual_info.get('depth_x')
                
                if current_depth_x is not None:
                    logger.info(f"当前距离人员: {current_depth_x:.3f}m")
                    
                    # 检查是否到达目标距离
                    if current_depth_x <= target_distance:
                        logger.info(f"已到达目标距离 {target_distance:.1f}m，停止移动")
                        self.sport_client.StopMove()
                        return True
                    
                    # 基于深度的前进速度控制
                    pid_output = self.dist_pid(current_depth_x)
                    vx = max(0.0, -pid_output)
                    if vx > 0.001:
                        vx = max(0.01, vx)
                    vx = min(vx, self.dist_pid.output_limits[1])
                else:
                    logger.warning("有边界框但无深度信息")
                    vx = 0.0
                
                # 像素对齐控制
                pixel_cx = visual_info['pixel_cx']
                image_width = visual_info['image_width']
                target_pixel_x = image_width / 2.0
                pixel_error = pixel_cx - target_pixel_x
                
                vyaw = 0.0
                if abs(pixel_error) > YAW_PIXEL_TOLERANCE_PHASE2:
                    vyaw = self.yaw_pid_visual(pixel_error)
                    if 0 < abs(vyaw) < MIN_EFFECTIVE_VYAW:
                        vyaw = math.copysign(MIN_EFFECTIVE_VYAW, vyaw)
                
                logger.info(f"接近人员 - vx: {vx:.3f}, vyaw: {vyaw:.3f}, 像素误差: {pixel_error:.1f}")
                self.sport_client.Move(vx, 0.0, vyaw)
                
            elif visual_info and not visual_info.get('bbox_available'):
                logger.warning("接近人员时未检测到边界框")
                self.sport_client.StopMove()
                consecutive_no_bbox_count += 1
                if consecutive_no_bbox_count > MAX_CONSECUTIVE_NO_BBOX:
                    logger.error("接近人员失败：长时间无边界框")
                    return False
            else:
                logger.error("接近人员时获取视觉信息失败")
                self.sport_client.StopMove()
            
            time.sleep(NAVIGATION_CONTROL_PERIOD)
        
        logger.error("接近人员超时")
        self.sport_client.StopMove()
        return False
    
    def _get_average_depth(self, num_samples=10, max_attempts=20):
        """获取多次深度值的平均值"""
        logger.info(f"开始获取{num_samples}次zed相机深度值取平均...")
        depth_values = []
        attempts = 0

        while len(depth_values) < num_samples and attempts < max_attempts:
            visual_info = self.communication.request_visual_info()
            attempts += 1
            
            if visual_info and visual_info.get('bbox_available'):
                depth_x = visual_info.get('depth_x')
                if depth_x is not None:
                    depth_values.append(depth_x)
                    logger.debug(f"获取第 {len(depth_values)} 次深度值: {depth_x:.3f}m")
                else:
                    logger.debug("未获取到深度信息,重试...")
                time.sleep(0.1)  # 短暂延迟以获取不同帧
            else:
                logger.debug("未检测到目标,重试...")
                time.sleep(0.1)

        if len(depth_values) < num_samples:
            logger.warning(f"仅获取到 {len(depth_values)} 个有效深度值")
            if not depth_values:  # 如果完全没有获取到值
                logger.error("无法获取任何有效深度值")
                return None
        
        avg_depth = sum(depth_values) / len(depth_values)
        logger.info(f"{len(depth_values)}次深度值平均结果: {avg_depth:.3f}m")
        return avg_depth 