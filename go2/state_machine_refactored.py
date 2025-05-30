# state_machine_refactored.py
"""
重构后的机器人状态机
使用模块化设计，提高代码可读性和可维护性
"""

import time
import sys
import math
import logging
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from pitch import PitchController

# 导入重构后的模块
from robot_config import *
from robot_utils import setup_logging, convert_relative_to_global
from robot_communication import RobotCommunication
from robot_navigation import RobotNavigation
from robot_control import RobotControl

# 导入全局机器人状态
from robot_state import robot_state, HighStateHandler

class RobotStateMachine:
    """机器人状态机主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化机器人状态机")
        
        # 初始化组件
        self._init_sport_client()
        self._init_state_subscriber()
        self._init_pitch_controller()
        self._init_modules()
        
        # 状态管理
        self.state = State.CALIBRATE_SHIRT_POSITION
        self._next_state = None
        self.current_coord = [0.0, 0.0]
        self.target_list = TARGET_LIST.copy()
    
    def _init_sport_client(self):
        """初始化运动客户端"""
        try:
            self.sport_client = SportClient()
            self.sport_client.Init()
            self.logger.info("SportClient 初始化成功")
        except Exception as e:
            self.logger.error(f"SportClient 初始化失败: {e}")
            raise RuntimeError(f"SportClient 初始化失败: {e}")
    
    def _init_state_subscriber(self):
        """初始化状态订阅"""
        try:
            self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self.sub.Init(HighStateHandler, 20)
            time.sleep(2)  # 等待第一条消息
            if robot_state is None:
                self.logger.warning("未能接收到机器人状态，但继续运行")
            self.logger.info("状态订阅初始化成功")
        except Exception as e:
            self.logger.error(f"状态订阅初始化失败: {e}")
            raise RuntimeError(f"状态订阅初始化失败: {e}")
    
    def _init_pitch_controller(self):
        """初始化俯仰角控制器"""
        try:
            self.pitch_controller = PitchController(self.sport_client, interpolation_duration_s=1.0)
            self.pitch_controller.start_control()
            self.logger.info("PitchController 初始化成功")
        except Exception as e:
            self.logger.error(f"PitchController 初始化失败: {e}")
            self.pitch_controller = None
    
    def _init_modules(self):
        """初始化各个功能模块"""
        self.communication = RobotCommunication()
        self.navigation = RobotNavigation(self.sport_client)
        self.control = RobotControl(self.sport_client, self.pitch_controller)
    
    def run(self):
        """运行状态机主循环"""
        self.logger.info("开始运行状态机")
        
        while self.state != State.DONE:
            try:
                self._execute_current_state()
                
                # 防止紧密循环
                if self.state != State.WAIT:
                    time.sleep(0.05)
                    
            except RuntimeError as e:
                self.logger.error(f"状态执行错误: {e}")
                self.control.stop_movement()
                self.logger.info("机器人已停止，程序将终止")
                break
            except Exception as e:
                self.logger.error(f"未知错误: {e}")
                self.control.stop_movement()
                self.logger.info("机器人已停止，程序将终止")
                break
        
        self.logger.info("状态机运行结束")
    
    def _execute_current_state(self):
        """执行当前状态"""
        if self.state == State.WAIT:
            self._handle_wait_state()
        elif self.state == State.CALIBRATE_SHIRT_POSITION:
            self._handle_calibrate_shirt_position()
        elif self.state == State.NAVIGATE_TO_OBJECT:
            self._handle_navigate_to_object()
        elif self.state == State.FINE_ADJUST_AND_APPROACH:
            self._handle_fine_adjust_and_approach()
        elif self.state == State.GRASP_OBJECT:
            self._handle_grasp_object()
        elif self.state == State.SEND_NEXT_COMMAND:
            self._handle_send_next_command()
        elif self.state == State.NAVIGATE_TO_PERSON:
            self._handle_navigate_to_person()
        elif self.state == State.RELEASE_OBJECT:
            self._handle_release_object()
        elif self.state == State.LOWER_HEAD:
            self._handle_lower_head()
        else:
            self.logger.warning(f"未处理的状态: {self.state}")
            self.state = State.DONE
    
    def _handle_wait_state(self):
        """处理等待状态"""
        self.control.stop_movement()
        self.logger.info(f"进入WAIT状态，下一个状态: {self._next_state}")
        time.sleep(1)
        
        if self._next_state is not None:
            self.state = self._next_state
            self.logger.info(f"从WAIT状态转换到: {self.state}")
            self._next_state = None
    
    def _handle_calibrate_shirt_position(self):
        """处理校准衬衫位置状态"""
        self.logger.info("进入 CALIBRATE_SHIRT_POSITION 状态")
        
        # 从图像服务器请求校准数据
        calibrated_coord = self.communication.request_calibration()
        
        if not calibrated_coord:
            self.logger.error("无法从图像服务器获取校准数据")
            self.target_list[0]["coord"] = [0.6, 0.0]  # 使用默认值
            
            user_choice = self._get_user_choice_for_calibration_failure()
            if user_choice == 'retry':
                return  # 重试校准
            else:
                self._transition_to_state(State.NAVIGATE_TO_OBJECT)
                return
        
        # 校准成功
        avg_x, avg_y = calibrated_coord[0], calibrated_coord[1]
        self.logger.info(f"获取的衬衫坐标: [{avg_x:.3f}, {avg_y:.3f}]")
        
        # 用户确认
        if self._get_user_confirmation_for_calibration(avg_x, avg_y):
            self.target_list[0]["coord"] = calibrated_coord
            self.logger.info(f"衬衫目标更新为: {self.target_list[0]['coord']}")
            self._transition_to_state(State.NAVIGATE_TO_OBJECT)
    
    def _handle_navigate_to_object(self):
        """处理导航到物体状态"""
        self.logger.info("进入 NAVIGATE_TO_OBJECT 状态")
        
        # 检查机器人状态是否可用
        from robot_state import robot_state
        if robot_state is None:
            self.logger.error("机器人状态不可用")
            time.sleep(1)
            return
        
        # 计算导航目标
        object_relative_coord = self.target_list[0]["coord"]
        self.logger.info(f"物体相对坐标: {object_relative_coord}")
        
        try:
            object_global_coord = convert_relative_to_global(object_relative_coord, robot_state)
            self.logger.info(f"物体全局坐标: {object_global_coord}")
            
            # 计算机器人中心目标位置
            nav_goal = self._calculate_robot_center_target(object_global_coord)
            self.logger.info(f"导航目标: {nav_goal}")
            
            # 执行导航
            final_goal = self.navigation.navigate_to(nav_goal, robot_state)
            self.current_coord = final_goal
            self._transition_to_state(State.FINE_ADJUST_AND_APPROACH)
        except Exception as e:
            self.logger.error(f"导航失败: {e}")
            self.state = State.DONE
    
    def _handle_fine_adjust_and_approach(self):
        """处理精细调整和接近状态"""
        self.logger.info("进入 FINE_ADJUST_AND_APPROACH 状态")
        
        # 准备抓取姿态
        if not self.control.prepare_for_grasp():
            self.logger.error("准备抓取姿态失败")
            self.control.restore_default_posture()
            self.state = State.DONE
            return
        
        # 执行精细接近和对准
        from robot_state import robot_state
        if self.navigation.approach_and_confirm_alignment(robot_state):
            self.logger.info("精细调整完成")
            self.control.grasp_object()
            time.sleep(1)
            self.sport_client.StopMove()
            self._transition_to_state(State.GRASP_OBJECT)
        else:
            self.logger.error("精细调整失败")
            self.control.restore_default_posture()
            self.state = State.DONE
    
    def _handle_grasp_object(self):
        """处理抓取物体状态"""
        self.logger.info("进入 GRASP_OBJECT 状态")
        self.control.stop_movement()
        self.control.stop_movement()
        self.control.stop_movement()
        self.control.stop_movement()
        # 重新设置抓取姿态
        #self.control.prepare_for_grasp()
        
        # 执行抓取
        if self.control.grasp_object():
            self.logger.info("抓取成功")
            
            # 等待用户确认
            self._wait_for_user_confirmation("物体已抓取。输入 'ok' 继续")
            
            # 恢复姿态
            self.control.restore_default_posture()
            time.sleep(1)
            self._transition_to_state(State.SEND_NEXT_COMMAND)
        else:
            self.logger.warning("抓取失败")
            #self.control.restore_default_posture()
            self.state = State.DONE
    
    def _handle_send_next_command(self):
        """处理发送下一个命令状态"""
        self.logger.info("进入 SEND_NEXT_COMMAND 状态")
        
        # 直接等待接收方向命令
        self.logger.info("等待接收方向命令...")
        direction = self.communication.receive_direction_command()
        if direction:
            from robot_state import robot_state
            self.control.execute_direction_rotation(direction, robot_state)
            self.logger.info("方向转向完成")
        else:
            self.logger.warning("未接收到方向命令")
        
        # 转向结束后发送NEXT_TARGET命令到PC端
        if not self._send_next_to_vlm():
            self.logger.error("发送NEXT_TARGET命令失败，任务终止")
            self.state = State.DONE
            return
        
        # 等待PC端处理完成（可选的延迟）
        time.sleep(2.0)
        
        self._transition_to_state(State.NAVIGATE_TO_PERSON)
    
    def _handle_navigate_to_person(self):
        """处理导航到人员状态"""
        self.logger.info("进入 NAVIGATE_TO_PERSON 状态")
        
        # 使用像素对齐并接近到1.5m的方法
        try:
            success = self.navigation.approach_person_to_distance(target_distance=0.40)
            if success:
                self.logger.info("成功接近人员到1.5m距离")
                self._transition_to_state(State.RELEASE_OBJECT)
            else:
                self.logger.error("接近人员失败")
                self.state = State.DONE
        except Exception as e:
            self.logger.error(f"接近人员时发生错误: {e}")
            self.state = State.DONE
    
    def _handle_release_object(self):
        """处理释放物体状态"""
        self.logger.info("进入 RELEASE_OBJECT 状态")

        self.control.release_object()

        self.logger.info("任务完成")
        self.state = State.DONE

    
    def _transition_to_state(self, next_state):
        """转换到下一个状态"""
        self._next_state = next_state
        self.state = State.WAIT
    
    def _calculate_robot_center_target(self, object_global_coord):
        """计算机器人中心应该到达的目标位置"""
        from robot_state import robot_state
        if robot_state is None:
            raise RuntimeError("机器人状态不可用")
        
        current_px = robot_state.position[0]
        current_py = robot_state.position[1]
        
        delta_x = object_global_coord[0] - current_px
        delta_y = object_global_coord[1] - current_py
        approach_yaw = math.atan2(delta_y, delta_x)
        
        robot_center_x = object_global_coord[0] - CLAW_OFFSET_FORWARD * math.cos(approach_yaw)
        robot_center_y = object_global_coord[1] - CLAW_OFFSET_FORWARD * math.sin(approach_yaw)
        
        return [robot_center_x, robot_center_y]
    
    def _get_user_choice_for_calibration_failure(self):
        """获取校准失败时的用户选择"""
        while True:
            choice = input("无法连接到图像服务器。输入 'proceed' 使用默认坐标，或 'retry' 重试: ")
            if choice.lower() in ['proceed', 'retry']:
                return choice.lower()
            self.logger.info("输入无效，请输入 'proceed' 或 'retry'")
    
    def _get_user_confirmation_for_calibration(self, avg_x, avg_y):
        """获取校准结果的用户确认"""
        while True:
            user_input = input(f"校准完成。衬衫坐标: [{avg_x:.3f}, {avg_y:.3f}]。输入 'ok' 确认或 'retry' 重试: ")
            if user_input.lower() == 'ok':
                return True
            elif user_input.lower() == 'retry':
                return False
            else:
                self.logger.info("输入无效，请输入 'ok' 或 'retry'")
    
    def _wait_for_user_confirmation(self, message):
        """等待用户确认"""
        while True:
            user_input = input(f"{message}: ")
            if user_input.lower() == 'ok':
                break
            self.logger.info("输入无效，请输入 'ok'")
    
    def _send_next_to_vlm(self):
        """发送NEXT_TARGET命令到PC端"""
        self.logger.info("向PC端发送NEXT_TARGET命令")
        
        success = self.communication.send_next_target_command()
        if success:
            self.logger.info("NEXT_TARGET命令发送成功")
        else:
            self.logger.error("NEXT_TARGET命令发送失败")
            
        return success
    
    def _get_next_target_from_vlm(self):
        """从图像服务器获取人员坐标"""
        self.logger.info("从图像服务器获取人员坐标")
        
        person_coord = self.communication.request_person_coordinates()
        if person_coord:
            self.logger.info(f"成功获取人员坐标: {person_coord}")
            return person_coord
        else:
            self.logger.error("获取人员坐标失败，使用默认坐标")
            return self.target_list[1]["coord"]  # 使用默认坐标作为备选
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理资源...")
        
        # 停止机器人移动
        try:
            if hasattr(self, 'sport_client') and self.sport_client:
                self.logger.info("停止机器人移动...")
                self.sport_client.StopMove()
                self.logger.info("机器人移动已停止")
        except Exception as e:
            self.logger.error(f"停止机器人移动时发生错误: {e}")
        
        # 关闭俯仰角控制器
        try:
            if hasattr(self, 'pitch_controller') and self.pitch_controller:
                self.logger.info("停止 PitchController...")
                self.pitch_controller.stop_control(transition_to_zero_pitch=True)
                self.logger.info("PitchController 已停止")
        except Exception as e:
            self.logger.error(f"停止 PitchController 时发生错误: {e}")
        
        self.logger.info("清理完成")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)
    
    # 设置日志
    setup_logging(LOG_LEVEL, LOG_FILE, LOG_FORMAT)
    
    print("警告：请确保机器人周围没有障碍物。")
    
    state_machine = None
    try:
        # 初始化通道工厂
        ChannelFactoryInitialize(0, sys.argv[1])
        
        # 创建状态机
        state_machine = RobotStateMachine()
        
        # 等待用户确认开始
        input("机器人状态机初始化完毕。按 Enter 键开始执行...")
        
        # 运行状态机
        state_machine.run()
        
    except Exception as e:
        logging.critical(f"主程序发生严重错误: {e}", exc_info=True)
    finally:
        if state_machine:
            state_machine.cleanup()
        logging.info("程序退出")

if __name__ == "__main__":
    main() 