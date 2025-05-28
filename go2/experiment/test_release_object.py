#!/usr/bin/env python3
# test_release_object.py
"""
独立测试释放物体功能的脚本
模拟状态机中的 _handle_release_object 方法
"""

import time
import sys
import logging
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 导入控制模块
try:
    from robot_control import RobotControl
    from pitch import PitchController
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保相关模块文件存在")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/home/unitree/test_release_object.log'),
        logging.StreamHandler()
    ]
)

class ReleaseObjectTester:
    """释放物体功能测试类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化释放物体测试器")
        
        # 初始化运动客户端
        self._init_sport_client()
        
        # 初始化俯仰角控制器
        self._init_pitch_controller()
        
        # 初始化控制模块
        self._init_control_module()
    
    def _init_sport_client(self):
        """初始化运动客户端"""
        try:
            self.sport_client = SportClient()
            self.sport_client.Init()
            self.logger.info("SportClient 初始化成功")
        except Exception as e:
            self.logger.error(f"SportClient 初始化失败: {e}")
            raise RuntimeError(f"SportClient 初始化失败: {e}")
    
    def _init_pitch_controller(self):
        """初始化俯仰角控制器"""
        try:
            self.pitch_controller = PitchController(self.sport_client, interpolation_duration_s=2.0)
            self.pitch_controller.start_control()
            self.logger.info("PitchController 初始化成功")
        except Exception as e:
            self.logger.error(f"PitchController 初始化失败: {e}")
            self.pitch_controller = None
    
    def _init_control_module(self):
        """初始化控制模块"""
        try:
            self.control = RobotControl(self.sport_client, self.pitch_controller)
            self.logger.info("RobotControl 模块初始化成功")
        except Exception as e:
            self.logger.error(f"RobotControl 模块初始化失败: {e}")
            raise RuntimeError(f"RobotControl 模块初始化失败: {e}")
    
    def test_release_object(self):
        """测试释放物体功能（模拟状态机中的方法）"""
        self.logger.info("开始测试释放物体功能")
        
        try:
            # 步骤1: 记录开始
            self.logger.info("进入 RELEASE_OBJECT 测试状态")
            
            # 步骤2: 机器人坐下
            self.logger.info("让机器人坐下...")
            self.sport_client.Sit()
            
            # 步骤3: 等待稳定
            self.logger.info("等待5秒让机器人稳定...")
            time.sleep(5)
            
            # 步骤4: 释放物体
            self.logger.info("执行释放物体命令...")
            self.control.release_object()
            
            # 步骤5: 恢复站立姿态
            self.logger.info("让机器人恢复站立姿态...")
            self.sport_client.RiseSit()
            time.sleep(2)  # 等待站立完成
            
            # 步骤6: 完成
            self.logger.info("释放物体测试完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"释放物体测试失败: {e}")
            return False
    
    def test_step_by_step(self):
        """分步测试，每步都需要用户确认"""
        self.logger.info("开始分步测试释放物体功能")
        
        try:
            # 步骤1: 确认开始
            input("按 Enter 键开始测试释放物体功能...")
            self.logger.info("进入 RELEASE_OBJECT 测试状态")
            
            # 步骤2: 机器人坐下
            input("按 Enter 键让机器人坐下...")
            self.logger.info("让机器人坐下...")
            self.sport_client.Sit()
            
            # 步骤3: 等待稳定
            input("按 Enter 键开始等待稳定（5秒）...")
            self.logger.info("等待5秒让机器人稳定...")
            for i in range(5, 0, -1):
                print(f"等待中... {i}秒")
                time.sleep(1)
            
            # 步骤4: 释放物体
            input("按 Enter 键执行释放物体命令...")
            self.logger.info("执行释放物体命令...")
            self.control.release_object()
            
            # 步骤5: 恢复站立姿态
            input("按 Enter 键让机器人恢复站立姿态...")
            self.logger.info("让机器人恢复站立姿态...")
            self.sport_client.RiseSit()
            self.logger.info("等待2秒让机器人站立...")
            for i in range(2, 0, -1):
                print(f"站立中... {i}秒")
                time.sleep(1)
            
            # 步骤6: 完成
            self.logger.info("释放物体测试完成")
            print("测试完成！")
            
            return True
            
        except Exception as e:
            self.logger.error(f"分步测试失败: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理资源...")
        
        try:
            if hasattr(self, 'sport_client') and self.sport_client:
                self.logger.info("停止机器人移动...")
                self.sport_client.StopMove()
        except Exception as e:
            self.logger.error(f"停止机器人移动时发生错误: {e}")
        
        try:
            if hasattr(self, 'pitch_controller') and self.pitch_controller:
                self.logger.info("停止 PitchController...")
                self.pitch_controller.stop_control(transition_to_zero_pitch=True)
        except Exception as e:
            self.logger.error(f"停止 PitchController 时发生错误: {e}")
        
        self.logger.info("清理完成")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} networkInterface [mode]")
        print("mode: 'auto' 为自动测试，'step' 为分步测试（默认为step）")
        sys.exit(-1)
    
    # 获取测试模式
    mode = 'step'
    if len(sys.argv) >= 3:
        mode = sys.argv[2].lower()
    
    print("警告：请确保机器人周围没有障碍物。")
    print("这个测试将让机器人坐下并释放物体。")
    
    tester = None
    try:
        # 初始化通道工厂
        ChannelFactoryInitialize(0, sys.argv[1])
        
        # 创建测试器
        tester = ReleaseObjectTester()
        
        # 等待用户确认开始
        input("释放物体测试器初始化完毕。按 Enter 键开始测试...")
        
        # 根据模式运行测试
        if mode == 'auto':
            print("运行自动测试模式...")
            success = tester.test_release_object()
        else:
            print("运行分步测试模式...")
            success = tester.test_step_by_step()
        
        if success:
            print("测试成功完成！")
        else:
            print("测试失败！")
        
    except Exception as e:
        logging.critical(f"测试程序发生严重错误: {e}", exc_info=True)
    finally:
        if tester:
            tester.cleanup()
        logging.info("测试程序退出")

if __name__ == "__main__":
    main() 