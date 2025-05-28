#!/usr/bin/env python3
"""
简单测试脚本：测试阶段A的运动方向
"""

import time
import sys
import logging
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# 机器人状态（全局变量）
robot_state = None

def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg

class SimpleStageATest:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化 SportClient
        self.sport_client = SportClient()
        self.sport_client.Init()
        
        # 初始化状态订阅
        self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sub.Init(HighStateHandler, 20)
        time.sleep(2)
        
        # 高度调整相关常量
        self.BODY_HEIGHT_REL_MIN = -0.18
        self.FOOT_RAISE_REL_MIN = -0.06

    def set_body_height_relative(self, relative_height: float):
        """设置机身相对高度"""
        self.logger.info(f"设置机身相对高度: {relative_height:.3f}m")
        ret = self.sport_client.BodyHeight(relative_height)
        if ret == 0:
            self.logger.info("机身高度设置成功")
            return True
        else:
            self.logger.error(f"机身高度设置失败，错误码: {ret}")
            return False

    def set_foot_raise_height_relative(self, relative_height: float):
        """设置抬足相对高度"""
        self.logger.info(f"设置抬足相对高度: {relative_height:.3f}m")
        ret = self.sport_client.FootRaiseHeight(relative_height)
        if ret == 0:
            self.logger.info("抬足高度设置成功")
            return True
        else:
            self.logger.error(f"抬足高度设置失败，错误码: {ret}")
            return False

    def continuous_move(self, vx, vy, vyaw, duration_seconds, control_period=0.02):
        """持续发送Move命令"""
        self.logger.info(f"开始移动: vx={vx}, vy={vy}, vyaw={vyaw}, 持续{duration_seconds}秒")
        
        start_time = time.time()
        step_count = 0
        
        while time.time() - start_time < duration_seconds:
            self.sport_client.Move(vx, vy, vyaw)
            step_count += 1
            
            # 每0.5秒打印一次进度
            if step_count % 25 == 0:  # 25 * 0.02s = 0.5s
                elapsed = time.time() - start_time
                self.logger.info(f"移动进度: {elapsed:.1f}s / {duration_seconds}s")
            
            time.sleep(control_period)
        
        # 停止移动
        self.sport_client.StopMove()
        self.logger.info("移动完成，机器人已停止")

    def test_stage_a_movements(self):
        """测试阶段A的运动方向"""
        self.logger.info("=== 开始阶段A运动测试 ===")
        
        # 1. 调整机身和抬足高度
        self.logger.info("步骤1: 调整机身和抬足高度")
        self.set_body_height_relative(self.BODY_HEIGHT_REL_MIN)
        time.sleep(2)
        self.set_foot_raise_height_relative(self.FOOT_RAISE_REL_MIN)
        time.sleep(2)
        
        self.sport_client.StandDown()
        time.sleep(2)
        
        # 4. 恢复默认高度
        self.logger.info("步骤4: 恢复默认高度")
        self.set_body_height_relative(0.0)
        time.sleep(1)
        self.set_foot_raise_height_relative(0.0)
        time.sleep(1)
        
        self.logger.info("=== 阶段A运动测试完成 ===")

    def run_test(self):
        """运行测试"""
        try:
            self.test_stage_a_movements()
        except Exception as e:
            self.logger.error(f"测试过程中发生错误: {e}")
        finally:
            self.sport_client.StopMove()
            self.logger.info("测试结束，机器人已停止")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)
    
    try:
        ChannelFactoryInitialize(0, sys.argv[1])
        test = SimpleStageATest()
        test.run_test()
    except Exception as e:
        logging.error(f"程序发生错误: {e}")
    finally:
        logging.info("程序退出") 