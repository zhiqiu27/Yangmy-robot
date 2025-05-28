#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_height_adjustment.py
脚本用于测试 Go2 机器人的 BodyHeight 和 FootRaiseHeight 功能。
"""

import sys
import time
import logging
import math
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ # For potential state display

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Go2 默认高度常量 (来自文档)
DEFAULT_BODY_HEIGHT_ABS = 0.33  # 米
DEFAULT_FOOT_RAISE_HEIGHT_ABS = 0.09 # 米

# 参数范围常量 (来自文档)
BODY_HEIGHT_REL_MIN = -0.18
BODY_HEIGHT_REL_MAX = 0.03
FOOT_RAISE_REL_MIN = -0.06
FOOT_RAISE_REL_MAX = 0.03

# 全局机器人状态 (可选，用于显示)
robot_state = None

def SportModeStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg
    # logger.debug(f"Current body height from state: {msg.body_height:.3f}m") # 这个状态可能不直接反映 BodyHeight() 的设定值

def main():
    if len(sys.argv) < 2:
        logger.error(f"用法: python3 {sys.argv[0]} networkInterface")
        logger.error("例如: python3 test_height_adjustment.py eth0")
        sys.exit(1)

    network_interface = sys.argv[1]
    logger.info(f"正在使用网络接口: {network_interface}")

    # 初始化 ChannelFactory
    try:
        ChannelFactoryInitialize(0, network_interface)
        logger.info("ChannelFactory 初始化成功。")
    except Exception as e:
        logger.error(f"ChannelFactory 初始化失败: {e}")
        sys.exit(1)

    # 初始化 SportClient
    sport_client = SportClient()
    try:
        sport_client.Init()
        logger.info("SportClient 初始化成功。")
    except Exception as e:
        logger.error(f"SportClient 初始化失败: {e}")
        sys.exit(1)
    
    # 可选：订阅机器人状态以获取一些反馈，但这可能不直接反映这些特定参数的设定值
    # sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    # sub.Init(SportModeStateHandler, 20)
    # logger.info("机器人状态订阅已启动 (可选功能)。")
    # time.sleep(0.5) # 等待一下状态

    logger.warning("重要提示：在测试这些功能前，请确保机器人处于安全、稳定的站立姿态！")
    input("按 Enter 键继续...")

    current_body_height_abs = DEFAULT_BODY_HEIGHT_ABS
    current_foot_raise_height_abs = DEFAULT_FOOT_RAISE_HEIGHT_ABS

    while True:
        print("\n" + "="*30)
        print("选择要测试的功能:")
        print(f"1. BodyHeight (当前近似绝对高度: {current_body_height_abs:.3f} m)")
        print(f"2. FootRaiseHeight (当前近似绝对抬足高度: {current_foot_raise_height_abs:.3f} m)")
        print("3. 查看当前机器人状态 (如果订阅了)")
        print("q. 退出")
        choice = input("请输入选项: ").strip().lower()
        print("="*30)

        if choice == '1':
            try:
                height_str = input(f"请输入机身相对高度值 (范围: [{BODY_HEIGHT_REL_MIN}, {BODY_HEIGHT_REL_MAX}] m): ").strip()
                relative_height = float(height_str)

                if not (BODY_HEIGHT_REL_MIN <= relative_height <= BODY_HEIGHT_REL_MAX):
                    logger.error(f"输入值 {relative_height:.3f} 超出允许范围 [{BODY_HEIGHT_REL_MIN}, {BODY_HEIGHT_REL_MAX}]。")
                    continue
                
                logger.info(f"调用 BodyHeight({relative_height:.3f}) ...")
                ret = sport_client.BodyHeight(relative_height)
                time.sleep(1)
                # 只有在最低高度时才添加俯仰角
                if relative_height == -0.18:
                    # 添加一个小的俯仰角 (10度转换为弧度)
                    pitch_degrees = 12.0
                    pitch_radians = math.radians(pitch_degrees)
                    logger.info(f"机身已调整到最低位置，添加俯仰角: {pitch_degrees}度 ({pitch_radians:.3f}弧度)")
                    sport_client.Euler(0, pitch_radians, 0)
                else:
                    logger.info("机身高度不是最低位置，不添加俯仰角")
                if ret == 0:
                    logger.info(f"BodyHeight({relative_height:.3f}) 调用成功！")
                    current_body_height_abs = DEFAULT_BODY_HEIGHT_ABS + relative_height
                    logger.info(f"机器人新的近似绝对机身高度约为: {current_body_height_abs:.3f} m")
                else:
                    logger.error(f"BodyHeight({relative_height:.3f}) 调用失败，错误码: {ret}")

            except ValueError:
                logger.error("输入无效，请输入一个数字。")
            except Exception as e:
                logger.error(f"执行 BodyHeight 时发生错误: {e}")

        elif choice == '2':
            try:
                height_str = input(f"请输入抬足相对高度值 (范围: [{FOOT_RAISE_REL_MIN}, {FOOT_RAISE_REL_MAX}] m): ").strip()
                relative_height = float(height_str)

                if not (FOOT_RAISE_REL_MIN <= relative_height <= FOOT_RAISE_REL_MAX):
                    logger.error(f"输入值 {relative_height:.3f} 超出允许范围 [{FOOT_RAISE_REL_MIN}, {FOOT_RAISE_REL_MAX}]。")
                    continue

                logger.info(f"调用 FootRaiseHeight({relative_height:.3f}) ...")
                ret = sport_client.FootRaiseHeight(relative_height)
                if ret == 0:
                    logger.info(f"FootRaiseHeight({relative_height:.3f}) 调用成功！")
                    current_foot_raise_height_abs = DEFAULT_FOOT_RAISE_HEIGHT_ABS + relative_height
                    logger.info(f"机器人新的近似绝对抬足高度约为: {current_foot_raise_height_abs:.3f} m")
                else:
                    logger.error(f"FootRaiseHeight({relative_height:.3f}) 调用失败，错误码: {ret}")

            except ValueError:
                logger.error("输入无效，请输入一个数字。")
            except Exception as e:
                logger.error(f"执行 FootRaiseHeight 时发生错误: {e}")
        
        elif choice == '3':
            if robot_state:
                logger.info(f"当前机器人状态 SportModeState_:")
                logger.info(f"  - 模式: {robot_state.mode}")
                logger.info(f"  - 机身高度 (来自状态): {robot_state.body_height:.3f} m") # 注意：这可能与 BodyHeight() 的设定不完全一致
                logger.info(f"  - 足端位置 (FL): {robot_state.foot_position_body[0].x:.3f}, {robot_state.foot_position_body[0].y:.3f}, {robot_state.foot_position_body[0].z:.3f}")
                # 可以添加更多你想查看的状态信息
            else:
                logger.info("机器人状态尚未接收，或未订阅状态。")


        elif choice == 'q':
            logger.info("正在退出测试脚本...")
            break
        else:
            logger.warning("无效选项，请重新输入。")
        
        time.sleep(0.1) # 防止CPU占用过高

    # 清理 (虽然SportClient通常在程序结束时会自动处理，但显式调用是个好习惯)
    # SportClient没有明确的shutdown/close方法，依赖于析构函数
    logger.info("测试脚本结束。")

if __name__ == "__main__":
    main() 