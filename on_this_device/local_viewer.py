#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_viewer.py
本地查看ZED相机的RGB图像和深度图像
按 'q' 键退出，按 's' 键保存当前帧
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class LocalViewer:
    def __init__(self, camera_fps=24, camera_resolution_str="VGA"):
        logger.info("初始化本地图像查看器...")
        
        self.camera_fps = camera_fps
        
        # 设置相机分辨率
        if camera_resolution_str.upper() == "VGA":
            self.camera_resolution = sl.RESOLUTION.VGA
        elif camera_resolution_str.upper() == "HD720":
            self.camera_resolution = sl.RESOLUTION.HD720
        elif camera_resolution_str.upper() == "HD1080":
            self.camera_resolution = sl.RESOLUTION.HD1080
        else:
            logger.warning(f"不支持的相机分辨率: {camera_resolution_str}. 使用默认VGA.")
            self.camera_resolution = sl.RESOLUTION.VGA

        # 初始化ZED相机
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        self.init_params.camera_fps = self.camera_fps
        self.init_params.camera_resolution = self.camera_resolution
        
        self.runtime_params = sl.RuntimeParameters()
        
        # 图像矩阵
        self.rgb_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.xyz_mat = sl.Mat()
        
        self.frame_count = 0

    def open_camera(self):
        """打开ZED相机"""
        logger.info("正在打开ZED相机...")
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED相机打开失败: {status}")
            raise RuntimeError(f"ZED相机打开失败: {status}")
        logger.info("ZED相机打开成功")
        
        # 获取相机信息
        camera_info = self.zed.get_camera_information()
        logger.info(f"相机分辨率: {camera_info.camera_configuration.resolution.width}x{camera_info.camera_configuration.resolution.height}")
        logger.info(f"相机FPS: {camera_info.camera_configuration.fps}")

    def process_depth_for_display(self, depth_map):
        """处理深度图像用于显示"""
        # 将深度值转换为可视化的图像
        depth_display = depth_map.copy()
        
        # 移除无效值
        depth_display[~np.isfinite(depth_display)] = 0
        
        # 限制深度范围 (0-10米)
        max_depth = 10.0
        depth_display = np.clip(depth_display, 0, max_depth)
        
        # 归一化到0-255
        depth_normalized = (depth_display / max_depth * 255).astype(np.uint8)
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored

    def add_info_overlay(self, image, info_text):
        """在图像上添加信息覆盖层"""
        overlay = image.copy()
        
        # 添加半透明背景
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 添加文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        y_offset = 30
        for line in info_text:
            cv2.putText(image, line, (15, y_offset), font, font_scale, color, thickness)
            y_offset += 20
        
        return image

    def save_frame(self, rgb_image, depth_colored, depth_raw):
        """保存当前帧"""
        timestamp = int(time.time())
        
        # 保存RGB图像
        rgb_filename = f"rgb_frame_{timestamp}.jpg"
        cv2.imwrite(rgb_filename, rgb_image)
        
        # 保存深度可视化图像
        depth_vis_filename = f"depth_vis_{timestamp}.jpg"
        cv2.imwrite(depth_vis_filename, depth_colored)
        
        # 保存原始深度数据
        depth_raw_filename = f"depth_raw_{timestamp}.npy"
        np.save(depth_raw_filename, depth_raw)
        
        logger.info(f"已保存帧 {self.frame_count}: {rgb_filename}, {depth_vis_filename}, {depth_raw_filename}")

    def run(self):
        """运行主循环"""
        try:
            self.open_camera()
            
            logger.info("开始图像显示循环...")
            logger.info("按键说明:")
            logger.info("  'q' - 退出程序")
            logger.info("  's' - 保存当前帧")
            logger.info("  'r' - 重置帧计数")
            
            start_time = time.time()
            
            while True:
                # 获取图像
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # 获取RGB图像
                    self.zed.retrieve_image(self.rgb_mat, sl.VIEW.LEFT)
                    rgb_image = self.rgb_mat.get_data()
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)
                    
                    # 获取深度图像
                    self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
                    self.zed.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZ)
                    
                    depth_raw = self.depth_mat.get_data()
                    xyz_data = self.xyz_mat.get_data()
                    
                    # 处理深度图像用于显示
                    depth_colored = self.process_depth_for_display(depth_raw)
                    
                    # 计算FPS
                    self.frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # 添加信息覆盖层
                    info_text = [
                        f"Frame: {self.frame_count}",
                        f"FPS: {fps:.1f}",
                        f"Resolution: {rgb_image.shape[1]}x{rgb_image.shape[0]}"
                    ]
                    
                    rgb_with_info = self.add_info_overlay(rgb_image.copy(), info_text)
                    depth_with_info = self.add_info_overlay(depth_colored.copy(), info_text)
                    
                    # 显示图像
                    cv2.imshow('RGB Image', rgb_with_info)
                    cv2.imshow('Depth Image', depth_with_info)
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("用户按下 'q' 键，退出程序")
                        break
                    elif key == ord('s'):
                        self.save_frame(rgb_image, depth_colored, depth_raw)
                    elif key == ord('r'):
                        self.frame_count = 0
                        start_time = time.time()
                        logger.info("帧计数已重置")
                
                else:
                    logger.warning("ZED grab失败")
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        except Exception as e:
            logger.error(f"运行时错误: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        logger.info("正在清理资源...")
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        # 关闭ZED相机
        if self.zed and self.zed.is_opened():
            self.zed.close()
            logger.info("ZED相机已关闭")
        
        logger.info("资源清理完成")

def main():
    """主函数"""
    logger.info("启动本地图像查看器")
    
    # 解析命令行参数
    fps = 24
    resolution = "VGA"
    
    if len(sys.argv) > 1:
        try:
            fps = int(sys.argv[1])
        except ValueError:
            logger.warning(f"无效的FPS值: {sys.argv[1]}，使用默认值24")
    
    if len(sys.argv) > 2:
        resolution = sys.argv[2].upper()
        if resolution not in ["VGA", "HD720", "HD1080"]:
            logger.warning(f"无效的分辨率: {resolution}，使用默认值VGA")
            resolution = "VGA"
    
    logger.info(f"配置: FPS={fps}, 分辨率={resolution}")
    
    # 创建并运行查看器
    viewer = LocalViewer(camera_fps=fps, camera_resolution_str=resolution)
    viewer.run()

if __name__ == "__main__":
    main() 