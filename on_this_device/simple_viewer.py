#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_viewer.py
简化版本的ZED相机图像查看器
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time

def main():
    # 初始化ZED相机
    zed = sl.Camera()
    
    # 设置初始化参数
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 24
    
    # 打开相机
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"相机打开失败: {status}")
        return
    
    print("相机打开成功!")
    print("按键说明:")
    print("  'q' - 退出")
    print("  's' - 保存图像")
    
    # 创建图像矩阵
    rgb_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    frame_count = 0
    
    try:
        while True:
            # 获取新帧
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 获取RGB图像
                zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
                rgb_image = rgb_mat.get_data()
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)
                
                # 获取深度图像
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_raw = depth_mat.get_data()
                
                # 处理深度图像用于显示
                depth_display = depth_raw.copy()
                depth_display[~np.isfinite(depth_display)] = 0
                depth_display = np.clip(depth_display, 0, 10.0)  # 限制到10米
                depth_normalized = (depth_display / 10.0 * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # 添加帧计数信息
                frame_count += 1
                cv2.putText(rgb_image, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(depth_colored, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow('RGB', rgb_image)
                cv2.imshow('Depth', depth_colored)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f'rgb_{timestamp}.jpg', rgb_image)
                    cv2.imwrite(f'depth_{timestamp}.jpg', depth_colored)
                    np.save(f'depth_raw_{timestamp}.npy', depth_raw)
                    print(f"已保存图像: rgb_{timestamp}.jpg, depth_{timestamp}.jpg")
            
            else:
                print("获取帧失败")
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n程序被中断")
    
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        zed.close()
        print("相机已关闭")

if __name__ == "__main__":
    main() 