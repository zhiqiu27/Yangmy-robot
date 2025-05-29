#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_zed_stream.py
最简单的ZED图像传输脚本
只负责获取ZED相机图像并通过socket发送
"""

import socket
import struct
import cv2
import pyzed.sl as sl
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 端口配置（与原代码保持一致）
IMAGE_PORT = 12345

class SimpleZEDStreamer:
    def __init__(self, image_port=IMAGE_PORT):
        self.image_port = image_port
        self.zed = sl.Camera()
        self.server_socket = None
        self.client_conn = None
        
        # ZED初始化参数
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        self.init_params.camera_fps = 24
        self.init_params.camera_resolution = sl.RESOLUTION.VGA
        
        self.runtime_params = sl.RuntimeParameters()
        self.img_mat = sl.Mat()

    def _open_camera(self):
        """打开ZED相机"""
        logger.info("正在打开ZED相机...")
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED相机打开失败: {status}")
            raise RuntimeError(f"ZED相机打开失败: {status}")
        logger.info("ZED相机打开成功")

    def _setup_socket(self):
        """设置socket服务器"""
        logger.info(f"设置图像服务器，端口: {self.image_port}")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.image_port))
        self.server_socket.listen(1)
        logger.info(f"图像服务器监听端口 {self.image_port}")

    def _wait_for_client(self):
        """等待客户端连接"""
        logger.info("等待客户端连接...")
        self.client_conn, addr = self.server_socket.accept()
        logger.info(f"客户端已连接: {addr}")

    def _send_image(self, image_data):
        """发送图像数据"""
        try:
            # 发送图像大小
            image_size = len(image_data)
            self.client_conn.sendall(struct.pack('>I', image_size))
            
            # 发送图像数据
            if image_size > 0:
                self.client_conn.sendall(image_data)
            return True
        except Exception as e:
            logger.error(f"发送图像失败: {e}")
            return False

    def run(self):
        """主运行循环"""
        try:
            self._open_camera()
            self._setup_socket()
            self._wait_for_client()
            
            logger.info("开始图像传输...")
            frame_count = 0
            start_time = time.time()
            
            while True:
                # 获取图像
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(self.img_mat, sl.VIEW.LEFT)
                    
                    # 转换图像格式
                    bgr_image = self.img_mat.get_data()
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
                    
                    # 编码为JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                    result, frame_jpeg = cv2.imencode('.jpg', rgb_image, encode_param)
                    
                    if result:
                        # 发送图像
                        if not self._send_image(frame_jpeg.tobytes()):
                            logger.warning("图像发送失败，等待重新连接...")
                            self._wait_for_client()
                            continue
                        
                        frame_count += 1
                        
                        # 每2秒报告一次帧率
                        if frame_count % 48 == 0:  # 24fps * 2秒
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed
                            logger.info(f"传输帧率: {fps:.2f} FPS")
                            frame_count = 0
                            start_time = time.time()
                    else:
                        logger.error("JPEG编码失败")
                else:
                    logger.warning("ZED图像获取失败")
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logger.info("用户中断，正在关闭...")
        except Exception as e:
            logger.error(f"运行时错误: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """关闭资源"""
        logger.info("正在关闭ZED图像传输器...")
        
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
            
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        if self.zed and self.zed.is_opened():
            logger.info("正在关闭ZED相机...")
            self.zed.close()
            
        logger.info("ZED图像传输器已关闭")

if __name__ == "__main__":
    streamer = SimpleZEDStreamer()
    streamer.run() 