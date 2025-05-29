#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zed_data_server.py
ZED数据服务器：负责图像传输和点云数据服务
"""

import socket
import struct
import cv2
import pyzed.sl as sl
import logging
import threading
import time
import json
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [ZEDServer] %(message)s')
logger = logging.getLogger(__name__)

# 端口配置
IMAGE_PORT = 12345
POINTCLOUD_PORT = 12350

class ZEDDataServer:
    def __init__(self, image_port=IMAGE_PORT, pointcloud_port=POINTCLOUD_PORT):
        self.image_port = image_port
        self.pointcloud_port = pointcloud_port
        
        # ZED相机配置
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        self.init_params.camera_fps = 24
        self.init_params.camera_resolution = sl.RESOLUTION.VGA
        
        self.runtime_params = sl.RuntimeParameters()
        self.img_mat = sl.Mat()
        self.xyz_mat = sl.Mat()
        
        # 数据缓存和锁
        self.data_lock = threading.Lock()
        self.latest_image = None
        self.latest_xyz_data = None
        self.latest_timestamp = 0
        self.latest_frame_id = 0  # 添加帧ID
        
        # 图像传输状态
        self.last_sent_frame_id = -1
        
        # 服务器socket
        self.image_server_socket = None
        self.pointcloud_server_socket = None
        self.image_conn = None
        
        # 控制标志
        self.stop_event = threading.Event()
        self.threads = []

    def _open_camera(self):
        """打开ZED相机"""
        logger.info("正在打开ZED相机...")
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"ZED相机打开失败: {status}")
            raise RuntimeError(f"ZED相机打开失败: {status}")
        logger.info("ZED相机打开成功")

    def _setup_servers(self):
        """设置服务器socket"""
        # 图像服务器
        logger.info(f"设置图像服务器，端口: {self.image_port}")
        self.image_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.image_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.image_server_socket.bind(('0.0.0.0', self.image_port))
        self.image_server_socket.listen(1)
        
        # 点云服务器
        logger.info(f"设置点云服务器，端口: {self.pointcloud_port}")
        self.pointcloud_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pointcloud_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.pointcloud_server_socket.bind(('0.0.0.0', self.pointcloud_port))
        self.pointcloud_server_socket.listen(5)  # 允许多个点云请求
        
        logger.info("服务器设置完成")

    def _zed_main_loop(self):
        """ZED数据获取和图像传输主循环"""
        logger.info("ZED主循环启动")
        frame_count = 0
        start_time = time.time()
        
        while not self.stop_event.is_set():
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                # 获取图像和深度数据
                self.zed.retrieve_image(self.img_mat, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZ)
                
                # 更新缓存数据
                with self.data_lock:
                    self.latest_image = self.img_mat.get_data()  # 不复制，直接引用
                    self.latest_xyz_data = self.xyz_mat.get_data()  # 不复制，直接引用
                    self.latest_timestamp = time.time()
                    self.latest_frame_id += 1
                    current_frame_id = self.latest_frame_id
                
                # 如果有图像连接，直接发送图像
                if self.image_conn and current_frame_id > self.last_sent_frame_id:
                    self._send_current_image()
                    self.last_sent_frame_id = current_frame_id
                
                frame_count += 1
                
                # 每2秒报告一次帧率
                if frame_count % 48 == 0:  # 24fps * 2秒
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.debug(f"ZED数据获取帧率: {fps:.2f} FPS")
                    frame_count = 0
                    start_time = time.time()
            else:
                logger.warning("ZED grab失败")
                time.sleep(0.01)
        
        logger.info("ZED主循环结束")

    def _send_current_image(self):
        """发送当前图像（在ZED主循环中调用，已持有锁）"""
        try:
            if self.latest_image is None:
                return
            
            # 转换和编码图像
            rgb_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGRA2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            result, frame_jpeg = cv2.imencode('.jpg', rgb_image, encode_param)
            
            if result:
                # 发送图像大小
                image_size = len(frame_jpeg)
                self.image_conn.sendall(struct.pack('>I', image_size))
                
                # 发送图像数据
                if image_size > 0:
                    self.image_conn.sendall(frame_jpeg.tobytes())
            else:
                logger.error("JPEG编码失败")
                
        except Exception as e:
            logger.warning(f"图像发送失败: {e}")
            # 连接断开，清理连接
            try:
                self.image_conn.close()
            except:
                pass
            self.image_conn = None
            self.last_sent_frame_id = -1

    def _image_server_thread(self):
        """图像服务器线程（只负责接受连接）"""
        logger.info("图像服务线程启动")
        
        while not self.stop_event.is_set():
            try:
                # 等待客户端连接
                logger.info("等待图像客户端连接...")
                conn, addr = self.image_server_socket.accept()
                logger.info(f"图像客户端已连接: {addr}")
                
                # 设置连接
                self.image_conn = conn
                self.last_sent_frame_id = -1  # 重置帧ID
                
                # 等待连接断开
                while not self.stop_event.is_set() and self.image_conn:
                    time.sleep(0.1)
                        
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"图像服务错误: {e}")
                    time.sleep(1)
        
        logger.info("图像服务线程结束")

    def _pointcloud_server_thread(self):
        """点云数据服务线程"""
        logger.info("点云服务线程启动")
        
        while not self.stop_event.is_set():
            try:
                # 等待点云请求连接
                conn, addr = self.pointcloud_server_socket.accept()
                logger.debug(f"点云客户端连接: {addr}")
                
                # 处理点云请求
                self._handle_pointcloud_request(conn)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"点云服务错误: {e}")
                    time.sleep(0.1)
        
        logger.info("点云服务线程结束")

    def _handle_pointcloud_request(self, conn):
        """处理单个点云请求"""
        try:
            # 接收请求数据
            data = conn.recv(1024)
            if not data:
                return
            
            request = json.loads(data.decode('utf-8'))
            bbox = request.get('bbox')
            
            if not bbox or len(bbox) != 4:
                response = {"status": "error", "message": "Invalid bbox"}
                conn.sendall(json.dumps(response).encode('utf-8'))
                return
            
            # 获取当前点云数据（需要复制以避免线程安全问题）
            with self.data_lock:
                if self.latest_xyz_data is not None:
                    xyz_data = self.latest_xyz_data.copy()  # 这里需要复制
                    timestamp = self.latest_timestamp
                else:
                    response = {"status": "error", "message": "No pointcloud data available"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    return
            
            # 计算bbox区域的3D坐标
            xyz_coords = self._calculate_bbox_xyz(bbox, xyz_data)
            
            # 发送响应
            response = {
                "status": "success" if xyz_coords else "no_valid_points",
                "xyz_coords": xyz_coords,
                "timestamp": timestamp
            }
            
            conn.sendall(json.dumps(response).encode('utf-8'))
            logger.debug(f"点云请求处理完成: {bbox} -> {xyz_coords}")
            
        except Exception as e:
            logger.error(f"处理点云请求时出错: {e}")
            try:
                response = {"status": "error", "message": str(e)}
                conn.sendall(json.dumps(response).encode('utf-8'))
            except:
                pass
        finally:
            try:
                conn.close()
            except:
                pass

    def _calculate_bbox_xyz(self, bbox, xyz_data):
        """计算bbox区域的3D坐标"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            img_h, img_w = xyz_data.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"无效的bbox: {bbox}")
                return None
            
            # 提取bbox区域的点云数据
            patch = xyz_data[y1:y2 + 1, x1:x2 + 1, :3]
            
            # 过滤有效点
            mask = np.isfinite(patch[..., 0]) & np.isfinite(patch[..., 1]) & np.isfinite(patch[..., 2])
            if not np.any(mask):
                logger.debug(f"bbox区域无有效深度点: {bbox}")
                return None
            
            # 计算中位数坐标
            median_coord = np.median(patch[mask], axis=0)
            return [float(median_coord[0]), float(median_coord[1]), float(median_coord[2])]
            
        except Exception as e:
            logger.error(f"计算bbox 3D坐标时出错: {e}")
            return None

    def start(self):
        """启动ZED数据服务器"""
        logger.info("启动ZED数据服务器...")
        
        try:
            self._open_camera()
            self._setup_servers()
            
            # 启动各个线程
            threads_config = [
                ("ZEDMainLoop", self._zed_main_loop),
                ("ImageServer", self._image_server_thread),
                ("PointcloudServer", self._pointcloud_server_thread)
            ]
            
            for name, target in threads_config:
                thread = threading.Thread(target=target, name=name)
                thread.daemon = True
                thread.start()
                self.threads.append(thread)
                logger.info(f"线程 {name} 已启动")
            
            logger.info("ZED数据服务器启动完成")
            
            # 主线程保持运行
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("收到中断信号，正在关闭...")
                
        except Exception as e:
            logger.error(f"启动失败: {e}")
            raise
        finally:
            self.shutdown()

    def shutdown(self):
        """关闭ZED数据服务器"""
        logger.info("正在关闭ZED数据服务器...")
        
        self.stop_event.set()
        
        # 等待线程结束
        for thread in self.threads:
            if thread.is_alive():
                logger.info(f"等待线程 {thread.name} 结束...")
                thread.join(timeout=2)
        
        # 关闭socket
        if self.image_conn:
            try: self.image_conn.close()
            except: pass
        if self.image_server_socket:
            try: self.image_server_socket.close()
            except: pass
        if self.pointcloud_server_socket:
            try: self.pointcloud_server_socket.close()
            except: pass
        
        # 关闭ZED相机
        if self.zed and self.zed.is_opened():
            logger.info("关闭ZED相机...")
            self.zed.close()
        
        logger.info("ZED数据服务器已关闭")

if __name__ == "__main__":
    server = ZEDDataServer()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        server.shutdown() 