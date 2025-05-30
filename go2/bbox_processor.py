#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bbox_processor.py
Bbox处理器：负责接收bbox、获取点云数据、处理命令
"""

import socket
import json
import logging
import threading
import time
import requests
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [BboxProcessor] %(message)s')
logger = logging.getLogger(__name__)

# 端口配置
BBOX_PORT = 12346
COMMAND_PORT = 50001
ZED_POINTCLOUD_PORT = 12350
DIRECTION_MESSAGE_PORT = 12348  # 接收PC端方向命令的端口

class BboxProcessor:
    def __init__(self, bbox_port=BBOX_PORT, command_port=COMMAND_PORT, 
                 zed_host='localhost', zed_pointcloud_port=ZED_POINTCLOUD_PORT,
                 direction_port=DIRECTION_MESSAGE_PORT):
        self.bbox_port = bbox_port
        self.command_port = command_port
        self.direction_port = direction_port
        self.zed_host = zed_host
        self.zed_pointcloud_port = zed_pointcloud_port
        
        # 服务器socket
        self.bbox_server_socket = None
        self.command_server_socket = None
        self.direction_server_socket = None
        
        # 控制标志
        self.stop_event = threading.Event()
        self.threads = []
        
        # 状态数据
        self.latest_bbox = None
        self.latest_xyz_coords = None
        self.bbox_lock = threading.Lock()

    def _setup_servers(self):
        """设置服务器socket"""
        # Bbox接收服务器
        logger.info(f"设置Bbox服务器，端口: {self.bbox_port}")
        self.bbox_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bbox_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bbox_server_socket.bind(('0.0.0.0', self.bbox_port))
        self.bbox_server_socket.listen(5)
        
        # 命令服务器
        logger.info(f"设置命令服务器，端口: {self.command_port}")
        self.command_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.command_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.command_server_socket.bind(('0.0.0.0', self.command_port))
        self.command_server_socket.listen(5)
        
        # 方向服务器
        logger.info(f"设置方向服务器，端口: {self.direction_port}")
        self.direction_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.direction_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.direction_server_socket.bind(('0.0.0.0', self.direction_port))
        self.direction_server_socket.listen(5)
        
        logger.info("Bbox处理器服务器设置完成")

    def _get_pointcloud_data(self, bbox) -> Optional[Dict[str, Any]]:
        """从ZED数据服务器获取点云数据"""
        try:
            # 连接到ZED点云服务
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)  # 1秒超时
            sock.connect((self.zed_host, self.zed_pointcloud_port))
            
            # 发送bbox请求
            request = {"bbox": bbox}
            sock.sendall(json.dumps(request).encode('utf-8'))
            
            # 接收响应
            response_data = sock.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            sock.close()
            return response
            
        except Exception as e:
            logger.error(f"获取点云数据失败: {e}")
            return None

    def _bbox_server_thread(self):
        """Bbox接收服务线程"""
        logger.info("Bbox服务线程启动")
        
        while not self.stop_event.is_set():
            try:
                # 等待bbox客户端连接
                conn, addr = self.bbox_server_socket.accept()
                logger.debug(f"Bbox客户端连接: {addr}")
                
                # 处理bbox请求
                self._handle_bbox_request(conn)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Bbox服务错误: {e}")
                    time.sleep(0.1)
        
        logger.info("Bbox服务线程结束")

    def _handle_bbox_request(self, conn):
        """处理单个bbox请求"""
        try:
            # 接收bbox数据
            data = conn.recv(1024)
            if not data:
                return
            
            request = json.loads(data.decode('utf-8'))
            bbox = request.get('bbox')
            
            if not bbox or len(bbox) != 4:
                response = {"status": "error", "message": "Invalid bbox"}
                conn.sendall(json.dumps(response).encode('utf-8'))
                return
            
            logger.debug(f"收到bbox: {bbox}")
            
            # 从ZED服务器获取点云数据
            pointcloud_response = self._get_pointcloud_data(bbox)
            
            if pointcloud_response and pointcloud_response.get('status') == 'success':
                xyz_coords = pointcloud_response.get('xyz_coords')
                
                # 更新状态
                with self.bbox_lock:
                    self.latest_bbox = bbox
                    self.latest_xyz_coords = xyz_coords
                
                # 发送成功响应
                response = {
                    "status": "success",
                    "bbox": bbox,
                    "xyz_coords": xyz_coords,
                    "timestamp": pointcloud_response.get('timestamp')
                }
                
                #logger.info(f"Bbox处理成功: {bbox} -> {xyz_coords}")
                
            else:
                # 点云获取失败
                response = {
                    "status": "error",
                    "message": "Failed to get pointcloud data",
                    "bbox": bbox
                }
                logger.warning(f"Bbox点云获取失败: {bbox}")
            
            conn.sendall(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"处理bbox请求时出错: {e}")
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

    def _command_server_thread(self):
        """命令服务线程"""
        logger.info("命令服务线程启动")
        
        while not self.stop_event.is_set():
            try:
                # 等待命令客户端连接
                conn, addr = self.command_server_socket.accept()
                logger.debug(f"命令客户端连接: {addr}")
                
                # 处理命令请求
                self._handle_command_request(conn)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"命令服务错误: {e}")
                    time.sleep(0.1)
        
        logger.info("命令服务线程结束")

    def _handle_command_request(self, conn):
        """处理命令请求"""
        try:
            # 接收命令数据
            data = conn.recv(1024)
            if not data:
                return
            
            request = json.loads(data.decode('utf-8'))
            command = request.get('command')
            
            logger.debug(f"收到命令: {command}")
            
            # 根据命令类型处理
            if command == 'get_status':
                response = self._get_status()
            elif command == 'get_latest_detection':
                response = self._get_latest_detection()
            elif command == 'reset':
                response = self._reset_state()
            elif command == 'next_target':
                response = self._handle_next_target()
            else:
                response = {
                    "status": "error",
                    "message": f"Unknown command: {command}"
                }
            
            conn.sendall(json.dumps(response).encode('utf-8'))
            logger.debug(f"命令处理完成: {command}")
            
        except Exception as e:
            logger.error(f"处理命令请求时出错: {e}")
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

    def _get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        with self.bbox_lock:
            has_detection = self.latest_bbox is not None and self.latest_xyz_coords is not None
            
            return {
                "status": "success",
                "processor_status": "running",
                "has_detection": has_detection,
                "latest_bbox": self.latest_bbox,
                "latest_xyz_coords": self.latest_xyz_coords,
                "timestamp": time.time()
            }

    def _get_latest_detection(self) -> Dict[str, Any]:
        """获取最新检测结果"""
        with self.bbox_lock:
            if self.latest_bbox is not None and self.latest_xyz_coords is not None:
                return {
                    "status": "success",
                    "bbox": self.latest_bbox,
                    "xyz_coords": self.latest_xyz_coords,
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "no_detection",
                    "message": "No detection available"
                }

    def _reset_state(self) -> Dict[str, Any]:
        """重置状态"""
        with self.bbox_lock:
            self.latest_bbox = None
            self.latest_xyz_coords = None
        
        logger.info("状态已重置")
        return {
            "status": "success",
            "message": "State reset successfully"
        }

    def _handle_next_target(self) -> Dict[str, Any]:
        """处理next_target命令 - 通知切换到下一个检测目标"""
        logger.info("收到next_target命令，准备切换检测目标")
        
        # 清除当前检测状态，为新目标做准备
        with self.bbox_lock:
            self.latest_bbox = None
            self.latest_xyz_coords = None
        
        logger.info("已清除当前检测状态，等待新目标")
        
        # 向远端服务器发送NEXT_TARGET命令
        success = self._send_next_target_to_remote()
        
        if success:
            return {
                "status": "success",
                "message": "Target switched successfully and remote server notified",
                "timestamp": time.time()
            }
        else:
            return {
                "status": "partial_success",
                "message": "Target switched locally but failed to notify remote server",
                "timestamp": time.time()
            }
    
    def _send_next_target_to_remote(self) -> bool:
        """向远端服务器发送NEXT_TARGET命令"""
        # 使用与image_server.py相同的配置
        remote_host = "192.168.3.70"  # VIEWER_PC1_IP
        remote_port = 12347           # VIEWER_ORDER_PORT
        
        try:
            logger.info(f"向远端服务器发送NEXT_TARGET命令: {remote_host}:{remote_port}")
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  # 5秒超时
                s.connect((remote_host, remote_port))
                
                # 发送纯文本字符串 "NEXT_TARGET"
                s.sendall(b"NEXT_TARGET")
                logger.info("已成功发送NEXT_TARGET命令到远端服务器")
                
                return True
                
        except Exception as e:
            logger.error(f"向远端服务器发送NEXT_TARGET命令失败: {e}")
            return False

    def _direction_server_thread(self):
        """方向命令服务线程"""
        logger.info("方向命令服务线程启动")
        
        while not self.stop_event.is_set():
            try:
                # 等待方向命令客户端连接
                conn, addr = self.direction_server_socket.accept()
                logger.debug(f"方向命令客户端连接: {addr}")
                
                # 处理方向命令请求
                self._handle_direction_request(conn, addr)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"方向命令服务错误: {e}")
                    time.sleep(0.1)
        
        logger.info("方向命令服务线程结束")

    def _handle_direction_request(self, conn, addr):
        """处理方向命令请求"""
        logger.info(f"方向命令连接来自 {addr}")
        try:
            data = conn.recv(1024)
            if data:
                direction = data.decode('utf-8').strip()
                logger.info(f"收到方向命令: '{direction}'")
                
                # 有效方向列表
                valid_directions = [
                    "forward", "backward", "left", "right",
                    "front-left", "front-right", "back-left", "back-right"
                ]
                
                if direction in valid_directions:
                    logger.info(f"处理方向命令: {direction}")
                    # 转发方向命令到机器人端口50002
                    success = self._forward_direction_to_robot(direction)
                    if success:
                        conn.sendall(f"ACK: Direction {direction} received and processed".encode('utf-8'))
                    else:
                        conn.sendall(f"ERROR: Failed to forward direction {direction}".encode('utf-8'))
                else:
                    logger.warning(f"无效方向命令: {direction}")
                    conn.sendall(f"ERROR: Invalid direction {direction}".encode('utf-8'))
        except Exception as e:
            logger.error(f"处理方向命令时出错 {addr}: {e}")
        finally:
            conn.close()

    def _forward_direction_to_robot(self, direction):
        """转发方向命令到机器人端口50002"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(('localhost', 50002))
                s.sendall(direction.encode('utf-8'))
                response = s.recv(1024)
                logger.info(f"方向命令 '{direction}' 已转发到机器人，响应: {response.decode()}")
                return True
        except Exception as e:
            logger.error(f"转发方向命令 '{direction}' 到机器人失败: {e}")
            return False

    def start(self):
        """启动Bbox处理器"""
        logger.info("启动Bbox处理器...")
        
        try:
            self._setup_servers()
            
            # 启动各个线程
            threads_config = [
                ("BboxServer", self._bbox_server_thread),
                ("CommandServer", self._command_server_thread),
                ("DirectionServer", self._direction_server_thread)
            ]
            
            for name, target in threads_config:
                thread = threading.Thread(target=target, name=name)
                thread.daemon = True
                thread.start()
                self.threads.append(thread)
                logger.info(f"线程 {name} 已启动")
            
            logger.info("Bbox处理器启动完成")
            logger.info(f"Bbox服务端口: {self.bbox_port}")
            logger.info(f"命令服务端口: {self.command_port}")
            logger.info(f"方向命令服务端口: {self.direction_port}")
            logger.info(f"ZED点云服务: {self.zed_host}:{self.zed_pointcloud_port}")
            
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
        """关闭Bbox处理器"""
        logger.info("正在关闭Bbox处理器...")
        
        self.stop_event.set()
        
        # 等待线程结束
        for thread in self.threads:
            if thread.is_alive():
                logger.info(f"等待线程 {thread.name} 结束...")
                thread.join(timeout=2)
        
        # 关闭socket
        if self.bbox_server_socket:
            try: self.bbox_server_socket.close()
            except: pass
        if self.command_server_socket:
            try: self.command_server_socket.close()
            except: pass
        if self.direction_server_socket:
            try: self.direction_server_socket.close()
            except: pass
        
        logger.info("Bbox处理器已关闭")

if __name__ == "__main__":
    processor = BboxProcessor()
    try:
        processor.start()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        processor.shutdown() 