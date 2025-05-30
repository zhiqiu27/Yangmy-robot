import socket
import struct
import threading
import json
import time
import cv2
import numpy as np
from collections import deque
from config import *

class NetworkPerformanceMonitor:
    """网络性能监控器"""
    def __init__(self):
        self.received_frames = 0
        self.received_bytes = 0
        self.last_fps_time = time.time()
        self.fps_history = deque(maxlen=PERFORMANCE_HISTORY_SIZE)
        self.bandwidth_history = deque(maxlen=PERFORMANCE_HISTORY_SIZE)
        
    def update_stats(self, frame_size):
        """更新统计信息"""
        self.received_frames += 1
        self.received_bytes += frame_size
        
        current_time = time.time()
        if current_time - self.last_fps_time >= RECEIVED_FPS_INTERVAL:
            # 计算FPS
            fps = self.received_frames / (current_time - self.last_fps_time)
            self.fps_history.append(fps)
            
            # 计算带宽 (MB/s)
            bandwidth = (self.received_bytes / (1024 * 1024)) / (current_time - self.last_fps_time)
            self.bandwidth_history.append(bandwidth)
            
            # 重置计数器
            self.received_frames = 0
            self.received_bytes = 0
            self.last_fps_time = current_time
            
            return fps, bandwidth
        return None, None
    
    def get_average_fps(self):
        """获取平均FPS"""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0
    
    def get_average_bandwidth(self):
        """获取平均带宽"""
        return sum(self.bandwidth_history) / len(self.bandwidth_history) if self.bandwidth_history else 0.0

class OptimizedImageReceiver:
    """优化的图像接收器（基于remote_camera_demo）"""
    
    def __init__(self, host, port, on_frame_callback=None):
        self.host = host
        self.port = port
        self.on_frame_callback = on_frame_callback
        
        self.socket = None
        self.running = False
        self.thread = None
        
        # 性能监控
        self.performance_monitor = NetworkPerformanceMonitor()
        
        # 网络诊断
        self.connection_attempts = 0
        self.last_connection_time = 0
        self.connection_errors = []
        
        print(f"OptimizedImageReceiver initialized for {host}:{port}")
    
    def create_optimized_socket(self):
        """创建优化的socket连接"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 网络优化设置
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, NETWORK_RCVBUF_SIZE)
            if ENABLE_TCP_NODELAY:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            sock.settimeout(IMAGE_SOCKET_TIMEOUT)
            
            return sock
            
        except Exception as e:
            print(f"Socket创建失败: {e}")
            return None
    
    def connect_with_retry(self):
        """带重试的连接"""
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                self.connection_attempts += 1
                print(f"尝试连接图像服务器 {self.host}:{self.port} (第{attempt+1}次)")
                
                self.socket = self.create_optimized_socket()
                if not self.socket:
                    continue
                
                self.socket.connect((self.host, self.port))
                self.last_connection_time = time.time()
                print(f"图像服务器连接成功！")
                return True
                
            except Exception as e:
                error_msg = f"连接失败 (第{attempt+1}次): {e}"
                print(error_msg)
                self.connection_errors.append((time.time(), error_msg))
                
                if self.socket:
                    try:
                        self.socket.close()
                    except:
                        pass
                    self.socket = None
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # 指数退避
        
        return False
    
    def receive_frame_data(self):
        """接收完整帧数据"""
        try:
            # 1. 接收帧大小 (4字节)
            size_data = b''
            while len(size_data) < 4:
                chunk = self.socket.recv(4 - len(size_data))
                if not chunk:
                    return None
                size_data += chunk
            
            frame_size = struct.unpack('!I', size_data)[0]
            
            # 2. 接收帧数据
            frame_data = b''
            remaining = frame_size
            
            while remaining > 0:
                chunk_size = min(remaining, NETWORK_CHUNK_SIZE)
                chunk = self.socket.recv(chunk_size)
                if not chunk:
                    return None
                frame_data += chunk
                remaining -= len(chunk)
            
            return frame_data, frame_size
            
        except socket.timeout:
            print("图像接收超时")
            return None
        except Exception as e:
            print(f"帧数据接收错误: {e}")
            return None
    
    def decode_frame(self, frame_data):
        """解码帧数据"""
        try:
            # 解码JPEG数据
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"帧解码错误: {e}")
            return None
    
    def receiver_worker(self):
        """接收器工作线程"""
        print("图像接收线程启动")
        
        while self.running:
            try:
                # 连接服务器
                if not self.connect_with_retry():
                    print("无法连接到图像服务器，退出接收线程")
                    break
                
                # 接收循环
                while self.running and self.socket:
                    result = self.receive_frame_data()
                    if result is None:
                        print("图像接收中断，尝试重连...")
                        break
                    
                    frame_data, frame_size = result
                    
                    # 解码帧
                    frame = self.decode_frame(frame_data)
                    if frame is None:
                        continue
                    
                    # 更新性能统计
                    fps, bandwidth = self.performance_monitor.update_stats(frame_size)
                    if fps is not None:
                        print(f"[Network] 接收FPS: {fps:.1f}, 带宽: {bandwidth:.1f} MB/s")
                    
                    # 回调处理
                    if self.on_frame_callback:
                        self.on_frame_callback(frame)
                
                # 连接断开，清理socket
                if self.socket:
                    try:
                        self.socket.close()
                    except:
                        pass
                    self.socket = None
                
                # 如果还在运行，等待后重试
                if self.running:
                    print("等待重连...")
                    time.sleep(3.0)
                    
            except Exception as e:
                print(f"接收线程错误: {e}")
                if self.running:
                    time.sleep(3.0)
        
        print("图像接收线程结束")
    
    def start(self):
        """启动接收器"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self.receiver_worker, daemon=True)
        self.thread.start()
        print("图像接收器启动")
    
    def stop(self):
        """停止接收器"""
        if not self.running:
            return
            
        print("停止图像接收器...")
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=THREAD_JOIN_TIMEOUT)
        
        print("图像接收器已停止")
    
    def get_network_stats(self):
        """获取网络统计信息"""
        return {
            'avg_fps': self.performance_monitor.get_average_fps(),
            'avg_bandwidth': self.performance_monitor.get_average_bandwidth(),
            'connection_attempts': self.connection_attempts,
            'last_connection_time': self.last_connection_time,
            'recent_errors': self.connection_errors[-5:] if self.connection_errors else []
        }

class NetworkHandler:
    """增强的网络处理器"""
    
    def __init__(self):
        # 图像接收器
        self.image_receiver = None
        
        # 其他网络组件
        self.bbox_socket = None
        self.json_server_socket = None
        self.order_listener_socket = None
        
        # 线程
        self.json_receiver_thread = None
        self.order_listener_thread = None
        
        # 回调函数
        self.on_frame_received = None
        self.on_json_received = None
        self.on_command_received = None
        self.on_3d_coords_received = None
        
        # 控制标志
        self.running = False
        
        # bbox发送统计
        self.bbox_send_count = 0
        self.bbox_send_errors = 0
        self.last_bbox_send_time = 0
        
        print("NetworkHandler initialized")
    
    def setup_image_receiver(self, on_frame_callback):
        """设置图像接收器"""
        self.on_frame_received = on_frame_callback
        self.image_receiver = OptimizedImageReceiver(
            IMAGE_SERVER_HOST, 
            IMAGE_PORT, 
            on_frame_callback=self.on_frame_received
        )
    
    def setup_bbox_sender(self):
        """设置bbox发送器（ZED协议使用临时连接）"""
        # ZED协议使用临时连接，不需要预先设置持久socket
        print("Bbox发送器设置完成（ZED协议模式）")
        return True
    
    def setup_json_receiver(self, on_json_callback):
        """设置JSON接收器"""
        self.on_json_received = on_json_callback
        try:
            self.json_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.json_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.json_server_socket.bind((JSON_SERVER_HOST, JSON_SERVER_PORT))
            self.json_server_socket.listen(1)
            print(f"JSON服务器监听 {JSON_SERVER_HOST}:{JSON_SERVER_PORT}")
            return True
        except Exception as e:
            print(f"JSON接收器设置失败: {e}")
            return False
    
    def setup_command_listener(self, on_command_callback):
        """设置命令监听器"""
        self.on_command_received = on_command_callback
        try:
            self.order_listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.order_listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.order_listener_socket.bind((ORDER_LISTENER_HOST, ORDER_PORT))
            self.order_listener_socket.listen(1)
            print(f"命令监听器监听 {ORDER_LISTENER_HOST}:{ORDER_PORT}")
            return True
        except Exception as e:
            print(f"命令监听器设置失败: {e}")
            return False
    
    def json_receiver_worker(self):
        """JSON接收工作线程"""
        print("JSON接收线程启动")
        
        while self.running:
            try:
                print("等待JSON客户端连接...")
                client_socket, addr = self.json_server_socket.accept()
                print(f"JSON客户端连接: {addr}")
                
                with client_socket:
                    while self.running:
                        try:
                            # 原始系统协议：先接收4字节长度，再接收JSON数据
                            # 1. 接收消息长度（4字节，大端序）
                            raw_msglen = self.receive_all(client_socket, 4)
                            if not raw_msglen:
                                print("[JSON] 无法接收消息长度")
                                break
                            
                            msglen = struct.unpack('>I', raw_msglen)[0]
                            print(f"[JSON] 准备接收 {msglen} 字节的JSON数据")
                            
                            # 2. 接收JSON数据
                            if msglen == 0:
                                print("[JSON] 收到零长度消息，假设为空目标")
                                json_data = "{}"
                            elif msglen > 0:
                                json_data_bytes = self.receive_all(client_socket, msglen)
                                if not json_data_bytes:
                                    print("[JSON] 无法接收JSON数据")
                                    break
                                json_data = json_data_bytes.decode('utf-8')
                                print(f"[JSON] 收到数据: {json_data}")
                            else:
                                print(f"[JSON] 无效的消息长度: {msglen}")
                                break
                            
                            # 3. 处理JSON数据
                            if json_data and self.on_json_received:
                                self.on_json_received(json_data)
                                
                        except struct.error as e:
                            print(f"[JSON] 协议解析错误: {e}")
                            break
                        except UnicodeDecodeError as e:
                            print(f"[JSON] 编码错误: {e}")
                            break
                        except Exception as e:
                            print(f"[JSON] 接收错误: {e}")
                            break
                
                print("JSON客户端断开连接")
                
            except Exception as e:
                if self.running:
                    print(f"JSON服务器错误: {e}")
                    time.sleep(1.0)
        
        print("JSON接收线程结束")
    
    def receive_all(self, sock, n):
        """接收指定字节数的数据（与原始系统兼容）"""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def command_listener_worker(self):
        """命令监听工作线程"""
        print("命令监听线程启动")
        
        while self.running:
            try:
                print("等待命令客户端连接...")
                client_socket, addr = self.order_listener_socket.accept()
                print(f"命令客户端连接: {addr}")
                
                with client_socket:
                    while self.running:
                        try:
                            data = client_socket.recv(1024).decode('utf-8')
                            if not data:
                                break
                            
                            command = data.strip()
                            print(f"[Command] 收到命令: {command}")
                            
                            # 处理原始系统的命令格式
                            if command == "NEXT_TARGET":
                                if self.on_command_received:
                                    response = self.on_command_received("switch_target")
                                    if "Not enough targets" in response:
                                        client_socket.send(b"ERROR: Not enough targets\n")
                                    elif "Already on second target" in response:
                                        client_socket.send(b"INFO: Already on second target or no further switch\n")
                                    elif "Switched to target" in response:
                                        client_socket.send(b"OK: Switching to next target\n")
                                    else:
                                        client_socket.send(f"RESULT: {response}\n".encode('utf-8'))
                            else:
                                # 处理其他命令
                                if self.on_command_received:
                                    response = self.on_command_received(command)
                                    if response:
                                        client_socket.send(f"{response}\n".encode('utf-8'))
                                    
                        except Exception as e:
                            print(f"[Command] 接收错误: {e}")
                            break
                
                print("命令客户端断开连接")
                
            except Exception as e:
                if self.running:
                    print(f"命令监听器错误: {e}")
                    time.sleep(1.0)
        
        print("命令监听线程结束")
    
    def send_bbox_data(self, bbox_data):
        """发送bbox数据到ZED服务器（按照ZED协议格式）"""
        if not bbox_data:
            return False
        
        current_time = time.time()
        
        # 限制发送频率
        if current_time - self.last_bbox_send_time < BBOX_SEND_INTERVAL:
            return True
        
        try:
            # 创建临时socket连接（ZED协议：处理完请求后自动关闭）
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(BBOX_SOCKET_TIMEOUT)
                sock.connect((SERVER_HOST_IP, BBOX_PORT))
                
                # 按照ZED协议发送JSON格式
                x1, y1, x2, y2 = map(int, bbox_data)
                bbox_json = {
                    "bbox": [x1, y1, x2, y2]
                }
                
                message = json.dumps(bbox_json).encode('utf-8')
                sock.send(message)
                
                # 接收ZED服务器的响应
                response_data = sock.recv(4096)
                if response_data:
                    try:
                        response = json.loads(response_data.decode('utf-8'))
                        if response.get("status") == "success":
                            xyz_coords = response.get("xyz_coords")
                            if xyz_coords:
                                #print(f"[ZED] 目标3D坐标: {xyz_coords} (x={xyz_coords[0]:.3f}m, y={xyz_coords[1]:.3f}m, z={xyz_coords[2]:.3f}m)")
                                # 可以通过回调函数传递3D坐标给其他模块
                                if hasattr(self, 'on_3d_coords_received') and self.on_3d_coords_received:
                                    self.on_3d_coords_received(xyz_coords, response.get("timestamp"))
                        else:
                            print(f"[ZED] 目标检测失败: {response}")
                    except json.JSONDecodeError as e:
                        print(f"[ZED] 响应解析错误: {e}")
                
                self.bbox_send_count += 1
                self.last_bbox_send_time = current_time
                
                # 定期报告发送统计
                if self.bbox_send_count % 100 == 0:
                    print(f"[ZED] 已发送 {self.bbox_send_count} 次bbox查询，错误 {self.bbox_send_errors} 次")
                
                return True
                
        except Exception as e:
            self.bbox_send_errors += 1
            
            # 智能错误记录
            error_type = type(e).__name__
            if not hasattr(self, '_last_bbox_error_type') or \
               self._last_bbox_error_type != error_type or \
               self.bbox_send_errors % 10 == 0:
                print(f"[ZED] Bbox发送错误: {e}")
                self._last_bbox_error_type = error_type
            
            return False
    
    def send_direction_command(self, direction):
        """发送方向指令到图像服务器"""
        if direction not in VALID_DIRECTIONS:
            print(f"无效方向指令: {direction}")
            return False
        
        try:
            # 创建临时socket发送指令
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect((IMAGE_SERVER_HOST, IMAGE_SERVER_MESSAGE_PORT))
                sock.send(direction.encode('utf-8'))
                print(f"[Direction] 发送方向指令: {direction}")
                return True
                
        except Exception as e:
            print(f"方向指令发送失败: {e}")
            return False
    
    def start_all(self):
        """启动所有网络组件"""
        if self.running:
            return
        
        print("启动网络处理器...")
        self.running = True
        
        # 启动图像接收器
        if self.image_receiver:
            self.image_receiver.start()
        
        # 启动JSON接收线程
        if self.json_server_socket:
            self.json_receiver_thread = threading.Thread(
                target=self.json_receiver_worker, daemon=True
            )
            self.json_receiver_thread.start()
        
        # 启动命令监听线程
        if self.order_listener_socket:
            self.order_listener_thread = threading.Thread(
                target=self.command_listener_worker, daemon=True
            )
            self.order_listener_thread.start()
        
        print("网络处理器启动完成")
    
    def stop_all(self):
        """停止所有网络组件"""
        if not self.running:
            return
        
        print("停止网络处理器...")
        self.running = False
        
        # 停止图像接收器
        if self.image_receiver:
            self.image_receiver.stop()
        
        # 关闭所有socket
        for sock_name, sock in [
            ('json_server_socket', self.json_server_socket),
            ('order_listener_socket', self.order_listener_socket)
        ]:
            if sock:
                try:
                    sock.close()
                    print(f"{sock_name} 已关闭")
                except:
                    pass
        
        # 等待线程结束
        for thread_name, thread in [
            ('json_receiver_thread', self.json_receiver_thread),
            ('order_listener_thread', self.order_listener_thread)
        ]:
            if thread and thread.is_alive():
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
                print(f"{thread_name} 已结束")
        
        print("网络处理器已停止")
    
    def get_network_status(self):
        """获取网络状态"""
        status = {
            'running': self.running,
            'bbox_send_count': self.bbox_send_count,
            'bbox_send_errors': self.bbox_send_errors,
        }
        
        if self.image_receiver:
            status.update(self.image_receiver.get_network_stats())
        
        return status
    
    def query_zed_system_status(self):
        """查询ZED系统状态（端口50001）"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5.0)
                sock.connect((SERVER_HOST_IP, 50001))
                
                command = {"command": "get_status"}
                message = json.dumps(command).encode('utf-8')
                sock.send(message)
                
                response_data = sock.recv(4096)
                if response_data:
                    response = json.loads(response_data.decode('utf-8'))
                    print(f"[ZED System] 状态: {response}")
                    return response
                    
        except Exception as e:
            print(f"[ZED System] 状态查询失败: {e}")
            return None
    
    def get_zed_latest_detection(self):
        """获取ZED最新检测结果（端口50001）"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5.0)
                sock.connect((SERVER_HOST_IP, 50001))
                
                command = {"command": "get_latest_detection"}
                message = json.dumps(command).encode('utf-8')
                sock.send(message)
                
                response_data = sock.recv(4096)
                if response_data:
                    response = json.loads(response_data.decode('utf-8'))
                    print(f"[ZED System] 最新检测: {response}")
                    return response
                    
        except Exception as e:
            print(f"[ZED System] 最新检测查询失败: {e}")
            return None
    
    def reset_zed_system(self):
        """重置ZED系统（端口50001）"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5.0)
                sock.connect((SERVER_HOST_IP, 50001))
                
                command = {"command": "reset"}
                message = json.dumps(command).encode('utf-8')
                sock.send(message)
                
                response_data = sock.recv(4096)
                if response_data:
                    response = json.loads(response_data.decode('utf-8'))
                    print(f"[ZED System] 重置结果: {response}")
                    return response
                    
        except Exception as e:
            print(f"[ZED System] 重置失败: {e}")
            return None 