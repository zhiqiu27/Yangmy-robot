import cv2
import torch
import numpy as np
import socket
import struct
import threading
import queue
import time
from PIL import Image
from torchvision.transforms.functional import to_tensor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

class RemoteCameraCutieDemo:
    def __init__(self, server_host='192.168.3.11', image_port=12345):
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = 480
        
        # 远程摄像头设置
        self.server_host = server_host
        self.image_port = image_port
        self.image_socket = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.processing_queue = queue.Queue(maxsize=5)
        self.running = False
        self.image_thread = None
        self.processing_thread = None  # 回退到单线程
        self.last_frame = None
        self.last_processed_frame = None
        
        # 鼠标框选相关变量
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.bbox = None
        self.mask_created = False
        self.frame_count = 0
        
        # 性能监控
        self.receive_fps_counter = 0
        self.receive_fps_start_time = time.time()
        self.display_fps_counter = 0
        self.display_fps_start_time = time.time()
        self.processing_fps_counter = 0
        self.processing_fps_start_time = time.time()
        
        # 网络性能监控
        self.total_bytes_received = 0
        self.network_start_time = time.time()
        self.frame_sizes = []
        self.receive_times = []
        self.decode_times = []
        
        # 性能优化设置
        self.skip_frames = True
        
        # 显示窗口
        cv2.namedWindow('Remote Camera Cutie Demo', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Remote Camera Cutie Demo', self.mouse_callback)
        
        print("初始化完成!")

    def connect_to_server(self):
        """连接到远程摄像头服务器"""
        try:
            self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.image_socket.settimeout(5.0)
            self.image_socket.connect((self.server_host, self.image_port))
            
            # 激进的网络优化
            self.image_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.image_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            self.image_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
            
            print("服务器连接成功")
            return True
        except Exception as e:
            print(f"连接服务器失败: {e}")
            return False

    def receive_all(self, sock, n):
        """接收指定长度的数据 - 带性能监控"""
        start_time = time.time()
        data = bytearray()
        while len(data) < n:
            try:
                chunk_size = min(n - len(data), 16384)
                packet = sock.recv(chunk_size)
                if not packet:
                    return None, 0
                data.extend(packet)
            except socket.timeout:
                print(f"接收超时！已接收 {len(data)}/{n} 字节")
                return None, time.time() - start_time
            except socket.error as e:
                print(f"接收错误: {e}")
                return None, time.time() - start_time
        
        receive_time = time.time() - start_time
        return data, receive_time

    def update_receive_fps(self):
        """更新接收FPS计数"""
        self.receive_fps_counter += 1
        current_time = time.time()
        if current_time - self.receive_fps_start_time > 5.0:  # 改为5秒
            fps = self.receive_fps_counter / (current_time - self.receive_fps_start_time)
            processing_fps = self.processing_fps_counter / (current_time - self.processing_fps_start_time) if (current_time - self.processing_fps_start_time) > 0 else 0
            display_fps = self.display_fps_counter / (current_time - self.display_fps_start_time) if (current_time - self.display_fps_start_time) > 0 else 0
            
            print(f"平均帧数 - 接收: {fps:.1f} FPS | 处理: {processing_fps:.1f} FPS | 显示: {display_fps:.1f} FPS")
            
            self.receive_fps_counter = 0
            self.receive_fps_start_time = current_time
            self.processing_fps_counter = 0
            self.processing_fps_start_time = current_time
            self.display_fps_counter = 0
            self.display_fps_start_time = current_time

    def print_network_diagnostics(self):
        """打印详细网络诊断信息"""
        print("\n=== 网络诊断信息 ===")
        if self.frame_sizes:
            print(f"总接收帧数: {len(self.frame_sizes)}")
            print(f"总接收数据: {self.total_bytes_received/1024/1024:.2f} MB")
            print(f"最小帧大小: {min(self.frame_sizes)/1024:.1f} KB")
            print(f"最大帧大小: {max(self.frame_sizes)/1024:.1f} KB")
            print(f"平均帧大小: {np.mean(self.frame_sizes)/1024:.1f} KB")
        
        if self.receive_times:
            print(f"最快接收时间: {min(self.receive_times)*1000:.1f} ms")
            print(f"最慢接收时间: {max(self.receive_times)*1000:.1f} ms")
            print(f"平均接收时间: {np.mean(self.receive_times)*1000:.1f} ms")
        
        if self.decode_times:
            print(f"最快解码时间: {min(self.decode_times)*1000:.1f} ms")
            print(f"最慢解码时间: {max(self.decode_times)*1000:.1f} ms")
            print(f"平均解码时间: {np.mean(self.decode_times)*1000:.1f} ms")
        
        print("==================\n")

    def image_receiver_thread(self):
        """图像接收线程"""
        consecutive_errors = 0
        
        while self.running:
            try:
                frame_start_time = time.time()
                
                # 接收图像大小
                img_size_data, size_receive_time = self.receive_all(self.image_socket, 4)
                if not img_size_data:
                    consecutive_errors += 1
                    if consecutive_errors > 2:
                        print("连续接收失败，退出接收线程")
                        break
                    continue
                
                img_size = struct.unpack('>I', img_size_data)[0]
                if img_size == 0:
                    continue
                
                # 记录帧大小
                self.frame_sizes.append(img_size)
                if len(self.frame_sizes) > 100:  # 只保留最近100帧的数据
                    self.frame_sizes.pop(0)
                
                # 接收图像数据
                img_data_jpeg, data_receive_time = self.receive_all(self.image_socket, img_size)
                if not img_data_jpeg:
                    consecutive_errors += 1
                    continue
                
                total_receive_time = size_receive_time + data_receive_time
                self.receive_times.append(total_receive_time)
                if len(self.receive_times) > 50:
                    self.receive_times.pop(0)
                
                # 解码图像
                decode_start_time = time.time()
                img_np_arr = np.frombuffer(img_data_jpeg, dtype=np.uint8)
                decoded_frame = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
                decode_time = time.time() - decode_start_time
                
                self.decode_times.append(decode_time)
                if len(self.decode_times) > 50:
                    self.decode_times.pop(0)
                
                if decoded_frame is not None:
                    # 更新统计
                    self.total_bytes_received += img_size + 4
                    
                    # 队列管理
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    try:
                        self.frame_queue.put_nowait(decoded_frame)
                        consecutive_errors = 0
                        self.update_receive_fps()
                    except queue.Full:
                        pass
                            
            except Exception as e:
                consecutive_errors += 1
                print(f"接收线程异常: {e}")
                if consecutive_errors > 3:
                    print(f"接收线程错误过多，退出")
                    break
                continue

    def processing_thread_func(self):
        """异步处理线程"""
        
        while self.running:
            try:
                frame = self.processing_queue.get(timeout=1.0)
                if frame is None:
                    break
                
                if self.mask_created and torch.cuda.is_available():
                    processed_frame = self.process_frame(frame)
                    self.last_processed_frame = processed_frame
                    self.processing_fps_counter += 1
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理线程错误: {e}")
                continue

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            if self.start_point and self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.bbox = (x1, y1, x2, y2)

    def create_mask_from_bbox(self, frame_shape, bbox):
        """从边界框创建mask"""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1
        return mask

    @torch.inference_mode()
    @torch.amp.autocast(device_type='cuda')
    def process_frame(self, frame):
        """处理单帧图像"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor(frame_rgb).cuda().float()
        
        if not self.mask_created and self.bbox is not None:
            mask_np = self.create_mask_from_bbox(frame.shape, self.bbox)
            mask_tensor = torch.from_numpy(mask_np).cuda()
            objects = [1]
            output_prob = self.processor.step(image_tensor, mask_tensor, objects=objects)
            self.mask_created = True

        elif self.mask_created:
            output_prob = self.processor.step(image_tensor)
        else:
            return frame
        
        mask = self.processor.output_prob_to_mask(output_prob)
        mask_np = mask.cpu().numpy()
        
        return self.visualize_result(frame, mask_np)

    def visualize_result(self, frame, mask):
        """可视化分割结果"""
        colored_mask = np.zeros_like(frame)
        colored_mask[mask == 1] = [0, 255, 0]
        
        alpha = 0.3
        result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result

    def run(self):
        """主运行循环"""
        if not self.connect_to_server():
            return
        
        self.running = True
        
        self.image_thread = threading.Thread(target=self.image_receiver_thread)
        self.image_thread.daemon = True
        self.image_thread.start()
        
        self.processing_thread = threading.Thread(target=self.processing_thread_func)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        try:
            while True:
                display_frame = None
                new_frame = None
                
                try:
                    new_frame = self.frame_queue.get_nowait()
                    self.last_frame = new_frame
                    
                    if self.skip_frames and self.mask_created:
                        try:
                            # 只保留最新的几帧，清理积压
                            while self.processing_queue.qsize() > 2:  # 改为2，保持更少的积压
                                try:
                                    self.processing_queue.get_nowait()
                                except queue.Empty:
                                    break
                            self.processing_queue.put_nowait(new_frame)
                        except queue.Full:
                            # 如果队列满了，移除最老的帧
                            try:
                                self.processing_queue.get_nowait()
                                self.processing_queue.put_nowait(new_frame)
                            except queue.Empty:
                                pass
                        
                        display_frame = self.last_processed_frame if self.last_processed_frame is not None else new_frame
                    else:
                        if torch.cuda.is_available():
                            display_frame = self.process_frame(new_frame)
                        else:
                            display_frame = new_frame
                    
                except queue.Empty:
                    if self.last_frame is None:
                        waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(waiting_frame, "Waiting for camera data...", 
                                  (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.imshow('Remote Camera Cutie Demo', waiting_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        continue
                    else:
                        display_frame = self.last_processed_frame if self.last_processed_frame is not None else self.last_frame
                
                if display_frame is not None:
                    if self.drawing and self.start_point and self.end_point:
                        cv2.rectangle(display_frame, self.start_point, self.end_point, (255, 0, 0), 2)
                    elif self.bbox and not self.mask_created:
                        x1, y1, x2, y2 = self.bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(display_frame, "Press any key to start tracking", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if not self.mask_created and not self.bbox:
                        cv2.putText(display_frame, "Draw bounding box around target object", 
                                  (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    elif self.mask_created:
                        status_text = f"Tracking... Frame: {self.frame_count}"
                        if self.skip_frames:
                            status_text += " (Skip Mode)"
                        cv2.putText(display_frame, status_text, 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        self.frame_count += 1
                    
                    if not torch.cuda.is_available():
                        cv2.putText(display_frame, "CUDA not available - CPU mode", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.imshow('Remote Camera Cutie Demo', display_frame)
                    
                    if new_frame is not None:
                        self.display_fps_counter += 1
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.bbox = None
                    self.mask_created = False
                    self.frame_count = 0
                    self.last_processed_frame = None
                    self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
                    self.processor.max_internal_size = 480
                elif key == ord('s'):
                    self.skip_frames = not self.skip_frames
                elif key == ord('d'):
                    self.print_network_diagnostics()
                elif key != 255 and self.bbox and not self.mask_created:
                    pass
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_queue.put_nowait(None)
            self.processing_thread.join(timeout=2.0)
        
        if self.image_thread and self.image_thread.is_alive():
            self.image_thread.join(timeout=2.0)
        
        if self.image_socket:
            try:
                self.image_socket.close()
            except:
                pass
        
        cv2.destroyAllWindows()

def main():
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("警告: CUDA不可用，将使用CPU模式（速度较慢）")
    
    demo = RemoteCameraCutieDemo(server_host='192.168.3.11', image_port=12345)
    demo.run()

if __name__ == "__main__":
    main()