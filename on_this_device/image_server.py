import socket
import time
import numpy as np
import json
import logging
import threading

# 尝试导入 PositionDetector，如果失败则记录错误但允许脚本尝试运行（如果仅用于测试网络部分）
try:
    from pos_detect import PositionDetector
except ImportError:
    PositionDetector = None
    logging.error("错误：未能导入 PositionDetector。请确保 pos_detect.py 文件存在且在PYTHONPATH中。")
    logging.warning("ImageProcessorServer 将在没有实际位置检测功能的情况下运行。")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (ImageServer) %(message)s',
    handlers=[
        logging.FileHandler('/home/unitree/image_server.log'), # 日志文件路径
        logging.StreamHandler()
    ]
)

# 全局常量
CALIBRATION_SAMPLES = 5
CALIBRATION_DELAY = 0.5  # Seconds
HOST = 'localhost' # 对于同一台机器上的进程间通信，使用localhost
PORT = 50001       # 确保这个端口未被其他应用占用

class ImageProcessorServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.pos_detector = None
        if PositionDetector:
            try:
                self.pos_detector = PositionDetector()
                self.pos_detector.start()
                self.logger.info("PositionDetector 初始化并启动成功。")
            except Exception as e:
                self.logger.error(f"PositionDetector 初始化失败: {e}")
                self.pos_detector = None # 确保在失败时self.pos_detector为None
        else:
            self.logger.warning("PositionDetector 类未加载，无法进行实际的位置检测。")

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 允许快速重用地址
        self.running = False

    def _collect_calibration_data(self):
        """
        收集标定数据样本并计算平均值。
        返回: [avg_x, avg_y] 或 None（如果失败）。
        """
        if not self.pos_detector:
            self.logger.error("PositionDetector 不可用，无法收集校准数据。")
            return None

        collected_coords = []
        self.logger.info(f"开始收集 {CALIBRATION_SAMPLES} 个样本用于 'shirt' 位置校准...")
        for i in range(CALIBRATION_SAMPLES):
            try:
                xy = self.pos_detector.get_current_object_xy()
                if xy:
                    self.logger.info(f"校准样本 {i+1}/{CALIBRATION_SAMPLES}: {xy}")
                    collected_coords.append(xy)
                else:
                    self.logger.warning(f"校准样本 {i+1}/{CALIBRATION_SAMPLES}: 未获取到坐标。")
            except Exception as e:
                self.logger.error(f"获取校准样本 {i+1} 时出错: {e}")
            time.sleep(CALIBRATION_DELAY)
        
        if not collected_coords:
            self.logger.error("校准失败：未收集到任何有效坐标。")
            return None

        try:
            avg_x = np.mean([coord[0] for coord in collected_coords])
            avg_y = np.mean([coord[1] for coord in collected_coords])
            # Convert numpy floats to standard Python floats for JSON serialization
            calibrated_coord = [float(avg_x), float(avg_y)]
            self.logger.info(f"平均计算得到的 'shirt' 坐标 (相机坐标系): [{calibrated_coord[0]:.3f}, {calibrated_coord[1]:.3f}]")
            return calibrated_coord
        except Exception as e:
            self.logger.error(f"计算平均坐标时出错: {e}")
            return None

    def handle_client_connection(self, conn, addr):
        self.logger.info(f"与客户端 {addr} 建立连接。")
        try:
            while True: # 保持连接，直到客户端关闭或出错
                data = conn.recv(1024)
                if not data:
                    self.logger.info(f"客户端 {addr} 断开连接。")
                    break 
                
                command = data.decode().strip()
                self.logger.info(f"从 {addr} 收到命令: {command}")

                response = {}
                if command == "REQUEST_CALIBRATION":
                    raw_calibrated_coord = self._collect_calibration_data()
                    if raw_calibrated_coord:
                        # Defensively ensure all coordinates are Python floats before JSON serialization
                        final_calibrated_coord = [float(c) for c in raw_calibrated_coord]
                        response = {"status": "success", "calibrated_coord": final_calibrated_coord}
                    else:
                        response = {"status": "error", "message": "Calibration failed or PositionDetector unavailable."}
                elif command == "REQUEST_VISUAL_INFO":
                    if self.pos_detector:
                        visual_info = self.pos_detector.get_current_visual_info()
                        if visual_info and visual_info.get('bbox_available') and visual_info.get('pixel_cx') is not None and visual_info.get('image_width') is not None:
                            # Attempt to get depth_x as well
                            current_object_xy = self.pos_detector.get_current_object_xy()
                            depth_x_value = None
                            if current_object_xy:
                                depth_x_value = float(current_object_xy[0])
                            
                            # Ensure numeric types are standard Python types for JSON
                            response = {
                                "status": "success",
                                "pixel_cx": float(visual_info['pixel_cx']),
                                "image_width": int(visual_info['image_width']),
                                "bbox_available": True,
                                "depth_x": depth_x_value # Will be None if current_object_xy was None
                            }
                        elif visual_info and not visual_info.get('bbox_available'):
                             response = {"status": "success", "bbox_available": False, "message": "No bounding box currently available."}
                        else:
                            response = {"status": "error", "message": "Failed to get valid visual info or bbox not available."}
                    else:
                        response = {"status": "error", "message": "PositionDetector not available in image server."}
                elif command == "PING": # 可选：添加一个PING命令来测试连接
                    response = {"status": "success", "message": "PONG"}
                else:
                    response = {"status": "error", "message": "Unknown command"}
                
                conn.sendall(json.dumps(response).encode())
                self.logger.info(f"已向 {addr} 发送响应: {response}")

        except socket.error as e:
            self.logger.error(f"与客户端 {addr} 的套接字通信错误: {e}")
        except Exception as e:
            self.logger.error(f"处理客户端 {addr} 请求时发生意外错误: {e}")
        finally:
            conn.close()
            self.logger.info(f"与客户端 {addr} 的连接已关闭。")

    def run(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1) # 通常只期望一个机器人控制器客户端
            self.running = True
            self.logger.info(f"图像处理服务器正在监听 {self.host}:{self.port}")

            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    # 为每个客户端创建一个新线程来处理，避免阻塞主服务器循环
                    # 对于这个特定场景，机器人控制器是唯一客户端，线程可能不是严格必需的，但这是良好实践
                    client_thread = threading.Thread(target=self.handle_client_connection, args=(conn, addr))
                    client_thread.daemon = True # 确保主程序退出时线程也会退出
                    client_thread.start()
                except socket.error as e:
                    if self.running: # 只有在服务器应该运行时才记录错误
                        self.logger.error(f"接受连接时出错: {e}")
                    break # 发生套接字错误时（例如端口已占用后关闭），退出循环
                except Exception as e:
                    if self.running:
                        self.logger.error(f"服务器主循环发生意外错误: {e}")
                    break


        except Exception as e:
            self.logger.critical(f"服务器未能启动或意外终止: {e}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        self.logger.info("开始关闭图像处理服务器...")
        if self.server_socket:
            try:
                self.server_socket.close() # 关闭服务器套接字
                self.logger.info("服务器套接字已关闭。")
            except Exception as e:
                self.logger.error(f"关闭服务器套接字时出错: {e}")
        
        if self.pos_detector:
            try:
                self.pos_detector.shutdown()
                self.logger.info("PositionDetector 已关闭。")
            except Exception as e:
                self.logger.error(f"关闭 PositionDetector 时出错: {e}")
        self.logger.info("图像处理服务器已关闭。")

if __name__ == "__main__":
    server = ImageProcessorServer()
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("收到 KeyboardInterrupt，服务器正在关闭...")
    finally:
        server.shutdown() 