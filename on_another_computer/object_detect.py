import socket
import cv2
import numpy as np
import struct
import torch
import supervision as sv
from florence2_local import Florence2
from autodistill.detection import CaptionOntology

class ObjectDetectServer:
    def __init__(self, host='0.0.0.0', port=12346):
        # 初始化网络服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"视频检测服务器启动于 {host}:{port}")
        
        # 初始化目标检测模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        ontology = CaptionOntology({"tennis ball": "a green bottle"})
        local_model_path = "D:/models"
        print("加载 Florence-2 模型...")
        self.model = Florence2(ontology=ontology, model_id=local_model_path)
        print("模型加载成功")

    def start(self):
        """开始接收连接"""
        conn, addr = self.server_socket.accept()
        print(f"视频连接来自: {addr}")
        return conn

    def receive_frame(self, conn):
        """接收一帧视频数据"""
        try:
            # 接收帧大小
            data = conn.recv(4)
            if not data:
                return None
            frame_size = struct.unpack('>I', data)[0]

            # 接收帧数据
            frame_data = b''
            while len(frame_data) < frame_size:
                packet = conn.recv(min(4096, frame_size - len(frame_data)))
                if not packet:
                    return None
                frame_data += packet

            # 解码图像
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (640, 480))
            return frame
        except Exception as e:
            print(f"接收帧错误: {e}")
            return None

    def detect_objects(self, frame):
        """对接收到的帧进行目标检测"""
        try:
            # 保存临时图像用于检测
            temp_image_path = "temp_frame.jpg"
            cv2.imwrite(temp_image_path, frame)
            detections = self.model.predict(temp_image_path, confidence=0.5)
            torch.cuda.empty_cache()  # 清理显存
            return detections
        except Exception as e:
            print(f"检测错误: {e}")
            return None

    def draw_detection(self, frame, detections):
        """在图像上绘制检测结果"""
        if detections is not None and len(detections) > 0:
            for detection in detections.xyxy:
                x_min, y_min, x_max, y_max = map(int, detection)
                # 绘制边界框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # 绘制目标中心点
                target_center_x = (x_min + x_max) // 2
                target_center_y = (y_min + y_max) // 2
                cv2.circle(frame, (target_center_x, target_center_y), 5, (0, 0, 255), -1)
                # 绘制图像中心点
                img_center_x = frame.shape[1] // 2
                img_center_y = frame.shape[0] // 2
                cv2.circle(frame, (img_center_x, img_center_y), 5, (255, 0, 0), -1)

    def display_frame(self, frame):
        """显示处理后的帧"""
        cv2.imshow('Object Detection Stream', frame)

    def should_exit(self):
        """检查是否应该退出"""
        return cv2.waitKey(1) & 0xFF == ord('q')

    def run(self):
        """运行主循环"""
        conn = self.start()
        try:
            while True:
                # 接收帧
                frame = self.receive_frame(conn)
                if frame is None:
                    break

                # 目标检测
                detections = self.detect_objects(frame)
                
                # 绘制检测结果
                self.draw_detection(frame, detections)
                
                # 显示结果
                self.display_frame(frame)
                
                # 检查退出条件
                if self.should_exit():
                    break
        finally:
            conn.close()
            self.server_socket.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    server = ObjectDetectServer()
    server.run() 