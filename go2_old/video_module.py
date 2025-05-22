import socket
import threading
import cv2
import numpy as np
import struct
from unitree_sdk2py.go2.video.video_client import VideoClient

class VideoModule:
    def __init__(self, video_ip, video_port):
        self.video_client = VideoClient()
        self.video_client.SetTimeout(3.0)
        self.video_client.Init()

        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_ip = video_ip
        self.video_port = video_port
        self.running = False
        self.thread = None
        print("视频模块已初始化")

    def video_stream(self):
        """视频传输线程"""
        try:
            self.video_socket.connect((self.video_ip, self.video_port))
            print(f"视频已连接到 {self.video_ip}:{self.video_port}")

            while self.running:
                code, data = self.video_client.GetImageSample()
                if code != 0:
                    print("获取图像失败，错误码:", code)
                    break

                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                _, encoded_frame = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                frame_data = encoded_frame.tobytes()

                frame_size = len(frame_data)
                self.video_socket.send(struct.pack('>I', frame_size))
                self.video_socket.sendall(frame_data)

        except Exception as e:
            print(f"视频传输错误: {e}")
        finally:
            self.video_socket.close()
            print("视频传输已关闭")

    def start(self):
        """启动视频传输"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.video_stream)
            self.thread.start()
            print("视频传输已启动")
        else:
            print("视频传输已在运行")

    def stop(self):
        """停止视频传输"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            print("视频传输已停止")