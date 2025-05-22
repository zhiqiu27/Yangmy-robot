import socket
import sys
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from sport_module import SportModule
from video_module import VideoModule

class Go2MainServer:
    def __init__(self, host='0.0.0.0', port=12345, video_ip='192.168.3.70', video_port=12346):
        # 初始化运动和视频模块
        self.sport_module = SportModule()
        self.video_module = VideoModule(video_ip, video_port)

        # 初始化TCP服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"主服务器启动于 {host}:{port}")

        self.running = True

    def handle_client(self, conn, addr):
        """处理客户端指令"""
        print(f"连接来自: {addr}")
        while self.running:
            try:
                # 接收指令（假设最大1024字节）
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    break

                print(f"收到指令: {data}")
                # 解析指令
                parts = data.split('|')
                if len(parts) < 2:
                    print("无效指令格式")
                    continue

                cmd_type, cmd_data = parts[0], parts[1]
                if cmd_type == 'M':  # 运动指令
                    try:
                        vx, vy, vyaw = map(float, cmd_data.split(','))
                        self.sport_module.move(vx, vy, vyaw)
                    except ValueError:
                        print("运动指令参数错误")
                elif cmd_type == 'V':  # 视频指令
                    if cmd_data == 'start':
                        self.video_module.start()
                    elif cmd_data == 'stop':
                        self.video_module.stop()
                    else:
                        print("未知视频指令")
                else:
                    print("未知指令类型")

            except Exception as e:
                print(f"处理指令错误: {e}")
                break

        conn.close()
        print(f"连接 {addr} 已关闭")

    def run(self):
        """启动服务器"""
        conn, addr = self.server_socket.accept()
        client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
        client_thread.start()

        try:
            client_thread.join()
        except KeyboardInterrupt:
            self.running = False
            self.video_module.stop()
            self.server_socket.close()
            print("主服务器已关闭")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: 请确保机器人周围没有障碍物。")
    input("按Enter键继续...")

    ChannelFactoryInitialize(0, sys.argv[1])
    server = Go2MainServer(video_ip='192.168.3.70')  # 替换为PC的IP
    server.run()