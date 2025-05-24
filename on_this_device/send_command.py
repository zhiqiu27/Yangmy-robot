import socket
import json

# ImageServer (在同一台PC2上) 的地址和端口
IMAGE_SERVER_HOST = 'localhost'
IMAGE_SERVER_PORT = 50001  # 这是 image_server.py 监听的端口

def send_command_to_image_server(command_to_send):
    """
    连接到本地的 ImageServer 并发送一个命令。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"连接到本地 ImageServer ({IMAGE_SERVER_HOST}:{IMAGE_SERVER_PORT})...")
            s.connect((IMAGE_SERVER_HOST, IMAGE_SERVER_PORT))
            print(f"成功连接到 ImageServer。")
            
            print(f"发送命令: '{command_to_send}' 给 ImageServer...")
            s.sendall(command_to_send.encode('utf-8'))
            
            # 等待 ImageServer 的响应
            # 设置一个超时，以防 ImageServer 没有立即响应
            s.settimeout(5.0) # 5 秒超时
            response_data = s.recv(2048) # 增加缓冲区大小以防响应较长
            
            if response_data:
                try:
                    response_json = json.loads(response_data.decode('utf-8'))
                    print(f"ImageServer 响应: {response_json}")
                except json.JSONDecodeError:
                    print(f"ImageServer 原始响应 (非JSON): {response_data.decode('utf-8', errors='ignore')}")
                except Exception as e_decode:
                    print(f"解码 ImageServer 响应时出错: {e_decode}")

            else:
                print("ImageServer 未返回响应或响应为空。")
            
    except ConnectionRefusedError:
        print(f"错误: 无法连接到 ImageServer @ {IMAGE_SERVER_HOST}:{IMAGE_SERVER_PORT}。")
        print("请确保 ImageServer (image_server.py) 正在机器人PC2上运行。")
    except socket.timeout:
        print("错误: 等待 ImageServer 响应超时。")
    except Exception as e:
        print(f"与 ImageServer 通信时发生错误: {e}")

if __name__ == "__main__":
    print("----------------------------------------------------")
    print("   PC2 终端命令发送器 (用于间接触发 PC1 Viewer)   ")
    print("----------------------------------------------------")
    print("此脚本会向运行在机器人PC2本地的 ImageServer 发送命令。")
    print("ImageServer 随后会将 'NEXT_TARGET' 命令发送给 PC1 上的 Viewer。")
    print("\n输入 'next' 来触发 PC1 Viewer 切换到下一个目标。")
    print("输入 'ping' 来测试与 ImageServer 的连接。")
    print("输入 'quit' 来退出此脚本。")
    print("----------------------------------------------------\n")
    
    while True:
        user_input = input("命令 (next, ping, quit) > ").strip().lower()
        
        if user_input == 'quit':
            print("正在退出终端命令发送器...")
            break
        elif user_input == 'next':
            # 这个命令发送给 image_server.py，
            # image_server.py 再把 "NEXT_TARGET" 命令发送给 PC1 上的 Viewer
            send_command_to_image_server("TRIGGER_VIEWER_NEXT_TARGET")
        elif user_input == 'ping':
            # 发送 PING 给 image_server.py 以测试连接
            send_command_to_image_server("PING")
        else:
            print(f"未知命令: '{user_input}'. 请输入 'next', 'ping', 或 'quit'.")
        print("-" * 20) # 分隔符