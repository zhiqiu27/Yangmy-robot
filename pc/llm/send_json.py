import socket
import struct
import json

# Configuration for connecting to the viewer script
JSON_VIEWER_HOST = '127.0.0.1'  # IP of the machine running other_pc_image_viewer.py
JSON_VIEWER_PORT = 65430         # Must match the port in other_pc_image_viewer.py

def send_json_to_viewer(json_string):
    """Sends a JSON string to the viewer application over TCP."""
    try:
        # Ensure what we're sending is a valid JSON string, even if it comes as a Python dict
        if isinstance(json_string, dict):
            data_to_send = json.dumps(json_string)
        elif isinstance(json_string, str):
            # Try to parse and re-dump to ensure it's valid JSON and consistently formatted
            try:
                parsed_json = json.loads(json_string)
                data_to_send = json.dumps(parsed_json)
            except json.JSONDecodeError:
                print(f"[Sender] Invalid JSON string provided: {json_string}. Not sending.")
                return False # Indicate failure to send
        else:
            print(f"[Sender] Data to send is not a string or dict: {type(json_string)}. Not sending.")
            return False

        data_bytes = data_to_send.encode('utf-8')
        data_len = len(data_bytes)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"[Sender] Connecting to viewer at {JSON_VIEWER_HOST}:{JSON_VIEWER_PORT}...")
            s.connect((JSON_VIEWER_HOST, JSON_VIEWER_PORT))
            print("[Sender] Connected to viewer.")
            
            # Send the length of the JSON data first (4 bytes, big-endian unsigned integer)
            s.sendall(struct.pack('>I', data_len))
            # Send the JSON data itself
            s.sendall(data_bytes)
            print(f"[Sender] Sent {data_len} bytes of JSON data: {data_to_send}")
        return True # Indicate success
    except socket.error as e:
        print(f"[Sender] Socket error when trying to send to viewer: {e}")
        return False
    except Exception as e:
        print(f"[Sender] An unexpected error occurred: {e}")
        return False

def send_target_agent_json(target_entities):
    """发送TargetAgent格式的JSON数据"""
    target_json = {
        "target_entities": target_entities
    }
    print(f"[TargetAgent] Sending target entities: {target_entities}")
    return send_json_to_viewer(target_json)

def send_direction_agent_json(direction):
    """发送DirectionAgent格式的JSON数据"""
    direction_json = {
        "direction": direction
    }
    print(f"[DirectionAgent] Sending direction: {direction}")
    return send_json_to_viewer(direction_json)

if __name__ == "__main__":
    print("=== JSON发送脚本 ===")
    print("选择要发送的JSON类型:")
    print("1. TargetAgent JSON (目标实体)")
    print("2. DirectionAgent JSON (方向命令)")
    print("3. 自定义JSON")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # TargetAgent JSON示例
        print("\n=== TargetAgent JSON ===")
        print("示例格式: [\"cup\", \"bottle\"]")
        targets_input = input("请输入目标实体列表 (用逗号分隔): ").strip()
        
        if targets_input:
            # 解析输入的目标列表
            targets = [target.strip().strip('"\'') for target in targets_input.split(',')]
            success = send_target_agent_json(targets)
        else:
            # 使用默认示例
            success = send_target_agent_json(["cup", "bottle"])
            
    elif choice == "2":
        # DirectionAgent JSON示例
        print("\n=== DirectionAgent JSON ===")
        print("有效方向: forward, backward, left, right, front-left, front-right, back-left, back-right")
        direction_input = input("请输入方向: ").strip()
        
        valid_directions = [
            "forward", "backward", "left", "right",
            "front-left", "front-right", "back-left", "back-right"
        ]
        
        if direction_input in valid_directions:
            success = send_direction_agent_json(direction_input)
        else:
            print(f"无效方向: {direction_input}")
            print(f"有效方向: {valid_directions}")
            success = False
            
    elif choice == "3":
        # 自定义JSON
        print("\n=== 自定义JSON ===")
        json_input = input("请输入JSON字符串: ").strip()
        
        try:
            # 验证JSON格式
            json.loads(json_input)
            success = send_json_to_viewer(json_input)
        except json.JSONDecodeError:
            print("无效的JSON格式")
            success = False
    else:
        print("无效选择")
        success = False
    
    if success:
        print("✓ JSON发送成功")
    else:
        print("✗ JSON发送失败") 