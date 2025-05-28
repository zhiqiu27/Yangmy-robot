# 更新 main.py

import os
from openai import OpenAI
from target_agent import TargetAgent
from direction_agent import DirectionAgent
from handoff_agent import HandoffAgent
from voice_input import listen_and_transcribe
import socket  # Added for TCP client
import struct  # Added for packing data length
import json    # Added for potentially re-formatting/validating JSON before sending

# Configuration for connecting to the viewer script
JSON_VIEWER_HOST = '127.0.0.1'  # IP of the machine running other_pc_image_viewer.py
JSON_VIEWER_PORT = 65430         # Must match the port in other_pc_image_viewer.py

# 初始化客户端
client = OpenAI(
    api_key='sk-899bf830799742489da5ecaf5f3aecb3',
    base_url="https://api.deepseek.com"
)

# 初始化 Agents
target_agent = TargetAgent(client)
direction_agent = DirectionAgent(client)
agents = [target_agent, direction_agent]

# 初始化 Handoff Agent
handoff_agent = HandoffAgent(client, agents)

# Pick up the trash and throw it in the bin
# The trash can is behind you to the right

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

# 主控逻辑
def main():
    while True:
        try:
            user_input = input("User command (or 'q'): ").strip()
            if user_input.lower() == "q":
                break

            # input("Press Enter to speak...")
            # user_input = listen_and_transcribe()
            # if user_input.strip().lower() == "exit":
            #     break

            chosen_agent_name = handoff_agent.route(user_input)

            if chosen_agent_name == target_agent.name:
                result = target_agent.extract_targets(user_input) # This should return a JSON string
                print("[TargetAgent Result]:", result)
                # Attempt to send this result to the viewer
                if result: # Ensure there is a result to send
                    print("[Main] Attempting to send TargetAgent result to viewer...")
                    if send_json_to_viewer(result):
                        print("[Main] Successfully sent data to viewer.")
                    else:
                        print("[Main] Failed to send data to viewer. Viewer might not be running or listening.")
            elif chosen_agent_name == direction_agent.name:
                result = direction_agent.extract_direction(user_input)
                print("[DirectionAgent Result]:", result)
                # Send DirectionAgent result to viewer as well
                if result: # Ensure there is a result to send
                    print("[Main] Attempting to send DirectionAgent result to viewer...")
                    if send_json_to_viewer(result):
                        print("[Main] Successfully sent DirectionAgent data to viewer.")
                    else:
                        print("[Main] Failed to send DirectionAgent data to viewer. Viewer might not be running or listening.")
            else:
                print(f"[System]: Unrecognized agent name: {chosen_agent_name}")
        except Exception as e:
            print("[Error]:", str(e))

if __name__ == "__main__":
    main()
