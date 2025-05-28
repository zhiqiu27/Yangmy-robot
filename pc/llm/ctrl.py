import time
import enum
import socket
import random

# -------------------------------
# 模拟目标列表（从LLM生成）
target_list = [
    {"name": "shirt", "coord": (1.5, 0.8)},
    {"name": "person", "coord": (0.2, 2.0)}  # 假设坐标来自VLM识别
]

# -------------------------------
# 状态定义
class State(enum.Enum):
    IDLE = 0
    NAVIGATE_TO_OBJECT = 1
    LOWER_HEAD = 2
    GRASP_OBJECT = 3
    SEND_NEXT_COMMAND = 4
    NAVIGATE_TO_PERSON = 5
    RELEASE_OBJECT = 6
    DONE = 7

# -------------------------------
# 模拟控制模块
class RobotController:
    def __init__(self):
        self.state = State.IDLE
        self.target_index = 0
        self.current_coord = (0.0, 0.0)

    def navigate_to(self, target_coord):
        print(f"[NAVIGATE] Walking to {target_coord} ...")
        time.sleep(1)
        self.current_coord = target_coord
        print(f"[NAVIGATE] Arrived at {target_coord}")

    def lower_head(self):
        print("[ACTION] Lowering head (pitch down)...")
        time.sleep(0.5)

    def grasp(self):
        print("[ACTION] Grasping object...")
        time.sleep(0.5)
        return True  # 假设成功

    def release(self):
        print("[ACTION] Releasing object...")
        time.sleep(0.5)

    def send_next_to_vlm(self):
        print("[SOCKET] Sending 'next' to VLM (simulated)")
        time.sleep(0.3)

    def get_next_target_from_vlm(self):
        # 模拟从VLM获取下一个坐标
        print("[VLM] Getting next target coordinates...")
        time.sleep(0.3)
        return target_list[1]["coord"]

    def run(self):
        self.state = State.NAVIGATE_TO_OBJECT
        while self.state != State.DONE:
            if self.state == State.NAVIGATE_TO_OBJECT:
                coord = target_list[0]["coord"]
                self.navigate_to(coord)
                self.state = State.LOWER_HEAD

            elif self.state == State.LOWER_HEAD:
                self.lower_head()
                self.state = State.GRASP_OBJECT

            elif self.state == State.GRASP_OBJECT:
                success = self.grasp()
                if success:
                    print("[STATUS] Grasp success!")
                    self.state = State.SEND_NEXT_COMMAND
                else:
                    print("[STATUS] Grasp failed. Retrying...")
                    self.state = State.LOWER_HEAD

            elif self.state == State.SEND_NEXT_COMMAND:
                self.send_next_to_vlm()
                next_coord = self.get_next_target_from_vlm()
                target_list[1]["coord"] = next_coord  # 更新目标
                self.state = State.NAVIGATE_TO_PERSON

            elif self.state == State.NAVIGATE_TO_PERSON:
                coord = target_list[1]["coord"]
                self.navigate_to(coord)
                self.state = State.RELEASE_OBJECT

            elif self.state == State.RELEASE_OBJECT:
                self.release()
                print("[MISSION] Task complete. Back to idle.")
                self.state = State.DONE

            time.sleep(0.2)

# -------------------------------
# 执行主控逻辑
if __name__ == "__main__":
    controller = RobotController()
    controller.run()
