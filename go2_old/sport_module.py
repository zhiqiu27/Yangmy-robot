from unitree_sdk2py.go2.sport.sport_client import SportClient

class SportModule:
    def __init__(self):
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("运动模块已初始化")

    def move(self, vx, vy, vyaw):
        """执行运动指令"""
        try:
            self.sport_client.Move(vx, vy, vyaw)
            print(f"执行运动: vx={vx}, vy={vy}, vyaw={vyaw}")
        except Exception as e:
            print(f"运动执行错误: {e}")