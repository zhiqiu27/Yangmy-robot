import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

if __name__ == "__main__":
    ChannelFactoryInitialize(0)
    rsc = RobotStateClient()
    rsc.SetTimeout(3.0)
    rsc.Init()


    print("##################ServiceSwitch###################")
    code = rsc.ServiceSwitch("sport_mode", True)
    if code != 0:
        print("service start sport_mode error. code:", code)
    else:
        print("service start sport_mode success. code:", code)

