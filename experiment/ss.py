#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

class SimpleRobotController:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('robot_controller', anonymous=False)
        
        # 存储最新的位置信息
        self.object_position = None
        self.object_detected = False
        
        # 订阅话题
        rospy.Subscriber('/vision/object_position', Point, self.position_callback)
        rospy.Subscriber('/vision/detection_status', Bool, self.detection_callback)
        
        print("Robot controller started. Listening for object positions...")
    
    def position_callback(self, msg):
        """当收到位置更新时的回调函数"""
        self.object_position = (msg.x, msg.y, msg.z)
        print(f"Received position: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})")
    
    def detection_callback(self, msg):
        """当收到检测状态更新时的回调函数"""
        self.object_detected = msg.data
        if not self.object_detected:
            self.object_position = None
            print("Object lost from view")
    
    def run(self):
        """主循环，处理收到的数据"""
        rate = rospy.Rate(1)  # 1Hz
        
        while not rospy.is_shutdown():
            if self.object_detected and self.object_position:
                x, y, z = self.object_position
                print(f"Processing object at ({x:.2f}, {y:.2f}, {z:.2f})")
                # 这里可以添加你的控制逻辑
            else:
                print("Waiting for object detection...")
            
            rate.sleep()

if __name__ == '__main__':
    controller = SimpleRobotController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass