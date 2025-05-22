#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import random
import time

def simple_position_detector():
    # 初始化ROS节点
    rospy.init_node('position_detector', anonymous=False)
    
    # 创建发布者
    position_pub = rospy.Publisher('/vision/object_position', Point, queue_size=10)
    detection_pub = rospy.Publisher('/vision/detection_status', Bool, queue_size=10)
    
    # 设置发布频率
    rate = rospy.Rate(10)  # 10Hz
    
    print("Position detector started. Publishing to /vision/object_position")
    
    while not rospy.is_shutdown():
        # 生成模拟数据
        detected = random.random() > 0.2  # 80%的概率检测到物体
        
        # 发布检测状态
        detection_pub.publish(Bool(detected))
        
        if detected:
            # 创建位置消息
            pos = Point()
            pos.x = 0.6 + 0.05 * random.random()  # 模拟位置抖动
            pos.y = 0.05 * random.random()
            pos.z = 0.5
            
            # 发布位置
            position_pub.publish(pos)
            print(f"Published position: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        else:
            print("No object detected")
            
        rate.sleep()

if __name__ == '__main__':
    try:
        simple_position_detector()
    except rospy.ROSInterruptException:
        pass