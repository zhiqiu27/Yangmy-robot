
"""
使用电脑摄像头测试 CutieTracker 的简单脚本

功能：
- 从摄像头捕获实时视频
- 允许用户手动绘制初始检测框
- 每5秒可重新绘制检测框更新掩码
- 支持按 'r' 键强制更新掩码
- 显示追踪结果（边界框、中心点、对象ID）

依赖：
- opencv-python
- torch
- cutie_tracker.py（需放在同一目录或正确配置路径）
"""

import cv2
import numpy as np
import torch
from cutie_tracker import CutieTracker
import time

# 全局变量
drawing = False  # 是否正在绘制检测框
bbox = []  # 存储当前绘制的检测框 [x1, y1, x2, y2]
bboxes = []  # 存储所有检测框
labels = []  # 存储所有标签
ref_point = []  # 临时存储鼠标点击的起点
update_interval = 5.0  # 掩码更新间隔（秒）
last_update_time = time.time()  # 上次更新时间

def draw_bounding_box(event, x, y, flags, param):
    """鼠标回调函数，用于手动绘制检测框"""
    global drawing, bbox, ref_point, bboxes, labels, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ref_point = [x, y]
        bbox = [x, y, x, y]  # 初始化检测框

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox[2], bbox[3] = x, y  # 更新检测框右下角坐标

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox[2], bbox[3] = x, y
        # 确保 x1 < x2, y1 < y2
        x1, y1, x2, y2 = min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])
        bboxes.append([x1, y1, x2, y2])
        labels.append(len(bboxes))  # 自动分配标签（从1开始）
        print(f"添加检测框: {bboxes[-1]}, 标签: {labels[-1]}")

def draw_tracked_objects(frame, tracker_results):
    """在帧上绘制追踪结果"""
    if not tracker_results or not tracker_results.get('bboxes'):
        return frame

    bboxes = tracker_results.get('bboxes', [])
    object_ids = tracker_results.get('object_ids', [])
    centers = tracker_results.get('centers', [])
    areas = tracker_results.get('areas', [])

    for i, bbox in enumerate(bboxes):
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        obj_id_text = f"TrackID: {object_ids[i]}" if i < len(object_ids) else "TrackID: N/A"
        
        # 绘制边界框（绿色）
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签和面积
        label_text = f"{obj_id_text}, Area: {int(areas[i]) if i < len(areas) else 0}"
        text_y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, label_text, (x1, text_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 绘制中心点（红色）
        if i < len(centers):
            center_x, center_y = map(int, centers[i])
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    
    return frame

def main():
    global drawing, bbox, bboxes, labels, last_update_time

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    # 初始化 CutieTracker
    tracker = CutieTracker(max_internal_size=480, device='cuda' if torch.cuda.is_available() else 'cpu', update_interval=5.0)
    print("CutieTracker 初始化完成")

    # 设置窗口和鼠标回调
    cv2.namedWindow('CutieTracker Test')
    cv2.setMouseCallback('CutieTracker Test', draw_bounding_box)

    initialized = False  # 是否已初始化追踪
    force_update = False  # 是否强制更新掩码

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            break

        # 复制帧用于显示
        display_frame = frame.copy()

        # 如果正在绘制检测框，实时显示
        if drawing:
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        # 检查是否需要更新掩码
        current_time = time.time()
        need_update = (current_time - last_update_time >= update_interval) or force_update

        # 运行追踪
        try:
            if not initialized:
                if bboxes and labels:
                    print("初始化追踪...")
                    tracker_results = tracker.track(frame, bboxes=bboxes, labels=labels, force_update=True)
                    initialized = True
                    last_update_time = current_time
                    bboxes, labels = [], []  # 清空检测框和标签
                else:
                    cv2.putText(display_frame, "请绘制初始检测框", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if need_update and bboxes and labels:
                    print("更新掩码...")
                    tracker_results = tracker.track(frame, bboxes=bboxes, labels=labels, force_update=force_update)
                    last_update_time = current_time
                    bboxes, labels = [], []  # 清空检测框和标签
                else:
                    tracker_results = tracker.track(frame, force_update=force_update)
                
                # 绘制追踪结果
                display_frame = draw_tracked_objects(display_frame, tracker_results)
        
        except Exception as e:
            print(f"追踪错误: {e}")
            tracker.reset()
            initialized = False
            bboxes, labels = [], []

        # 显示提示信息
        if need_update and not bboxes:
            cv2.putText(display_frame, "请绘制新检测框以更新掩码", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.imshow('CutieTracker Test', display_frame)

        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("退出...")
            break
        elif key == ord('r'):
            print("触发强制更新掩码")
            force_update = True
            cv2.putText(display_frame, "请绘制新检测框", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            force_update = False

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    tracker.reset()
    print("程序结束")

if __name__ == "__main__":
    main()
