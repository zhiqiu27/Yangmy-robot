#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zed_bbox_to_3d.py
实时读取 ZED-Mini，配合 2D 目标检测，输出目标在
RIGHT_HANDED_Z_UP_X_FWD 相机坐标系下的 (前 X, 左 Y, 上 Z) 位置。
按  q  键退出。
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl


# ----------------------------------------------------------------------
# 1) 目标检测（示例：固定一个 200×200 框；请替换为 Florence-2/YOLO）
# ----------------------------------------------------------------------
def run_detection(bgr_img):
    h, w, _ = bgr_img.shape
    cx, cy, size = w // 2, h // 2, 200
    return [(cx - size // 2, cy - size // 2,
             cx + size // 2, cy + size // 2)]


# ----------------------------------------------------------------------
# 2) 计算检测框内点云中位数坐标 (x, y, z)
# ----------------------------------------------------------------------
def bbox_center_xyz(bbox, xyz):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(xyz.shape[1] - 1, x2), min(xyz.shape[0] - 1, y2)
    patch = xyz[y1:y2 + 1, x1:x2 + 1, :3]           # (H,W,3)
    mask  = np.isfinite(patch[..., 0])               # 只看 X 是否有效
    if not np.any(mask):
        return None
    return tuple(np.median(patch[mask], axis=0))     # (x, y, z)


# ----------------------------------------------------------------------
# 3) 主程序
# ----------------------------------------------------------------------
def main():
    # ---------- 打开相机 ----------
    zed = sl.Camera()
    ip = sl.InitParameters()
    ip.depth_mode             = sl.DEPTH_MODE.NEURAL
    ip.coordinate_units       = sl.UNIT.METER
    ip.coordinate_system      = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    ip.depth_maximum_distance = 10.0
    ip.camera_fps             = 10          # (注：仅 SVO/USB3.0 一些模式支持改 FPS)

    # 必须调用 open()
    if zed.open(ip) != sl.ERROR_CODE.SUCCESS:
        print("Camera open failed"); return

    # ---------- 运行时参数 ----------
    rt = sl.RuntimeParameters()
    rt.confidence_threshold         = 95
    rt.texture_confidence_threshold = 120
    rt.enable_fill_mode             = False

    img_mat, xyz_mat = sl.Mat(), sl.Mat()
    print("Running...  press [q] to quit")
    while True:
        if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue

        # 读取左目图像和 XYZ 点云
        zed.retrieve_image(img_mat,  sl.VIEW.LEFT)
        zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)

        rgb = cv2.cvtColor(img_mat.get_data(), cv2.COLOR_BGRA2BGR)
        xyz = xyz_mat.get_data()                         # H×W×4

        # 目标检测 + 坐标估计
        for box in run_detection(rgb):
            coord = bbox_center_xyz(box, xyz)
            if coord is None:
                continue
            x, y, z = coord   # (前, 左, 上) 方向
            label = f"({x:+.2f},{y:+.2f},{z:+.2f}) m"

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("ZED Left", rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    zed.close()
    print("Finished.")


if __name__ == "__main__":
    main()
