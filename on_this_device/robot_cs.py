import cv2
import numpy as np
import math

# 初始状态变量
xs, ys, ws, hs = 0, 0, 0, 0
xo, yo = 0, 0
selectObject = False
trackObject = 0
track_window = None
roi_hist = None

# === VLM 注入模拟 ===
vlm_bbox = None
vlm_updated = False
last_good_bbox = None  # 上一个可信 bbox

# === 跳变检测函数 ===
def bbox_center_area(bbox):
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2, w * h)

def is_abrupt_jump(new_bbox, old_bbox, pos_thresh=50, area_thresh=0.5):
    if old_bbox is None:
        return False
    cx1, cy1, area1 = bbox_center_area(old_bbox)
    cx2, cy2, area2 = bbox_center_area(new_bbox)
    dist = math.hypot(cx2 - cx1, cy2 - cy1)
    area_ratio = abs(area2 - area1) / (area1 + 1e-5)
    return dist > pos_thresh or area_ratio > area_thresh

# === 鼠标框选函数（调试用） ===
def onMouse(event, x, y, flags, param):
    global xs, ys, ws, hs, xo, yo, selectObject, trackObject
    if selectObject:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1

# === 初始化窗口 ===
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.namedWindow('imshow')
cv2.setMouseCallback('imshow', onMouse)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))

    # === 模拟远程 VLM 输出（测试用，实际应通过 socket 更新） ===
    # Example: simulate new bbox every 100 frames
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 100 == 0:
        fake_x, fake_y, fake_w, fake_h = 200, 150, 100, 100
        vlm_candidate_bbox = (fake_x, fake_y, fake_w, fake_h)
        if not is_abrupt_jump(vlm_candidate_bbox, last_good_bbox):
            vlm_bbox = vlm_candidate_bbox
            vlm_updated = True
            last_good_bbox = vlm_candidate_bbox
        else:
            print("[VLM Warning] bbox jump detected — update skipped.")

    # === 若有新的VLM bbox，重新初始化跟踪器 ===
    if vlm_updated:
        x, y, w, h = vlm_bbox
        track_window = (x, y, w, h)
        hsv_roi = hsv[y:y + h, x:x + w]
        maskroi = mask[y:y + h, x:x + w]
        roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        trackObject = 1
        vlm_updated = False

    # === 鼠标框选初始化 ===
    if trackObject == -1:
        track_window = (xs, ys, ws, hs)
        hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
        maskroi = mask[ys:ys + hs, xs:xs + ws]
        roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        trackObject = 1

    # === CamShift 主跟踪 ===
    if trackObject == 1 and track_window is not None:
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        dst &= mask
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.intp(pts)
        frame = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # === 可视化鼠标选择框 ===
    if selectObject and ws > 0 and hs > 0:
        cv2.imshow('imshow1', frame[ys:ys + hs, xs:xs + ws])
        cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])

    cv2.imshow('imshow', frame)
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
