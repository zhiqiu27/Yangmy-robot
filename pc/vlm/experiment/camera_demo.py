import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

class CameraCutieDemo:
    def __init__(self):
        print("初始化Cutie模型...")
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = 480
        
        # 摄像头设置
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 鼠标框选相关变量
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.bbox = None
        self.mask_created = False
        self.frame_count = 0
        
        # 显示窗口
        cv2.namedWindow('Camera Cutie Demo', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Camera Cutie Demo', self.mouse_callback)
        
        print("初始化完成!")
        print("使用说明:")
        print("1. 按住鼠标左键拖拽框选目标对象")
        print("2. 松开鼠标后开始自动分割追踪")
        print("3. 按 'r' 重新框选")
        print("4. 按 'q' 退出")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            if self.start_point and self.end_point:
                # 确保坐标顺序正确
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                if x2 - x1 > 10 and y2 - y1 > 10:  # 确保框选区域足够大
                    self.bbox = (x1, y1, x2, y2)
                    print(f"框选区域: ({x1}, {y1}) -> ({x2}, {y2})")

    def create_mask_from_bbox(self, frame_shape, bbox):
        """从边界框创建mask"""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1  # 对象ID为1
        return mask

    @torch.inference_mode()
    @torch.amp.autocast(device_type='cuda')
    def process_frame(self, frame):
        """处理单帧图像"""
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor
        image_tensor = to_tensor(frame_rgb).cuda().float()
        
        if not self.mask_created and self.bbox is not None:
            # 第一帧：创建mask并初始化
            mask_np = self.create_mask_from_bbox(frame.shape, self.bbox)
            mask_tensor = torch.from_numpy(mask_np).cuda()
            
            # 获取对象列表
            objects = [1]  # 只有一个对象，ID为1
            
            # 处理第一帧
            output_prob = self.processor.step(image_tensor, mask_tensor, objects=objects)
            self.mask_created = True
            print("开始追踪对象...")
            
        elif self.mask_created:
            # 后续帧：自动传播
            output_prob = self.processor.step(image_tensor)
        else:
            return frame
        
        # 转换输出为mask
        mask = self.processor.output_prob_to_mask(output_prob)
        mask_np = mask.cpu().numpy()
        
        return self.visualize_result(frame, mask_np)

    def visualize_result(self, frame, mask):
        """可视化分割结果"""
        # 创建彩色mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask == 1] = [0, 255, 0]  # 绿色表示对象
        
        # 混合原图和mask
        alpha = 0.3
        result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        
        # 绘制对象轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result

    def run(self):
        """主运行循环"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 处理帧
                if torch.cuda.is_available():
                    processed_frame = self.process_frame(frame)
                else:
                    processed_frame = frame
                    if not self.mask_created:
                        cv2.putText(processed_frame, "CUDA not available - CPU mode", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 绘制当前框选框
                if self.drawing and self.start_point and self.end_point:
                    cv2.rectangle(processed_frame, self.start_point, self.end_point, (255, 0, 0), 2)
                elif self.bbox and not self.mask_created:
                    x1, y1, x2, y2 = self.bbox
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(processed_frame, "Press any key to start tracking", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 显示状态信息
                if not self.mask_created and not self.bbox:
                    cv2.putText(processed_frame, "Draw bounding box around target object", 
                              (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                elif self.mask_created:
                    cv2.putText(processed_frame, f"Tracking... Frame: {self.frame_count}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.frame_count += 1
                
                cv2.imshow('Camera Cutie Demo', processed_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 重置
                    self.bbox = None
                    self.mask_created = False
                    self.frame_count = 0
                    self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
                    self.processor.max_internal_size = 480
                    print("重置完成，请重新框选目标")
                elif key != 255 and self.bbox and not self.mask_created:
                    # 任意键开始追踪
                    pass
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("资源清理完成")

def main():
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("警告: CUDA不可用，将使用CPU模式（速度较慢）")
    
    # 创建并运行演示
    demo = CameraCutieDemo()
    demo.run()

if __name__ == "__main__":
    main() 