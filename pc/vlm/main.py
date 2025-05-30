import cv2
import time
import queue
import threading
import signal
import sys

from config import *
from hybrid_manager import HybridManager
from network_handler import NetworkHandler

class HybridViewer:
    """混合检测跟踪查看器 - 新的4线程架构"""
    
    def __init__(self):
        # 核心组件
        self.hybrid_manager = HybridManager()
        self.network_handler = NetworkHandler()
        
        # 帧队列（主线程和接收线程之间）
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        
        # 控制标志
        self.running = False
        self.shutdown_event = threading.Event()
        
        # 显示相关
        self.current_frame = None
        self.display_frame = None
        self.last_bbox = None
        
        # 性能统计
        self.display_fps = 0.0
        self.last_display_time = time.time()
        self.display_frame_count = 0
        
        # 状态信息
        self.system_status = "初始化中..."
        self.network_status = {}
        
        print("HybridViewer initialized")
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print(f"\n收到信号 {signum}，开始关闭...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def on_frame_received(self, frame):
        """图像接收回调"""
        if frame is None:
            return
        
        try:
            # 清理积压帧（保持实时性）
            while self.frame_queue.qsize() >= FRAME_QUEUE_MAXSIZE - 1:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 添加新帧
            self.frame_queue.put_nowait(frame)
            
        except queue.Full:
            # 队列满，丢弃最旧的帧
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
    
    def on_json_received(self, json_data):
        """JSON数据接收回调"""
        if self.hybrid_manager:
            self.hybrid_manager.process_json_data(json_data)
    
    def on_command_received(self, command):
        """命令接收回调"""
        if self.hybrid_manager:
            response = self.hybrid_manager.handle_command(command)
            return response
        return "System not ready"
    
    def on_direction_command(self, direction):
        """方向指令回调"""
        if self.network_handler:
            self.network_handler.send_direction_command(direction)
    
    def on_3d_coords_received(self, xyz_coords, timestamp):
        """3D坐标接收回调"""
        print(f"[3D Coords] 收到目标3D坐标: x={xyz_coords[0]:.3f}m, y={xyz_coords[1]:.3f}m, z={xyz_coords[2]:.3f}m")
        # 这里可以添加3D坐标的处理逻辑，比如：
        # - 发送给机器人控制系统
        # - 保存到日志文件
        # - 触发导航算法等
    
    def initialize_system(self):
        """初始化系统"""
        print("=== 初始化混合检测跟踪系统 ===")
        
        # 1. 设置信号处理
        self.setup_signal_handlers()
        
        # 2. 设置混合管理器回调
        self.hybrid_manager.on_direction_command = self.on_direction_command
        
        # 3. 设置网络处理器
        print("设置网络组件...")
        
        # 设置3D坐标回调
        self.network_handler.on_3d_coords_received = self.on_3d_coords_received
        
        # 图像接收器
        self.network_handler.setup_image_receiver(self.on_frame_received)
        
        # bbox发送器
        if not self.network_handler.setup_bbox_sender():
            print("警告: Bbox发送器设置失败")
        
        # JSON接收器
        if not self.network_handler.setup_json_receiver(self.on_json_received):
            print("警告: JSON接收器设置失败")
        
        # 命令监听器
        if not self.network_handler.setup_command_listener(self.on_command_received):
            print("警告: 命令监听器设置失败")
        
        print("系统初始化完成")
        return True
    
    def start_system(self):
        """启动系统"""
        print("=== 启动系统 ===")
        
        # 启动网络处理器（包含所有网络线程）
        self.network_handler.start_all()
        
        # 设置运行标志
        self.running = True
        self.system_status = "系统运行中"
        
        print("系统启动完成")
        print("等待图像数据...")
    
    def update_display_fps(self):
        """更新显示FPS"""
        self.display_frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_display_time >= FPS_CALC_INTERVAL:
            self.display_fps = self.display_frame_count / (current_time - self.last_display_time)
            self.display_frame_count = 0
            self.last_display_time = current_time
    
    def create_status_overlay(self, frame):
        """创建状态叠加层"""
        if frame is None:
            return None
        
        overlay_frame = frame.copy()
        
        # # 基本信息
        # y_offset = 30
        # line_height = 25
        
        # # 系统状态
        # status_text = f"状态: {self.hybrid_manager.system_state}"
        # cv2.putText(overlay_frame, status_text, (10, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # y_offset += line_height
        
        # # 当前目标
        # if self.hybrid_manager.current_target_class:
        #     target_text = f"目标: {self.hybrid_manager.current_target_class}"
        #     cv2.putText(overlay_frame, target_text, (10, y_offset), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        #     y_offset += line_height
        
        # # 处理模式
        # mode_text = f"模式: {'异步' if self.hybrid_manager.use_async_mode else '同步'}"
        # cv2.putText(overlay_frame, mode_text, (10, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # y_offset += line_height
        
        # # FPS信息
        # fps_text = f"显示FPS: {self.display_fps:.1f}"
        # cv2.putText(overlay_frame, fps_text, (10, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        # y_offset += line_height
        
        # # 网络统计
        # if hasattr(self.network_handler, 'image_receiver') and self.network_handler.image_receiver:
        #     net_stats = self.network_handler.image_receiver.get_network_stats()
        #     net_fps = net_stats.get('avg_fps', 0)
        #     bandwidth = net_stats.get('avg_bandwidth', 0)
            
        #     net_text = f"网络: {net_fps:.1f}FPS, {bandwidth:.1f}MB/s"
        #     cv2.putText(overlay_frame, net_text, (10, y_offset), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 2)
        #     y_offset += line_height
        
        # # bbox信息
        # if self.last_bbox:
        #     x1, y1, x2, y2 = map(int, self.last_bbox)
        #     bbox_text = f"Bbox: ({x1},{y1})-({x2},{y2})"
        #     cv2.putText(overlay_frame, bbox_text, (10, y_offset), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
        
        # # 右上角显示帧计数
        # if self.hybrid_manager.system_state == STATE_TRACKING:
        #     frame_count_text = f"Frame: {self.hybrid_manager.tracking_frame_count}"
        #     text_size = cv2.getTextSize(frame_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        #     cv2.putText(overlay_frame, frame_count_text, 
        #                (overlay_frame.shape[1] - text_size[0] - 10, 30), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay_frame
    
    def process_keyboard_input(self):
        """处理键盘输入"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print("用户请求退出")
            return False
        elif key == ord('r'):  # 重置
            print("用户请求重置")
            self.hybrid_manager.switch_to_detection_mode()
        elif key == ord('s'):  # 切换目标
            print("用户请求切换目标")
            self.hybrid_manager.handle_command("switch_target")
        elif key == ord('i'):  # 显示信息
            status = self.hybrid_manager.handle_command("status")
            print(f"系统状态: {status}")
        elif key == ord('h'):  # 帮助
            print("键盘快捷键:")
            print("  q/ESC - 退出")
            print("  r - 重置系统")
            print("  s - 切换目标")
            print("  i - 显示状态信息")
            print("  z - 查询ZED系统状态")
            print("  x - 获取ZED最新检测")
            print("  c - 重置ZED系统")
            print("  h - 显示帮助")
        elif key == ord('z'):  # ZED系统状态
            print("查询ZED系统状态...")
            if self.network_handler:
                self.network_handler.query_zed_system_status()
        elif key == ord('x'):  # ZED最新检测
            print("获取ZED最新检测结果...")
            if self.network_handler:
                self.network_handler.get_zed_latest_detection()
        elif key == ord('c'):  # 重置ZED系统
            print("重置ZED系统...")
            if self.network_handler:
                self.network_handler.reset_zed_system()
        
        return True
    
    def main_loop(self):
        """主循环 - 在主线程中运行"""
        print("=== 进入主循环 ===")
        
        # 创建显示窗口
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        
        # 等待第一帧
        print("等待第一帧图像...")
        while self.running and self.frame_queue.empty():
            time.sleep(0.1)
            if not self.process_keyboard_input():
                return
        
        print("开始处理图像流...")
        
        while self.running:
            try:
                # 1. 获取最新帧
                try:
                    self.current_frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    # 超时，检查是否需要退出
                    if not self.process_keyboard_input():
                        break
                    continue
                
                # 2. 处理帧（检测/跟踪）
                self.display_frame, bbox_result = self.hybrid_manager.process_frame(self.current_frame)
                
                # 3. 更新bbox
                if bbox_result is not None:
                    self.last_bbox = bbox_result
                    # 发送bbox数据
                    self.network_handler.send_bbox_data(bbox_result)
                
                # 4. 创建显示帧
                if self.display_frame is not None:
                    final_frame = self.create_status_overlay(self.display_frame)
                    
                    # 5. 显示帧
                    cv2.imshow(WINDOW_NAME, final_frame)
                    
                    # 6. 更新显示FPS
                    self.update_display_fps()
                
                # 7. 更新性能统计
                self.hybrid_manager.update_performance_stats()
                
                # 8. 处理键盘输入
                if not self.process_keyboard_input():
                    break
                
            except Exception as e:
                print(f"主循环错误: {e}")
                time.sleep(0.1)
        
        print("主循环结束")
    
    def run(self):
        """运行系统"""
        try:
            # 初始化
            if not self.initialize_system():
                print("系统初始化失败")
                return False
            
            # 启动
            self.start_system()
            
            # 主循环
            self.main_loop()
            
        except KeyboardInterrupt:
            print("\n收到中断信号")
        except Exception as e:
            print(f"系统运行错误: {e}")
        finally:
            self.shutdown()
        
        return True
    
    def shutdown(self):
        """关闭系统"""
        if not self.running:
            return
        
        print("=== 关闭系统 ===")
        self.running = False
        self.shutdown_event.set()
        
        # 关闭显示窗口
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # 关闭混合管理器
        if self.hybrid_manager:
            self.hybrid_manager.shutdown()
        
        # 关闭网络处理器
        if self.network_handler:
            self.network_handler.stop_all()
        
        print("系统关闭完成")

def main():
    """主函数"""
    print("=== 混合检测跟踪系统启动 ===")
    print("Florence2 + Cutie 混合架构")
    print("按 'h' 查看键盘快捷键")
    print("=" * 40)
    
    # 创建并运行系统
    viewer = HybridViewer()
    success = viewer.run()
    
    if success:
        print("系统正常退出")
    else:
        print("系统异常退出")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main() 