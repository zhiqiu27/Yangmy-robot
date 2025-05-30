import threading
import queue
import time
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor

# Florence2 imports
from object_detect import initialize_detection_model, run_detection, update_model_ontology

# Cutie imports  
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from config import *

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.fps_history = []
        self.gpu_usage_history = []
        self.last_fps_time = time.time()
        self.frame_count = 0
        
    def update_fps(self):
        """更新FPS统计"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= FPS_CALC_INTERVAL:
            fps = self.frame_count / (current_time - self.last_fps_time)
            self.fps_history.append(fps)
            
            if len(self.fps_history) > PERFORMANCE_HISTORY_SIZE:
                self.fps_history.pop(0)
                
            self.frame_count = 0
            self.last_fps_time = current_time
            return fps
        return None
    
    def get_average_fps(self):
        """获取平均FPS"""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0
    
    def should_use_async_mode(self, queue_size=0):
        """判断是否应该使用异步处理模式"""
        # 1. FPS检查
        avg_fps = self.get_average_fps()
        if avg_fps > 0 and avg_fps < TARGET_FPS * FPS_THRESHOLD_RATIO:
            return True
            
        # 2. 队列积压检查
        if queue_size > QUEUE_BACKLOG_THRESHOLD:
            return True
            
        # 3. GPU使用率检查
        if torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
                if gpu_util > GPU_USAGE_THRESHOLD:
                    return True
            except:
                pass
                
        return False

class HybridManager:
    """混合检测跟踪管理器 - 整合Florence2和Cutie"""
    
    def __init__(self):
        # 系统状态
        self.system_state = STATE_DETECTING
        self.state_lock = threading.Lock()
        
        # 模型对象
        self.florence2_model = None
        self.florence2_device = None
        self.florence2_ontology = None
        
        self.cutie_model = None
        self.cutie_processor = None
        
        # 目标管理
        self.target_entities_list = []
        self.current_target_index = 0
        self.current_target_class = None
        
        # 跟踪状态
        self.tracking_initialized = False
        self.tracking_frame_count = 0
        self.last_mask = None
        self.last_bbox = None
        
        # 处理模式
        self.use_async_mode = False
        self.last_mode_check = time.time()
        
        # 队列
        self.processing_queue = queue.Queue(maxsize=PROCESSING_QUEUE_MAXSIZE)
        self.json_command_queue = queue.Queue(maxsize=JSON_COMMAND_QUEUE_MAXSIZE)
        self.system_command_queue = queue.Queue(maxsize=SYSTEM_COMMAND_QUEUE_MAXSIZE)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 回调函数
        self.on_direction_command = None
        
        # 控制事件
        self.shutdown_event = threading.Event()
        
        # 异步处理线程
        self.async_processing_thread = None
        self.async_thread_running = False
        
        print("HybridManager initialized")
    
    def start_async_processing_thread(self):
        """启动异步处理线程"""
        if self.async_processing_thread and self.async_processing_thread.is_alive():
            return
            
        self.async_thread_running = True
        self.async_processing_thread = threading.Thread(
            target=self.async_processing_worker, 
            daemon=True,
            name="AsyncProcessingThread"
        )
        self.async_processing_thread.start()
        print("异步处理线程启动")
    
    def stop_async_processing_thread(self):
        """停止异步处理线程"""
        if not self.async_thread_running:
            return
            
        self.async_thread_running = False
        
        # 发送停止信号
        try:
            self.processing_queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self.async_processing_thread and self.async_processing_thread.is_alive():
            self.async_processing_thread.join(timeout=2.0)
        
        print("异步处理线程停止")
    
    def async_processing_worker(self):
        """异步处理工作线程"""
        print("异步处理工作线程启动")
        
        while self.async_thread_running:
            try:
                frame = self.processing_queue.get(timeout=1.0)
                
                # 停止信号
                if frame is None:
                    break
                
                # 只在跟踪模式下处理
                if self.system_state == STATE_TRACKING and self.tracking_initialized:
                    mask = self.run_cutie_tracking(frame)
                    # mask和bbox会自动更新到self.last_mask和self.last_bbox
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"异步处理错误: {e}")
                continue
        
        print("异步处理工作线程结束")
    
    def monitor_gpu_memory(self):
        """监控GPU内存使用"""
        if not ENABLE_GPU_MEMORY_MONITORING or not torch.cuda.is_available():
            return
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage_ratio = allocated / total
            
            if usage_ratio > GPU_MEMORY_WARNING_THRESHOLD:
                print(f"警告: GPU内存使用率 {usage_ratio*100:.1f}% ({allocated:.1f}GB/{total:.1f}GB)")
                
                if AUTO_CLEAR_GPU_CACHE:
                    torch.cuda.empty_cache()
                    print("GPU缓存已清理")
                    
        except Exception as e:
            print(f"GPU内存监控错误: {e}")
    
    def switch_to_detection_mode(self):
        """切换到检测模式"""
        print("切换到检测模式...")
        
        with self.state_lock:
            # 停止异步处理线程
            self.stop_async_processing_thread()
            
            # 完全清理Cutie资源 - 切换目标时清空
            if self.cutie_processor:
                del self.cutie_processor
                self.cutie_processor = None
                print("[Detection] 清理Cutie processor")
                
            if self.cutie_model:
                del self.cutie_model
                self.cutie_model = None
                print("[Detection] 清理Cutie model")
            
            # 清理跟踪状态
            self.tracking_initialized = False
            self.tracking_frame_count = 0
            self.last_mask = None
            self.last_bbox = None
            
            # 清理GPU缓存
            if torch.cuda.is_available() and AUTO_CLEAR_GPU_CACHE:
                torch.cuda.empty_cache()
                print("[Detection] GPU缓存已清理")
            
            # 准备ontology
            if self.current_target_class:
                # 使用当前目标创建ontology
                target_object_name = self.current_target_class
                target_object_caption = f"a {target_object_name}"
                ontology_caption = {target_object_name: target_object_caption}
                print(f"[Detection] 使用目标: '{target_object_name}' 描述: '{target_object_caption}'")
            else:
                # 使用默认ontology
                ontology_caption = DEFAULT_ONTOLOGY
                print(f"[Detection] 使用默认ontology: {ontology_caption}")
            
            # 检查是否需要重新加载模型（ontology变化）
            need_reload = False
            if not self.florence2_model or not self.florence2_ontology:
                need_reload = True
                print("[Detection] 模型未初始化，需要加载")
            else:
                # 检查ontology是否变化
                current_ontology_dict = {}
                if hasattr(self.florence2_ontology, 'prompts'):
                    current_ontology_dict = self.florence2_ontology.prompts
                elif hasattr(self.florence2_ontology, '_prompts'):
                    current_ontology_dict = self.florence2_ontology._prompts
                
                if current_ontology_dict != ontology_caption:
                    need_reload = True
                    print(f"[Detection] Ontology变化，需要重新加载")
                    print(f"[Detection] 当前: {current_ontology_dict}")
                    print(f"[Detection] 新的: {ontology_caption}")
                else:
                    print("[Detection] Ontology未变化，使用现有模型")
            
            if need_reload:
                # 清理现有Florence2模型
                if self.florence2_model:
                    print("[Detection] 清理现有Florence2模型...")
                    del self.florence2_model
                    self.florence2_model = None
                    del self.florence2_ontology
                    self.florence2_ontology = None
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 重新初始化Florence2模型
                print("[Detection] 重新初始化Florence2模型...")
                self.florence2_model, self.florence2_device, self.florence2_ontology = \
                    initialize_detection_model(
                        model_path=FLORENCE2_MODEL_PATH,
                        ontology_caption=ontology_caption,
                        force_reinit=True  # 强制重新初始化
                    )
                
                if self.florence2_model:
                    print("[Detection] Florence2模型重新加载完成")
                else:
                    print("[Detection] Florence2模型重新加载失败")
                    return False
            
            self.system_state = STATE_DETECTING
            
        self.monitor_gpu_memory()
        print(f"[Detection] 切换完成，当前状态: {self.system_state}")
        return True
    
    def switch_to_tracking_mode(self, initial_bbox, frame):
        """切换到跟踪模式"""
        print("切换到跟踪模式...")
        
        with self.state_lock:
            # 清理Florence2资源
            if self.florence2_model:
                del self.florence2_model
                self.florence2_model = None
                del self.florence2_ontology
                self.florence2_ontology = None
                print("[Tracking] 清理Florence2模型")
            
            # 清理GPU缓存
            if torch.cuda.is_available() and AUTO_CLEAR_GPU_CACHE:
                torch.cuda.empty_cache()
                print("[Tracking] GPU缓存已清理")
            
            # 重新初始化Cutie（每次都重新加载，避免状态冲突）
            print("[Tracking] 重新初始化Cutie模型...")
            try:
                # 清理Hydra全局状态
                try:
                    from hydra.core.global_hydra import GlobalHydra
                    if GlobalHydra.instance().is_initialized():
                        GlobalHydra.instance().clear()
                        print("[Tracking] 清理Hydra全局状态")
                except Exception as e:
                    print(f"[Tracking] Hydra清理警告: {e}")
                
                # 重新加载Cutie模型
                self.cutie_model = get_default_model()
                self.cutie_processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
                self.cutie_processor.max_internal_size = CUTIE_MAX_INTERNAL_SIZE
                print("[Tracking] Cutie模型重新加载完成")
                
            except Exception as e:
                print(f"[Tracking] Cutie模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # 初始化跟踪
            success = self.initialize_cutie_with_bbox(frame, initial_bbox)
            if success:
                self.system_state = STATE_TRACKING
                self.tracking_initialized = True
                self.tracking_frame_count = 0
                
                # 启动异步处理线程
                self.start_async_processing_thread()
                
                print(f"[Tracking] 跟踪初始化成功，bbox: {initial_bbox}")
            else:
                print("[Tracking] 跟踪初始化失败，保持检测模式")
                self.system_state = STATE_DETECTING
                
        self.monitor_gpu_memory()
        return success
    
    def initialize_cutie_with_bbox(self, frame, bbox):
        """使用bbox初始化Cutie跟踪"""
        try:
            if bbox is None or len(bbox) != 4:
                print(f"[Tracking] 无效的bbox: {bbox}")
                return False
                
            x1, y1, x2, y2 = map(int, bbox)
            print(f"[Tracking] 初始化bbox: ({x1}, {y1}, {x2}, {y2})")
            
            # 验证bbox范围
            h, w = frame.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                print(f"[Tracking] bbox超出图像范围或无效: frame({w}x{h}), bbox({x1},{y1},{x2},{y2})")
                return False
            
            # 创建初始mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            print(f"[Tracking] 创建初始mask，大小: {mask.shape}, 非零像素: {np.sum(mask)}")
            
            # 转换为Cutie需要的格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = to_tensor(frame_rgb).cuda().float()
            mask_tensor = torch.from_numpy(mask).cuda()
            print(f"[Tracking] 张量转换完成，image: {image_tensor.shape}, mask: {mask_tensor.shape}")
            
            # 初始化跟踪
            objects = [1]
            print(f"[Tracking] 开始Cutie初始化，objects: {objects}")
            output_prob = self.cutie_processor.step(image_tensor, mask_tensor, objects=objects)
            print(f"[Tracking] Cutie step完成，output_prob shape: {output_prob.shape}")
            
            # 保存初始结果
            initial_mask = self.cutie_processor.output_prob_to_mask(output_prob)
            self.last_mask = initial_mask.cpu().numpy()
            self.last_bbox = self.mask_to_bbox(self.last_mask)
            
            print(f"[Tracking] 初始化成功，last_mask shape: {self.last_mask.shape}, last_bbox: {self.last_bbox}")
            return True
            
        except Exception as e:
            print(f"[Tracking] Cutie初始化错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_florence2_detection(self, frame):
        """运行Florence2检测"""
        if not self.florence2_model:
            return None
            
        try:
            detections = run_detection(frame, self.florence2_model)
            return detections
        except Exception as e:
            print(f"Florence2检测错误: {e}")
            return None
    
    def select_best_target(self, detections):
        """从检测结果中选择最佳目标"""
        if not detections or not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
            return None
            
        # 简单策略：选择置信度最高的目标
        if hasattr(detections, 'confidence') and detections.confidence is not None:
            best_idx = np.argmax(detections.confidence)
            return detections.xyxy[best_idx]
        else:
            # 如果没有置信度，选择面积最大的
            areas = []
            for bbox in detections.xyxy:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)
            best_idx = np.argmax(areas)
            return detections.xyxy[best_idx]
    
    @torch.inference_mode()
    @torch.amp.autocast(device_type='cuda')
    def run_cutie_tracking(self, frame):
        """运行Cutie跟踪"""
        if not self.cutie_processor or not self.tracking_initialized:
            return None
            
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = to_tensor(frame_rgb).cuda().float()
            
            output_prob = self.cutie_processor.step(image_tensor)
            mask = self.cutie_processor.output_prob_to_mask(output_prob)
            mask_np = mask.cpu().numpy()
            
            self.last_mask = mask_np
            self.last_bbox = self.mask_to_bbox(mask_np)
            self.tracking_frame_count += 1
            
            return mask_np
            
        except Exception as e:
            print(f"Cutie跟踪错误: {e}")
            return None
    
    def mask_to_bbox(self, mask):
        """将mask转换为bbox（使用方法一：最小外接矩形）"""
        if mask is None:
            return None
            
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (x_min, y_min, x_max, y_max)
    
    def visualize_tracking_result(self, frame, mask):
        """可视化跟踪结果"""
        if mask is None:
            return frame
            
        try:
            # 创建彩色mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask == 1] = [0, 255, 0]  # 绿色
            
            # 混合显示
            alpha = 0.3
            result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
            
            # 绘制轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            # 绘制bbox
            if self.last_bbox:
                x1, y1, x2, y2 = map(int, self.last_bbox)
                cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 显示中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
            
            return result
            
        except Exception as e:
            print(f"可视化错误: {e}")
            return frame
    
    def determine_processing_strategy(self):
        """确定处理策略"""
        current_time = time.time()
        
        # 定期检查是否需要切换模式
        if current_time - self.last_mode_check > PERFORMANCE_CHECK_INTERVAL:
            queue_size = self.processing_queue.qsize() if hasattr(self, 'processing_queue') else 0
            should_async = self.performance_monitor.should_use_async_mode(queue_size)
            
            if should_async != self.use_async_mode:
                self.use_async_mode = should_async
                mode_name = "异步" if should_async else "同步"
                print(f"切换到{mode_name}处理模式")
            
            self.last_mode_check = current_time
        
        # 根据系统状态确定策略
        if self.system_state == STATE_DETECTING:
            return "sync_detection"
        elif self.system_state == STATE_SWITCHING:
            return "sync_detection"
        elif self.system_state == STATE_TRACKING:
            if self.tracking_frame_count < 5:  # 前几帧用同步确保稳定
                return "sync_tracking"
            elif self.use_async_mode:
                return "async_tracking"
            else:
                return "sync_tracking"
        
        return "sync_tracking"
    
    def process_frame(self, frame):
        """处理单帧图像"""
        if frame is None:
            return frame, None
            
        strategy = self.determine_processing_strategy()
        
        if strategy == "sync_detection":
            return self.sync_detection_process(frame)
        elif strategy == "sync_tracking":
            return self.sync_tracking_process(frame)
        elif strategy == "async_tracking":
            return self.async_tracking_process(frame)
        
        return frame, None
    
    def sync_detection_process(self, frame):
        """同步检测处理"""
        #print(f"[Detection] 开始检测，当前目标: {self.current_target_class}")
        detections = self.run_florence2_detection(frame)
        
        if detections:
            print(f"[Detection] 检测结果: {detections}")
            selected_bbox = self.select_best_target(detections)
            if selected_bbox is not None:
                print(f"[Detection] 检测到目标，bbox: {selected_bbox}")
                # 切换到跟踪模式
                success = self.switch_to_tracking_mode(selected_bbox, frame)
                if success:
                    print(f"[Detection] 跟踪模式切换成功，开始跟踪")
                    # 立即进行一次跟踪并显示结果
                    mask = self.run_cutie_tracking(frame)
                    if mask is not None:
                        print(f"[Detection] 初始跟踪成功，mask shape: {mask.shape}")
                        display_frame = self.visualize_tracking_result(frame, mask)
                        return display_frame, self.last_bbox
                    else:
                        print(f"[Detection] 初始跟踪失败")
                else:
                    print(f"[Detection] 跟踪模式切换失败")
            else:
                print(f"[Detection] 未选择到合适的目标")
        else:
            #print(f"[Detection] 未检测到任何目标")
            pass
        
        # 检测失败或未找到目标 - 只显示状态文本，不显示检测框
        display_frame = frame.copy()
        # status_text = f"{DETECTING_TEXT} 目标: {self.current_target_class}"
        # cv2.putText(display_frame, status_text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return display_frame, None
    
    def sync_tracking_process(self, frame):
        """同步跟踪处理"""
        mask = self.run_cutie_tracking(frame)
        
        if mask is not None:
            # 实时显示跟踪结果
            display_frame = self.visualize_tracking_result(frame, mask)
            
            # # 添加状态文本
            # status_text = f"{TRACKING_TEXT} Frame: {self.tracking_frame_count} {SYNC_MODE_TEXT}"
            # cv2.putText(display_frame, status_text, (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return display_frame, self.last_bbox
        else:
            # 跟踪失败，切换回检测模式
            print(f"[Tracking] 跟踪失败，切换回检测模式")
            self.switch_to_detection_mode()
            return frame, None
    
    def async_tracking_process(self, frame):
        """异步跟踪处理"""
        # 发送到处理队列
        try:
            # 清理积压
            while self.processing_queue.qsize() > QUEUE_BACKLOG_THRESHOLD:
                try:
                    self.processing_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.processing_queue.put_nowait(frame.copy())
        except queue.Full:
            pass
        
        # 使用最新的跟踪结果实时显示
        if self.last_mask is not None:
            display_frame = self.visualize_tracking_result(frame, self.last_mask)
            
            # # 添加状态文本
            # status_text = f"{TRACKING_TEXT} Frame: {self.tracking_frame_count} {ASYNC_MODE_TEXT}"
            # cv2.putText(display_frame, status_text, (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return display_frame, self.last_bbox
        else:
            return frame, None
    
    def get_bbox_data(self):
        """获取当前bbox数据"""
        return self.last_bbox
    
    def process_json_data(self, json_data):
        """处理JSON数据"""
        try:
            # 确保json_data是字符串
            if isinstance(json_data, bytes):
                json_data = json_data.decode('utf-8')
            
            json_data = json_data.strip()
            if not json_data:
                print("[JSON] 收到空数据")
                return
                
            print(f"[JSON] 解析数据: {json_data}")
            data = json.loads(json_data)
            
            # 处理DirectionAgent的方向指令
            if "direction" in data:
                direction_value = data.get("direction", "")
                print(f"[JSON] 收到方向指令: '{direction_value}'")
                if self.on_direction_command:
                    self.on_direction_command(direction_value)
                return
            
            # 处理TargetAgent的目标信息
            if "target_entities" in data:
                new_target_entities_list = data.get("target_entities", [])
                print(f"[JSON] 收到目标列表: {new_target_entities_list}")
                
                # 验证目标列表
                if not new_target_entities_list or not isinstance(new_target_entities_list, list) or len(new_target_entities_list) == 0:
                    print(f"[JSON] 无效的目标列表: {new_target_entities_list}")
                    return
                
                # 更新目标列表 - 总是从第一个目标开始
                self.target_entities_list = new_target_entities_list
                self.current_target_index = 0  # 总是从第一个目标开始
                self.current_target_class = self.target_entities_list[0]
                
                print(f"[JSON] 设置主要目标: '{self.current_target_class}' (index: 0)")
                if len(self.target_entities_list) > 1:
                    print(f"[JSON] 备选目标: {self.target_entities_list[1:]} (可通过切换命令访问)")
                
                # 切换到检测模式（这会重新加载模型和ontology）
                self.switch_to_detection_mode()
                return
            
            print(f"[JSON] 未知JSON格式: {json_data}")
            print("[JSON] 期望包含 'target_entities' (TargetAgent) 或 'direction' (DirectionAgent) 字段")
                    
        except json.JSONDecodeError as e:
            print(f"[JSON] JSON解析错误: {e}")
            print(f"[JSON] 原始数据: '{json_data}'")
        except Exception as e:
            print(f"[JSON] 处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_command(self, command):
        """处理外部命令"""
        command = command.strip().lower()
        
        if command == "reset":
            print("[Command] 重置系统")
            # 重置到第一个目标
            if self.target_entities_list:
                self.current_target_index = 0
                self.current_target_class = self.target_entities_list[0]
                print(f"[Command] 重置到第一个目标: {self.current_target_class}")
            self.switch_to_detection_mode()
            return "System reset"
            
        elif command == "switch_target" or command == "next_target":
            if not self.target_entities_list or len(self.target_entities_list) < 2:
                print("[Command] 目标列表不足，无法切换")
                return "Not enough targets to switch"
            
            if self.current_target_index == 1:
                print("[Command] 已经在第二个目标上")
                return "Already on second target"
            
            # 切换到第二个目标 (index 1)
            self.current_target_index = 1
            self.current_target_class = self.target_entities_list[1]
            print(f"[Command] 切换到第二个目标: {self.current_target_class}")
            self.switch_to_detection_mode()
            return f"Switched to target: {self.current_target_class}"
                
        elif command == "status":
            target_info = f"Target: {self.current_target_class} (index: {self.current_target_index})"
            if len(self.target_entities_list) > 1:
                target_info += f", Available: {self.target_entities_list}"
            return f"State: {self.system_state}, {target_info}, Mode: {'Async' if self.use_async_mode else 'Sync'}"
            
        return "Unknown command"
    
    def update_performance_stats(self):
        """更新性能统计"""
        fps = self.performance_monitor.update_fps()
        if fps is not None:
            avg_fps = self.performance_monitor.get_average_fps()
            print(f"[Performance] 当前FPS: {fps:.1f}, 平均FPS: {avg_fps:.1f}")
    
    def shutdown(self):
        """关闭管理器"""
        print("关闭HybridManager...")
        
        self.shutdown_event.set()
        
        # 停止异步处理线程
        self.stop_async_processing_thread()
        
        # 清理模型资源
        with self.state_lock:
            if self.florence2_model:
                del self.florence2_model
                self.florence2_model = None
                
            if self.cutie_model:
                del self.cutie_model
                self.cutie_model = None
                
            if self.cutie_processor:
                del self.cutie_processor
                self.cutie_processor = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("HybridManager关闭完成") 