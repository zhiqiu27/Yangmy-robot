"""
Cutie Tracker - 独立的视频对象追踪器
可以在其他项目中直接调用

使用示例:
    from cutie_tracker import CutieTracker
    
    tracker = CutieTracker(update_interval=5.0)  # 设置5秒更新间隔
    
    # 第一帧提供检测框
    result = tracker.track(image, bboxes=[[100,50,200,150]], labels=[1])
    print(f"中心点: {result['centers']}")  # [(150, 100)]
    
    # 后续帧自动追踪
    result = tracker.track(next_image)
    print(f"新中心点: {result['centers']}")
    
    # 强制更新掩码
    result = tracker.track(next_image, bboxes=[[110,60,210,160]], labels=[1], force_update=True)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from typing import List, Tuple, Optional, Dict, Any, Union
import time

# 添加Cutie路径（如果需要）
# sys.path.append('/path/to/cutie')

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


class CutieTracker:
    """
    Cutie视频对象追踪器
    
    功能:
    - 基于检测框初始化对象追踪
    - 返回对象的中心点坐标
    - 支持多对象同时追踪
    - 动态添加/删除对象
    - 支持定时更新掩码（默认每5秒）
    - 支持强制更新掩码
    """
    
    def __init__(self, 
                 max_internal_size: int = 480, 
                 device: str = 'cuda',
                 model_path: Optional[str] = None,
                 update_interval: float = 5.0):
        """
        初始化追踪器
        
        Args:
            max_internal_size: 内部处理的最大图像尺寸，-1表示保持原尺寸
            device: 计算设备 'cuda' 或 'cpu'
            model_path: 自定义模型路径，None表示使用默认模型
            update_interval: 掩码更新的时间间隔（秒）
        """
        self.device = device
        self.max_internal_size = max_internal_size
        self.model_path = model_path
        self.update_interval = update_interval  # 新增：掩码更新间隔
        
        # 初始化Cutie模型
        print("正在加载Cutie模型...")
        if model_path is not None:
            self.cutie_model = self._load_custom_model(model_path)
        else:
            self.cutie_model = get_default_model()
        
        self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
        self.processor.max_internal_size = max_internal_size
        
        # 状态管理
        self.initialized = False
        self.frame_count = 0
        self.last_update_time = time.time()  # 新增：记录上次更新时间
        
        print("Cutie追踪器初始化完成")
    
    def _load_custom_model(self, model_path: str):
        """加载自定义模型"""
        from hydra import compose, initialize
        from omegaconf import open_dict
        from cutie.model.cutie import CUTIE
        from cutie.inference.utils.args_utils import get_dataset_cfg
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化配置
        initialize(version_base='1.3.2', config_path="../config", job_name="eval_config")
        cfg = compose(config_name="eval_config")
        
        # 设置自定义权重路径
        with open_dict(cfg):
            cfg['weights'] = model_path
        get_dataset_cfg(cfg)
        
        # 加载模型
        cutie = CUTIE(cfg).cuda().eval() if self.device == 'cuda' else CUTIE(cfg).eval()
        model_weights = torch.load(cfg.weights, map_location=self.device)
        cutie.load_weights(model_weights)
        
        print(f"已加载自定义模型: {model_path}")
        return cutie
    
    def track(self, 
              image: Union[np.ndarray, Image.Image, torch.Tensor],
              bboxes: Optional[List[List[int]]] = None,
              labels: Optional[List[int]] = None,
              force_update: bool = False) -> Dict[str, Any]:
        """
        追踪对象并返回中心点，定期更新初始掩码
        
        Args:
            image: 输入图像 (H,W,3) numpy数组 或 PIL图像 或 (3,H,W) tensor
            bboxes: 检测框列表 [[x1,y1,x2,y2], ...] (可选)
            labels: 对象标签列表 [1,2,3,...] (可选)
            force_update: 是否强制更新掩码（忽略时间间隔）
        
        Returns:
            {
                'centers': [(x1,y1), (x2,y2), ...],  # 对象中心点坐标
                'object_ids': [1, 2, ...],           # 对象ID列表  
                'masks': torch.Tensor,                # (H,W) 分割掩码
                'areas': [area1, area2, ...],         # 对象面积(像素数)
                'bboxes': [[x1,y1,x2,y2], ...]      # 基于掩码的边界框
            }
        """
        # 预处理图像
        image_tensor = self._preprocess_image(image)
        
        # 检查是否需要更新掩码
        current_time = time.time()
        need_update = force_update or (
            self.initialized and 
            (current_time - self.last_update_time) >= self.update_interval
        )
        
        # 第一帧或需要更新掩码
        if (bboxes is not None and labels is not None) or need_update:
            if bboxes is not None and labels is not None:
                # 使用提供的检测框和标签更新掩码
                mask, objects = self._bboxes_to_mask(bboxes, labels, 
                                                  image_tensor.shape[-2:])
                output_prob = self.processor.step(image_tensor, mask, objects=objects)
                self.initialized = True
                self.last_update_time = current_time  # 更新时间戳
            elif need_update and self.initialized:
                # 未提供新检测框，打印警告并继续追踪
                print("警告：需要更新掩码但未提供新检测框，使用上一帧结果继续追踪")
                output_prob = self.processor.step(image_tensor)
            else:
                raise ValueError("首次调用必须提供bboxes和labels参数")
        elif self.initialized:
            # 后续帧，使用记忆进行追踪
            output_prob = self.processor.step(image_tensor)
        else:
            raise ValueError("首次调用必须提供bboxes和labels参数")
        
        # 转换为最终掩码
        final_mask = self.processor.output_prob_to_mask(output_prob)
        
        # 计算结果
        result = self._analyze_mask(final_mask)
        
        self.frame_count += 1
        return result
    
    def reset(self):
        """重置追踪器状态"""
        self.processor.clear_memory()
        self.initialized = False
        self.frame_count = 0
        self.last_update_time = time.time()  # 重置时间戳
        print("追踪器已重置")
    
    def remove_objects(self, object_ids: List[int]):
        """删除指定的对象"""
        self.processor.delete_objects(object_ids)
        print(f"已删除对象: {object_ids}")
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """预处理输入图像"""
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
            image = to_tensor(image)
        elif isinstance(image, Image.Image):
            image = to_tensor(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                pass
            elif image.dim() == 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)
            else:
                raise ValueError(f"不支持的tensor形状: {image.shape}")
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        image = image.float()
        if self.device == 'cuda' and torch.cuda.is_available():
            image = image.cuda()
        
        return image
    
    def _bboxes_to_mask(self, bboxes: List[List[int]], labels: List[int], 
                       image_size: Tuple[int, int]) -> Tuple[torch.Tensor, List[int]]:
        """将检测框转换为掩码"""
        height, width = image_size
        mask = torch.zeros((height, width), dtype=torch.long)
        
        objects = []
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1), width-1))
            y1 = max(0, min(int(y1), height-1))
            x2 = max(x1+1, min(int(x2), width))
            y2 = max(y1+1, min(int(y2), height))
            
            object_id = i + 1
            mask[y1:y2, x1:x2] = object_id
            objects.append(object_id)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            mask = mask.cuda()
        
        return mask, objects
    
    def _analyze_mask(self, mask: torch.Tensor) -> Dict[str, Any]:
        """分析掩码，计算中心点等信息"""
        unique_ids = torch.unique(mask)
        object_ids = unique_ids[unique_ids != 0].cpu().tolist()
        
        centers = []
        areas = []
        bboxes = []
        
        for obj_id in object_ids:
            object_pixels = (mask == obj_id)
            if object_pixels.sum() > 0:
                area = object_pixels.sum().item()
                areas.append(area)
                
                y_coords, x_coords = torch.where(object_pixels)
                x1 = x_coords.min().item()
                x2 = x_coords.max().item()
                y1 = y_coords.min().item()
                y2 = y_coords.max().item()
                bboxes.append([x1, y1, x2, y2])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((center_x, center_y))
            else:
                areas.append(0)
                bboxes.append([0, 0, 0, 0])
                centers.append((0, 0))
        
        return {
            'centers': centers,
            'object_ids': object_ids,
            'masks': mask.cpu(),
            'areas': areas,
            'bboxes': bboxes
        }


def demo():
    """演示用法"""
    print("Cutie追踪器演示")
    
    tracker = CutieTracker(max_internal_size=480, update_interval=5.0)
    
    dummy_image = torch.rand(3, 480, 640)
    bboxes = [[100, 50, 200, 150], [300, 100, 400, 250]]
    labels = [1, 2]
    
    print("\n第一帧追踪:")
    result = tracker.track(dummy_image, bboxes=bboxes, labels=labels)
    print(f"对象ID: {result['object_ids']}")
    print(f"中心点: {result['centers']}")
    print(f"面积: {result['areas']}")
    print(f"边界框: {result['bboxes']}")
    
    # 模拟第二帧（自动追踪）
    dummy_image2 = torch.rand(3, 480, 640)
    print("\n第二帧追踪:")
    result = tracker.track(dummy_image2)
    print(f"对象ID: {result['object_ids']}")
    print(f"中心点: {result['centers']}")
    
    # 模拟强制更新
    print("\n强制更新掩码:")
    result = tracker.track(dummy_image2, bboxes=bboxes, labels=labels, force_update=True)
    print(f"对象ID: {result['object_ids']}")
    print(f"中心点: {result['centers']}")


if __name__ == "__main__":
    demo()