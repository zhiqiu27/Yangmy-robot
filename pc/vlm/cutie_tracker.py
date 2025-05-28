"""
Cutie Tracker - 独立的视频对象追踪器
可以在其他项目中直接调用

使用示例:
    from cutie_tracker import CutieTracker
    
    tracker = CutieTracker()
    
    # 第一帧提供检测框
    result = tracker.track(image, bboxes=[[100,50,200,150]], labels=[1])
    print(f"中心点: {result['centers']}")  # [(150, 100)]
    
    # 后续帧自动追踪
    result = tracker.track(next_image)
    print(f"新中心点: {result['centers']}")
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from typing import List, Tuple, Optional, Dict, Any, Union

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
    """
    
    def __init__(self, 
                 max_internal_size: int = 480, 
                 device: str = 'cuda',
                 model_path: Optional[str] = None):
        """
        初始化追踪器
        
        Args:
            max_internal_size: 内部处理的最大图像尺寸，-1表示保持原尺寸
            device: 计算设备 'cuda' 或 'cpu'
            model_path: 自定义模型路径，None表示使用默认模型
        """
        self.device = device
        self.max_internal_size = max_internal_size
        self.model_path = model_path
        
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
              labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        追踪对象并返回中心点
        
        Args:
            image: 输入图像 (H,W,3) numpy数组 或 PIL图像 或 (3,H,W) tensor
            bboxes: 检测框列表 [[x1,y1,x2,y2], ...] (可选)
            labels: 对象标签列表 [1,2,3,...] (可选)
        
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
        
        # 第一帧或提供了新的检测框
        if bboxes is not None and labels is not None:
            mask, objects = self._bboxes_to_mask(bboxes, labels, 
                                                image_tensor.shape[-2:])
            output_prob = self.processor.step(image_tensor, mask, objects=objects)
            self.initialized = True
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
        print("追踪器已重置")
    
    def remove_objects(self, object_ids: List[int]):
        """删除指定的对象"""
        self.processor.delete_objects(object_ids)
        print(f"已删除对象: {object_ids}")
    
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """预处理输入图像"""
        if isinstance(image, np.ndarray):
            # numpy数组 (H,W,3) -> PIL -> tensor
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
            image = to_tensor(image)
        elif isinstance(image, Image.Image):
            # PIL图像 -> tensor
            image = to_tensor(image)
        elif isinstance(image, torch.Tensor):
            # 已经是tensor
            if image.dim() == 3 and image.shape[0] == 3:
                # (3,H,W) 格式正确
                pass
            elif image.dim() == 3 and image.shape[2] == 3:
                # (H,W,3) -> (3,H,W)
                image = image.permute(2, 0, 1)
            else:
                raise ValueError(f"不支持的tensor形状: {image.shape}")
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
        
        # 确保数据类型和设备
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
            
            # 确保坐标在图像范围内
            x1 = max(0, min(int(x1), width-1))
            y1 = max(0, min(int(y1), height-1))
            x2 = max(x1+1, min(int(x2), width))
            y2 = max(y1+1, min(int(y2), height))
            
            # 对象ID从1开始
            object_id = i + 1
            mask[y1:y2, x1:x2] = object_id
            objects.append(object_id)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            mask = mask.cuda()
        
        return mask, objects
    
    def _analyze_mask(self, mask: torch.Tensor) -> Dict[str, Any]:
        """分析掩码，计算中心点等信息"""
        # 获取所有对象ID（除了背景0）
        unique_ids = torch.unique(mask)
        object_ids = unique_ids[unique_ids != 0].cpu().tolist()
        
        centers = []
        areas = []
        bboxes = []
        
        for obj_id in object_ids:
            # 找到对象像素
            object_pixels = (mask == obj_id)
            
            if object_pixels.sum() > 0:
                # 计算面积
                area = object_pixels.sum().item()
                areas.append(area)
                
                # 找到对象的坐标
                y_coords, x_coords = torch.where(object_pixels)
                
                # 计算边界框
                x1 = x_coords.min().item()
                x2 = x_coords.max().item()
                y1 = y_coords.min().item()
                y2 = y_coords.max().item()
                bboxes.append([x1, y1, x2, y2])
                
                # 计算边界框中心
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((center_x, center_y))
            else:
                # 对象不存在
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
    
    # 创建追踪器
    tracker = CutieTracker(max_internal_size=480)
    
    # 模拟第一帧（提供检测框）
    dummy_image = torch.rand(3, 480, 640)  # 随机图像
    bboxes = [[100, 50, 200, 150], [300, 100, 400, 250]]  # 两个检测框
    labels = [1, 2]  # 对象标签
    
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


if __name__ == "__main__":
    demo() 