# VLM + CamShift 跟踪系统

这是一个整合了VLM（Vision Language Model）检测和CamShift跟踪的完整目标跟踪系统。

## 系统架构

```
VLM检测结果 → CamShift矫正 → 持续跟踪 → 输出bbox中心点
```

## 主要组件

### 1. `vlm_camshift_tracker.py` - 核心跟踪器
- **VLMCamShiftTracker**: 主要跟踪器类
  - 使用Florence-2模型进行VLM检测
  - 使用CamShift算法进行实时跟踪
  - 自动处理跳变检测和跟踪重新初始化
- **TrackerServer**: 网络服务器，输出bbox中心点数据

### 2. `integrated_tracker_viewer.py` - 集成查看器
- 接收网络图像流
- 整合VLM检测和CamShift跟踪
- 实时显示跟踪结果
- 支持JSON配置和目标切换

### 3. `center_point_client.py` - 中心点客户端
- 接收跟踪器输出的bbox中心点数据
- 支持同步和异步接收模式

### 4. `robot_cs.py` - 原始CamShift实现（参考）
- 基础的CamShift跟踪实现
- 包含跳变检测功能

## 功能特性

### VLM检测
- 使用Florence-2模型进行目标检测
- 支持自定义目标类别（通过JSON配置）
- 可配置检测间隔和置信度阈值
- 自动跳变检测，避免误检测干扰

### CamShift跟踪
- 基于颜色直方图的实时跟踪
- 自适应目标大小和旋转
- 高帧率跟踪（不依赖VLM检测频率）
- 跟踪置信度评估

### 网络接口
- **端口12345**: 图像接收
- **端口12346**: bbox数据发送
- **端口12349**: 中心点数据输出（新增）
- **端口65430**: JSON配置接收
- **端口12347**: 目标切换命令
- **端口12348**: 方向命令发送

## 使用方法

### 1. 启动集成跟踪器查看器

```bash
python integrated_tracker_viewer.py
```

系统启动后会：
- 连接到图像服务器（192.168.3.11:12345）
- 等待JSON配置数据
- 启动各种网络服务

### 2. 发送目标配置JSON

发送到端口65430的JSON格式：

```json
{
    "target_entities": ["cup", "bottle"]
}
```

### 3. 接收中心点数据

```bash
python center_point_client.py
```

或在代码中使用：

```python
from center_point_client import CenterPointClient

client = CenterPointClient(server_host='127.0.0.1', server_port=12349)
client.start_receiving_async()

# 获取最新中心点
center = client.get_center_point()
if center:
    cx, cy = center
    print(f"Target center: ({cx}, {cy})")
```

### 4. 目标切换

发送到端口12347：
```
NEXT_TARGET
```

### 5. 方向命令

发送JSON到端口65430：
```json
{
    "direction": "forward"
}
```

支持的方向：`forward`, `backward`, `left`, `right`, `front-left`, `front-right`, `back-left`, `back-right`

## 配置参数

### VLMCamShiftTracker参数

```python
tracker = VLMCamShiftTracker(
    model_path="D:/models",                    # Florence-2模型路径
    ontology_caption={"object": "a object"}   # 检测目标配置
)

# 可调参数
tracker.vlm_detection_interval = 0.5    # VLM检测间隔（秒）
tracker.pos_thresh = 50                 # 位置跳变阈值（像素）
tracker.area_thresh = 0.5               # 面积变化阈值（比例）
tracker.confidence_threshold = 0.5      # VLM检测置信度阈值
```

## 输出数据格式

### 中心点数据（端口12349）

```json
{
    "center_x": 320,
    "center_y": 240,
    "bbox": [300, 220, 40, 40],
    "confidence": 0.85,
    "timestamp": 1703123456.789,
    "status": "tracking"
}
```

### bbox数据（端口12346）

格式：`x1,y1,x2,y2`（xyxy格式）

## 性能优化

### VLM检测优化
- 使用快速模式（`fast_mode=True`）
- 减少beam search（`num_beams=1`）
- 降低最大token数（`max_new_tokens=512`）
- 启用早停（`early_stopping=True`）

### CamShift跟踪优化
- 高频率跟踪（不受VLM检测频率限制）
- 轻量级颜色直方图计算
- 自适应ROI更新

### 系统优化
- 多线程架构（VLM检测、图像接收、显示分离）
- 队列缓冲机制
- GPU内存自动清理

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径：`D:/models`
   - 确保Florence-2模型文件存在
   - 检查CUDA环境

2. **网络连接问题**
   - 检查图像服务器IP：`192.168.3.11`
   - 确保端口未被占用
   - 检查防火墙设置

3. **跟踪效果不佳**
   - 调整`pos_thresh`和`area_thresh`参数
   - 增加VLM检测频率（减少`vlm_detection_interval`）
   - 提高置信度阈值（`confidence_threshold`）

4. **性能问题**
   - 启用GPU加速
   - 调整图像分辨率
   - 优化VLM检测参数

### 调试模式

在代码中添加调试输出：

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 打印跟踪状态
tracker.debug_mode = True
```

## 扩展功能

### 添加新的检测模型
1. 在`vlm_camshift_tracker.py`中修改模型初始化
2. 实现新的检测接口
3. 更新数据格式转换

### 添加新的跟踪算法
1. 继承`VLMCamShiftTracker`类
2. 重写`update_camshift_tracker`方法
3. 实现新的跟踪逻辑

### 自定义网络协议
1. 修改`TrackerServer`类
2. 实现新的数据序列化格式
3. 更新客户端接收逻辑

## 依赖项

```
opencv-python
numpy
torch
transformers
autodistill
supervision
```

## 许可证

本项目遵循MIT许可证。 