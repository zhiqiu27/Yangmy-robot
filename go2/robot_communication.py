# robot_communication.py
"""
机器人通信模块
处理与新架构服务的通信 (zed_data_server + bbox_processor)
"""

import socket
import json
import logging
import time
from robot_config import (
    IMAGE_SERVER_HOST, IMAGE_SERVER_PORT, DIRECTION_COMMAND_PORT, NEXT_TARGET_PORT,
    ZED_DATA_SERVER_HOST, ZED_IMAGE_PORT, ZED_POINTCLOUD_PORT,
    BBOX_PROCESSOR_HOST, BBOX_PROCESSOR_PORT, BBOX_COMMAND_PORT
)

logger = logging.getLogger(__name__)

class RobotCommunication:
    """处理机器人与外部服务的通信 - 新架构版本"""
    
    def __init__(self):
        # 保持旧配置以兼容某些功能
        self.image_server_host = IMAGE_SERVER_HOST
        self.image_server_port = IMAGE_SERVER_PORT
        self.direction_port = DIRECTION_COMMAND_PORT
        self.next_target_port = NEXT_TARGET_PORT
        
        # 新架构配置
        self.zed_host = ZED_DATA_SERVER_HOST
        self.zed_image_port = ZED_IMAGE_PORT
        self.zed_pointcloud_port = ZED_POINTCLOUD_PORT
        self.bbox_host = BBOX_PROCESSOR_HOST
        self.bbox_port = BBOX_PROCESSOR_PORT
        self.bbox_command_port = BBOX_COMMAND_PORT
    
    def request_visual_info(self):
        """从bbox_processor请求最新的检测信息"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((self.bbox_host, self.bbox_command_port))
                logger.debug("已连接到bbox_processor (请求视觉信息)")
                
                # 发送获取最新检测的命令
                command = {"command": "get_latest_detection"}
                s.sendall(json.dumps(command).encode('utf-8'))
                logger.debug("已发送 'get_latest_detection' 命令")
                
                response_data = s.recv(1024)
                if not response_data:
                    logger.error("从bbox_processor收到空响应 (视觉信息)")
                    return None
                
                response = json.loads(response_data.decode())
                logger.debug(f"从bbox_processor收到响应 (视觉信息): {response}")
                
                return self._parse_bbox_visual_response(response)
                
        except socket.timeout:
            logger.warning(f"连接bbox_processor超时")
            return None
        except socket.error as e:
            logger.error(f"与bbox_processor通信时发生套接字错误: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON响应时出错: {e}")
            return None
        except Exception as e:
            logger.error(f"请求视觉信息时发生意外错误: {e}")
            return None
    
    def request_calibration(self):
        """通过bbox_processor获取多个检测样本实现校准"""
        logger.info(f"开始校准流程，将收集多个检测样本...")
        
        calibration_samples = []
        max_attempts = 10
        required_samples = 3
        
        for attempt in range(max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5)
                    s.connect((self.bbox_host, self.bbox_command_port))
                    logger.debug(f"校准尝试 {attempt + 1}: 已连接到bbox_processor")
                    
                    # 发送获取最新检测的命令
                    command = {"command": "get_latest_detection"}
                    s.sendall(json.dumps(command).encode('utf-8'))
                    
                    response_data = s.recv(1024)
                    if not response_data:
                        logger.warning(f"校准尝试 {attempt + 1}: 收到空响应")
                        continue
                    
                    response = json.loads(response_data.decode())
                    logger.debug(f"校准尝试 {attempt + 1}: 收到响应 {response}")
                    
                    if response.get("status") == "success" and response.get("xyz_coords"):
                        xyz_coords = response["xyz_coords"]
                        # 只取x, y坐标用于校准
                        calibration_coord = [float(xyz_coords[0]), float(xyz_coords[1])]
                        calibration_samples.append(calibration_coord)
                        logger.info(f"校准样本 {len(calibration_samples)}: {calibration_coord}")
                        
                        if len(calibration_samples) >= required_samples:
                            break
                    else:
                        logger.debug(f"校准尝试 {attempt + 1}: 无有效检测结果")
                
                # 等待一段时间再获取下一个样本
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"校准尝试 {attempt + 1} 失败: {e}")
                continue
        
        if len(calibration_samples) >= required_samples:
            # 计算平均坐标作为校准结果
            avg_x = sum(sample[0] for sample in calibration_samples) / len(calibration_samples)
            avg_y = sum(sample[1] for sample in calibration_samples) / len(calibration_samples)
            calibrated_coord = [avg_x, avg_y]
            
            logger.info(f"校准成功，使用 {len(calibration_samples)} 个样本")
            logger.info(f"校准坐标: {calibrated_coord}")
            return calibrated_coord
        else:
            logger.error(f"校准失败，只收集到 {len(calibration_samples)} 个样本，需要至少 {required_samples} 个")
            return None
    
    def receive_direction_command(self, timeout=500):
        """接收方向命令"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', self.direction_port))
                s.listen(1)
                s.settimeout(timeout)
                
                logger.info(f"等待方向命令 (端口 {self.direction_port})")
                conn, addr = s.accept()
                
                data = conn.recv(1024)
                if data:
                    direction = data.decode('utf-8').strip()
                    logger.info(f"收到方向命令: {direction}")
                    conn.sendall(b"ACK")
                    return direction
                return None
                
        except socket.timeout:
            logger.warning("等待方向命令超时")
            return None
        except Exception as e:
            logger.error(f"接收方向命令失败: {e}")
            return None
    
    def _parse_visual_response(self, response):
        """解析视觉信息响应"""
        if response.get("status") == "success":
            if response.get("bbox_available") and 'pixel_cx' in response and 'image_width' in response:
                return {
                    'pixel_cx': float(response['pixel_cx']),
                    'image_width': int(response['image_width']),
                    'bbox_available': True,
                    'depth_x': response.get('depth_x')
                }
            elif not response.get("bbox_available"):
                logger.info("视觉信息：图像服务器报告当前无可用边界框")
                return {'bbox_available': False, 'depth_x': None}
            else:
                logger.error(f"图像服务器成功响应但视觉信息格式不正确: {response}")
                return None
        else:
            logger.error(f"图像服务器报告错误: {response.get('message', '未知错误')}")
            return None
    
    def _parse_calibration_response(self, response):
        """解析校准数据响应"""
        if response.get("status") == "success" and "calibrated_coord" in response:
            calibrated_coord = response["calibrated_coord"]
            calibrated_coord = [float(c) for c in calibrated_coord]
            logger.info(f"成功从图像服务器获取校准坐标: {calibrated_coord}")
            return calibrated_coord
        else:
            logger.error(f"图像服务器报告错误或响应格式不正确: {response.get('message', '未知错误')}")
            return None
    
    def send_next_target_command(self):
        """发送NEXT_TARGET命令到PC端 - 适配新架构的JSON命令格式"""
        logger.info(f"向bbox_processor发送next_target命令")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((self.bbox_host, self.bbox_command_port))  # 连接到bbox_processor
                logger.info("已连接到bbox_processor")
                
                # 发送JSON格式的命令
                command = {"command": "next_target"}
                s.sendall(json.dumps(command).encode('utf-8'))
                logger.info("已发送 'next_target' 命令")
                
                # 等待JSON响应
                response_data = s.recv(1024)
                if response_data:
                    response = json.loads(response_data.decode())
                    logger.info(f"bbox_processor响应: {response}")
                    return response.get("status") == "success"
                else:
                    logger.error("从bbox_processor收到空响应")
                    return False
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return False
        except Exception as e:
            logger.error(f"发送next_target命令失败: {e}")
            return False
    
    def request_person_coordinates(self):
        """从图像服务器请求人员坐标"""
        logger.info("向图像服务器请求人员坐标...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((self.image_server_host, self.image_server_port))
                logger.info("已连接到图像服务器")
                
                s.sendall(b"REQUEST_PERSON_COORD")
                logger.info("已发送 'REQUEST_PERSON_COORD' 命令")
                
                response_data = s.recv(1024)
                if not response_data:
                    logger.error("从图像服务器收到空响应")
                    return None
                
                response = json.loads(response_data.decode())
                logger.info(f"从图像服务器收到人员坐标响应: {response}")
                
                return self._parse_person_coord_response(response)
                
        except socket.timeout:
            logger.error(f"连接图像服务器或接收人员坐标数据超时")
            return None
        except socket.error as e:
            logger.error(f"与图像服务器通信时发生套接字错误: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"解析人员坐标JSON响应时出错: {e}")
            return None
        except Exception as e:
            logger.error(f"请求人员坐标时发生意外错误: {e}")
            return None
    
    def _parse_person_coord_response(self, response):
        """解析人员坐标响应"""
        if response.get("status") == "success" and "person_coord" in response:
            person_coord = response["person_coord"]
            person_coord = [float(c) for c in person_coord]
            logger.info(f"成功从图像服务器获取人员坐标: {person_coord}")
            return person_coord
        else:
            logger.error(f"图像服务器报告错误或人员坐标响应格式不正确: {response.get('message', '未知错误')}")
            return None
    
    def get_image_dimensions(self):
        """获取当前图像尺寸 - 从ZED数据服务器获取"""
        # 这是一个占位方法，实际实现可能需要连接到ZED服务器
        # 或者从配置中获取固定值
        # 目前返回VGA分辨率的默认值
        return {"width": 640, "height": 480}
    
    def _get_bbox_status(self):
        """获取bbox_processor的状态信息"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((self.bbox_host, self.bbox_command_port))
                
                command = {"command": "get_status"}
                s.sendall(json.dumps(command).encode('utf-8'))
                
                response_data = s.recv(1024)
                if response_data:
                    response = json.loads(response_data.decode())
                    return response
                return None
                
        except Exception as e:
            logger.error(f"获取bbox_processor状态失败: {e}")
            return None

    def _parse_bbox_visual_response(self, response):
        """解析bbox_processor视觉信息响应"""
        if response.get("status") == "success":
            # bbox_processor返回bbox和xyz_coords
            bbox = response.get("bbox")
            xyz_coords = response.get("xyz_coords")
            
            if bbox and xyz_coords:
                # 计算bbox中心的像素坐标
                x1, y1, x2, y2 = bbox
                pixel_cx = (x1 + x2) / 2.0
                
                # 动态获取图像宽度
                image_dims = self.get_image_dimensions()
                image_width = image_dims["width"]
                
                return {
                    'pixel_cx': float(pixel_cx),
                    'image_width': int(image_width),
                    'bbox_available': True,
                    'depth_x': float(xyz_coords[0]) if xyz_coords else None,
                    'bbox': bbox,
                    'xyz_coords': xyz_coords
                }
            else:
                logger.error(f"bbox_processor响应缺少必要数据: {response}")
                return None
        elif response.get("status") == "no_detection":
            logger.info("视觉信息：bbox_processor报告当前无检测结果")
            return {'bbox_available': False, 'depth_x': None}
        else:
            logger.error(f"bbox_processor报告错误: {response.get('message', '未知错误')}")
            return None 