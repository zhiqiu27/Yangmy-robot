# robot_communication.py
"""
机器人通信模块
处理与图像服务器和方向命令的通信
"""

import socket
import json
import logging
import time
from robot_config import IMAGE_SERVER_HOST, IMAGE_SERVER_PORT, DIRECTION_COMMAND_PORT, NEXT_TARGET_PORT

logger = logging.getLogger(__name__)

class RobotCommunication:
    """处理机器人与外部服务的通信"""
    
    def __init__(self):
        self.image_server_host = IMAGE_SERVER_HOST
        self.image_server_port = IMAGE_SERVER_PORT
        self.direction_port = DIRECTION_COMMAND_PORT
        self.next_target_port = NEXT_TARGET_PORT
    
    def request_visual_info(self):
        """从图像服务器请求视觉信息"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((self.image_server_host, self.image_server_port))
                logger.debug("已连接到图像服务器 (请求视觉信息)")
                
                s.sendall(b"REQUEST_VISUAL_INFO")
                logger.debug("已发送 'REQUEST_VISUAL_INFO' 命令")
                
                response_data = s.recv(1024)
                if not response_data:
                    logger.error("从图像服务器收到空响应 (视觉信息)")
                    return None
                
                response = json.loads(response_data.decode())
                logger.debug(f"从图像服务器收到响应 (视觉信息): {response}")
                
                return self._parse_visual_response(response)
                
        except socket.timeout:
            logger.warning(f"连接图像服务器超时")
            return None
        except socket.error as e:
            logger.error(f"与图像服务器通信时发生套接字错误: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON响应时出错: {e}")
            return None
        except Exception as e:
            logger.error(f"请求视觉信息时发生意外错误: {e}")
            return None
    
    def request_calibration(self):
        """从图像服务器请求校准数据"""
        logger.info(f"向图像服务器请求校准数据...")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((self.image_server_host, self.image_server_port))
                logger.info("已连接到图像服务器")
                
                s.sendall(b"REQUEST_CALIBRATION")
                logger.info("已发送 'REQUEST_CALIBRATION' 命令")
                
                response_data = s.recv(1024)
                if not response_data:
                    logger.error("从图像服务器收到空响应")
                    return None
                
                response = json.loads(response_data.decode())
                logger.info(f"从图像服务器收到响应: {response}")
                
                return self._parse_calibration_response(response)
                
        except socket.timeout:
            logger.error(f"连接图像服务器或接收数据超时")
            return None
        except socket.error as e:
            logger.error(f"与图像服务器通信时发生套接字错误: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON响应时出错: {e}")
            return None
        except Exception as e:
            logger.error(f"请求校准数据时发生意外错误: {e}")
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
        """发送NEXT_TARGET命令到PC端"""
        logger.info(f"向PC端发送NEXT_TARGET命令 (端口 {self.next_target_port})")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((self.image_server_host, self.next_target_port))
                logger.info("已连接到PC端NEXT_TARGET服务")
                
                s.sendall(b"NEXT_TARGET")
                logger.info("已发送 'NEXT_TARGET' 命令")
                
                # 等待确认响应
                response_data = s.recv(1024)
                if response_data:
                    response = response_data.decode().strip()
                    logger.info(f"PC端响应: {response}")
                    return response == "ACK"
                else:
                    logger.error("PC端无响应")
                    return False
                    
        except socket.timeout:
            logger.error(f"连接PC端NEXT_TARGET服务超时")
            return False
        except socket.error as e:
            logger.error(f"与PC端NEXT_TARGET服务通信时发生套接字错误: {e}")
            return False
        except Exception as e:
            logger.error(f"发送NEXT_TARGET命令时发生意外错误: {e}")
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