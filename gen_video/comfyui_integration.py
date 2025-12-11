#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI 集成模块
通过 API 调用 ComfyUI 生成视频
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image


class ComfyUIAPI:
    """ComfyUI API 客户端"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        """
        初始化 ComfyUI API 客户端
        
        Args:
            server_url: ComfyUI 服务器地址
        """
        self.server_url = server_url
        self.client_id = str(time.time())
    
    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """
        提交任务到 ComfyUI 队列
        
        Args:
            prompt: ComfyUI 工作流 JSON
        
        Returns:
            任务 ID
        """
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        response = requests.post(
            f"{self.server_url}/prompt",
            data=data
        )
        return response.json()["prompt_id"]
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        获取任务历史
        
        Args:
            prompt_id: 任务 ID
        
        Returns:
            任务历史
        """
        response = requests.get(f"{self.server_url}/history/{prompt_id}")
        return response.json()
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Image.Image:
        """
        从 ComfyUI 获取生成的图像
        
        Args:
            filename: 文件名
            subfolder: 子文件夹
            folder_type: 文件夹类型（output/input/temp）
        
        Returns:
            PIL Image
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = requests.get(f"{self.server_url}/view", params=data)
        return Image.open(BytesIO(response.content))
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> bool:
        """
        等待任务完成
        
        Args:
            prompt_id: 任务 ID
            timeout: 超时时间（秒）
        
        Returns:
            是否成功完成
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return True
            time.sleep(1)
        return False
    
    def create_animatediff_workflow(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 16,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, Any]:
        """
        创建 AnimateDiff 工作流
        
        Args:
            image_path: 输入图像路径
            prompt: 提示词
            negative_prompt: 负面提示词
            num_frames: 帧数
            fps: 帧率
            width: 宽度
            height: 高度
        
        Returns:
            ComfyUI 工作流 JSON
        """
        # 这里需要根据实际的 ComfyUI AnimateDiff 工作流结构来构建
        # 这是一个示例，实际结构需要根据 ComfyUI-AnimateDiff-Evolved 的节点来调整
        workflow = {
            "1": {
                "inputs": {
                    "image": image_path,
                },
                "class_type": "LoadImage"
            },
            # ... 更多节点
        }
        return workflow


def test_comfyui_connection(server_url: str = "http://127.0.0.1:8188") -> bool:
    """
    测试 ComfyUI 连接
    
    Args:
        server_url: ComfyUI 服务器地址
    
    Returns:
        是否连接成功
    """
    try:
        response = requests.get(f"{server_url}/system_stats", timeout=5)
        return response.status_code == 200
    except:
        return False

