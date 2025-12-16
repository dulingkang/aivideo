#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ComfyUI AnimateDiff 图生视频
"""

import json
import requests
import time
import sys
from pathlib import Path

# 添加路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen_video.comfyui_integration import ComfyUIAPI, test_comfyui_connection

def create_simple_animatediff_workflow(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 16,
    width: int = 512,
    height: int = 512,
) -> dict:
    """
    创建简单的 AnimateDiff 工作流（图生视频）
    
    注意：这是一个基础示例，实际工作流可能需要更多节点
    """
    # 将图像路径转换为绝对路径
    image_path = str(Path(image_path).resolve())
    
    workflow = {
        "1": {
            "inputs": {
                "image": image_path,
            },
            "class_type": "LoadImage"
        },
        "2": {
            "inputs": {
                "text": prompt,
                "clip": ["3", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 0]
            },
            "class_type": "CLIPTextEncode"
        },
        # ... 需要更多节点来构建完整的 AnimateDiff 工作流
    }
    
    return workflow

def test_comfyui_animatediff():
    """测试 ComfyUI AnimateDiff"""
    print("=" * 60)
    print("ComfyUI AnimateDiff 测试")
    print("=" * 60)
    
    # 测试连接
    print("\n[1] 测试 ComfyUI 连接...")
    if not test_comfyui_connection():
        print("✗ ComfyUI 服务器未运行，请先启动服务器")
        print("  启动命令: bash gen_video/启动ComfyUI服务器.sh")
        return False
    
    print("✓ ComfyUI 服务器连接成功")
    
    # 检查是否有测试图像
    print("\n[2] 检查测试图像...")
    test_image = Path("/vepfs-dev/shawn/vid/fanren/gen_video/outputs/images/test_scenes/scene_001.png")
    if not test_image.exists():
        print(f"⚠ 测试图像不存在: {test_image}")
        print("  ℹ 请先生成测试图像，或使用其他图像路径")
        return False
    
    print(f"✓ 找到测试图像: {test_image}")
    
    # 获取节点信息
    print("\n[3] 获取 ComfyUI 节点信息...")
    try:
        api = ComfyUIAPI()
        response = requests.get("http://127.0.0.1:8188/object_info", timeout=10)
        if response.status_code == 200:
            node_info = response.json()
            print("✓ 节点信息获取成功")
            
            # 检查 AnimateDiff 相关节点
            animatediff_nodes = []
            for node_type in node_info.get("nodes", {}):
                if "AnimateDiff" in node_type or "animatediff" in node_type.lower():
                    animatediff_nodes.append(node_type)
            
            if animatediff_nodes:
                print(f"✓ 找到 {len(animatediff_nodes)} 个 AnimateDiff 相关节点:")
                for node in animatediff_nodes[:5]:  # 只显示前5个
                    print(f"  - {node}")
            else:
                print("⚠ 未找到 AnimateDiff 节点（可能需要重启服务器）")
        else:
            print(f"⚠ 获取节点信息失败: {response.status_code}")
    except Exception as e:
        print(f"⚠ 获取节点信息出错: {e}")
    
    print("\n" + "=" * 60)
    print("✓ ComfyUI AnimateDiff 测试完成")
    print("\n提示：")
    print("1. 访问 Web UI: http://127.0.0.1:8188")
    print("2. 在 Web UI 中手动构建 AnimateDiff 工作流")
    print("3. 或使用 API 调用（需要构建完整的工作流 JSON）")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_comfyui_animatediff()

