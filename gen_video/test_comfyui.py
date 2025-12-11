#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ComfyUI 连接和基本功能
"""

import sys
import time
import requests
from pathlib import Path

def test_comfyui_connection(server_url: str = "http://127.0.0.1:8188", timeout: int = 5):
    """测试 ComfyUI 连接"""
    try:
        response = requests.get(f"{server_url}/system_stats", timeout=timeout)
        if response.status_code == 200:
            print("✓ ComfyUI 服务器连接成功")
            stats = response.json()
            print(f"  - 系统信息: {stats}")
            return True
        else:
            print(f"✗ ComfyUI 服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到 ComfyUI 服务器")
        print("  ℹ 请先启动 ComfyUI 服务器:")
        print("    cd /vepfs-dev/shawn/vid/fanren/ComfyUI")
        print("    source /vepfs-dev/shawn/venv/py312/bin/activate")
        print("    python main.py --port 8188")
        return False
    except Exception as e:
        print(f"✗ 连接测试失败: {e}")
        return False

def test_animatediff_models():
    """测试 AnimateDiff 模型是否存在"""
    models_path = Path("/vepfs-dev/shawn/vid/fanren/ComfyUI/models/animatediff_models")
    if models_path.exists():
        files = list(models_path.glob("*.safetensors"))
        if files:
            print(f"✓ 找到 {len(files)} 个 AnimateDiff 模型文件")
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
            return True
        else:
            print("⚠ AnimateDiff 模型目录存在，但未找到模型文件")
            return False
    else:
        print("⚠ AnimateDiff 模型目录不存在")
        return False

def test_comfyui_nodes():
    """测试 ComfyUI 节点（通过 API）"""
    server_url = "http://127.0.0.1:8188"
    try:
        response = requests.get(f"{server_url}/object_info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print("✓ ComfyUI 节点信息获取成功")
            
            # 检查 AnimateDiff 相关节点
            if "AnimateDiff" in str(info):
                print("  ✓ 检测到 AnimateDiff 相关节点")
            else:
                print("  ⚠ 未检测到 AnimateDiff 相关节点（可能需要重启服务器）")
            
            return True
        else:
            print(f"✗ 获取节点信息失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 节点测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ComfyUI 测试")
    print("=" * 60)
    
    print("\n[1] 测试 ComfyUI 服务器连接...")
    connected = test_comfyui_connection()
    
    if connected:
        print("\n[2] 测试 ComfyUI 节点...")
        test_comfyui_nodes()
    
    print("\n[3] 检查 AnimateDiff 模型...")
    test_animatediff_models()
    
    print("\n" + "=" * 60)
    if connected:
        print("✓ ComfyUI 测试完成，服务器运行正常")
        print("\n访问 Web UI: http://127.0.0.1:8188")
    else:
        print("⚠ 请先启动 ComfyUI 服务器")
    print("=" * 60)

