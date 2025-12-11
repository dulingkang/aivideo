#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 API 与 ModelManager 集成
"""

import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"
API_KEY = "test-key-123"

def test_model_manager_api():
    """测试 ModelManager API"""
    print("="*80)
    print("测试 ModelManager API 集成")
    print("="*80)
    
    # 测试 1: 健康检查
    print("\n1. 健康检查...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health")
        print(f"   ✅ 状态码: {response.status_code}")
        print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        print("   提示: 请先启动 API 服务: python gen_video/api/mvp_main.py")
        return
    
    # 测试 2: 获取模型状态
    print("\n2. 获取模型状态...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/models/status",
            headers={"X-API-Key": API_KEY}
        )
        if response.status_code == 200:
            print("   ✅ 模型状态:")
            models = response.json().get("models", {})
            for name, info in models.items():
                status = "✅" if info["exists"] else "❌"
                print(f"      {status} {name}: 存在={info['exists']}, 已加载={info['loaded']}")
        else:
            print(f"   ⚠️  状态码: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  失败: {e} (可能 API 不支持此端点)")
    
    # 测试 3: 使用 ModelManager 生成图像
    print("\n3. 测试 ModelManager 生成图像...")
    print("   任务: host_face (科普主持人脸)")
    print("   提示词: 一位温暖亲和的中国科普主持人，正面对镜头，专业形象")
    
    form_data = {
        "prompt": "一位温暖亲和的中国科普主持人，正面对镜头，专业形象",
        "task": "host_face",
        "use_model_manager": "true",
        "width": 1024,
        "height": 1024,
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/images/generate",
            data=form_data,
            headers={"X-API-Key": API_KEY},
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 生成成功!")
            print(f"   任务ID: {result.get('task_id')}")
            print(f"   图像URL: {result.get('image_url')}")
            if result.get('metadata'):
                print(f"   使用的模型: {result['metadata'].get('model_used', 'N/A')}")
                print(f"   任务类型: {result['metadata'].get('task', 'N/A')}")
        else:
            print(f"   ❌ 失败: {response.status_code}")
            print(f"   错误: {response.text}")
    except requests.exceptions.Timeout:
        print("   ⏳ 请求超时（生成可能需要较长时间）")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    print("\n提示:")
    print("1. 确保 API 服务已启动: python gen_video/api/mvp_main.py")
    print("2. 使用 ModelManager: 设置 use_model_manager=true 和 task 参数")
    print("3. 前端可以添加任务类型选择器")

if __name__ == "__main__":
    test_model_manager_api()

