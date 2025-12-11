#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP API 测试脚本
"""

import requests
import json
import time
from pathlib import Path

# API配置
BASE_URL = "http://localhost:8000"
API_KEY = "test-key-123"  # 免费版测试Key

def test_health():
    """测试健康检查"""
    print("🔍 测试健康检查...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_quota():
    """测试配额查询"""
    print("\n🔍 测试配额查询...")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/v1/quota", headers=headers)
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_image_generation():
    """测试图像生成"""
    print("\n🎨 测试图像生成...")
    
    headers = {"X-API-Key": API_KEY}
    
    payload = {
        "prompt": "一个美丽的风景，山峦起伏，云雾缭绕，阳光透过云层",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
        "negative_prompt": "模糊，低质量",
    }
    
    print(f"   提示词: {payload['prompt']}")
    print("   正在生成（可能需要30-60秒）...")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/v1/images/generate",
        headers=headers,
        json=payload
    )
    elapsed_time = time.time() - start_time
    
    print(f"   状态码: {response.status_code}")
    print(f"   耗时: {elapsed_time:.1f}秒")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ 生成成功!")
        print(f"   任务ID: {result['task_id']}")
        print(f"   图像URL: {result['image_url']}")
        print(f"   文件大小: {result['file_size'] / 1024:.1f} KB")
        print(f"   剩余配额: {result['quota_remaining']}")
        
        # 尝试下载图像
        if result.get('image_url'):
            image_url = f"{BASE_URL}{result['image_url']}"
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                output_path = Path("test_output.png")
                output_path.write_bytes(img_response.content)
                print(f"   💾 图像已保存到: {output_path}")
        
        return result['task_id']
    else:
        print(f"   ❌ 生成失败: {response.text}")
        return None

def test_multiple_images():
    """测试多次生成（测试配额）"""
    print("\n🎨 测试多次图像生成（测试配额限制）...")
    
    headers = {"X-API-Key": API_KEY}
    
    for i in range(3):
        print(f"\n   第 {i+1} 次生成...")
        payload = {
            "prompt": f"测试图像 {i+1}，简洁的抽象艺术",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,  # 减少步数以加快速度
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/images/generate",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 成功，剩余配额: {result['quota_remaining']}")
        elif response.status_code == 429:
            print(f"   ⚠️  配额已用完: {response.json()['detail']}")
            break
        else:
            print(f"   ❌ 失败: {response.text}")

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 MVP API 测试")
    print("=" * 60)
    
    # 测试健康检查
    if not test_health():
        print("\n❌ 健康检查失败，请确保API服务正在运行")
        print("   启动命令: python gen_video/api/mvp_main.py")
        return
    
    # 测试配额查询
    test_quota()
    
    # 测试图像生成
    task_id = test_image_generation()
    
    # 测试多次生成
    # test_multiple_images()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
    print("\n💡 提示:")
    print("   - 查看API文档: http://localhost:8000/docs")
    print("   - 测试API Key: test-key-123 (免费版)")
    print("   - 演示API Key: demo-key-456 (付费版)")

if __name__ == "__main__":
    main()

