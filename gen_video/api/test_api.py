#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API的简单脚本
用于验证API是否正常工作
"""
import requests
import time
import json

API_BASE_URL = "http://localhost:8000"
API_KEY = "test-token"  # 简化版，实际应该使用JWT

def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("🔍 测试健康检查...")
    response = requests.get(f"{API_BASE_URL}/api/v1/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.status_code == 200

def test_image_generation():
    """测试图像生成"""
    print("\n" + "=" * 60)
    print("🎨 测试图像生成...")
    
    payload = {
        "prompt": "xianxia fantasy, Han Li, calm cultivator, medium shot, front view",
        "width": 1536,
        "height": 864,
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/api/v1/images/generate",
        headers=headers,
        json=payload
    )
    
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"任务ID: {result['task_id']}")
        print(f"状态: {result['status']}")
        print(f"预计时间: {result['estimated_time']}秒")
        return result['task_id']
    else:
        print(f"错误: {response.text}")
        return None

def test_task_status(task_id: str):
    """测试任务状态查询"""
    print("\n" + "=" * 60)
    print(f"📊 查询任务状态: {task_id}")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    max_wait = 300  # 最多等待5分钟
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/tasks/{task_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"状态: {status['status']}, 进度: {status.get('progress', 0)}%")
            
            if status['status'] == 'completed':
                print(f"✅ 任务完成！")
                print(f"结果: {json.dumps(status.get('result'), indent=2, ensure_ascii=False)}")
                return True
            elif status['status'] == 'failed':
                print(f"❌ 任务失败: {status.get('error')}")
                return False
        
        time.sleep(2)
    
    print("⏰ 超时：任务未在5分钟内完成")
    return False

def main():
    print("🧪 API测试脚本")
    print("=" * 60)
    print(f"API地址: {API_BASE_URL}")
    print()
    
    # 1. 健康检查
    if not test_health():
        print("\n❌ 健康检查失败，API可能未启动")
        return
    
    # 2. 图像生成
    task_id = test_image_generation()
    if task_id:
        # 3. 查询任务状态
        test_task_status(task_id)
    else:
        print("\n❌ 图像生成任务提交失败")

if __name__ == "__main__":
    import sys
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ 连接失败：请确保API服务器已启动 (http://localhost:8000)")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试中断")
        sys.exit(1)

