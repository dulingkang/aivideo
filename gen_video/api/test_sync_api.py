#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试同步API的脚本（不依赖Redis）
"""
import requests
import time
import json

API_BASE_URL = "http://localhost:8000"
API_KEY = "test-token"

def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("🔍 测试健康检查...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"模式: {result.get('mode', 'unknown')}")
        print(f"状态: {result.get('status', 'unknown')}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败：请确保API服务器已启动")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def test_image_generation():
    """测试图像生成（同步模式）"""
    print("\n" + "=" * 60)
    print("🎨 测试图像生成（同步模式）...")
    print("⚠️  注意：同步模式会等待生成完成，可能需要30-60秒")
    print()
    
    payload = {
        "prompt": "xianxia fantasy, Han Li, calm cultivator, medium shot, front view, facing camera",
        "width": 1536,
        "height": 864,
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"📤 提交生成请求...")
    print(f"   提示词: {payload['prompt'][:50]}...")
    print(f"   分辨率: {payload['width']}x{payload['height']}")
    print()
    print("⏳ 等待生成中（这可能需要30-60秒）...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/images/generate",
            headers=headers,
            json=payload,
            timeout=300  # 5分钟超时
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ 请求完成（耗时: {elapsed_time:.1f}秒）")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 生成结果:")
            print(f"   任务ID: {result['task_id']}")
            print(f"   状态: {result['status']}")
            print(f"   图像路径: {result['image_path']}")
            print(f"   分辨率: {result['width']}x{result['height']}")
            print(f"   文件大小: {result.get('file_size', 0) / 1024:.1f} KB")
            
            return result
        else:
            print(f"❌ 错误: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"\n❌ 超时：生成时间超过5分钟")
        return None
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return None

def main():
    print("🧪 API测试脚本（同步模式，不依赖Redis）")
    print("=" * 60)
    print(f"API地址: {API_BASE_URL}")
    print()
    
    # 1. 健康检查
    if not test_health():
        print("\n❌ 健康检查失败")
        print("\n💡 提示：")
        print("   1. 确保API服务器已启动")
        print("   2. 使用同步模式: python gen_video/api/main_sync.py")
        return
    
    # 2. 图像生成
    print("\n" + "=" * 60)
    print("准备测试图像生成...")
    print("=" * 60)
    print("\n⚠️  警告：同步模式会阻塞等待生成完成")
    print("   图像生成可能需要30-60秒，请耐心等待")
    print("\n是否继续？(按Enter继续，Ctrl+C取消)")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试取消")
        return
    
    result = test_image_generation()
    
    if result:
        print("\n" + "=" * 60)
        print("✅ 测试完成！")
        print("=" * 60)
        print(f"\n📁 生成的图像: {result['image_path']}")
        print("\n💡 提示：")
        print("   1. 可以查看生成的图像文件")
        print("   2. 如果需要异步模式，需要安装Redis")
        print("   3. 然后使用 main.py（异步模式）")
    else:
        print("\n❌ 测试失败")
        print("\n💡 可能的原因：")
        print("   1. 生成器初始化失败")
        print("   2. GPU不可用或内存不足")
        print("   3. 模型文件缺失")
        print("\n请检查日志以获取详细错误信息")

if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试中断")
        sys.exit(1)

