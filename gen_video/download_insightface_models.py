#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 InsightFace 模型文件
支持使用代理下载
"""

import os
import sys
from pathlib import Path
import subprocess

def download_insightface_models(use_proxy=False):
    """
    下载 InsightFace 模型文件
    
    Args:
        use_proxy: 是否使用 proxychains4 代理
    """
    print("=" * 60)
    print("📥 下载 InsightFace 模型文件")
    print("=" * 60)
    
    # 模型下载信息
    models = {
        "antelopev2": {
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
            "size": "约 500MB",
            "description": "InsightFace AntelopeV2 模型（推荐用于 InstantID）"
        }
    }
    
    print("\n📋 需要下载的模型：")
    for name, info in models.items():
        print(f"  - {name}: {info['description']}")
        print(f"    大小: {info['size']}")
        print(f"    URL: {info['url']}")
    
    print("\n💡 下载方式：")
    print("  1. 自动下载（需要网络连接）")
    print("  2. 手动下载（使用 proxychains4 代理）")
    print("  3. 从其他位置复制模型文件")
    
    # 方法1: 使用 Python 直接下载（如果网络可用）
    if not use_proxy:
        try:
            print("\n🔄 尝试自动下载...")
            import insightface
            # 尝试初始化，会自动下载模型
            app = insightface.app.FaceAnalysis(name='antelopev2')
            app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ 模型下载成功！")
            return True
        except Exception as e:
            print(f"❌ 自动下载失败: {e}")
            print("💡 建议使用 proxychains4 代理下载")
    
    # 方法2: 使用 proxychains4 下载
    if use_proxy:
        print("\n🔄 使用 proxychains4 下载...")
        try:
            # 使用 wget 通过代理下载
            download_dir = Path.home() / ".insightface" / "models"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            url = models["antelopev2"]["url"]
            zip_path = download_dir / "antelopev2.zip"
            
            cmd = ["proxychains4", "wget", "-O", str(zip_path), url]
            print(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 下载成功！")
                # 解压
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                print("✅ 解压完成！")
                return True
            else:
                print(f"❌ 下载失败: {result.stderr}")
        except Exception as e:
            print(f"❌ 代理下载失败: {e}")
    
    # 方法3: 手动下载说明
    print("\n📝 手动下载说明：")
    print("  1. 下载模型文件：")
    print(f"     URL: {models['antelopev2']['url']}")
    print("  2. 解压到目录：")
    print("     ~/.insightface/models/antelopev2/")
    print("  3. 或者解压到项目目录：")
    print("     gen_video/models/instantid/antelopev2/")
    
    return False

if __name__ == "__main__":
    use_proxy = "--proxy" in sys.argv or "-p" in sys.argv
    
    if use_proxy:
        print("使用 proxychains4 代理下载")
    else:
        print("尝试自动下载（不使用代理）")
    
    success = download_insightface_models(use_proxy=use_proxy)
    
    if success:
        print("\n✅ 模型下载完成！")
        print("现在可以使用 InstantID 了")
    else:
        print("\n⚠️  模型下载失败")
        print("请手动下载模型文件，或使用 proxychains4 代理")


