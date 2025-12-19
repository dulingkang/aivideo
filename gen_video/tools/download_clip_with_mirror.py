#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用镜像源下载 CLIP 模型（加速下载）

支持多种镜像源：
1. ModelScope（魔搭社区，国内推荐）
2. HuggingFace 镜像站
3. 直接下载（如果网络可用）
"""

import os
import sys
from pathlib import Path

# 设置 HuggingFace 缓存目录
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
if not os.path.exists(hf_home):
    hf_home = "/vepfs-dev/shawn/.cache/huggingface"
    os.environ["HF_HOME"] = hf_home

os.environ["TRANSFORMERS_CACHE"] = hf_home
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")

print(f"📦 HuggingFace 缓存目录: {hf_home}")
print(f"📥 开始下载 CLIP 模型: openai/clip-vit-large-patch14")
print("")

model_id = "openai/clip-vit-large-patch14"

# 方法 1: 尝试使用 ModelScope（国内镜像，速度快）
def download_with_modelscope():
    """使用 ModelScope 下载"""
    try:
        print("🔄 方法 1: 尝试使用 ModelScope（魔搭社区）...")
        from modelscope import snapshot_download
        
        # ModelScope 上的 CLIP 模型 ID
        modelscope_id = "AI-ModelScope/clip-vit-large-patch14"
        print(f"   下载地址: {modelscope_id}")
        
        # 下载到 HuggingFace 缓存目录
        cache_dir = os.path.join(hf_home, "hub")
        model_dir = snapshot_download(
            modelscope_id,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        print(f"   ✅ ModelScope 下载成功: {model_dir}")
        return True
    except ImportError:
        print("   ⚠️  ModelScope 未安装，跳过")
        print("   💡 安装命令: pip install modelscope")
        return False
    except Exception as e:
        print(f"   ⚠️  ModelScope 下载失败: {e}")
        return False

# 方法 2: 使用 HuggingFace 镜像站
def download_with_hf_mirror():
    """使用 HuggingFace 镜像站下载"""
    try:
        print("🔄 方法 2: 尝试使用 HuggingFace 镜像站...")
        
        # 设置镜像站环境变量
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        from transformers import CLIPTokenizer, CLIPTextModel
        
        print("   1️⃣ 下载 CLIP Tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        print(f"   ✓ Tokenizer 下载成功")
        
        print("   2️⃣ 下载 CLIP Text Model...")
        model = CLIPTextModel.from_pretrained(model_id)
        print(f"   ✓ Model 下载成功")
        
        return True
    except Exception as e:
        print(f"   ⚠️  镜像站下载失败: {e}")
        return False

# 方法 3: 直接下载（使用 proxychains4）
def download_direct():
    """直接下载（需要代理）"""
    try:
        print("🔄 方法 3: 直接下载（需要代理）...")
        from transformers import CLIPTokenizer, CLIPTextModel
        
        print("   1️⃣ 下载 CLIP Tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        print(f"   ✓ Tokenizer 下载成功")
        
        print("   2️⃣ 下载 CLIP Text Model...")
        print("   （这可能需要几分钟，请耐心等待...）")
        model = CLIPTextModel.from_pretrained(model_id)
        print(f"   ✓ Model 下载成功")
        
        return True
    except Exception as e:
        print(f"   ⚠️  直接下载失败: {e}")
        return False

# 主流程
print("=" * 60)
print("🚀 开始尝试多种下载方式...")
print("=" * 60)
print("")

success = False

# 尝试方法 1: ModelScope
if not success:
    success = download_with_modelscope()

# 尝试方法 2: HuggingFace 镜像站
if not success:
    success = download_with_hf_mirror()

# 尝试方法 3: 直接下载
if not success:
    success = download_direct()

if success:
    # 验证缓存路径
    cache_path = os.path.join(hf_home, "hub", "models--openai--clip-vit-large-patch14")
    if os.path.exists(cache_path):
        print(f"\n✅ CLIP 模型已下载到缓存: {cache_path}")
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_path)
            for filename in filenames
        )
        print(f"   文件大小: {total_size / 1024 / 1024:.2f} MB")
        print("\n✅ 下载完成！现在可以在离线环境中使用 CLIP 模型了。")
    else:
        print("\n⚠️  缓存路径不存在，但模型已下载")
else:
    print("\n❌ 所有下载方法都失败了")
    print("\n💡 建议：")
    print("   1. 安装 ModelScope: pip install modelscope")
    print("   2. 或使用 proxychains4 运行: proxychains4 python3 tools/download_clip_model.py")
    print("   3. 或手动从百度网盘下载后放到缓存目录")
    sys.exit(1)

