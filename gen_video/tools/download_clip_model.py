#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 CLIP 模型到本地缓存

用于在离线环境中预先下载 CLIP 模型，避免运行时网络连接问题。

使用方法：
1. 直接运行（需要网络）：
   python3 tools/download_clip_model.py

2. 通过 proxychains4 运行（需要代理）：
   proxychains4 python3 tools/download_clip_model.py

3. 使用镜像站（推荐，速度快）：
   export HF_ENDPOINT=https://hf-mirror.com
   python3 tools/download_clip_model.py

4. 使用快速下载脚本（自动尝试多种方式）：
   ./tools/download_clip_fast.sh
"""

import os
import sys
from pathlib import Path
import shutil

# 设置 HuggingFace 缓存目录
# 优先使用项目配置的缓存目录
hf_home = os.environ.get("HF_HOME")
if not hf_home or not os.path.exists(hf_home):
    # 尝试使用项目配置的缓存目录
    hf_home = "/vepfs-dev/shawn/.cache/huggingface"
    if not os.path.exists(hf_home):
        # 回退到用户目录
        hf_home = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = hf_home

os.environ["TRANSFORMERS_CACHE"] = hf_home
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")

# 同时清理可能的其他缓存位置
other_cache_paths = [
    "/root/.cache/huggingface",
    os.path.expanduser("~/.cache/huggingface"),
    "/vepfs-dev/shawn/.cache/huggingface"
]

# 检查是否设置了镜像站
hf_endpoint = os.environ.get("HF_ENDPOINT", "")
if hf_endpoint:
    print(f"🌐 使用镜像站: {hf_endpoint}")

print(f"📦 HuggingFace 缓存目录: {hf_home}")
print(f"📥 开始下载 CLIP 模型: openai/clip-vit-large-patch14")
print(f"💡 提示：")
print(f"   - 如果下载慢，可以设置镜像站: export HF_ENDPOINT=https://hf-mirror.com")
print(f"   - 或使用快速下载脚本: ./tools/download_clip_fast.sh")
print(f"   - 或使用 proxychains4: proxychains4 python3 tools/download_clip_model.py")
print("")

model_id = "openai/clip-vit-large-patch14"
cache_path = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")

# 清理所有可能的缓存位置（包括 /root/.cache）
print(f"🔍 检查并清理所有可能的缓存位置...")
for cache_base in other_cache_paths:
    if cache_base and os.path.exists(cache_base):
        # 检查两种可能的路径结构
        for subpath in ["hub", ""]:
            if subpath:
                other_cache = os.path.join(cache_base, subpath, f"models--{model_id.replace('/', '--')}")
            else:
                other_cache = os.path.join(cache_base, f"models--{model_id.replace('/', '--')}")
            if os.path.exists(other_cache):
                print(f"   ⚠️  发现其他缓存位置: {other_cache}")
                try:
                    print(f"   🗑️  清理: {other_cache}")
                    shutil.rmtree(other_cache)
                    print(f"   ✅ 已清理")
                except Exception as e:
                    print(f"   ⚠️  清理失败: {e}")

# 检查并清理可能损坏的缓存
if os.path.exists(cache_path):
    print(f"⚠️  检测到现有缓存，检查是否损坏...")
    snapshots_path = os.path.join(cache_path, "snapshots")
    if os.path.exists(snapshots_path):
        # 检查 model.safetensors 文件
        for root, dirs, files in os.walk(snapshots_path):
            for file in files:
                if file == "model.safetensors":
                    file_path = os.path.join(root, file)
                    # 检查是否是符号链接
                    if os.path.islink(file_path):
                        real_path = os.readlink(file_path)
                        if not os.path.isabs(real_path):
                            real_path = os.path.join(os.path.dirname(file_path), real_path)
                        # 解析相对路径
                        if real_path.startswith("../../blobs/"):
                            blob_path = os.path.join(cache_path, "blobs", real_path.split("/")[-1])
                            if os.path.exists(blob_path):
                                file_size = os.path.getsize(blob_path)
                                # CLIP 模型文件应该大约 500MB，如果太小可能是损坏的
                                if file_size < 100 * 1024 * 1024:  # 小于 100MB
                                    print(f"   ⚠️  发现可能损坏的文件: {blob_path} (大小: {file_size / 1024 / 1024:.2f} MB)")
                                    print(f"   🗑️  删除损坏的文件...")
                                    try:
                                        os.remove(blob_path)
                                        # 删除符号链接
                                        os.remove(file_path)
                                    except Exception as e:
                                        print(f"   ⚠️  删除失败: {e}")

try:
    from transformers import CLIPTokenizer, CLIPTextModel
    
    # 如果设置了镜像站，显示信息
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"🌐 使用镜像站: {hf_endpoint}")
    
    print("1️⃣ 下载 CLIP Tokenizer...")
    print("   （如果网络不可用，请使用 proxychains4 或镜像站）")
    
    max_retries = 3
    tokenizer = None
    for attempt in range(max_retries):
        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_id)
            print(f"   ✓ Tokenizer 下载成功")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if not hf_endpoint and attempt == 0:
                    print(f"   💡 提示：可以设置镜像站加速: export HF_ENDPOINT=https://hf-mirror.com")
                print(f"   🔄 重试中...")
            else:
                raise
    
    print("2️⃣ 下载 CLIP Text Model...")
    print("   （这可能需要几分钟，请耐心等待...）")
    if hf_endpoint:
        print(f"   🌐 使用镜像站: {hf_endpoint}（应该会更快）")
    
    model = None
    for attempt in range(max_retries):
        try:
            model = CLIPTextModel.from_pretrained(model_id)
            print(f"   ✓ Model 下载成功")
            break
        except Exception as e:
            error_msg = str(e)
            is_corrupted = any(keyword in error_msg for keyword in [
                "invalid JSON", "EOF", "SafetensorError", "deserializing header"
            ])
            
            if attempt < max_retries - 1:
                if is_corrupted:
                    print(f"   ⚠️  检测到文件损坏错误 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}")
                    print(f"   🔄 清理损坏的缓存并重试...")
                    # 清理可能损坏的模型文件 - 删除所有缓存位置的整个缓存目录
                    print(f"   🗑️  清理所有缓存位置的损坏文件...")
                    for cache_base in [hf_home] + [p for p in other_cache_paths if p and p != hf_home]:
                        if not os.path.exists(cache_base):
                            continue
                        # 检查两种可能的路径结构
                        for subpath in ["hub", ""]:
                            if subpath:
                                cache_to_clean = os.path.join(cache_base, subpath, f"models--{model_id.replace('/', '--')}")
                            else:
                                cache_to_clean = os.path.join(cache_base, f"models--{model_id.replace('/', '--')}")
                            if os.path.exists(cache_to_clean):
                                try:
                                    print(f"   🗑️  删除缓存目录: {cache_to_clean}")
                                    shutil.rmtree(cache_to_clean)
                                    print(f"   ✅ 已清理: {cache_to_clean}")
                                except Exception as cleanup_error:
                                    print(f"   ⚠️  清理失败 {cache_to_clean}: {cleanup_error}")
                                    # 如果删除整个目录失败，尝试只删除 snapshots 和 blobs
                                    for subdir in ["snapshots", "blobs"]:
                                        subdir_path = os.path.join(cache_to_clean, subdir)
                                        if os.path.exists(subdir_path):
                                            try:
                                                shutil.rmtree(subdir_path)
                                                print(f"   ✅ {subdir} 目录已清理")
                                            except Exception:
                                                pass
                        # 也清理 locks
                        for lock_subpath in ["hub/.locks", ".locks"]:
                            lock_path = os.path.join(cache_base, lock_subpath, f"models--{model_id.replace('/', '--')}")
                            if os.path.exists(lock_path):
                                try:
                                    shutil.rmtree(lock_path)
                                    print(f"   ✅ 锁文件已清理: {lock_path}")
                                except Exception:
                                    pass
                else:
                    print(f"   ⚠️  下载失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}")
                print(f"   🔄 重试中...")
            else:
                raise
    
    # 验证缓存路径
    if os.path.exists(cache_path):
        print(f"\n✅ CLIP 模型已下载到缓存: {cache_path}")
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_path)
            for filename in filenames
        )
        print(f"   文件大小: {total_size / 1024 / 1024:.2f} MB")
    else:
        print(f"\n⚠️  缓存路径不存在，但模型已下载")
    
    print("\n✅ 下载完成！现在可以在离线环境中使用 CLIP 模型了。")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n💡 建议：")
    print(f"   1. 检查网络连接和代理设置")
    print(f"   2. 如果使用代理，确保 proxychains4 配置正确")
    print(f"   3. 如果文件损坏，可以手动删除缓存目录后重试：")
    print(f"      rm -rf {cache_path}")
    sys.exit(1)
