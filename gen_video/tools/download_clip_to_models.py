#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 CLIP 模型到 models 目录

将 openai/clip-vit-large-patch14 下载到 models/clip/openai-clip-vit-large-patch14

支持镜像站加速：
export HF_ENDPOINT=https://hf-mirror.com
python3 tools/download_clip_to_models.py
"""

import os
import sys
from pathlib import Path
import shutil
import json

# 项目根目录
project_root = Path(__file__).parent.parent
models_dir = project_root / "models" / "clip" / "openai-clip-vit-large-patch14"

# 如果目录已存在，询问是否覆盖
if models_dir.exists():
    print(f"⚠️  目标目录已存在: {models_dir}")
    print(f"   是否覆盖？(y/n): ", end="")
    response = input().strip().lower()
    if response != 'y':
        print("   取消下载")
        sys.exit(0)
    print(f"   🗑️  删除旧目录...")
    shutil.rmtree(models_dir)

models_dir.mkdir(parents=True, exist_ok=True)

# 检查是否设置了镜像站
hf_endpoint = os.environ.get("HF_ENDPOINT", "")
if hf_endpoint:
    print(f"🌐 使用镜像站: {hf_endpoint}")

print(f"📦 目标目录: {models_dir}")
print(f"📥 开始下载 CLIP 模型: openai/clip-vit-large-patch14")
print(f"💡 提示：")
print(f"   - 如果下载慢，可以设置镜像站: export HF_ENDPOINT=https://hf-mirror.com")
print(f"   - 或使用 proxychains4: proxychains4 python3 tools/download_clip_to_models.py")
print("")

model_id = "openai/clip-vit-large-patch14"

try:
    from transformers import CLIPTokenizer, CLIPTextModel
    
    print("1️⃣ 下载 CLIP Tokenizer...")
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
        print(f"   ⚠️  注意：如果镜像站文件有问题，下载后可能损坏")
    
    model = None
    for attempt in range(max_retries):
        try:
            # 使用 resume_download=True 支持断点续传
            model = CLIPTextModel.from_pretrained(
                model_id,
                resume_download=True  # 如果文件已存在但损坏，会重新下载
            )
            print(f"   ✓ Model 下载成功")
            break
        except Exception as e:
            error_str = str(e)
            # 检查是否是 safetensors 文件损坏
            is_corrupted = any(keyword in error_str for keyword in [
                "SafetensorError", "invalid JSON", "EOF", "deserializing header"
            ])
            
            if attempt < max_retries - 1:
                print(f"   ⚠️  下载失败 (尝试 {attempt + 1}/{max_retries}): {error_str[:100]}")
                
                if is_corrupted:
                    print(f"   🔍 检测到文件损坏，清理缓存后重试...")
                    # 清理 HuggingFace 缓存中的损坏文件
                    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                    if not os.path.exists(hf_home):
                        hf_home = "/vepfs-dev/shawn/.cache/huggingface"
                    
                    # 所有可能的缓存路径
                    cache_bases = [hf_home]
                    if os.path.exists(os.path.expanduser("~/.cache/huggingface")):
                        cache_bases.append(os.path.expanduser("~/.cache/huggingface"))
                    if os.path.exists("/root/.cache/huggingface"):
                        cache_bases.append("/root/.cache/huggingface")
                    
                    cache_name = f"models--{model_id.replace('/', '--')}"
                    
                    for cache_base in cache_bases:
                        if not os.path.exists(cache_base):
                            continue
                        
                        # 检查两种可能的路径结构
                        for subpath in ["hub", ""]:
                            if subpath:
                                cache_path = os.path.join(cache_base, subpath, cache_name)
                            else:
                                cache_path = os.path.join(cache_base, cache_name)
                            
                            if os.path.exists(cache_path):
                                print(f"      🗑️  清理缓存目录: {cache_path}")
                                try:
                                    # 先尝试删除整个目录
                                    shutil.rmtree(cache_path)
                                    print(f"      ✅ 已删除: {cache_path}")
                                except Exception as cleanup_error:
                                    # 如果删除整个目录失败，尝试只删除 snapshots 和 blobs
                                    print(f"      ⚠️  删除整个目录失败，尝试删除子目录...")
                                    for subdir in ["snapshots", "blobs"]:
                                        subdir_path = os.path.join(cache_path, subdir)
                                        if os.path.exists(subdir_path):
                                            try:
                                                shutil.rmtree(subdir_path)
                                                print(f"      ✅ {subdir} 目录已清理")
                                            except Exception:
                                                pass
                                    
                                    # 也尝试删除损坏的 safetensors 文件
                                    for root, dirs, files in os.walk(cache_path):
                                        for file in files:
                                            if file.endswith(".safetensors"):
                                                file_path = os.path.join(root, file)
                                                try:
                                                    # 验证文件完整性
                                                    with open(file_path, "rb") as f:
                                                        header_len_bytes = f.read(8)
                                                        if len(header_len_bytes) < 8:
                                                            raise ValueError("文件头不完整")
                                                        header_len = int.from_bytes(header_len_bytes, "little")
                                                        if header_len <= 0 or header_len > 1024 * 1024:  # 限制最大 1MB
                                                            raise ValueError("文件头长度异常")
                                                        header_json = f.read(header_len).decode("utf-8")
                                                        json.loads(header_json)  # 验证 JSON
                                                except Exception:
                                                    # 文件损坏，删除它
                                                    print(f"      🗑️  删除损坏文件: {file_path}")
                                                    try:
                                                        os.remove(file_path)
                                                    except:
                                                        pass
                        
                        # 也清理 locks
                        for lock_subpath in ["hub/.locks", ".locks"]:
                            lock_path = os.path.join(cache_base, lock_subpath, cache_name)
                            if os.path.exists(lock_path):
                                try:
                                    shutil.rmtree(lock_path)
                                    print(f"      ✅ 锁文件已清理: {lock_path}")
                                except Exception:
                                    pass
                    
                    # 清理 transformers 缓存
                    try:
                        import torch
                        if hasattr(torch, 'cuda'):
                            torch.cuda.empty_cache()
                    except:
                        pass
                    
                    print(f"   🔄 清理完成，重新下载...")
                else:
                    print(f"   🔄 重试中...")
            else:
                raise
    
    print("3️⃣ 验证下载的模型文件完整性...")
    # 验证模型文件是否完整
    try:
        # 尝试重新加载验证
        from transformers import CLIPTextModel
        test_model = CLIPTextModel.from_pretrained(model_id, local_files_only=True)
        print(f"   ✓ 模型文件完整性验证通过")
        del test_model
    except Exception as verify_error:
        error_str = str(verify_error)
        if any(keyword in error_str for keyword in ["SafetensorError", "invalid JSON", "EOF", "deserializing header"]):
            print(f"   ⚠️  模型文件损坏！错误: {verify_error}")
            print(f"   💡 可能原因：")
            print(f"      1. 镜像站文件本身有问题")
            print(f"      2. 下载过程中网络中断")
            print(f"      3. 磁盘空间不足导致写入不完整")
            print(f"   🔄 清理缓存并重新下载（不使用镜像站）...")
            
            # 清理缓存
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            if not os.path.exists(hf_home):
                hf_home = "/vepfs-dev/shawn/.cache/huggingface"
            
            cache_name = f"models--{model_id.replace('/', '--')}"
            cache_path = os.path.join(hf_home, "hub", cache_name)
            if os.path.exists(cache_path):
                print(f"      🗑️  清理缓存: {cache_path}")
                try:
                    shutil.rmtree(cache_path)
                except:
                    pass
            
            # 尝试不使用镜像站重新下载（使用断点续传）
            if hf_endpoint:
                print(f"   🔄 禁用镜像站，使用官方源重新下载（断点续传）...")
                os.environ.pop("HF_ENDPOINT", None)
                model = CLIPTextModel.from_pretrained(
                    model_id,
                    resume_download=True  # 断点续传，只下载缺失或损坏的部分
                )
                print(f"   ✓ 使用官方源下载成功")
            else:
                raise
        else:
            raise
    
    print("4️⃣ 保存到 models 目录...")
    # 保存 tokenizer
    print("   保存 Tokenizer...")
    tokenizer.save_pretrained(str(models_dir))
    print(f"   ✓ Tokenizer 已保存")
    
    # 保存 model
    print("   保存 Model...")
    model.save_pretrained(str(models_dir))
    print(f"   ✓ Model 已保存")
    
    # 验证保存的文件
    print("5️⃣ 验证保存的文件完整性...")
    try:
        # 验证 safetensors 文件
        safetensors_files = list(models_dir.glob("*.safetensors"))
        for safetensors_file in safetensors_files:
            print(f"   验证: {safetensors_file.name}...")
            with open(safetensors_file, "rb") as f:
                # 读取文件头长度（8字节）
                header_len_bytes = f.read(8)
                if len(header_len_bytes) < 8:
                    raise ValueError(f"文件头不完整: {safetensors_file}")
                
                header_len = int.from_bytes(header_len_bytes, "little")
                if header_len <= 0 or header_len > 10 * 1024 * 1024:  # 限制最大 10MB
                    raise ValueError(f"文件头长度异常: {header_len} bytes")
                
                # 读取并验证 JSON
                header_json = f.read(header_len).decode("utf-8")
                header_data = json.loads(header_json)
                
                # 检查文件大小
                file_size = safetensors_file.stat().st_size
                expected_size = header_len + 8  # 至少是头部大小
                for tensor_info in header_data.values():
                    if isinstance(tensor_info, dict) and "data_offsets" in tensor_info:
                        offsets = tensor_info["data_offsets"]
                        expected_size = max(expected_size, offsets[1] + 8)
                
                if file_size < expected_size:
                    raise ValueError(f"文件大小不完整: {file_size} < {expected_size}")
                
                print(f"      ✓ {safetensors_file.name} 完整性验证通过 ({file_size / 1024 / 1024:.2f} MB)")
    except Exception as verify_error:
        print(f"   ⚠️  文件验证失败: {verify_error}")
        print(f"   💡 建议重新下载")
        raise
    
    # 验证文件
    required_files = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "model.safetensors"  # 或 model.safetensors.index.json
    ]
    
    print("\n4️⃣ 验证文件...")
    all_exist = True
    missing_files = []
    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            file_size = file_path.stat().st_size / 1024 / 1024
            print(f"   ✓ {file} ({file_size:.2f} MB)")
        else:
            # 检查是否有 .index.json 文件
            if file == "model.safetensors":
                index_file = models_dir / "model.safetensors.index.json"
                if index_file.exists():
                    print(f"   ✓ model.safetensors.index.json (分片模型)")
                    continue
            print(f"   ⚠️  {file} 不存在")
            missing_files.append(file)
            all_exist = False
    
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(models_dir)
        for filename in filenames
    )
    
    if all_exist:
        print(f"\n✅ CLIP 模型已下载到: {models_dir}")
        print(f"   总文件大小: {total_size / 1024 / 1024:.2f} MB")
        print("\n✅ 下载完成！现在可以在离线环境中使用本地 CLIP 模型了。")
        print(f"\n💡 代码会自动使用此模型，路径: {models_dir}")
    else:
        print(f"\n⚠️  部分文件缺失: {', '.join(missing_files)}")
        print(f"   但主要文件已下载，总大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"   如果加载失败，请检查缺失的文件")
    
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

