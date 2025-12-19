#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复损坏的 CLIP 模型文件

如果文件已下载但损坏，尝试修复或重新下载损坏的部分
"""

import os
import sys
import json
import shutil
from pathlib import Path

def check_safetensors_file(file_path):
    """检查 safetensors 文件是否损坏"""
    try:
        with open(file_path, "rb") as f:
            # 读取文件头长度（8字节）
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return False, "文件头不完整"
            
            header_len = int.from_bytes(header_len_bytes, "little")
            if header_len <= 0 or header_len > 10 * 1024 * 1024:  # 限制最大 10MB
                return False, f"文件头长度异常: {header_len} bytes"
            
            # 读取并验证 JSON
            header_json = f.read(header_len).decode("utf-8")
            header_data = json.loads(header_json)
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            expected_size = header_len + 8  # 至少是头部大小
            for tensor_info in header_data.values():
                if isinstance(tensor_info, dict) and "data_offsets" in tensor_info:
                    offsets = tensor_info["data_offsets"]
                    expected_size = max(expected_size, offsets[1] + 8)
            
            if file_size < expected_size:
                return False, f"文件大小不完整: {file_size} < {expected_size}"
            
            return True, f"文件完整 ({file_size / 1024 / 1024:.2f} MB)"
    except json.JSONDecodeError as e:
        return False, f"JSON 解析失败: {e}"
    except Exception as e:
        return False, f"验证失败: {e}"

def find_corrupted_files(cache_path):
    """查找损坏的文件"""
    corrupted_files = []
    
    for root, dirs, files in os.walk(cache_path):
        for file in files:
            if file.endswith(".safetensors"):
                file_path = os.path.join(root, file)
                is_valid, message = check_safetensors_file(file_path)
                if not is_valid:
                    corrupted_files.append((file_path, message))
    
    return corrupted_files

def fix_corrupted_clip(model_id="openai/clip-vit-large-patch14", use_mirror=False):
    """修复损坏的 CLIP 模型"""
    print("🔧 检查并修复损坏的 CLIP 模型文件")
    print(f"   模型: {model_id}")
    print("")
    
    # 查找缓存目录
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    if not os.path.exists(hf_home):
        hf_home = "/vepfs-dev/shawn/.cache/huggingface"
    
    cache_name = f"models--{model_id.replace('/', '--')}"
    cache_path = os.path.join(hf_home, "hub", cache_name)
    
    if not os.path.exists(cache_path):
        print(f"❌ 缓存目录不存在: {cache_path}")
        print("   💡 请先运行下载脚本")
        return False
    
    print(f"📦 检查缓存目录: {cache_path}")
    print("")
    
    # 查找损坏的文件
    print("1️⃣ 扫描文件...")
    corrupted_files = find_corrupted_files(cache_path)
    
    if not corrupted_files:
        print("   ✓ 所有文件完整，无需修复")
        
        # 尝试加载验证
        print("\n2️⃣ 尝试加载模型验证...")
        try:
            from transformers import CLIPTextModel
            model = CLIPTextModel.from_pretrained(model_id, local_files_only=True)
            print("   ✓ 模型可以正常加载，无需修复")
            return True
        except Exception as e:
            error_str = str(e)
            if any(keyword in error_str for keyword in ["SafetensorError", "invalid JSON", "EOF"]):
                print(f"   ⚠️  虽然文件看起来完整，但加载失败: {e}")
                print(f"   💡 可能是文件格式问题，需要重新下载")
                corrupted_files = [("unknown", "加载失败")]
            else:
                print(f"   ⚠️  加载失败: {e}")
                return False
    
    print(f"   ⚠️  发现 {len(corrupted_files)} 个损坏的文件:")
    for file_path, message in corrupted_files:
        file_size = os.path.getsize(file_path) / 1024 / 1024 if os.path.exists(file_path) else 0
        print(f"      - {os.path.basename(file_path)} ({file_size:.2f} MB): {message}")
    
    print("\n3️⃣ 修复损坏的文件...")
    print("   💡 策略：删除损坏的文件，让 transformers 重新下载")
    
    # 删除损坏的文件
    for file_path, message in corrupted_files:
        if file_path != "unknown":
            print(f"   🗑️  删除: {os.path.basename(file_path)}")
            try:
                os.remove(file_path)
                print(f"      ✓ 已删除")
            except Exception as e:
                print(f"      ⚠️  删除失败: {e}")
    
    # 如果整个目录有问题，清理整个缓存
    if len(corrupted_files) > 0 and corrupted_files[0][0] == "unknown":
        print(f"   🗑️  清理整个缓存目录...")
        try:
            shutil.rmtree(cache_path)
            print(f"      ✓ 已清理")
        except Exception as e:
            print(f"      ⚠️  清理失败: {e}")
    
    print("\n4️⃣ 重新下载损坏的文件...")
    
    # 设置镜像站
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("   🌐 使用镜像站: https://hf-mirror.com")
    else:
        os.environ.pop("HF_ENDPOINT", None)
        print("   🌐 使用官方源（更可靠）")
    
    try:
        from transformers import CLIPTextModel
        
        print("   下载中...")
        # 使用 resume_download=True 可以断点续传
        model = CLIPTextModel.from_pretrained(
            model_id,
            resume_download=True,  # 断点续传
            local_files_only=False
        )
        print("   ✓ 下载完成")
        
        # 再次验证
        print("\n5️⃣ 验证修复结果...")
        model = CLIPTextModel.from_pretrained(model_id, local_files_only=True)
        print("   ✓ 模型可以正常加载，修复成功！")
        
        return True
    except Exception as e:
        print(f"   ❌ 下载失败: {e}")
        print(f"\n💡 建议：")
        print(f"   1. 清理缓存后重新下载: ./tools/clean_corrupted_clip.sh")
        print(f"   2. 使用官方源: unset HF_ENDPOINT && python3 tools/download_clip_to_models.py")
        print(f"   3. 使用 proxychains4: proxychains4 python3 tools/download_clip_to_models.py")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="修复损坏的 CLIP 模型文件")
    parser.add_argument("--use-mirror", action="store_true", help="使用镜像站（可能不稳定）")
    args = parser.parse_args()
    
    success = fix_corrupted_clip(use_mirror=args.use_mirror)
    sys.exit(0 if success else 1)

