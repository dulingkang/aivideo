#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 CLIP 模型文件完整性

检查已下载的 CLIP 模型文件是否完整和有效
"""

import os
import sys
import json
from pathlib import Path

def verify_safetensors_file(file_path):
    """验证 safetensors 文件完整性"""
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

def verify_clip_model(model_path):
    """验证 CLIP 模型目录"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    print(f"📦 检查模型路径: {model_path}")
    print("")
    
    # 检查必需文件
    required_files = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    
    all_ok = True
    
    print("1️⃣ 检查 Tokenizer 文件...")
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"   ✓ {file} ({file_size / 1024:.2f} KB)")
        else:
            print(f"   ❌ {file} 不存在")
            all_ok = False
    
    print("\n2️⃣ 检查 Model 文件...")
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        print("   ❌ 未找到 .safetensors 文件")
        all_ok = False
    else:
        for safetensors_file in safetensors_files:
            print(f"   验证: {safetensors_file.name}...")
            is_valid, message = verify_safetensors_file(safetensors_file)
            if is_valid:
                print(f"      ✓ {message}")
            else:
                print(f"      ❌ {message}")
                all_ok = False
    
    print("\n3️⃣ 尝试加载模型验证...")
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        
        print("   加载 Tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(str(model_path), local_files_only=True)
        print("   ✓ Tokenizer 加载成功")
        
        print("   加载 Model...")
        model = CLIPTextModel.from_pretrained(str(model_path), local_files_only=True)
        print("   ✓ Model 加载成功")
        
        print("\n✅ 模型文件完整且可正常加载！")
        return True
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        return False

if __name__ == "__main__":
    # 默认检查 models 目录
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "clip" / "openai-clip-vit-large-patch14"
    
    success = verify_clip_model(model_path)
    sys.exit(0 if success else 1)

