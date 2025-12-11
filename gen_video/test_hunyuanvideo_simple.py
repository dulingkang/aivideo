#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HunyuanVideo简单测试脚本（只测试模型加载）
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("HunyuanVideo模型加载测试")
    print("=" * 60)
    
    # 检查模型文件
    print("\n1. 检查模型文件...")
    model_path = project_root / "gen_video" / "models" / "hunyuan-video"
    if not model_path.exists():
        print(f"  ✗ 模型目录不存在: {model_path}")
        return False
    
    print(f"  ✓ 模型目录存在: {model_path}")
    
    # 检查model_index.json
    model_index = model_path / "model_index.json"
    if model_index.exists():
        import json
        with open(model_index, 'r') as f:
            index = json.load(f)
        class_name = index.get('_class_name', '')
        print(f"  ✓ 模型类型: {class_name}")
        
        # 判断版本
        if 'HunyuanVideo15' in class_name:
            print("  ℹ 检测到：HunyuanVideo 1.5版本")
            use_v15 = True
        else:
            print("  ℹ 检测到：HunyuanVideo原版")
            use_v15 = False
    else:
        print("  ⚠ 未找到model_index.json，将使用配置的版本")
        use_v15 = False
    
    # 测试加载
    print("\n2. 测试模型加载...")
    try:
        import torch
        from diffusers import HunyuanVideoImageToVideoPipeline
        
        # 检查是否支持1.5版本
        try:
            from diffusers import HunyuanVideo15ImageToVideoPipeline
            has_v15 = True
        except ImportError:
            has_v15 = False
            print("  ℹ diffusers版本不支持1.5，使用原版")
        
        if use_v15 and has_v15:
            print("  尝试加载HunyuanVideo 1.5...")
            try:
                pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
            except (ValueError, OSError):
                pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16
                )
        else:
            print("  尝试加载HunyuanVideo原版...")
            try:
                pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
            except (ValueError, OSError):
                # 如果没有fp16变体，使用默认
                pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16
                )
        
        print("  ✓ Pipeline创建成功")
        
        # 检查显存
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ℹ GPU显存: {total_memory:.1f}GB")
            
            if total_memory < 16:
                print("  ⚠ 显存可能不足，建议使用CPU offload")
            else:
                print("  ✓ 显存充足，可以加载到GPU")
        
        print("\n" + "=" * 60)
        print("✅ 模型加载测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if not success:
        sys.exit(1)

