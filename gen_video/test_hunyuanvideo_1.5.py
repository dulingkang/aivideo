#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HunyuanVideo 1.5 测试脚本（测试480p_i2v模型）
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 优先使用项目中的diffusers（支持1.5版本）
diffusers_path = project_root / "gen_video" / "diffusers" / "src"
if diffusers_path.exists():
    sys.path.insert(0, str(diffusers_path))
    print(f"  ℹ 使用项目中的diffusers: {diffusers_path}")

def test_hunyuanvideo_1_5():
    """测试HunyuanVideo 1.5模型加载"""
    print("=" * 60)
    print("HunyuanVideo 1.5 模型测试（480p_i2v）")
    print("=" * 60)
    
    # 检查模型文件
    print("\n1. 检查模型文件...")
    model_path = project_root / "gen_video" / "models" / "hunyuan-video-1.5"
    # 使用已下载的模型（480p_i2v只有配置，权重未下载）
    transformer_subfolder = "transformer/480p_i2v_distilled"  # 或 "transformer/720p_i2v"
    
    if not model_path.exists():
        print(f"  ✗ 模型目录不存在: {model_path}")
        return False
    
    print(f"  ✓ 模型目录存在: {model_path}")
    
    transformer_path = model_path / transformer_subfolder
    if not transformer_path.exists():
        print(f"  ✗ Transformer子目录不存在: {transformer_path}")
        return False
    
    print(f"  ✓ Transformer子目录存在: {transformer_subfolder}")
    
    # 测试加载
    print("\n2. 测试模型加载...")
    try:
        import torch
        from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Transformer3DModel
        
        print(f"  加载transformer: {transformer_subfolder}")
        transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
            str(model_path),
            subfolder=transformer_subfolder,
            torch_dtype=torch.float16
        )
        print("  ✓ Transformer加载成功")
        
        print("  加载完整pipeline...")
        try:
            pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                str(model_path),
                transformer=transformer,
                torch_dtype=torch.float16,
                variant="fp16"
            )
        except (ValueError, OSError):
            pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                str(model_path),
                transformer=transformer,
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
        print("✅ HunyuanVideo 1.5 模型加载测试通过！")
        print("=" * 60)
        print(f"\n使用的模型: {transformer_subfolder}")
        print("可以开始生成视频了！")
        return True
        
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        print("  ℹ 提示: diffusers版本可能不支持HunyuanVideo 1.5")
        print("  ℹ 尝试升级: pip install -U diffusers")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hunyuanvideo_1_5()
    if not success:
        sys.exit(1)
