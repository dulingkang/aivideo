#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Hunyuan-DiT 是否正确加载
"""

import sys
from pathlib import Path

# 添加 HunyuanDiT 到路径
hunyuan_dir = Path(__file__).parent / "HunyuanDiT"
sys.path.insert(0, str(hunyuan_dir))

print("="*80)
print("测试 Hunyuan-DiT 加载")
print("="*80)

try:
    from hydit.inference import End2End
    from hydit.config import get_args
    import argparse
    
    print("✅ 成功导入 Hunyuan-DiT 模块")
    
    # 设置模型路径
    models_root = Path("/vepfs-dev/shawn/vid/fanren/gen_video/models/hunyuan-dit")
    
    if not models_root.exists():
        print(f"❌ 模型目录不存在: {models_root}")
        sys.exit(1)
    
    print(f"✅ 模型目录存在: {models_root}")
    
    # 创建参数对象
    args = argparse.Namespace(
        model_root=str(models_root),
        enhance=False,  # 不使用增强模型
        infer_mode="torch",
        sampler="ddpm",
        load_4bit=False,
    )
    
    print("\n开始加载模型...")
    print("注意: 这可能需要一些时间...")
    
    # 加载模型
    gen = End2End(args, models_root)
    
    print("✅ Hunyuan-DiT 模型加载成功！")
    print(f"   类型: {type(gen).__name__}")
    
    # 测试生成（可选，如果 GPU 内存足够）
    print("\n" + "="*80)
    print("模型可以正常使用")
    print("="*80)
    print("\n使用示例:")
    print("  python sample_t2i.py --prompt '测试' --no-enhance")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n可能的原因:")
    print("1. 未安装依赖: pip install -r requirements.txt")
    print("2. 路径不正确")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

