#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 antelopev2 模型位置是否正确"""

import os
from pathlib import Path

print("="*60)
print("验证 antelopev2 模型位置")
print("="*60)

# 检查模型文件
model_dir = Path("models/antelopev2")
print(f"\n检查目录: {model_dir.absolute()}")

if model_dir.exists():
    onnx_files = list(model_dir.glob("*.onnx"))
    print(f"✓ 目录存在，找到 {len(onnx_files)} 个 .onnx 文件:")
    for f in onnx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    # 检查 InstantID 需要的文件
    required_files = [
        "1k3d68.onnx",
        "2d106det.onnx", 
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx"
    ]
    
    missing = []
    for req_file in required_files:
        if not (model_dir / req_file).exists():
            missing.append(req_file)
    
    if missing:
        print(f"\n⚠ 缺少文件: {', '.join(missing)}")
    else:
        print("\n✓ 所有必需文件都存在")
        
    # 测试 InsightFace 是否能找到模型
    try:
        from insightface.app import FaceAnalysis
        print("\n测试 InsightFace 加载...")
        # 使用当前目录作为 root
        app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
        print("✓ InsightFace 可以加载模型（使用 CPU 测试）")
        print(f"  模型路径: {model_dir.absolute()}")
    except Exception as e:
        print(f"\n⚠ InsightFace 测试失败: {e}")
        print("  提示: 可能需要安装 insightface: pip install insightface")
else:
    print(f"✗ 目录不存在: {model_dir.absolute()}")

print("\n" + "="*60)
print("验证完成")
print("="*60)

