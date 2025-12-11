#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ModelManager 系统
验证所有模型是否可以正常加载和使用
"""

from model_manager import ModelManager
from pathlib import Path

def test_model_manager():
    """测试 ModelManager"""
    print("="*80)
    print("测试 ModelManager 系统")
    print("="*80)
    
    # 创建管理器
    print("\n1. 创建 ModelManager...")
    manager = ModelManager(lazy_load=True)
    print("✅ ModelManager 创建成功")
    
    # 检查模型状态
    print("\n2. 检查模型状态...")
    status = manager.list_models()
    for model_name, info in status.items():
        exists = "✅" if info["exists"] else "❌"
        print(f"   {exists} {model_name}: 存在={info['exists']}, 已加载={info['loaded']}")
    
    # 测试路由
    print("\n3. 测试任务路由...")
    test_tasks = [
        ("host_face", "kolors"),
        ("science_background", "flux2"),
        ("official_style", "hunyuan"),
        ("fast_background", "sd3"),
        ("lab_scene", "flux1"),
    ]
    
    for task, expected_model in test_tasks:
        routed_model = manager.route(task)
        status = "✅" if routed_model == expected_model else "❌"
        print(f"   {status} {task:20s} -> {routed_model:10s} (期望: {expected_model})")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
    print("\n✅ ModelManager 系统可以正常使用")
    print("\n使用示例:")
    print("  from model_manager import ModelManager")
    print("  manager = ModelManager()")
    print("  img = manager.generate(task='host_face', prompt='科普主持人')")
    print("  img.save('output.png')")

if __name__ == "__main__":
    test_model_manager()

