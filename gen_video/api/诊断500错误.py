#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 500 错误
"""

import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

print("="*80)
print("诊断 500 错误")
print("="*80)

# 1. 检查 ModelManager 是否可以导入
print("\n1. 检查 ModelManager...")
try:
    from model_manager import ModelManager
    print("   ✅ ModelManager 可以导入")
except Exception as e:
    print(f"   ❌ ModelManager 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 检查模型路径
print("\n2. 检查模型路径...")
models_root = Path(__file__).parent.parent / "models"
model_paths = {
    "flux1": models_root / "flux1-dev",
    "flux2": models_root / "flux2-dev",
    "kolors": models_root / "kolors-base",
    "hunyuan": models_root / "hunyuan-dit" / "t2i",
    "sd3": models_root / "sd3-turbo",
}

for name, path in model_paths.items():
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024**3)
        print(f"   ✅ {name}: {size:.2f} GB")
    else:
        print(f"   ❌ {name}: 路径不存在 - {path}")

# 3. 测试 ModelManager 初始化
print("\n3. 测试 ModelManager 初始化...")
try:
    manager = ModelManager(models_root=str(models_root), lazy_load=True)
    print("   ✅ ModelManager 初始化成功")
    
    # 检查模型状态
    status = manager.list_models()
    print("\n   模型状态:")
    for name, info in status.items():
        exists = "✅" if info["exists"] else "❌"
        print(f"      {exists} {name}: 存在={info['exists']}")
        
except Exception as e:
    print(f"   ❌ ModelManager 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试路由
print("\n4. 测试任务路由...")
test_tasks = ["host_face", "science_background", "official_style", "fast_background"]
for task in test_tasks:
    try:
        model = manager.route(task)
        print(f"   ✅ {task} -> {model}")
    except Exception as e:
        print(f"   ❌ {task} -> 错误: {e}")

# 5. 测试 API 导入
print("\n5. 测试 API 导入...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.mvp_main import app, get_model_manager, MODEL_MANAGER_AVAILABLE
    print(f"   ✅ API 可以导入")
    print(f"   ✅ MODEL_MANAGER_AVAILABLE = {MODEL_MANAGER_AVAILABLE}")
    
    # 测试 get_model_manager
    if MODEL_MANAGER_AVAILABLE:
        try:
            mgr = get_model_manager()
            print(f"   ✅ get_model_manager() 可以调用")
        except Exception as e:
            print(f"   ❌ get_model_manager() 失败: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"   ❌ API 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("诊断完成")
print("="*80)
print("\n如果所有检查都通过，但仍有 500 错误，请:")
print("1. 查看 API 控制台的详细错误日志")
print("2. 检查是否使用了 ModelManager（use_model_manager=true）")
print("3. 如果 ModelManager 失败，应该自动回退到 ImageGenerator")

