#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查环境是否准备好运行API
"""
import sys
from pathlib import Path

def check_python():
    """检查Python版本"""
    print("=" * 60)
    print("🔍 检查Python环境")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    version_info = sys.version_info
    if version_info.major >= 3 and version_info.minor >= 8:
        print("✅ Python版本符合要求 (>=3.8)")
        return True
    else:
        print(f"❌ Python版本过低，需要 >= 3.8，当前: {version_info.major}.{version_info.minor}")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n" + "=" * 60)
    print("🔍 检查依赖包")
    print("=" * 60)
    
    required_packages = {
        "fastapi": "FastAPI框架",
        "uvicorn": "ASGI服务器",
        "pydantic": "数据验证",
    }
    
    missing = []
    for package, desc in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package}: {desc}")
        except ImportError:
            print(f"❌ {package}: 未安装 ({desc})")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_config():
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("🔍 检查配置文件")
    print("=" * 60)
    
    config_path = Path(__file__).parent / "gen_video" / "config.yaml"
    if config_path.exists():
        print(f"✅ 配置文件存在: {config_path}")
        return True
    else:
        print(f"❌ 配置文件不存在: {config_path}")
        return False

def check_gpu():
    """检查GPU"""
    print("\n" + "=" * 60)
    print("🔍 检查GPU")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠️  GPU不可用（使用CPU会很慢）")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查GPU")
        return False

def check_paths():
    """检查路径"""
    print("\n" + "=" * 60)
    print("🔍 检查路径")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    api_dir = project_root / "gen_video" / "api"
    outputs_dir = project_root / "outputs" / "api"
    
    if api_dir.exists():
        print(f"✅ API目录存在: {api_dir}")
    else:
        print(f"❌ API目录不存在: {api_dir}")
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录准备: {outputs_dir}")
    
    return True

def main():
    print("🧪 环境检查")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # 检查Python
    if not check_python():
        all_ok = False
    
    # 检查依赖
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_ok = False
        print(f"\n💡 安装缺失的依赖:")
        print(f"   pip install {' '.join(missing)}")
        print(f"   或: pip install -r gen_video/api/requirements.txt")
    
    # 检查配置
    if not check_config():
        all_ok = False
    
    # 检查GPU
    gpu_ok = check_gpu()
    
    # 检查路径
    check_paths()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 检查总结")
    print("=" * 60)
    
    if all_ok:
        print("✅ 环境检查通过，可以启动API服务器")
        print("\n🚀 启动命令:")
        print("   cd gen_video/api")
        print("   python3 main_sync.py")
        print("\n或使用启动脚本:")
        print("   ./start_sync_api.sh")
    else:
        print("❌ 环境检查未通过，请先解决上述问题")
    
    if not gpu_ok:
        print("\n⚠️  警告: GPU不可用，图像生成会很慢")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

