#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查完整视频生成流程的依赖
"""

import sys
from pathlib import Path

def check_module(module_name, import_name=None):
    """检查模块是否可导入"""
    if import_name is None:
        import_name = module_name
    try:
        __import__(import_name)
        return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"✗ {module_name}: {e}"

def check_file(path, name):
    """检查文件或目录是否存在"""
    path_obj = Path(path)
    if path_obj.exists():
        return True, f"✓ {name}"
    else:
        return False, f"✗ {name}: {path} 不存在"

def main():
    print("=" * 60)
    print("检查完整视频生成流程依赖")
    print("=" * 60)
    
    issues = []
    
    # 1. 检查核心模块
    print("\n【核心 Python 模块】")
    core_modules = [
        ('torch', 'torch'),
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('numpy', 'numpy'),
        ('yaml', 'yaml'),
        ('insightface', 'insightface'),
    ]
    for module_name, import_name in core_modules:
        ok, msg = check_module(module_name, import_name)
        print(f"  {msg}")
        if not ok:
            issues.append(f"缺少模块: {module_name}")
    
    # 2. 检查视频生成相关
    print("\n【视频生成模块】")
    video_modules = [
        ('imageio', 'imageio'),
        ('imageio_ffmpeg', 'imageio_ffmpeg'),
        ('omegaconf', 'omegaconf'),
        ('ffmpeg', 'ffmpeg'),
    ]
    for module_name, import_name in video_modules:
        ok, msg = check_module(module_name, import_name)
        print(f"  {msg}")
        if not ok:
            issues.append(f"缺少模块: {module_name}")
    
    # 3. 检查 InstantID 和 CosyVoice（通过路径）
    print("\n【InstantID 和 CosyVoice】")
    instantid_path = Path(__file__).parent.parent / "InstantID"
    if instantid_path.exists():
        try:
            import sys
            if str(instantid_path) not in sys.path:
                sys.path.insert(0, str(instantid_path))
            from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
            print("  ✓ InstantID: 可用")
        except ImportError as e:
            print(f"  ✗ InstantID: 无法导入 ({e})")
            issues.append("InstantID 无法导入")
    else:
        print("  ✗ InstantID: 目录不存在")
        issues.append("InstantID 目录不存在")
    
    cosyvoice_path = Path(__file__).parent.parent / "CosyVoice"
    if cosyvoice_path.exists():
        try:
            import sys
            if str(cosyvoice_path) not in sys.path:
                sys.path.insert(0, str(cosyvoice_path))
            from cosyvoice.cli.cosyvoice import CosyVoice2
            print("  ✓ CosyVoice: 可用")
        except ImportError as e:
            print(f"  ✗ CosyVoice: 无法导入 ({e})")
            issues.append("CosyVoice 无法导入")
    else:
        print("  ✗ CosyVoice: 目录不存在")
        issues.append("CosyVoice 目录不存在")
    
    # 4. 检查系统工具
    print("\n【系统工具】")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ ffmpeg: {version_line}")
        else:
            print("  ✗ ffmpeg: 无法运行")
            issues.append("ffmpeg 无法运行")
    except FileNotFoundError:
        print("  ✗ ffmpeg: 未安装")
        issues.append("ffmpeg 未安装")
    
    # 5. 检查项目模块
    print("\n【项目模块】")
    project_modules = [
        ('image_generator', 'image_generator'),
        ('video_generator', 'video_generator'),
        ('tts_generator', 'tts_generator'),
        ('subtitle_generator', 'subtitle_generator'),
        ('video_composer', 'video_composer'),
        ('main', 'main'),
    ]
    for module_name, import_name in project_modules:
        ok, msg = check_module(module_name, import_name)
        print(f"  {msg}")
        if not ok:
            issues.append(f"缺少项目模块: {module_name}")
    
    # 6. 检查 generative-models（可选，用于原生 SVD）
    print("\n【generative-models（可选）】")
    gen_models_path = Path(__file__).parent / "generative-models"
    if gen_models_path.exists():
        try:
            import sys
            if str(gen_models_path) not in sys.path:
                sys.path.insert(0, str(gen_models_path))
            from sgm.util import instantiate_from_config
            print("  ✓ generative-models: 可用")
        except ImportError as e:
            print(f"  ⚠ generative-models: 部分功能不可用 ({e})")
            print("    提示: 将使用 diffusers 作为替代方案")
    else:
        print("  ⚠ generative-models: 目录不存在（将使用 diffusers）")
    
    # 总结
    print("\n" + "=" * 60)
    if issues:
        print("发现的问题：")
        for issue in issues:
            print(f"  - {issue}")
        print("\n建议：")
        print("  1. 运行: pip install -r requirements.txt")
        print("  2. 确保 InstantID 和 CosyVoice 目录存在")
        print("  3. 安装 ffmpeg: sudo apt install ffmpeg")
        return 1
    else:
        print("✓ 所有依赖都已就绪！")
        print("\n可以开始运行完整流程：")
        print("  python run_pipeline.py --markdown <script.md> --image-dir <dir> --output <name>")
        return 0

if __name__ == "__main__":
    sys.exit(main())

