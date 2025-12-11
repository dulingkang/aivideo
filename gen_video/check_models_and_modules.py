#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查缺失的模型和模块
"""

import os
import sys
from pathlib import Path
import yaml

def resolve_path(path_str, base_dir=None):
    """解析路径（支持相对路径和绝对路径）"""
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    else:
        # 相对路径，需要基于base_dir或当前工作目录
        if base_dir:
            return base_dir / path_obj
        else:
            # 默认基于gen_video目录
            return Path(__file__).parent / path_obj

def check_file_exists(path, name, base_dir=None):
    """检查文件或目录是否存在"""
    path_obj = resolve_path(path, base_dir)
    if path_obj.exists():
        if path_obj.is_file():
            size = path_obj.stat().st_size / (1024 * 1024)  # MB
            return True, f"✓ {name}: 存在 ({size:.1f} MB)"
        else:
            return True, f"✓ {name}: 目录存在"
    else:
        # 如果是参考图像，尝试查找替代文件名
        if '参考图像' in name or 'face_image' in str(path).lower():
            # 尝试查找 hanli_mid.png 或其他常见名称
            parent_dir = path_obj.parent
            if parent_dir.exists():
                alternatives = ['hanli_mid.png', 'character_mid.png', 'face.png', 'reference.png']
                for alt in alternatives:
                    alt_path = parent_dir / alt
                    if alt_path.exists():
                        return True, f"✓ {name}: 存在（使用替代文件 {alt}）"
        return False, f"✗ {name}: 缺失 ({path_obj})"

def check_model_file(path, name, required_files=None, base_dir=None):
    """检查模型文件是否存在"""
    path_obj = resolve_path(path, base_dir)
    if not path_obj.exists():
        return False, f"✗ {name}: 目录不存在 ({path_obj})"
    
    if required_files:
        missing = []
        for req_file in required_files:
            req_path = path_obj / req_file
            if not req_path.exists():
                missing.append(req_file)
        if missing:
            return False, f"✗ {name}: 缺少文件 {missing}"
    
    return True, f"✓ {name}: 完整"

def check_python_module(module_name, import_name=None):
    """检查 Python 模块是否已安装"""
    if import_name is None:
        import_name = module_name
    
    try:
        __import__(import_name)
        return True, f"✓ {module_name}: 已安装"
    except ImportError:
        return False, f"✗ {module_name}: 未安装"

def main():
    print("=" * 60)
    print("检查模型和模块")
    print("=" * 60)
    
    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置基础目录（gen_video目录）
    base_dir = Path(__file__).parent
    
    missing_items = []
    all_items = []
    
    # 1. 检查图像生成模型
    print("\n【图像生成模型】")
    image_config = config.get('image', {})
    
    # InstantID 模型
    if image_config.get('engine') == 'instantid':
        instantid_config = image_config.get('instantid', {})
        
        # SDXL 基础模型
        sdxl_path = instantid_config.get('model_path', 'models/sdxl-base')
        ok, msg = check_model_file(sdxl_path, "SDXL 基础模型", 
                                   ['model_index.json', 'unet/diffusion_pytorch_model.safetensors'], base_dir)
        all_items.append(msg)
        if not ok:
            missing_items.append(("SDXL 基础模型", sdxl_path))
        
        # InstantID ControlNet
        controlnet_path = instantid_config.get('controlnet_path', 'models/instantid/ControlNet')
        ok, msg = check_model_file(controlnet_path, "InstantID ControlNet",
                                  ['ControlNetModel/diffusion_pytorch_model.safetensors'], base_dir)
        all_items.append(msg)
        if not ok:
            missing_items.append(("InstantID ControlNet", controlnet_path))
        
        # InstantID IP-Adapter
        ip_adapter_path = instantid_config.get('ip_adapter_path', 'models/instantid/ip-adapter')
        ok, msg = check_file_exists(f"{ip_adapter_path}/ip-adapter.bin", "InstantID IP-Adapter", base_dir)
        all_items.append(msg)
        if not ok:
            missing_items.append(("InstantID IP-Adapter", ip_adapter_path))
        
        # 面部参考图像
        face_image_path = instantid_config.get('face_image_path', '')
        if face_image_path:
            ok, msg = check_file_exists(face_image_path, "面部参考图像", base_dir)
            all_items.append(msg)
            if not ok:
                missing_items.append(("面部参考图像", face_image_path))
    
    # LoRA 模型
    lora_config = image_config.get('lora', {})
    if lora_config.get('enabled', False):
        lora_path = lora_config.get('weights_path', '')
        if lora_path:
            ok, msg = check_file_exists(lora_path, "LoRA 权重", base_dir)
            all_items.append(msg)
            if not ok:
                missing_items.append(("LoRA 权重", lora_path))
    
    # 2. 检查视频生成模型
    print("\n【视频生成模型】")
    video_config = config.get('video', {})
    svd_path = video_config.get('model_path', 'models/svd')
    ok, msg = check_model_file(svd_path, "SVD 模型", ['model_index.json'], base_dir)
    all_items.append(msg)
    if not ok:
        missing_items.append(("SVD 模型", svd_path))
    
    # 3. 检查 TTS 模型
    print("\n【TTS 模型】")
    tts_config = config.get('tts', {})
    if tts_config.get('engine') == 'cosyvoice':
        cosyvoice_config = tts_config.get('cosyvoice', {})
        model_name = cosyvoice_config.get('model_name', 'CosyVoice2-0.5B')
        model_path = cosyvoice_config.get('model_path', 'models/cosyvoice')
        
        # 检查 CosyVoice 模型（可能在 CosyVoice 仓库或 models 目录）
        cosyvoice_repo_path = Path(__file__).parent.parent / "CosyVoice" / "pretrained_models" / model_name
        models_path = Path(model_path) / model_name
        
        if cosyvoice_repo_path.exists():
            ok, msg = check_model_file(str(cosyvoice_repo_path), f"CosyVoice {model_name}",
                                       ['cosyvoice2.yaml' if 'CosyVoice2' in model_name else 'cosyvoice.yaml'])
            all_items.append(msg)
            if not ok:
                missing_items.append((f"CosyVoice {model_name}", str(cosyvoice_repo_path)))
        elif models_path.exists():
            ok, msg = check_model_file(str(models_path), f"CosyVoice {model_name}",
                                       ['cosyvoice2.yaml' if 'CosyVoice2' in model_name else 'cosyvoice.yaml'])
            all_items.append(msg)
            if not ok:
                missing_items.append((f"CosyVoice {model_name}", str(models_path)))
        else:
            all_items.append(f"✗ CosyVoice {model_name}: 未找到")
            missing_items.append((f"CosyVoice {model_name}", f"{cosyvoice_repo_path} 或 {models_path}"))
        
        # 检查参考音频
        prompt_speech = cosyvoice_config.get('prompt_speech', '')
        if prompt_speech:
            ok, msg = check_file_exists(prompt_speech, "CosyVoice 参考音频", base_dir)
            all_items.append(msg)
            if not ok:
                missing_items.append(("CosyVoice 参考音频", prompt_speech))
    
    # 4. 检查字幕模型
    print("\n【字幕模型】")
    subtitle_config = config.get('subtitle', {})
    model_size = subtitle_config.get('model_size', 'medium')
    model_dir = subtitle_config.get('model_dir', f'models/faster-whisper-{model_size}')
    ok, msg = check_model_file(model_dir, f"Whisper {model_size}",
                               ['model.bin', 'config.json'], base_dir)
    all_items.append(msg)
    if not ok:
        missing_items.append((f"Whisper {model_size}", model_dir))
    
    # 5. 检查后处理模型
    print("\n【后处理模型】")
    postprocess_config = config.get('postprocess', {})
    if postprocess_config.get('enabled', False):
        realesrgan_path = postprocess_config.get('model_path', 'models/realesrgan/RealESRGAN_x4plus_anime_6B.pth')
        ok, msg = check_file_exists(realesrgan_path, "Real-ESRGAN 模型", base_dir)
        all_items.append(msg)
        if not ok:
            missing_items.append(("Real-ESRGAN 模型", realesrgan_path))
    
    # 6. 检查背景音乐
    print("\n【资源文件】")
    composition_config = config.get('composition', {})
    bgm_config = composition_config.get('bgm', {})
    if bgm_config.get('enabled', False):
        tracks = bgm_config.get('tracks', {})
        for track_name, track_config in tracks.items():
            bgm_path = track_config.get('path', '')
            if bgm_path:
                ok, msg = check_file_exists(bgm_path, f"背景音乐 ({track_name})", base_dir)
                all_items.append(msg)
                if not ok:
                    missing_items.append((f"背景音乐 ({track_name})", bgm_path))
    
    # 7. 检查 Python 模块
    print("\n【Python 模块】")
    required_modules = [
        ('torch', 'torch'),
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers'),
        ('PIL', 'PIL'),
        ('cv2', 'cv2'),
        ('numpy', 'numpy'),
        ('yaml', 'yaml'),
        ('insightface', 'insightface'),
    ]
    
    # 检查 InstantID（通过路径导入，不是 pip 包）
    instantid_repo_path = Path(__file__).parent.parent / "InstantID"
    instantid_pipeline_file = instantid_repo_path / "pipeline_stable_diffusion_xl_instantid.py"
    if instantid_pipeline_file.exists():
        try:
            import sys
            if str(instantid_repo_path) not in sys.path:
                sys.path.insert(0, str(instantid_repo_path))
            from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
            all_items.append("✓ InstantID: 可用（通过路径导入）")
        except ImportError as e:
            all_items.append(f"✗ InstantID: 路径存在但无法导入 ({e})")
            missing_items.append(("InstantID", f"检查 {instantid_repo_path} 目录"))
    else:
        # 尝试检查是否安装了 instantid 包
        try:
            from instantid import InstantIDPipeline
            all_items.append("✓ InstantID: 已安装（pip 包）")
        except ImportError:
            all_items.append("✗ InstantID: 未找到（需要 InstantID 目录或 pip 包）")
            missing_items.append(("InstantID", f"确保 {instantid_repo_path} 存在或 pip install instantid"))
    
    # 检查 CosyVoice（通过路径导入，不是 pip 包）
    cosyvoice_repo_path = Path(__file__).parent.parent / "CosyVoice"
    cosyvoice_module_file = cosyvoice_repo_path / "cosyvoice" / "__init__.py"
    if cosyvoice_module_file.exists():
        try:
            import sys
            if str(cosyvoice_repo_path) not in sys.path:
                sys.path.insert(0, str(cosyvoice_repo_path))
            from cosyvoice.cli.cosyvoice import CosyVoice2
            all_items.append("✓ CosyVoice: 可用（通过路径导入）")
        except ImportError as e:
            all_items.append(f"✗ CosyVoice: 路径存在但无法导入 ({e})")
            missing_items.append(("CosyVoice", f"检查 {cosyvoice_repo_path} 目录"))
    else:
        all_items.append("✗ CosyVoice: 未找到（需要 CosyVoice 目录）")
        missing_items.append(("CosyVoice", f"确保 {cosyvoice_repo_path} 存在"))
    
    # 检查其他模块
    for module_name, import_name in required_modules:
        ok, msg = check_python_module(module_name, import_name)
        all_items.append(msg)
        if not ok:
            missing_items.append((module_name, f"pip install {module_name}"))
    
    # 打印所有检查结果
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    for msg in all_items:
        print(f"  {msg}")
    
    # 打印缺失项
    if missing_items:
        print("\n" + "=" * 60)
        print("缺失的模型和模块")
        print("=" * 60)
        for name, path in missing_items:
            print(f"  ✗ {name}: {path}")
        
        print("\n" + "=" * 60)
        print("补齐建议")
        print("=" * 60)
        print("运行以下命令下载缺失的模型：")
        print("  python download_stage1_models.py")
        print("\n安装缺失的 Python 模块：")
        for name, install_cmd in missing_items:
            if 'pip install' in str(install_cmd) or 'install' in str(install_cmd):
                print(f"  {install_cmd}")
        
        return 1
    else:
        print("\n" + "=" * 60)
        print("✓ 所有模型和模块都已就绪！")
        print("=" * 60)
        return 0

if __name__ == "__main__":
    sys.exit(main())

