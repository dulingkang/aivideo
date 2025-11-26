#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1+阶段2模型下载脚本
下载 InstantID、CosyVoice、WhisperX-large-v3、AnimateDiff-SDXL-1080P、Real-ESRGAN 等模型
支持通过 proxychains 使用代理下载
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# 启用实时输出刷新
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 启用 huggingface_hub 的进度显示
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # 禁用 hf_transfer，使用标准下载以显示进度
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # 启用进度条

# 检查代理配置（从环境变量读取）
HTTP_PROXY = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
HTTPS_PROXY = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

# 确保 huggingface_hub 使用代理
if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
    os.environ['http_proxy'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY
    os.environ['https_proxy'] = HTTPS_PROXY

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_instantid_models(config):
    """下载 InstantID 相关模型"""
    print("\n" + "="*60)
    print("下载 InstantID 模型")
    print("="*60)
    
    instantid_config = config['image']['instantid']
    base_model = instantid_config.get('base_model', 'Juggernaut-XL-v9-anime')
    model_path = Path(instantid_config.get('model_path', 'models/instantid'))
    controlnet_path = Path(instantid_config.get('controlnet_path', 'models/instantid/ControlNet'))
    ip_adapter_path = Path(instantid_config.get('ip_adapter_path', 'models/instantid/ip-adapter'))
    
    # 创建目录
    model_path.mkdir(parents=True, exist_ok=True)
    controlnet_path.mkdir(parents=True, exist_ok=True)
    ip_adapter_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载基础模型
        print(f"\n[1/3] 下载基础模型: {base_model}")
        if HTTP_PROXY or HTTPS_PROXY:
            print(f"  使用代理下载...")
        if not (model_path / "model_index.json").exists():
            print(f"  开始下载，这可能需要一些时间...")
            sys.stdout.flush()
            snapshot_download(
                repo_id=base_model,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"✓ 基础模型已下载到: {model_path}")
            sys.stdout.flush()
        else:
            print(f"✓ 基础模型已存在: {model_path}")
        
        # 下载 InstantID ControlNet（使用官方推荐方式）
        print(f"\n[2/3] 下载 InstantID ControlNet")
        controlnet_repo = "InstantX/InstantID"
        if HTTP_PROXY or HTTPS_PROXY:
            print(f"  使用代理下载...")
        
        # 检查 ControlNet 文件是否存在
        controlnet_config = controlnet_path / "ControlNetModel" / "config.json"
        controlnet_model = controlnet_path / "ControlNetModel" / "diffusion_pytorch_model.safetensors"
        
        if not controlnet_model.exists() or not controlnet_config.exists():
            try:
                print(f"  使用官方推荐方式下载（hf_hub_download）...")
                sys.stdout.flush()
                
                # 使用官方推荐的方式下载 ControlNet
                # 参考: https://github.com/instantX-research/InstantID
                hf_hub_download(
                    repo_id=controlnet_repo,
                    filename="ControlNetModel/config.json",
                    local_dir=str(controlnet_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                hf_hub_download(
                    repo_id=controlnet_repo,
                    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
                    local_dir=str(controlnet_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"✓ ControlNet 已下载到: {controlnet_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠ ControlNet 下载失败: {e}")
                print("  提示: 尝试使用备用方式（snapshot_download）...")
                try:
                    snapshot_download(
                        repo_id=controlnet_repo,
                        local_dir=str(controlnet_path),
                        allow_patterns=["ControlNetModel/*"],
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    print(f"✓ ControlNet 已下载到: {controlnet_path}")
                except Exception as e2:
                    print(f"✗ ControlNet 下载失败: {e2}")
                    print("  提示: 请手动下载或检查网络连接")
        else:
            print(f"✓ ControlNet 已存在: {controlnet_path}")
        
        # 下载 InstantID IP-Adapter（使用官方推荐方式）
        print(f"\n[3/3] 下载 InstantID IP-Adapter")
        if HTTP_PROXY or HTTPS_PROXY:
            print(f"  使用代理下载...")
        
        # 检查 IP-Adapter 文件是否存在（可能在 ip-adapter 目录或根目录）
        ip_adapter_file1 = ip_adapter_path / "ip-adapter.bin"
        ip_adapter_file2 = model_path / "ip-adapter.bin"
        
        if not ip_adapter_file1.exists() and not ip_adapter_file2.exists():
            try:
                print(f"  使用官方推荐方式下载（hf_hub_download）...")
                sys.stdout.flush()
                
                # 使用官方推荐的方式下载 IP-Adapter
                hf_hub_download(
                    repo_id=controlnet_repo,
                    filename="ip-adapter.bin",
                    local_dir=str(ip_adapter_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"✓ IP-Adapter 已下载到: {ip_adapter_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠ IP-Adapter 下载失败: {e}")
                print("  提示: 尝试使用备用方式（snapshot_download）...")
                try:
                    snapshot_download(
                        repo_id=controlnet_repo,
                        local_dir=str(ip_adapter_path),
                        allow_patterns=["ip-adapter.bin"],
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    print(f"✓ IP-Adapter 已下载到: {ip_adapter_path}")
                except Exception as e2:
                    print(f"✗ IP-Adapter 下载失败: {e2}")
                    print("  提示: 请手动下载或检查网络连接")
        else:
            if ip_adapter_file1.exists():
                print(f"✓ IP-Adapter 已存在: {ip_adapter_file1}")
            else:
                print(f"✓ IP-Adapter 已存在: {ip_adapter_file2}")
            
    except Exception as e:
        print(f"✗ InstantID 模型下载失败: {e}")
        print("  提示: 请检查网络连接或手动下载模型")

def download_cosyvoice_models(config):
    """下载 CosyVoice 模型（使用 ModelScope，根据 GitHub README）"""
    print("\n" + "="*60)
    print("下载 CosyVoice 模型（使用 ModelScope）")
    print("="*60)
    
    try:
        cosyvoice_config = config['tts']['cosyvoice']
        model_name = cosyvoice_config.get('model_name', 'CosyVoice2-0.5B')
        # 将模型名称转换为 ModelScope 格式
        modelscope_model_map = {
            'CosyVoice2-0.5B': 'iic/CosyVoice2-0.5B',
            'CosyVoice-2.0-0.5B': 'iic/CosyVoice2-0.5B',
            'CosyVoice-300M': 'iic/CosyVoice-300M',
            'CosyVoice-300M-SFT': 'iic/CosyVoice-300M-SFT',
            'CosyVoice-300M-Instruct': 'iic/CosyVoice-300M-Instruct',
        }
        modelscope_id = modelscope_model_map.get(model_name, f'iic/{model_name}')
        
        # 模型下载到 CosyVoice 仓库的 pretrained_models 目录
        cosyvoice_repo_path = Path(__file__).parent.parent / "CosyVoice"
        if cosyvoice_repo_path.exists():
            model_path = cosyvoice_repo_path / "pretrained_models" / model_name
        else:
            # 如果 CosyVoice 仓库不存在，下载到本地 models 目录
            model_path = Path(cosyvoice_config.get('model_path', 'models/cosyvoice')) / model_name
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n下载 CosyVoice 模型: {model_name}")
        print(f"  ModelScope ID: {modelscope_id}")
        print(f"  下载路径: {model_path}")
        
        # 检查模型是否已存在（检查 yaml 文件）
        # CosyVoice2 需要 cosyvoice2.yaml，CosyVoice 需要 cosyvoice.yaml
        if model_name.startswith('CosyVoice2'):
            expected_yaml = 'cosyvoice2.yaml'
        else:
            expected_yaml = 'cosyvoice.yaml'
        if (model_path / expected_yaml).exists():
            print(f"✓ CosyVoice 模型已存在: {model_path}")
            return
        
        # 使用 ModelScope 下载（根据 GitHub README）
        try:
            print(f"  使用 ModelScope 下载（首次下载可能需要较长时间）...")
            sys.stdout.flush()
            
            # 尝试导入 modelscope
            try:
                from modelscope import snapshot_download as ms_snapshot_download
            except ImportError:
                print("  ⚠ modelscope 未安装，尝试安装（使用中科大镜像）...")
                import subprocess
                # 使用中科大镜像安装，失败则使用官方源
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "modelscope",
                        "-i", "https://mirrors.ustc.edu.cn/pypi/simple",
                        "--trusted-host", "mirrors.ustc.edu.cn"
                    ])
                except subprocess.CalledProcessError:
                    print("  ⚠ 中科大镜像安装失败，尝试官方源...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
                from modelscope import snapshot_download as ms_snapshot_download
            
            ms_snapshot_download(
                modelscope_id,
                local_dir=str(model_path),
                cache_dir=None,
            )
            print(f"✓ CosyVoice 模型已下载到: {model_path}")
            sys.stdout.flush()
            
            # 可选：下载 ttsfrd 资源（用于更好的文本标准化）
            if model_name in ['CosyVoice2-0.5B', 'CosyVoice-300M']:
                print(f"\n  提示: 如需更好的文本标准化，可下载 CosyVoice-ttsfrd:")
                print(f"    from modelscope import snapshot_download")
                print(f"    snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')")
            
        except Exception as e:
            print(f"⚠ ModelScope 下载失败: {e}")
            print(f"  提示: 可以手动使用 git clone 下载:")
            print(f"    mkdir -p {model_path.parent}")
            print(f"    git clone https://www.modelscope.cn/{modelscope_id}.git {model_path}")
            print(f"  或访问: https://www.modelscope.cn/{modelscope_id}")
            
    except Exception as e:
        print(f"✗ CosyVoice 模型下载失败: {e}")
        print("  提示: 请检查网络连接或手动下载模型")
        print("  参考: https://github.com/FunAudioLLM/CosyVoice")

def download_whisperx_models(config):
    """下载 WhisperX 模型"""
    print("\n" + "="*60)
    print("下载 WhisperX 模型")
    print("="*60)
    
    subtitle_config = config['subtitle']
    model_size = subtitle_config.get('model_size', 'large-v3')
    model_dir = Path(subtitle_config.get('model_dir', f'models/faster-whisper-{model_size}'))
    align_model = subtitle_config.get('align_model', 'ZhangCheng/whisperx-align-zh')
    vad_model = subtitle_config.get('vad_model', 'pyannote/segmentation-3.0')
    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载 Whisper 模型
        print(f"\n[1/3] 下载 Whisper 模型: {model_size}")
        whisper_repo = f"openai/whisper-{model_size}"
        
        if not (model_dir / "model.bin").exists():
            try:
                print(f"  开始下载，这可能需要一些时间...")
                sys.stdout.flush()
                snapshot_download(
                    repo_id=whisper_repo,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"✓ Whisper 模型已下载到: {model_dir}")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠ Whisper 模型下载失败: {e}")
                print("  提示: WhisperX 会在首次使用时自动下载模型")
        else:
            print(f"✓ Whisper 模型已存在: {model_dir}")
        
        # 下载对齐模型
        print(f"\n[2/3] 下载对齐模型")
        align_dir = model_dir.parent / "whisperx-align-zh"
        align_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试多个可能的对齐模型仓库
        possible_align_repos = [
            align_model,  # 配置中的模型
            "m-bain/whisperx-align-zh",  # 可能的官方仓库
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",  # 替代方案
        ]
        
        if not (align_dir / "model.bin").exists() and not (align_dir / "pytorch_model.bin").exists():
            downloaded = False
            for repo_id in possible_align_repos:
                try:
                    print(f"  尝试从仓库下载: {repo_id}")
                    sys.stdout.flush()
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(align_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    # 检查是否下载成功
                    if (align_dir / "model.bin").exists() or (align_dir / "pytorch_model.bin").exists() or list(align_dir.glob("*.bin")):
                        print(f"✓ 对齐模型已下载到: {align_dir}")
                        sys.stdout.flush()
                        downloaded = True
                        break
                except Exception as e:
                    print(f"  尝试 {repo_id} 失败: {e}")
                    continue
            
            if not downloaded:
                print(f"⚠ 对齐模型自动下载失败")
                print("  提示: WhisperX 会在首次使用时根据 language_code='zh' 自动下载对齐模型")
                print("  或者您可以手动访问 HuggingFace 搜索 'whisperx align chinese' 找到正确的模型")
        else:
            print(f"✓ 对齐模型已存在: {align_dir}")
        
        # 下载 VAD 模型
        print(f"\n[3/3] 下载 VAD 模型: {vad_model}")
        vad_dir = model_dir.parent / "pyannote-segmentation"
        vad_dir.mkdir(parents=True, exist_ok=True)
        
        if not (vad_dir / "pytorch_model.bin").exists() and not list(vad_dir.glob("*.bin")):
            try:
                print(f"  开始下载，这可能需要一些时间...")
                print(f"  ⚠ 注意: {vad_model} 是受限模型，需要先接受使用条款")
                sys.stdout.flush()
                
                # 检查是否需要登录
                from huggingface_hub import whoami
                try:
                    user_info = whoami()
                    print(f"  当前已登录 HuggingFace 用户: {user_info.get('name', 'unknown')}")
                except Exception:
                    print(f"  ⚠ 未检测到 HuggingFace 登录")
                    print(f"  提示: 受限模型需要先登录并接受使用条款")
                
                snapshot_download(
                    repo_id=vad_model,
                    local_dir=str(vad_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"✓ VAD 模型已下载到: {vad_dir}")
                sys.stdout.flush()
            except Exception as e:
                error_msg = str(e)
                if "403" in error_msg or "gated" in error_msg.lower() or "restricted" in error_msg.lower():
                    print(f"⚠ VAD 模型下载失败: 需要授权访问")
                    print(f"  解决步骤:")
                    print(f"    1. 访问 https://huggingface.co/{vad_model}")
                    print(f"    2. 点击 'Agree and access repository' 接受使用条款")
                    print(f"    3. 确保已登录 HuggingFace 账号")
                    print(f"    4. 运行: huggingface-cli login")
                    print(f"    5. 重新运行此脚本，或让 WhisperX 在首次使用时自动下载")
                else:
                    print(f"⚠ VAD 模型下载失败: {e}")
                    print("  提示: WhisperX 会在首次使用时自动下载 VAD 模型")
        else:
            print(f"✓ VAD 模型已存在: {vad_dir}")
            
    except Exception as e:
        print(f"✗ WhisperX 模型下载失败: {e}")

def download_animatediff_models(config):
    """下载 AnimateDiff-SDXL 相关模型（阶段2）"""
    print("\n" + "="*60)
    print("下载 AnimateDiff-SDXL 模型（阶段2）")
    print("="*60)
    
    # 使用独立的 AnimateDiff 模型路径，不依赖 config 中的 video.model_path
    model_path = Path("models/animatediff-sdxl-1080p")
    
    # 创建目录
    model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # AnimateDiff-SDXL 需要以下组件：
        # 1. Motion Module (运动模块) - mm_sdxl_v10_beta.ckpt
        # 2. 或者使用 diffusers 格式的完整 pipeline
        
        print(f"\n下载 AnimateDiff-SDXL 模型")
        print(f"  目标路径: {model_path}")
        
        # 尝试下载 diffusers 格式的 AnimateDiff pipeline
        # 注意：AnimateDiff 在 diffusers 中可能需要配合 SDXL base model 使用
        possible_repos = [
            "guoyww/animatediff-motion-lora-sdxl",  # Motion LoRA for SDXL
            "guoyww/animatediff-motion-lora",  # Motion LoRA (通用)
            "guoyww/AnimateDiff",  # 官方仓库（可能包含模型）
        ]
        
        downloaded = False
        
        # 首先尝试下载 Motion Module (ckpt 格式)
        motion_module_path = model_path / "mm_sdxl_v10_beta.ckpt"
        if not motion_module_path.exists():
            print(f"\n[1/2] 下载 AnimateDiff SDXL Motion Module")
            print("  尝试从 HuggingFace 下载...")
            sys.stdout.flush()
            
            # 尝试多个可能的文件名
            possible_filenames = [
                "mm_sdxl_v10_beta.ckpt",  # SDXL motion module
                "mm_sdxl_v10.ckpt",
                "model_v15_sd15.ckpt",  # SD1.5 motion module (备选)
            ]
            
            downloaded_motion = False
            for filename in possible_filenames:
                try:
                    print(f"  尝试下载: {filename}")
                    hf_hub_download(
                        repo_id="guoyww/animatediff",
                        filename=filename,
                        local_dir=str(model_path),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    print(f"  ✓ 已下载 motion module: {filename}")
                    downloaded_motion = True
                    break
                except Exception as e:
                    print(f"  ⚠ {filename} 下载失败: {str(e)[:100]}")
                    continue
            
            if not downloaded_motion:
                # 尝试从 GitHub releases 下载（使用 urllib）
                print("  ⚠ HuggingFace 下载失败，尝试从 GitHub releases 下载...")
                try:
                    import urllib.request
                    github_urls = [
                        "https://github.com/guoyww/AnimateDiff/releases/download/mm_sdxl_v10_beta/mm_sdxl_v10_beta.ckpt",
                        "https://github.com/guoyww/AnimateDiff/releases/latest/download/mm_sdxl_v10_beta.ckpt",
                    ]
                    
                    for url in github_urls:
                        try:
                            print(f"  尝试从 GitHub 下载: {url}")
                            def reporthook(count, block_size, total_size):
                                if total_size > 0:
                                    percent = int(count * block_size * 100 / total_size)
                                    print(f"\r  下载进度: {percent}%", end="", flush=True)
                            
                            urllib.request.urlretrieve(url, str(motion_module_path), reporthook=reporthook)
                            print()  # 换行
                            print(f"  ✓ 已从 GitHub 下载 motion module: {motion_module_path}")
                            downloaded_motion = True
                            break
                        except Exception as e:
                            print(f"\n  ⚠ GitHub 下载失败: {str(e)[:100]}")
                            continue
                except Exception as e:
                    print(f"  ⚠ GitHub 下载尝试失败: {e}")
                
                if not downloaded_motion:
                    print("  ⚠ 自动下载失败，需要手动下载")
                    print("    手动下载步骤:")
                    print("    1. 访问 https://huggingface.co/guoyww/animatediff/tree/main")
                    print("    2. 或访问 https://github.com/guoyww/AnimateDiff/releases")
                    print("    3. 下载 mm_sdxl_v10_beta.ckpt")
                    print(f"    4. 放置到: {motion_module_path}")
        else:
            print(f"\n[1/2] AnimateDiff SDXL Motion Module")
            print(f"  ✓ Motion Module 已存在: {motion_module_path}")
        
        # 尝试下载 diffusers 格式的完整 pipeline
        print(f"\n[2/2] 尝试下载 diffusers 格式的 AnimateDiff pipeline")
        for repo_id in possible_repos:
            try:
                # 检查是否已有模型文件
                if (model_path / "model_index.json").exists() or (model_path / "pytorch_model.bin").exists():
                    print(f"✓ AnimateDiff 模型已存在: {model_path}")
                    downloaded = True
                    break
                
                print(f"  尝试从仓库下载: {repo_id}")
                sys.stdout.flush()
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                
                # 检查是否下载成功
                if (model_path / "model_index.json").exists() or list(model_path.glob("*.ckpt")) or list(model_path.glob("*.safetensors")):
                    print(f"✓ AnimateDiff 模型已下载到: {model_path}")
                    sys.stdout.flush()
                    downloaded = True
                    break
            except Exception as e:
                print(f"  尝试 {repo_id} 失败: {e}")
                continue
        
        if not downloaded:
            print(f"\n⚠ AnimateDiff-SDXL 模型自动下载失败")
            print("  提示: AnimateDiff-SDXL 可能需要手动下载和配置")
            print("  参考步骤:")
            print("    1. 访问 https://github.com/guoyww/AnimateDiff")
            print("    2. 下载 SDXL motion module: mm_sdxl_v10_beta.ckpt")
            print("    3. 或使用 diffusers 格式的 AnimateDiff pipeline")
            print(f"    4. 放置到: {model_path}")
            print("  注意: AnimateDiff 需要配合 SDXL base model 使用（已存在）")
            
    except Exception as e:
        print(f"✗ AnimateDiff 模型下载失败: {e}")
        import traceback
        traceback.print_exc()
        print("  提示: 这是阶段2的模型，如果当前不需要可以跳过")

def download_realesrgan_models(config):
    """下载 Real-ESRGAN 模型（阶段2）- 支持动漫模型和通用模型"""
    print("\n" + "="*60)
    print("下载 Real-ESRGAN 模型（阶段2）")
    print("="*60)
    
    # 从配置中读取 Real-ESRGAN 路径
    postprocess_config = config.get('gpu', {}).get('postprocess', {})
    if not postprocess_config:
        # 尝试从 composition 配置读取
        composition_config = config.get('composition', {}).get('postprocess', {})
        if composition_config:
            postprocess_config = composition_config
    
    model_path_str = postprocess_config.get('model_path', 'models/realesrgan/RealESRGAN_x4plus_anime_6B.pth')
    model_path = Path(model_path_str)
    model_dir = model_path.parent
    
    # 创建目录
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import urllib.request
    import urllib.error
    
    def download_with_progress(url, dest_path, model_name):
        """带进度的下载函数"""
        try:
            print(f"  从 GitHub Releases 下载: {model_name}")
            sys.stdout.flush()
            
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\r  下载进度: {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
            print()  # 换行
            print(f"✓ {model_name} 已下载到: {dest_path}")
            sys.stdout.flush()
            return True
        except Exception as e:
            print(f"\n⚠ 下载失败: {e}")
            return False
    
    # 定义要下载的模型列表
    models_to_download = [
        {
            "name": "RealESRGAN_x4plus_anime_6B",
            "path": model_dir / "RealESRGAN_x4plus_anime_6B.pth",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "description": "动漫专用模型（推荐先试，适合《凡人修仙传》等3D渲染动漫）"
        },
        {
            "name": "RealESRGAN_x4plus",
            "path": model_dir / "RealESRGAN_x4plus.pth",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "description": "通用模型（备选，如果动漫模型效果不理想可尝试）"
        }
    ]
    
    try:
        print(f"\n下载 Real-ESRGAN 模型（支持切换测试）")
        print("  提示: 两个模型都会下载，可在 config.yaml 中切换使用")
        
        downloaded_count = 0
        for model_info in models_to_download:
            model_name = model_info["name"]
            model_path_file = model_info["path"]
            download_url = model_info["url"]
            description = model_info["description"]
            
            print(f"\n[{downloaded_count + 1}/{len(models_to_download)}] {model_name}")
            print(f"  说明: {description}")
            
            if model_path_file.exists():
                print(f"  ✓ 模型已存在: {model_path_file}")
                downloaded_count += 1
                continue
            
            if download_with_progress(download_url, model_path_file, model_name):
                downloaded_count += 1
            else:
                print(f"  ⚠ 下载失败，可稍后手动下载:")
                print(f"    访问: {download_url}")
                print(f"    或访问: https://github.com/xinntao/Real-ESRGAN/releases")
                print(f"    下载后放置到: {model_path_file}")
        
        print(f"\n✓ Real-ESRGAN 模型下载完成 ({downloaded_count}/{len(models_to_download)})")
        print("  提示: 可在 config.yaml 中修改 model_path 切换模型进行对比测试")
            
    except Exception as e:
        print(f"✗ Real-ESRGAN 模型下载失败: {e}")
        print("  提示: 这是阶段2的模型，如果当前不需要可以跳过")

def main():
    """主函数"""
    print("="*60)
    print("阶段1+阶段2模型下载脚本")
    print("="*60)
    
    # 显示代理配置
    if HTTP_PROXY or HTTPS_PROXY:
        print(f"\n使用代理配置:")
        if HTTP_PROXY:
            print(f"  HTTP_PROXY={HTTP_PROXY}")
        if HTTPS_PROXY:
            print(f"  HTTPS_PROXY={HTTPS_PROXY}")
    else:
        print(f"\n提示: 未检测到代理配置，将直接连接下载")
    
    print("\n将下载以下模型：")
    print("\n【阶段1 - 已完成】")
    print("  1. InstantID (Juggernaut-XL-v9-anime + ControlNet + IP-Adapter)")
    print("  2. CosyVoice-2.0-0.5B")
    print("  3. WhisperX-large-v3 + 对齐模型 + VAD 模型")
    print("\n【阶段2 - 1080P方案】")
    print("  4. AnimateDiff-SDXL-1080P (替换 SVD)")
    print("  5. Real-ESRGAN-anime1080P-v2 (替换 x4plus)")
    print("\n注意: 模型文件较大，请确保有足够的磁盘空间和网络带宽")
    
    # 检查磁盘空间（简单检查）
    import shutil
    total, used, free = shutil.disk_usage(Path.cwd())
    free_gb = free / (1024**3)
    print(f"\n当前目录可用空间: {free_gb:.2f} GB")
    if free_gb < 50:
        print("⚠ 警告: 可用空间可能不足，建议至少 50GB 可用空间")
    
    # 加载配置
    try:
        config = load_config()
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}")
        sys.exit(1)
    
    # 下载模型
    try:
        # 阶段1模型
        download_instantid_models(config)
        download_cosyvoice_models(config)
        download_whisperx_models(config)
        
        # 阶段2模型（1080P方案）
        print("\n" + "="*60)
        print("开始下载阶段2模型（1080P方案）")
        print("="*60)
        download_animatediff_models(config)
        download_realesrgan_models(config)
        
        print("\n" + "="*60)
        print("✓ 模型下载完成！")
        print("="*60)
        print("\n下一步:")
        print("  1. 检查模型文件是否完整")
        print("  2. 运行测试脚本验证模型加载")
        print("  3. 阶段1模型可直接使用")
        print("  4. 阶段2模型需要更新代码以支持 AnimateDiff-SDXL-1080P")
        
    except KeyboardInterrupt:
        print("\n\n⚠ 下载被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 下载过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

