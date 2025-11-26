#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景图像生成器骨架

负责根据脚本数据与配置，生成场景配图。默认使用 Stable Diffusion XL
（diffusers pipeline），也预留其它模型接口（如 ControlNet / LoRA /
IP-Adapter 等），以便后续扩展。
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional

import torch
from pathlib import Path
import huggingface_hub
from scene_intent_analyzer import SceneIntentAnalyzer
from prompt import TokenEstimator, PromptParser, PromptOptimizer, PromptBuilder
import re


class ImageGenerator:
    """场景图像生成主类"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.image_config: Dict[str, Any] = self.config.get("image", {})
        self.paths_config: Dict[str, Any] = self.config.get("paths", {})

        self.device = torch.device(
            self.image_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pipeline = None
        self.img2img_pipeline = None
        self.sdxl_pipeline = None  # 用于存储普通的 SDXL pipeline（当使用 InstantID 引擎时）
        self.pipe_name: str = self.image_config.get("model_name", "")
        self.negative_prompt: str = self.image_config.get("negative_prompt", "")
        self.base_style_prompt: str = self.image_config.get("base_style_prompt", "")
        self.environment_prompt: str = self.image_config.get("environment_prompt", "")
        self.character_prompt: str = self.image_config.get("character_prompt", "")
        
        # 加载角色和场景模板
        self.character_profiles = self._load_character_profiles()
        self.scene_profiles = self._load_scene_profiles()
        
        # 初始化通用场景意图分析器
        self.intent_analyzer = SceneIntentAnalyzer()
        
        # 设置 ascii_only_prompt（需要在 Prompt 模块初始化之前设置）
        self.ascii_only_prompt: bool = bool(self.image_config.get("ascii_only_prompt", False))
        
        # 初始化 Prompt 模块组件
        self.token_estimator = TokenEstimator(
            ascii_only_prompt=self.ascii_only_prompt
        )
        self.prompt_parser = PromptParser()
        self.prompt_optimizer = PromptOptimizer(
            token_estimator=self.token_estimator,
            parser=self.prompt_parser,
            ascii_only_prompt=self.ascii_only_prompt
        )
        self.prompt_builder = PromptBuilder(
            token_estimator=self.token_estimator,
            parser=self.prompt_parser,
            optimizer=self.prompt_optimizer,
            intent_analyzer=self.intent_analyzer,
            character_profiles=self.character_profiles,
            scene_profiles=self.scene_profiles,
            ascii_only_prompt=self.ascii_only_prompt,
            identify_characters_fn=self._identify_characters_in_scene,
            needs_character_fn=self._needs_character,
            clip_tokenizer=None  # 可选，TokenEstimator 内部会处理
        )
        
        self.width: int = int(self.image_config.get("width", 1024))
        self.height: int = int(self.image_config.get("height", 1024))
        self.img2img_strength: float = float(self.image_config.get("img2img_strength", 0.6))
        self.use_img2img: bool = bool(self.image_config.get("use_img2img", False))
        self.ip_adapter_config: Dict[str, Any] = self.image_config.get("ip_adapter", {}) or {}
        self.use_ip_adapter: bool = bool(self.image_config.get("use_ip_adapter", False))
        self.ip_adapter_weight_names: List[str] = self._as_list(self.ip_adapter_config.get("weight_name", []))
        if not self.ip_adapter_weight_names:
            self.ip_adapter_weight_names = ["ip-adapter_sdxl.bin"]

        raw_scales = self._as_list(self.ip_adapter_config.get("scale", 0.6))
        if not raw_scales:
            raw_scales = [0.6]
        if len(raw_scales) == 1 and len(self.ip_adapter_weight_names) > 1:
            raw_scales = raw_scales * len(self.ip_adapter_weight_names)
        self.ip_adapter_scales: List[float] = [float(s) for s in raw_scales]
        # 注意：ascii_only_prompt 已在上方初始化 Prompt 模块之前定义

        self.lora_config: Dict[str, Any] = self.image_config.get("lora", {}) or {}
        self.use_lora: bool = bool(self.lora_config.get("enabled", False))
        self.lora_adapter_name: str = str(self.lora_config.get("adapter_name", "default") or "default")
        self.lora_alpha: float = float(self.lora_config.get("alpha", 1.0))
        self.lora_ip_scale_multiplier: float = float(
            self.lora_config.get("ip_adapter_scale_multiplier", 0.6)
        )

        self.reference_images: List[Path] = self._load_reference_images()
        self.face_reference_dir = self.image_config.get("face_reference_dir")
        self.face_reference_images: List[Path] = self._load_face_reference_images()
        
        # 角色参考图像目录（用于存储生成的参考图像）
        self.character_reference_dir = self.image_config.get("character_reference_dir")
        if not self.character_reference_dir:
            # 如果没有配置，尝试从 face_reference_dir 推断
            if self.face_reference_dir:
                face_ref_path = Path(self.face_reference_dir)
                # 检查是否有 character_references 子目录
                char_ref_path = face_ref_path.parent / "character_references"
                if char_ref_path.exists():
                    self.character_reference_dir = str(char_ref_path)
                else:
                    # 或者检查 face_reference_dir 本身是否包含参考图像
                    if any(Path(self.face_reference_dir).glob("*_reference.png")) or any(Path(self.face_reference_dir).glob("*_reference.jpg")):
                        self.character_reference_dir = self.face_reference_dir
                    else:
                        # 默认路径
                        self.character_reference_dir = "gen_video/character_references"
            else:
                # 默认路径
                self.character_reference_dir = "gen_video/character_references"
        
        # 加载角色参考图像映射
        self.character_reference_images: Dict[str, Path] = {}
        if self.character_reference_dir:
            self.character_reference_images = self._load_character_reference_images()
        
        # 检查使用的引擎
        self.engine = self.image_config.get("engine", "sdxl")  # 默认使用 sdxl
        self.instantid_config = self.image_config.get("instantid", {})
        self.sdxl_config = self.image_config.get("sdxl", {})
        
        # InstantID 特定配置
        if self.engine == "instantid":
            instantid_width = self.instantid_config.get("width", 1536)
            instantid_height = self.instantid_config.get("height", 864)
            self.width = int(instantid_width)
            self.height = int(instantid_height)
            self.face_image_path = self.instantid_config.get("face_image_path")
            self.face_cache_enabled = self.instantid_config.get("enable_face_cache", True)
            self.face_emb_scale = float(self.instantid_config.get("face_emb_scale", 0.8))
        elif self.engine == "sdxl":
            # 使用 SDXL 配置
            sdxl_width = self.sdxl_config.get("width", 1280)
            sdxl_height = self.sdxl_config.get("height", 720)
            if sdxl_width:
                self.width = int(sdxl_width)
            if sdxl_height:
                self.height = int(sdxl_height)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load_pipeline(self) -> None:
        """按配置加载图像生成模型。

        支持两种引擎：
        - instantid: InstantID-SDXL-1080P（新方案）
        - sdxl: Stable Diffusion XL（旧方案，兼容）
        """
        
        if self.engine == "instantid":
            self._load_instantid_pipeline()
        elif self.engine == "sdxl":
            self._load_sdxl_pipeline()
        else:
            raise ValueError(f"不支持的图像生成引擎: {self.engine}，请使用 'instantid' 或 'sdxl'")
    
    def _load_instantid_pipeline(self) -> None:
        """加载 InstantID-SDXL-1080P 模型"""
        try:
            from huggingface_hub import hf_hub_download  # noqa: F401
            if not hasattr(huggingface_hub, "cached_download"):
                def _cached_download(*args, **kwargs):
                    return hf_hub_download(*args, **kwargs)
                huggingface_hub.cached_download = _cached_download  # type: ignore
            
            # 尝试导入 InstantID
            # 方式1: 从 instantid 包导入（如果已安装）
            InstantIDPipeline = None
            try:
                from instantid import InstantIDPipeline
                print("✓ 从 instantid 包导入 InstantIDPipeline")
            except ImportError:
                # 方式2: 从克隆的 InstantID 仓库导入
                try:
                    instantid_repo_path = Path(__file__).parent.parent / "InstantID"
                    if instantid_repo_path.exists():
                        import sys
                        if str(instantid_repo_path) not in sys.path:
                            sys.path.insert(0, str(instantid_repo_path))
                        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
                        InstantIDPipeline = StableDiffusionXLInstantIDPipeline
                        print("✓ 从克隆的 InstantID 仓库导入 StableDiffusionXLInstantIDPipeline")
                    else:
                        raise ImportError("未找到 InstantID 仓库")
                except ImportError as e:
                    # 方式3: 提示安装
                    print("⚠ 无法导入 InstantID")
                    print("  请选择以下方式之一：")
                    print("  1. 安装 instantid 包: pip install instantid")
                    print("  2. 或从 GitHub 安装: pip install git+https://github.com/instantX-research/InstantID.git")
                    print("  3. 或确保 InstantID 仓库在 ../InstantID 目录下")
                    raise ImportError(f"无法导入 InstantID: {e}") from e
            
            base_model = self.instantid_config.get("base_model", "Juggernaut-XL-v9-anime")
            model_path = self.instantid_config.get("model_path")
            quantization = self.instantid_config.get("quantization", "fp8")
            
            # 确定量化类型
            # 注意: 标准 SDXL 模型可能不支持 FP8 variant，需要先尝试不使用 variant
            if quantization == "fp8":
                try:
                    # 检查是否支持 FP8
                    if hasattr(torch, "float8_e4m3fn"):
                        dtype = torch.float8_e4m3fn
                        variant = "fp8"  # 尝试使用 FP8 variant
                        use_variant = True
                    else:
                        print("⚠ PyTorch 不支持 FP8，回退到 FP16")
                        dtype = torch.float16
                        variant = None
                        use_variant = False
                except AttributeError:
                    print("⚠ PyTorch 版本过低，不支持 FP8，回退到 FP16")
                    dtype = torch.float16
                    variant = None
                    use_variant = False
            elif quantization == "fp16":
                dtype = torch.float16
                variant = None  # 标准模型通常不需要 variant 参数
                use_variant = False
            else:
                dtype = torch.float32
                variant = None
                use_variant = False
            
            print(f"加载 InstantID 模型: {base_model}")
            print(f"  量化类型: {quantization} ({dtype})")
            
            # 检查本地模型路径是否有效（必须包含 model_index.json 和核心模型文件）
            use_local_model = False
            if model_path:
                model_path_obj = Path(model_path)
                # 检查是否包含 model_index.json
                has_index = (model_path_obj / "model_index.json").exists()
                # 检查是否包含核心模型组件（至少要有 unet 或 vae）
                unet_dir = model_path_obj / "unet"
                vae_dir = model_path_obj / "vae"
                has_unet = unet_dir.exists() and (list(unet_dir.glob("*.safetensors")) or list(unet_dir.glob("*.bin")))
                has_vae = vae_dir.exists() and (list(vae_dir.glob("*.safetensors")) or list(vae_dir.glob("*.bin")))
                
                if has_index and (has_unet or has_vae):
                    use_local_model = True
                    print(f"  使用本地模型: {model_path}")
                elif has_index:
                    print(f"  ⚠ 本地路径存在但缺少核心模型文件（unet/vae）: {model_path}")
                    print(f"  将从 HuggingFace 下载基础模型: {base_model}")
                elif model_path_obj.exists():
                    print(f"  ⚠ 本地路径存在但缺少 model_index.json: {model_path}")
                    print(f"  将从 HuggingFace 下载基础模型: {base_model}")
            
            # 加载 ControlNet（InstantID 必需）
            controlnet_path = self.instantid_config.get("controlnet_path")
            if not controlnet_path:
                raise ValueError("InstantID 需要 controlnet_path 配置")
            
            controlnet_path_obj = Path(controlnet_path)
            if not controlnet_path_obj.is_absolute():
                config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
                controlnet_path_obj = config_dir / controlnet_path
                if not controlnet_path_obj.exists():
                    controlnet_path_obj = Path.cwd() / controlnet_path
            
            # 查找 ControlNet 模型目录（可能是 ControlNet 或 ControlNet/ControlNetModel）
            controlnet_model_path = None
            if (controlnet_path_obj / "config.json").exists():
                # 直接是模型目录
                controlnet_model_path = controlnet_path_obj
            elif (controlnet_path_obj / "ControlNetModel" / "config.json").exists():
                # 在 ControlNetModel 子目录中
                controlnet_model_path = controlnet_path_obj / "ControlNetModel"
            else:
                # 尝试查找包含 config.json 的子目录
                for subdir in controlnet_path_obj.iterdir():
                    if subdir.is_dir() and (subdir / "config.json").exists():
                        controlnet_model_path = subdir
                        break
            
            if not controlnet_model_path or not controlnet_model_path.exists():
                raise FileNotFoundError(
                    f"ControlNet 模型未找到: {controlnet_path}\n"
                    f"请确保路径包含 config.json 文件，或使用 ControlNet/ControlNetModel 结构"
                )
            
            print(f"  加载 ControlNet: {controlnet_model_path}")
            from diffusers.models import ControlNetModel
            controlnet = ControlNetModel.from_pretrained(
                str(controlnet_model_path),
                torch_dtype=dtype,
            )
            print("  ✓ ControlNet 加载成功")
            
            # 加载模型
            # 注意: InstantID Pipeline 需要传入 controlnet 参数
            if use_local_model:
                # 使用本地路径
                try:
                    self.pipeline = InstantIDPipeline.from_pretrained(
                        str(model_path),
                        controlnet=controlnet,
                        torch_dtype=dtype,
                    )
                except Exception as e:
                    # 如果 FP8 失败，自动回退到 FP16
                    if "Float8" in str(e) or "float8" in str(e).lower():
                        print(f"  ⚠ FP8 不支持，自动回退到 FP16")
                        self.pipeline = InstantIDPipeline.from_pretrained(
                            str(model_path),
                            controlnet=controlnet,
                            torch_dtype=torch.float16,
                        )
                    else:
                        raise
            else:
                # 从 HuggingFace 下载
                print(f"  从 HuggingFace 下载基础模型: {base_model}")
                print(f"  提示: 首次下载可能需要较长时间，请耐心等待...")
                
                # 尝试加载模型，如果失败则尝试不使用 variant 或备用模型
                try:
                    self.pipeline = InstantIDPipeline.from_pretrained(
                        base_model,
                        controlnet=controlnet,
                        torch_dtype=dtype,
                    )
                except Exception as e:
                    print(f"  ⚠ 加载模型失败: {e}")
                    # 如果使用 variant 失败，尝试不使用 variant
                    if use_variant and variant:
                        print(f"  尝试不使用 variant 参数（模型可能不支持 {variant} variant）...")
                        # 如果 FP8 失败，自动回退到 FP16
                        if "Float8" in str(e) or "float8" in str(e).lower():
                            print(f"  ⚠ FP8 不支持，自动回退到 FP16")
                            try:
                                self.pipeline = InstantIDPipeline.from_pretrained(
                                    base_model,
                                    controlnet=controlnet,
                                    torch_dtype=torch.float16,
                                )
                                print(f"  ✓ 成功使用 FP16 加载模型")
                            except Exception as e2:
                                raise RuntimeError(
                                    f"无法加载基础模型: {base_model}\n"
                                    f"FP8 错误: {e}\nFP16 错误: {e2}\n"
                                    f"请检查模型路径或网络连接。"
                                )
                        else:
                            raise
                    else:
                        raise
                
                # 如果配置了 model_path，保存下载的模型
                if model_path:
                    model_path_obj = Path(model_path)
                    model_path_obj.mkdir(parents=True, exist_ok=True)
                    print(f"  模型已缓存到 HuggingFace 缓存目录")
                    print(f"  提示: 如需保存到 {model_path}，请手动复制或使用 snapshot_download")
            
            # 加载 IP-Adapter（InstantID 必需）
            ip_adapter_path = self.instantid_config.get("ip_adapter_path")
            if not ip_adapter_path:
                raise ValueError("InstantID 需要 ip_adapter_path 配置")
            
            ip_adapter_path_obj = Path(ip_adapter_path)
            if not ip_adapter_path_obj.is_absolute():
                config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
                ip_adapter_path_obj = config_dir / ip_adapter_path
                if not ip_adapter_path_obj.exists():
                    ip_adapter_path_obj = Path.cwd() / ip_adapter_path
            
            # 查找 IP-Adapter 文件（可能是 .bin 或 .safetensors）
            ip_adapter_file = None
            if ip_adapter_path_obj.is_file():
                ip_adapter_file = ip_adapter_path_obj
            elif ip_adapter_path_obj.is_dir():
                # 查找 .bin 或 .safetensors 文件
                for ext in ['.bin', '.safetensors']:
                    found = list(ip_adapter_path_obj.glob(f'*{ext}'))
                    if found:
                        ip_adapter_file = found[0]
                        break
            
            if ip_adapter_file and ip_adapter_file.exists():
                print(f"  加载 IP-Adapter: {ip_adapter_file}")
                # InstantID 的 IP-Adapter 使用标准的 load_ip_adapter 方法
                # 注意：不同版本的 diffusers 有不同的参数要求
                try:
                    # 对于 InstantID pipeline，优先使用 load_ip_adapter_instantid
                    if hasattr(self.pipeline, "load_ip_adapter_instantid"):
                        # InstantID 特定的方法，优先使用
                        self.pipeline.load_ip_adapter_instantid(str(ip_adapter_file))
                        # 验证 IP-Adapter 是否正确加载
                        if not hasattr(self.pipeline, "image_proj_model_in_features"):
                            # 如果 image_proj_model_in_features 未设置，尝试手动设置
                            if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                                try:
                                    # InstantID 默认使用 512
                                    self.pipeline.image_proj_model_in_features = 512
                                    print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                                except Exception as e:
                                    raise RuntimeError(f"load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置且无法手动设置: {e}")
                            else:
                                raise RuntimeError("load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置，IP-Adapter 可能未正确加载")
                        print("  ✓ IP-Adapter 加载成功（使用 load_ip_adapter_instantid）")
                    elif hasattr(self.pipeline, "load_ip_adapter"):
                        # 标准 diffusers IP-Adapter 加载方法（用于非 InstantID pipeline）
                        # 检查 load_ip_adapter 的签名，确定需要哪些参数
                        import inspect
                        sig = inspect.signature(self.pipeline.load_ip_adapter)
                        params = list(sig.parameters.keys())
                        
                        # 如果方法需要 pretrained_model_name_or_path_or_dict 作为第一个参数
                        if "pretrained_model_name_or_path_or_dict" in params:
                            # 新版本：第一个参数是路径或字典，然后是 subfolder 和 weight_name
                            if ip_adapter_file.is_file():
                                # 如果是文件，pretrained_model_name_or_path_or_dict 是父目录
                                pretrained_path = str(ip_adapter_file.parent)
                                weight_name = ip_adapter_file.name
                            else:
                                # 如果是目录，pretrained_path 是目录路径
                                pretrained_path = str(ip_adapter_file)
                                weight_name = "ip-adapter.bin"
                            
                            # 检查是否需要 subfolder 参数
                            if "subfolder" in params:
                                self.pipeline.load_ip_adapter(
                                    pretrained_path,
                                    subfolder="",  # diffusers 要求字符串，空字符串表示根目录
                                    weight_name=weight_name
                                )
                            else:
                                self.pipeline.load_ip_adapter(
                                    pretrained_path,
                                    weight_name=weight_name
                                )
                            print(f"  ✓ IP-Adapter 加载成功（使用 load_ip_adapter，pretrained_path={pretrained_path}, weight_name={weight_name}）")
                        elif "subfolder" in params and "weight_name" in params:
                            # 中等版本：需要 subfolder 和 weight_name
                            if ip_adapter_file.is_file():
                                subfolder = str(ip_adapter_file.parent)
                                weight_name = ip_adapter_file.name
                            else:
                                subfolder = str(ip_adapter_file)
                                weight_name = "ip-adapter.bin"
                            
                            self.pipeline.load_ip_adapter(
                                subfolder=subfolder,
                                weight_name=weight_name
                            )
                            print(f"  ✓ IP-Adapter 加载成功（使用 load_ip_adapter，subfolder={subfolder}, weight_name={weight_name}）")
                        else:
                            # 旧版本，只需要文件路径
                            self.pipeline.load_ip_adapter(str(ip_adapter_file))
                            print("  ✓ IP-Adapter 加载成功（使用 load_ip_adapter，旧版本）")
                    elif hasattr(self.pipeline, "load_ip_adapter_instantid"):
                        # 如果存在 InstantID 特定的方法，使用它
                        self.pipeline.load_ip_adapter_instantid(str(ip_adapter_file))
                        # 验证 IP-Adapter 是否正确加载
                        if not hasattr(self.pipeline, "image_proj_model_in_features"):
                            raise RuntimeError("load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置，IP-Adapter 可能未正确加载")
                        print("  ✓ IP-Adapter 加载成功（使用 load_ip_adapter_instantid）")
                    else:
                        # InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载
                        # 或者需要通过其他方式加载
                        print("  ℹ InstantID pipeline 可能已经包含 IP-Adapter，或需要重新创建 pipeline")
                        # 验证 IP-Adapter 是否正确加载
                        if not hasattr(self.pipeline, "image_proj_model_in_features"):
                            print("  ⚠ 警告：未检测到 image_proj_model_in_features，IP-Adapter 可能未正确加载")
                        else:
                            print("  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model_in_features）")
                except Exception as e:
                    print(f"  ⚠ IP-Adapter 加载失败: {e}")
                    # 检查是否已经加载了 IP-Adapter
                    # 对于 InstantID，优先检查 image_proj_model_in_features
                    if hasattr(self.pipeline, "image_proj_model_in_features"):
                        print("  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model_in_features）")
                    elif hasattr(self.pipeline, "image_proj_model"):
                        # 如果有 image_proj_model 但没有 image_proj_model_in_features，说明加载不完整
                        print("  ⚠ 检测到 image_proj_model 但缺少 image_proj_model_in_features，IP-Adapter 加载不完整")
                        print("  ⚠ 尝试手动设置 image_proj_model_in_features...")
                        try:
                            # 尝试从 image_proj_model 推断 image_emb_dim
                            # InstantID 默认使用 512
                            self.pipeline.image_proj_model_in_features = 512
                            print("  ✓ 已手动设置 image_proj_model_in_features = 512")
                        except Exception as e2:
                            print(f"  ✗ 无法手动设置 image_proj_model_in_features: {e2}")
                    elif hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                        print("  ✓ 检测到 IP-Adapter 已加载（通过 ip_adapter_image_processor）")
                    elif hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
                        # 注意：InstantID 不使用 prepare_ip_adapter_image_embeds，这个检测可能不准确
                        print("  ⚠ 检测到 prepare_ip_adapter_image_embeds 方法，但这可能不是 InstantID 的 IP-Adapter")
                        print("  ⚠ 对于 InstantID，应该检查 image_proj_model_in_features")
                    else:
                        print("  ⚠ 警告：未检测到 IP-Adapter 的任何加载标志，后续生成可能会失败")
            else:
                raise FileNotFoundError(f"IP-Adapter 文件未找到: {ip_adapter_path}")
            
            # 移动到设备
            if self.image_config.get("enable_cpu_offload", False):
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
                if hasattr(self.pipeline, "enable_vae_tiling"):
                    self.pipeline.enable_vae_tiling()
            
            # 配置采样器（主流优化：使用 DPM++ 获得更好质量）
            scheduler_name = self.instantid_config.get("scheduler", "EulerDiscreteScheduler")
            if scheduler_name != "EulerDiscreteScheduler":
                try:
                    from diffusers import DPMSolverMultistepScheduler
                    
                    # 获取调度器配置（从当前调度器或默认配置）
                    scheduler_config = self.instantid_config.get("scheduler_config", {})
                    
                    # 使用 DPM++ 采样器
                    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipeline.scheduler.config,
                        algorithm_type=scheduler_config.get("algorithm_type", "dpmsolver++"),
                        solver_order=scheduler_config.get("solver_order", 2),
                        lower_order_final=scheduler_config.get("lower_order_final", True),
                        use_karras_sigmas=scheduler_config.get("use_karras_sigmas", True),
                    )
                    print(f"  ✓ 已切换采样器: {scheduler_name} (DPM++，质量更好)")
                except ImportError:
                    print(f"  ⚠ 无法导入 DPMSolverMultistepScheduler，使用默认采样器")
                    print(f"  提示: 请确保 diffusers >= 0.21.0: pip install --upgrade diffusers")
                except Exception as e:
                    print(f"  ⚠ 切换采样器失败: {e}，使用默认采样器")
            else:
                print(f"  使用默认采样器: EulerDiscreteScheduler")
            
            # 初始化 FaceAnalysis（InstantID 必需，用于提取面部特征）
            try:
                from insightface.app import FaceAnalysis
                import cv2
                import numpy as np
                
                # 查找 antelopev2 模型路径（按优先级查找）
                antelopev2_path = None
                antelopev2_root = None
                
                # 优先级1: 从配置中读取（如果配置了）
                config_antelopev2 = self.instantid_config.get("antelopev2_path")
                if config_antelopev2:
                    antelopev2_path = Path(config_antelopev2)
                    if not antelopev2_path.is_absolute():
                        config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
                        antelopev2_path = config_dir / antelopev2_path
                
                # 优先级2: 相对于当前脚本目录查找
                if not antelopev2_path or not antelopev2_path.exists():
                    script_dir = Path(__file__).parent
                    antelopev2_path = script_dir / "models" / "antelopev2"
                
                # 优先级3: 相对于配置目录查找
                if not antelopev2_path.exists():
                    if hasattr(self, 'config_path'):
                        config_dir = Path(self.config_path).parent
                        antelopev2_path = config_dir / "models" / "antelopev2"
                
                # 优先级4: 相对于当前工作目录查找
                if not antelopev2_path.exists():
                    antelopev2_path = Path.cwd() / "models" / "antelopev2"
                
                # 如果找到了 antelopev2 目录，使用它
                # 注意: FaceAnalysis 的 root 参数应该是 antelopev2 目录的父目录的父目录
                # 因为 FaceAnalysis 会在 root/models/antelopev2 中查找模型
                if antelopev2_path.exists() and antelopev2_path.is_dir():
                    # antelopev2_path 是 models/antelopev2
                    # root 应该是当前目录（antelopev2_path.parent.parent）
                    antelopev2_root = str(antelopev2_path.parent.parent)
                    print(f"  使用 antelopev2 模型: {antelopev2_path}")
                    print(f"  FaceAnalysis root: {antelopev2_root}")
                    self.face_analysis = FaceAnalysis(
                        name='antelopev2',
                        root=antelopev2_root,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                else:
                    # 使用默认路径（会从网上下载到 ~/.insightface/models/antelopev2）
                    print(f"  ⚠ 未找到本地 antelopev2 模型，将使用默认路径（可能从网上下载）")
                    print(f"    查找过的路径: {antelopev2_path}")
                    self.face_analysis = FaceAnalysis(
                        name='antelopev2',
                        root='./',
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                
                self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
                self.cv2 = cv2
                self.np = np
                print("  ✓ FaceAnalysis 初始化成功")
            except ImportError as e:
                print(f"  ⚠ FaceAnalysis 初始化失败: {e}")
                print("  提示: 请安装 insightface: pip install insightface")
                self.face_analysis = None
            except Exception as e:
                print(f"  ⚠ FaceAnalysis 初始化失败: {e}")
                self.face_analysis = None
            
            # 设置面部缓存（如果启用）
            if self.face_cache_enabled and self.face_image_path:
                face_image_path = Path(self.face_image_path)
                if face_image_path.exists():
                    if face_image_path.is_dir():
                        # 如果是目录，选择第一张图片
                        face_images = sorted(face_image_path.glob("*.png")) + sorted(face_image_path.glob("*.jpg"))
                        if face_images:
                            face_image_path = face_images[0]
                        else:
                            print("⚠ 面部参考目录中没有找到图片")
                            face_image_path = None
                    
                    if face_image_path and face_image_path.is_file():
                        print(f"设置面部缓存: {face_image_path}")
                        try:
                            # InstantID 的面部缓存 API
                            if hasattr(self.pipeline, "set_face_cache"):
                                self.pipeline.set_face_cache(str(face_image_path))
                            elif hasattr(self.pipeline, "prepare_face_emb"):
                                # 备用方法：准备面部嵌入
                                from PIL import Image
                                face_img = Image.open(face_image_path).convert("RGB")
                                self.pipeline.prepare_face_emb(face_img)
                        except Exception as e:
                            print(f"⚠ 设置面部缓存失败: {e}")
                            print("  将继续使用单次面部特征提取")
            
            self.pipe_name = base_model
            print("✓ InstantID 模型加载成功")
            if self.use_lora:
                self._load_lora()
            
        except ImportError as exc:
            raise RuntimeError("未安装 instantid，请先 `pip install instantid`.") from exc
        except Exception as e:
            raise RuntimeError(f"加载 InstantID 模型失败: {e}") from e
    
    def _load_sdxl_pipeline(self) -> None:
        """加载 SDXL 模型（旧方案，兼容）"""
        model_path = self.sdxl_config.get("model_path") or self.image_config.get("model_path")
        if not model_path:
            raise ValueError("image.sdxl.model_path 或 image.model_path 未配置，无法加载模型")

        try:
            from huggingface_hub import hf_hub_download  # noqa: F401
            if not hasattr(huggingface_hub, "cached_download"):
                def _cached_download(*args, **kwargs):
                    return hf_hub_download(*args, **kwargs)
                huggingface_hub.cached_download = _cached_download  # type: ignore

            from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

            dtype = torch.float16 if self.image_config.get("mixed_precision", "bf16") == "fp16" else torch.float32

            pipe_kwargs = {
                "torch_dtype": dtype,
            }

            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                **pipe_kwargs,
            )

            if self.image_config.get("enable_cpu_offload", True):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
                if self.image_config.get("enable_vae_tiling", True) and hasattr(self.pipeline, "enable_vae_tiling"):
                    self.pipeline.enable_vae_tiling()

            if self.use_img2img and self.reference_images:
                try:
                    self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**self.pipeline.components)
                except Exception:
                    self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_path,
                        **pipe_kwargs,
                    )

                if self.image_config.get("enable_cpu_offload", True):
                    self.img2img_pipeline.enable_model_cpu_offload()
                else:
                    self.img2img_pipeline = self.img2img_pipeline.to(self.device)
                    if self.image_config.get("enable_vae_tiling", True) and hasattr(self.img2img_pipeline, "enable_vae_tiling"):
                        self.img2img_pipeline.enable_vae_tiling()

            if self.use_ip_adapter:
                self._load_ip_adapter()

            if self.use_lora:
                self._load_lora()

            self.pipe_name = model_path
            print("✓ SDXL 模型加载成功")
        except ImportError as exc:
            raise RuntimeError("未安装 diffusers，请先 `pip install diffusers`.") from exc

    def _load_reference_images(self) -> List[Path]:
        ref_dir = self.image_config.get("reference_image_dir")
        pattern = self.image_config.get("reference_image_pattern", "*.jpg")
        if not ref_dir:
            return []

        ref_dir_path = Path(ref_dir)
        if not ref_dir_path.exists():
            return []

        images: List[Path] = sorted(ref_dir_path.glob(pattern))
        if not images:
            images = sorted(ref_dir_path.glob("*.png"))
        return images

    def _load_face_reference_images(self) -> List[Path]:
        if not self.face_reference_dir:
            return []

        dir_path = Path(self.face_reference_dir)
        if not dir_path.exists():
            return []

        face_images = sorted(dir_path.glob("*.png")) + sorted(dir_path.glob("*.jpg"))
        return face_images

    def _select_reference_image(self, scene: Dict[str, Any], index: int) -> Optional[Path]:
        """选择参考图像。
        
        注意：不使用场景中已有的 image_path（可能是输入目录中的旧图），
        只从配置的 reference_image_dir 中选择，确保使用高质量的参考图。
        """
        if not (self.use_img2img or self.use_ip_adapter):
            return None

        # 不从场景的 image_path 选择（可能是输入目录中的旧图，质量不好）
        # 只从配置的 reference_image_dir 中选择
        if self.reference_images:
            return self.reference_images[(index - 1) % len(self.reference_images)]
        return None

    def _load_character_reference_images(self) -> Dict[str, Path]:
        """加载角色参考图像映射（从 character_reference_dir 加载）"""
        character_refs = {}
        if not self.character_reference_dir:
            return character_refs
        
        ref_dir = Path(self.character_reference_dir)
        if not ref_dir.exists():
            return character_refs
        
        # 查找所有 {character_id}_reference.png 文件
        for ref_file in ref_dir.glob("*_reference.png"):
            # 提取角色ID（例如：huangliang_lingjun_reference.png -> huangliang_lingjun）
            char_id = ref_file.stem.replace("_reference", "")
            character_refs[char_id] = ref_file
        
        # 也支持 .jpg 格式
        for ref_file in ref_dir.glob("*_reference.jpg"):
            char_id = ref_file.stem.replace("_reference", "")
            character_refs[char_id] = ref_file
        
        if character_refs:
            print(f"  ✓ 已加载 {len(character_refs)} 个角色参考图像")
            for char_id, ref_path in sorted(character_refs.items()):
                print(f"    - {char_id}: {ref_path.name}")
        
        return character_refs
    
    def _select_face_reference_image(self, index: int, character_id: Optional[str] = None) -> Optional[Path]:
        """选择面部参考图像
        
        Args:
            index: 场景索引（用于循环选择）
            character_id: 角色ID（如果提供，优先使用对应的参考图像）
        """
        # 如果提供了角色ID，优先使用对应的参考图像
        if character_id and character_id in self.character_reference_images:
            ref_path = self.character_reference_images[character_id]
            print(f"  ✓ 使用角色参考图像: {character_id} -> {ref_path.name}")
            return ref_path
        
        # 如果没有找到对应的参考图像，使用默认的循环选择方式
        if not self.face_reference_images:
            return None
        return self.face_reference_images[(index - 1) % len(self.face_reference_images)]

    def _load_ip_adapter(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("pipeline 未加载，无法装载 IP-Adapter")

        model_path = self.ip_adapter_config.get("model_path")
        if not model_path:
            raise ValueError("image.ip_adapter.model_path 未配置，无法加载 IP-Adapter 权重")

        weight_name = self.ip_adapter_weight_names
        subfolder = self.ip_adapter_config.get("subfolder", "sdxl_models")

        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"未找到 IP-Adapter 权重目录: {path_obj}")

        def _load_into(pipe: Any) -> None:
            # 检查 IP-Adapter 是否已经加载
            ip_adapter_already_loaded = False
            if hasattr(pipe, "ip_adapter_image_processor") and pipe.ip_adapter_image_processor is not None:
                ip_adapter_already_loaded = True
                print(f"  ℹ IP-Adapter 已加载，跳过重新加载")
            elif hasattr(pipe, "prepare_ip_adapter_image_embeds"):
                # 尝试检查是否已加载（通过检查是否有 image_projection_layers）
                try:
                    # 如果 IP-Adapter 已加载，prepare_ip_adapter_image_embeds 应该可以调用
                    # 但这里只是检查，不实际调用
                    ip_adapter_already_loaded = False  # 保守起见，假设未加载
                except:
                    ip_adapter_already_loaded = False
            
            if not ip_adapter_already_loaded:
                try:
                    # 每次加载前都先尝试卸载旧的 IP-Adapter（如果存在）
                    # 这样可以确保每次都能成功加载新的 IP-Adapter
                    if hasattr(pipe, "disable_ip_adapter"):
                        try:
                            pipe.disable_ip_adapter()
                            print(f"  ℹ 已卸载旧的 IP-Adapter（如果存在）")
                        except:
                            pass
                    elif hasattr(pipe, "unload_ip_adapter"):
                        try:
                            pipe.unload_ip_adapter()
                            print(f"  ℹ 已卸载旧的 IP-Adapter（如果存在）")
                        except:
                            pass
                    
                    # 加载新的 IP-Adapter
                    pipe.load_ip_adapter(
                        str(path_obj),
                        subfolder=subfolder,
                        weight_name=weight_name,
                    )
                    print(f"  ✓ IP-Adapter 加载成功")
                except AttributeError as exc:
                    raise RuntimeError("当前 diffusers 版本不支持 IP-Adapter，请升级到 0.21.0 及以上版本。") from exc
                except Exception as e:
                    # 如果加载失败，可能是因为已经加载了不同的 IP-Adapter
                    error_msg = str(e).lower()
                    if "already loaded" in error_msg or "already exists" in error_msg:
                        print(f"  ℹ IP-Adapter 已加载，尝试强制重新加载...")
                        # 尝试强制卸载并重新加载
                        try:
                            if hasattr(pipe, "disable_ip_adapter"):
                                pipe.disable_ip_adapter()
                            elif hasattr(pipe, "unload_ip_adapter"):
                                pipe.unload_ip_adapter()
                            # 重新尝试加载
                            pipe.load_ip_adapter(
                                str(path_obj),
                                subfolder=subfolder,
                                weight_name=weight_name,
                            )
                            print(f"  ✓ IP-Adapter 强制重新加载成功")
                        except Exception as e2:
                            print(f"  ⚠ 强制重新加载失败: {e2}，使用已加载的 IP-Adapter")
                    else:
                        raise RuntimeError(f"IP-Adapter 加载失败: {e}") from e
            else:
                # 即使已加载，也尝试重新加载以确保使用最新的配置
                print(f"  ℹ IP-Adapter 已加载，但为了确保使用正确的适配器，尝试重新加载...")
                try:
                    # 先卸载
                    if hasattr(pipe, "disable_ip_adapter"):
                        pipe.disable_ip_adapter()
                    elif hasattr(pipe, "unload_ip_adapter"):
                        pipe.unload_ip_adapter()
                    # 重新加载
                    pipe.load_ip_adapter(
                        str(path_obj),
                        subfolder=subfolder,
                        weight_name=weight_name,
                    )
                    print(f"  ✓ IP-Adapter 重新加载成功")
                except Exception as e:
                    print(f"  ⚠ 重新加载失败: {e}，使用已加载的 IP-Adapter")

            if hasattr(pipe, "set_ip_adapter_scale"):
                scale = (
                    self.ip_adapter_scales
                    if len(self.ip_adapter_scales) > 1
                    else self.ip_adapter_scales[0]
                )
                pipe.set_ip_adapter_scale(scale)

        _load_into(self.pipeline)

        if self.img2img_pipeline is not None:
            _load_into(self.img2img_pipeline)

    def _load_lora(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("pipeline 未加载，无法装载 LoRA")

        weights_path = self.lora_config.get("weights_path")
        if not weights_path:
            raise ValueError("image.lora.weights_path 未配置，无法加载 LoRA 权重")

        path_obj = Path(weights_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"未找到 LoRA 权重文件: {path_obj}")

        adapter_name = self.lora_adapter_name or "default"
        alpha = self.lora_alpha
        load_kwargs: Dict[str, Any] = {}

        if bool(self.lora_config.get("force_cpu", False)):
            load_kwargs["device"] = torch.device("cpu")

        def _load_into(pipe: Any) -> None:
            pipe.load_lora_weights(
                str(path_obj),
                adapter_name=adapter_name,
                **load_kwargs,
            )
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters([adapter_name], adapter_weights=[alpha])

        print(f"  加载 LoRA 权重: {path_obj}")
        _load_into(self.pipeline)

        if self.img2img_pipeline is not None:
            _load_into(self.img2img_pipeline)

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------
    def _identify_characters_in_scene(self, scene: Dict[str, Any]) -> List[str]:
        """识别场景中的所有角色（不仅仅是韩立）。
        
        Returns:
            角色ID列表，例如 ['hanli', 'huangliang_lingjun', 'huan_cangqi']
        """
        identified_characters = []
        
        # 获取场景文本
        text_parts = [
            scene.get("title", ""),
            scene.get("description", ""),
            scene.get("prompt", ""),
            scene.get("narration", ""),
        ]
        combined_text = " ".join(str(p) for p in text_parts).lower()
        
        # 检查 visual.character_pose
        visual = scene.get("visual", {}) or {}
        if isinstance(visual, dict):
            character_pose = str(visual.get("character_pose", "")).lower()
            combined_text += " " + character_pose
        
        # 角色关键词映射（中文和英文）
        character_keywords = {
            "hanli": ["韩立", "han li", "hanli", "主角", "main character", "hero"],
            "huangliang_lingjun": ["黄粱灵君", "huangliang", "huangliang spirit lord", "黄粱"],
            "huan_cangqi": ["寰姓少年", "寰天奇", "huan cangqi", "huan tianqi", "huan youth", "寰"],
            "dumu_juren": ["独目巨人", "one-eyed giant", "giant", "巨人"],
        }
        
        # 检查每个角色
        for char_id, keywords in character_keywords.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    if char_id not in identified_characters:
                        identified_characters.append(char_id)
                    break
        
        return identified_characters
    
    def _needs_character(self, scene: Dict[str, Any]) -> bool:
        """判断场景是否需要主角（韩立）。
        
        检查场景描述、标题、提示词中是否包含主角相关关键词。
        同时识别纯环境场景（只有环境，不需要人物）。
        """
        # 首先检查是否有明确的"纯环境"标记
        if scene.get("environment_only") is True:
            return False
        
        # 检查 visual.composition 是否明确表示只有环境
        visual = scene.get("visual") or {}
        if isinstance(visual, dict):
            composition = str(visual.get("composition", "")).lower()
            # 如果 composition 中明确表示只有环境/物体，没有人物
            env_only_keywords = [
                "only environment", "environment only", "no character", "no person",
                "pure environment", "landscape only", "scene only",
                "只有环境", "纯环境", "无人物", "仅环境"
            ]
            if any(kw in composition for kw in env_only_keywords):
                return False
        
        keywords = ["韩立", "han li", "主角", "hero", "cultivator", "main character"]
        text_parts = [
            scene.get("title", ""),
            scene.get("description", ""),
            scene.get("prompt", ""),
            scene.get("action", ""),
        ]
        combined_text = " ".join(str(p) for p in text_parts).lower()
        
        # 检查是否有主角关键词
        has_character_keyword = False
        for keyword in keywords:
            if keyword.lower() in combined_text:
                has_character_keyword = True
                break
        
        # 检查 characters 字段
        has_characters_field = False
        characters = scene.get("characters", [])
        if characters:
            for char in characters:
                name = str(char.get("name", "")).lower()
                if any(kw in name for kw in ["han", "韩立", "主角"]):
                    has_characters_field = True
                    break
        
        # 如果明确有主角关键词或字段，返回 True
        if has_character_keyword or has_characters_field:
            return True
        
        # 检查是否是纯环境场景（描述中只有环境/物体，没有人物动作）
        # 纯环境场景的特征：
        # 1. description 和 prompt 中只描述环境、物体、现象，没有人物动作
        # 2. 没有 action 字段或 action 是观察类（observe_xxx）
        description = str(scene.get("description", "")).lower()
        prompt = str(scene.get("prompt", "")).lower()
        action = str(scene.get("action", "")).lower()
        
        # 纯环境关键词（表示只有环境/物体）
        env_only_indicators = [
            "出现", "显现", "露出", "透出", "发出", "传来", "形成", "扩大", "下坠",
            "appears", "revealed", "leaks", "emerges", "forms", "expands", "collapses"
        ]
        
        # 人物动作关键词（表示有人物）
        character_action_keywords = [
            "踏入", "停下", "站起", "望向", "半蹲", "跃下", "检查", "靠近", "准备",
            "walking", "stands", "gazes", "kneels", "jumps", "examines", "approaches", "prepares",
            "han li", "character", "person", "figure", "cultivator"
        ]
        
        # 如果描述和 prompt 中有环境指示词，但没有人物动作关键词
        has_env_indicator = any(ind in description or ind in prompt for ind in env_only_indicators)
        has_character_action = any(kw in description or kw in prompt or kw in action for kw in character_action_keywords)
        
        # 如果只有环境指示词，没有人物动作，且 action 是观察类或为空
        if has_env_indicator and not has_character_action:
            if not action or action.startswith("observe") or "observe" in action:
                print(f"  ✓ 检测到纯环境场景（无人物），跳过角色生成")
                return False
        
        return False

    # 注意：以下方法已迁移到 prompt.PromptBuilder
    # - _convert_motion_to_prompt
    # - _convert_camera_to_prompt
    # - _looks_like_camera_prompt
    # - _estimate_clip_tokens
    # - _analyze_prompt_importance
    # - _translate_chinese_to_english
    # - _smart_optimize_prompt
    # - _extract_first_keyword
    # - _extract_core_keywords
    # - _build_character_prompt
    # - _build_character_prompt_compact
    # 如需查看实现，请参考 prompt/builder.py, prompt/parser.py, prompt/optimizer.py, prompt/token_estimator.py

    def build_prompt(self, scene: Dict[str, Any], include_character: Optional[bool] = None, script_data: Dict[str, Any] = None, previous_scene: Optional[Dict[str, Any]] = None) -> str:
        """根据场景数据构建 prompt。
        
        通用版本：基于场景意图分析，智能构建Prompt，不依赖特殊规则。
        
        Args:
            scene: 场景数据字典
            include_character: 是否包含主角描述。None 时自动判断
            script_data: 脚本数据（用于场景模板匹配）
            previous_scene: 前一个场景（用于连贯性）
            
        Returns:
            构建好的prompt字符串
        """
        # 委托给 PromptBuilder
        return self.prompt_builder.build(
            scene=scene,
            include_character=include_character,
            script_data=script_data,
            previous_scene=previous_scene
        )
    
    # ======================================================================
    # 注意：旧的 build_prompt 实现已迁移到 prompt.PromptBuilder
    # 如需查看旧实现，请参考 prompt/builder.py
    # ======================================================================

    # ------------------------------------------------------------------
    # 图像生成核心
    # ------------------------------------------------------------------
    def generate_image(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        reference_image_path: Optional[Path] = None,
        face_reference_image_path: Optional[Path] = None,
        use_lora: Optional[bool] = None,
        scene: Optional[Dict[str, Any]] = None,
        init_image: Optional[Path] = None,  # 用于场景连贯性的前一个场景图像
    ) -> Path:
        """调用 pipeline 生成单张图像
        
        Args:
            use_lora: 是否使用 LoRA。None 时使用配置中的默认值（仅 SDXL）
        """
        
        # 检查场景中的角色，确定使用哪种生成方法
        use_text_to_image = False
        primary_character = None
        has_character_reference = False
        
        if scene:
            identified_characters = self._identify_characters_in_scene(scene)
            if identified_characters:
                primary_character = identified_characters[0]
                # 重要：韩立角色应该使用原来的高质量参考图（face_image_path），而不是生成的参考图
                # 这样可以确保人脸相似度和发型一致性
                if primary_character == "hanli":
                    # 韩立使用原来的参考图，不使用 character_reference_images
                    print(f"  ℹ 检测到角色: {primary_character}（韩立），使用原来的高质量参考图（跳过生成的参考图）")
                    # 不设置 face_reference_image_path，让它使用 self.face_image_path
                    # 这样会使用配置文件中的 韩立_mid.png
                elif primary_character in self.character_reference_images:
                    # 其他角色使用生成的参考图
                    has_character_reference = True
                    ref_path = self.character_reference_images[primary_character]
                    print(f"  ℹ 检测到角色: {primary_character}，找到参考图像: {ref_path.name}")
                    # 如果有参考图像，使用它作为 face_reference_image_path
                    if face_reference_image_path is None:
                        face_reference_image_path = ref_path
                else:
                    print(f"  ℹ 检测到角色: {primary_character}，未找到参考图像")
        
        # 根据引擎类型选择生成方法
        # 如果使用 InstantID 引擎，所有角色都可以使用 InstantID（只要有参考图像）
        if self.engine == "instantid":
            # 检查是否有参考图像
            # 对于韩立，即使没有 face_reference_image_path，也会使用 self.face_image_path
            # 所以需要检查 self.face_image_path 是否存在
            has_reference = (
                (face_reference_image_path is not None and Path(face_reference_image_path).exists()) or 
                has_character_reference or
                (primary_character == "hanli" and self.face_image_path and Path(self.face_image_path).exists())
            )
            
            if has_reference:
                # 有参考图像，使用 InstantID
                print(f"  ℹ 使用 InstantID 生成（角色: {primary_character or '未知'}）")
            else:
                # 没有参考图像，使用 SDXL 文生图
                print(f"  ⚠ 没有参考图像，无法使用 InstantID，切换到 SDXL 文生图")
                use_text_to_image = True
        
        # 根据引擎类型和是否有参考图像选择生成方法
        if self.engine == "instantid" and not use_text_to_image:
            # 韩立角色使用 InstantID（需要参考照片）
            # 如果之前使用了 SDXL 的 IP-Adapter，需要先卸载它
            # 因为 InstantID 和 SDXL 使用不同的 IP-Adapter
            if hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                # 检查是否是 SDXL 的 IP-Adapter（通过检查是否有 prepare_ip_adapter_image_embeds 方法）
                # InstantID 的 IP-Adapter 使用不同的加载方式
                try:
                    # 尝试卸载 SDXL 的 IP-Adapter
                    if hasattr(self.pipeline, "disable_ip_adapter"):
                        self.pipeline.disable_ip_adapter()
                        print(f"  ℹ 已卸载 SDXL 的 IP-Adapter，准备加载 InstantID 的 IP-Adapter")
                    elif hasattr(self.pipeline, "unload_ip_adapter"):
                        self.pipeline.unload_ip_adapter()
                        print(f"  ℹ 已卸载 SDXL 的 IP-Adapter，准备加载 InstantID 的 IP-Adapter")
                except Exception as e:
                    print(f"  ⚠ 无法卸载 SDXL 的 IP-Adapter: {e}，继续使用 InstantID")
            
            # 确保 InstantID 的 IP-Adapter 已加载
            # InstantID 的 IP-Adapter 在 _load_instantid_pipeline 中加载
            # 但如果之前卸载了，需要重新加载
            # 注意：InstantID 的 IP-Adapter 可能不需要单独加载，因为它可能在创建 pipeline 时已经自动加载
            # 或者通过其他方式集成，所以这里只是尝试加载，如果失败就跳过
            ip_adapter_loaded = False
            # 对于 InstantID，优先检查 image_proj_model_in_features（最准确的标志）
            if hasattr(self.pipeline, "image_proj_model_in_features"):
                ip_adapter_loaded = True
            elif hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                # 如果有 image_proj_model 但没有 image_proj_model_in_features，尝试手动设置
                try:
                    # InstantID 默认使用 512
                    self.pipeline.image_proj_model_in_features = 512
                    ip_adapter_loaded = True
                    print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                except Exception as e:
                    print(f"  ⚠ 无法手动设置 image_proj_model_in_features: {e}")
            elif hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                ip_adapter_loaded = True
            elif hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
                # 注意：InstantID 不使用 prepare_ip_adapter_image_embeds，这个检测可能不准确
                # 但为了兼容性，仍然检查
                ip_adapter_loaded = True
            
            if not ip_adapter_loaded:
                print(f"  ℹ InstantID 的 IP-Adapter 可能未加载，尝试加载...")
                try:
                    # 重新加载 InstantID 的 IP-Adapter
                    instantid_config = self.instantid_config
                    ip_adapter_path = instantid_config.get("ip_adapter_path")
                    if ip_adapter_path and Path(ip_adapter_path).exists():
                        ip_adapter_file = None
                        # 查找 IP-Adapter 文件
                        for pattern in ["*.safetensors", "*.bin", "*.pth"]:
                            files = list(Path(ip_adapter_path).glob(pattern))
                            if files:
                                ip_adapter_file = files[0]
                                break
                        
                        if ip_adapter_file and ip_adapter_file.exists():
                            print(f"  加载 InstantID IP-Adapter: {ip_adapter_file}")
                            # InstantID 的 IP-Adapter 使用标准的 load_ip_adapter 方法
                            try:
                                if hasattr(self.pipeline, "load_ip_adapter"):
                                    # 检查 load_ip_adapter 的签名，确定需要哪些参数
                                    import inspect
                                    sig = inspect.signature(self.pipeline.load_ip_adapter)
                                    params = list(sig.parameters.keys())
                                    
                                    # 如果方法需要 pretrained_model_name_or_path_or_dict 作为第一个参数
                                    if "pretrained_model_name_or_path_or_dict" in params:
                                        # 新版本：第一个参数是路径或字典，然后是 subfolder 和 weight_name
                                        if ip_adapter_file.is_file():
                                            pretrained_path = str(ip_adapter_file.parent)
                                            weight_name = ip_adapter_file.name
                                        else:
                                            pretrained_path = str(ip_adapter_file)
                                            weight_name = "ip-adapter.bin"
                                        
                                        # 检查是否需要 subfolder 参数
                                        if "subfolder" in params:
                                            self.pipeline.load_ip_adapter(
                                                pretrained_path,
                                                subfolder=None,  # InstantID 的 IP-Adapter 通常在根目录
                                                weight_name=weight_name
                                            )
                                        else:
                                            self.pipeline.load_ip_adapter(
                                                pretrained_path,
                                                weight_name=weight_name
                                            )
                                        print(f"  ✓ InstantID IP-Adapter 加载成功（使用 load_ip_adapter，pretrained_path={pretrained_path}, weight_name={weight_name}）")
                                        ip_adapter_loaded = True
                                    elif "subfolder" in params and "weight_name" in params:
                                        # 中等版本：需要 subfolder 和 weight_name
                                        if ip_adapter_file.is_file():
                                            subfolder = str(ip_adapter_file.parent)
                                            weight_name = ip_adapter_file.name
                                        else:
                                            subfolder = str(ip_adapter_file)
                                            weight_name = "ip-adapter.bin"
                                        
                                        self.pipeline.load_ip_adapter(
                                            subfolder=subfolder,
                                            weight_name=weight_name
                                        )
                                        print(f"  ✓ InstantID IP-Adapter 加载成功（使用 load_ip_adapter，subfolder={subfolder}, weight_name={weight_name}）")
                                        ip_adapter_loaded = True
                                    else:
                                        # 旧版本，只需要文件路径
                                        self.pipeline.load_ip_adapter(str(ip_adapter_file))
                                        print("  ✓ InstantID IP-Adapter 加载成功（使用 load_ip_adapter，旧版本）")
                                        ip_adapter_loaded = True
                                else:
                                    print("  ⚠ InstantID pipeline 不支持 load_ip_adapter 方法")
                                    print("  ℹ InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载，继续...")
                            except AttributeError as e:
                                print(f"  ⚠ 无法加载 InstantID IP-Adapter: {e}")
                                print("  ℹ InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载，继续...")
                            except Exception as e2:
                                print(f"  ⚠ 无法加载 InstantID IP-Adapter: {e2}")
                                print("  ℹ InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载，继续...")
                                # 检查是否已经加载了 IP-Adapter
                                if hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                                    print("  ✓ 检测到 IP-Adapter 已加载（通过 ip_adapter_image_processor）")
                                    ip_adapter_loaded = True
                                elif hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
                                    print("  ✓ 检测到 IP-Adapter 已加载（通过 prepare_ip_adapter_image_embeds）")
                                    ip_adapter_loaded = True
                        else:
                            print(f"  ⚠ 未找到 InstantID IP-Adapter 文件，假设已在创建 pipeline 时加载")
                    else:
                        print(f"  ⚠ InstantID IP-Adapter 路径未配置或不存在，假设已在创建 pipeline 时加载")
                except Exception as e:
                    print(f"  ⚠ 无法加载 InstantID 的 IP-Adapter: {e}")
                    print("  ℹ 假设 InstantID 的 IP-Adapter 已在创建 pipeline 时加载，继续...")
            else:
                print(f"  ✓ InstantID 的 IP-Adapter 已加载")
            
            try:
                return self._generate_image_instantid(
                    prompt, output_path, negative_prompt, guidance_scale, 
                    num_inference_steps, seed, face_reference_image_path, scene=scene,
                    init_image=init_image  # 传递前一个场景图像用于连贯性
                )
            except ValueError as e:
                # 如果 InstantID 无法识别人脸，自动回退到 SDXL 文生图
                error_msg = str(e)
                if "未检测到人脸" in error_msg or "no face" in error_msg.lower() or "face not detected" in error_msg.lower():
                    print(f"  ⚠ InstantID 无法识别人脸: {error_msg}")
                    print(f"  ℹ 自动回退到 SDXL 文生图（不使用参考图像）")
                    # 回退到 SDXL 文生图，禁用 IP-Adapter
                    return self._generate_image_sdxl(
                        prompt, output_path, negative_prompt, guidance_scale,
                        num_inference_steps, seed, reference_image_path=None, 
                        face_reference_image_path=None, use_lora=use_lora, scene=scene,
                        use_ip_adapter_override=False  # 禁用 IP-Adapter，使用纯文生图
                    )
                else:
                    # 其他类型的 ValueError，继续抛出
                    raise
        else:
            # 非韩立角色或其他情况使用 SDXL
            # 对于非韩立角色：
            # - 如果有参考图像，使用 IP-Adapter（使用参考图像）
            # - 如果没有参考图像，禁用 IP-Adapter（纯文生图）
            use_ip_adapter_for_this = self.use_ip_adapter
            if use_text_to_image and primary_character != "hanli":
                if has_character_reference:
                    # 有参考图像，使用 IP-Adapter
                    use_ip_adapter_for_this = True
                    print(f"  ℹ 非韩立角色有参考图像，启用 IP-Adapter（使用参考图像）")
                else:
                    # 没有参考图像，禁用 IP-Adapter（纯文生图）
                    use_ip_adapter_for_this = False
                    print(f"  ℹ 非韩立角色无参考图像，禁用 IP-Adapter（纯文生图）")
            
            return self._generate_image_sdxl(
                prompt, output_path, negative_prompt, guidance_scale,
                num_inference_steps, seed, reference_image_path, 
                face_reference_image_path, use_lora, scene=scene,
                use_ip_adapter_override=use_ip_adapter_for_this  # 传递 IP-Adapter 使用标志
            )
    
    def _generate_image_instantid(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        face_reference_image_path: Optional[Path] = None,
        scene: Optional[Dict[str, Any]] = None,
        init_image: Optional[Path] = None,  # 用于场景连贯性的前一个场景图像
    ) -> Path:
        """InstantID 图像生成
        
        注意：如果 prompt 中没有明确的远景关键词，会自动在开头添加，
        确保镜头足够远，避免只看到头部。
        """
        """使用 InstantID 生成图像"""
        if self.pipeline is None:
            raise RuntimeError("pipeline 未加载，请先调用 load_pipeline()")
        
        from PIL import Image
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 使用 InstantID 配置
        guidance = guidance_scale or self.instantid_config.get("guidance_scale", 7.5)
        base_steps = num_inference_steps or self.instantid_config.get("num_inference_steps", 40)  # 基础步数（使用 DPM++ 采样器时）
        
        # 检测是否是远景场景（提前检测，用于调整推理步数）
        prompt_lower = prompt.lower() if prompt else ""
        is_wide_shot_early = any(kw in prompt_lower for kw in [
            "wide shot", "full body", "full figure", "全身", "wide view", "full view", 
            "long shot", "extreme wide", "distant view", "far away", "establishing shot"
        ])
        is_full_body_early = any(kw in prompt_lower for kw in ["full body", "full figure", "全身"])
        # 确保 scene 是字典类型，避免字符串或其他类型导致的错误
        if scene and isinstance(scene, dict):
            visual = scene.get("visual", {})
            if isinstance(visual, dict):
                composition = visual.get("composition", {})
                if isinstance(composition, dict):
                    scene_camera = composition.get("camera", "").lower() if isinstance(composition.get("camera"), str) else ""
                    if not is_wide_shot_early:
                        is_wide_shot_early = any(kw in scene_camera for kw in [
                            "wide shot", "full body", "full figure", "全身", "wide view", "full view",
                            "long shot", "extreme wide", "distant view"
                        ])
                    if not is_full_body_early:
                        is_full_body_early = any(kw in scene_camera for kw in ["full body", "full figure", "全身"])
        
        # 如果是远景场景，增加推理步数以提高清晰度
        if is_wide_shot_early or is_full_body_early:
            steps = int(base_steps * 1.25)  # 增加25%的步数（40步 -> 50步，50步 -> 62.5步约63步）
            print(f"  ℹ 远景场景：推理步数从 {base_steps} 增加到 {steps}，提高清晰度")
        else:
            steps = base_steps
        
        guidance_rescale = self.instantid_config.get("guidance_rescale", None)  # CFG Rescale（可选）
        print(f"  图像生成参数: num_inference_steps={steps}, guidance_scale={guidance}", end="")
        if guidance_rescale is not None:
            print(f", guidance_rescale={guidance_rescale}")
        else:
            print()
        
        negative_prompt = (
            negative_prompt
            if negative_prompt is not None
            else self.negative_prompt
        )
        
        # 准备面部参考图像（InstantID 必需）
        # 优先级：1) face_reference_image_path 2) 韩立_face.png 3) 目录中的第一张
        face_image = None
        used_reference_path = None
        if face_reference_image_path and Path(face_reference_image_path).exists():
            face_image = Image.open(face_reference_image_path).convert("RGB")
            used_reference_path = face_reference_image_path
            print(f"  ✓ 使用传入的参考图像: {face_reference_image_path}")
        elif self.face_image_path:
            face_path = Path(self.face_image_path)
            if face_path.exists():
                if face_path.is_dir():
                    # 优先查找顺序：
                    # 1. 半身照（优先，包含完整发型信息）
                    # 2. 正脸照片（备选，用于面部嵌入）
                    preferred_names = [
                        '韩立_半身.png', 'half_body.png', 'half_body.jpg', '半身.png',  # 半身照（优先，发型完整）
                        '韩立_face.png', 'face.png', 'face.jpg', 'reference_face.png'  # 正脸照片（备选）
                    ]
                    face_image = None
                    for name in preferred_names:
                        preferred_path = face_path.parent / name
                        if preferred_path.exists():
                            face_image = Image.open(preferred_path).convert("RGB")
                            used_reference_path = preferred_path
                            print(f"  ✓ 使用优先面部参考图像: {preferred_path}")
                            if '半身' in name or 'half_body' in name.lower():
                                print(f"    (半身照：包含发型和姿势信息)")
                            break
                    
                    # 如果没找到优先图片，使用目录中的第一张
                    if not face_image:
                        face_images = sorted(face_path.glob("*.png")) + sorted(face_path.glob("*.jpg"))
                        if face_images:
                            face_image = Image.open(face_images[0]).convert("RGB")
                            used_reference_path = face_images[0]
                            print(f"  ✓ 使用目录中的第一张图片: {face_images[0]}")
                elif face_path.is_file():
                    face_image = Image.open(face_path).convert("RGB")
                    used_reference_path = face_path
                    print(f"  ✓ 使用配置文件中的参考图像: {face_path}")
        
        if not face_image:
            raise ValueError("InstantID 需要面部参考图像，请提供 face_reference_image_path 或配置 face_image_path")
        
        # 打印使用的参考图像信息（用于调试）
        if used_reference_path:
            print(f"  📸 InstantID 参考图像: {used_reference_path}")
            print(f"     图像尺寸: {face_image.size[0]}x{face_image.size[1]}")
        
        # 提取面部特征和关键点（InstantID 必需）
        if not hasattr(self, 'face_analysis') or self.face_analysis is None:
            raise RuntimeError("FaceAnalysis 未初始化，无法提取面部特征")
        
        # 调整图像大小（使用 InstantID 官方的 resize_img 函数，保持宽高比）
        def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
                      pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
            """InstantID 官方的 resize_img 函数，保持宽高比"""
            w, h = input_image.size
            if size is not None:
                w_resize_new, h_resize_new = size
            else:
                ratio = min_side / min(h, w)
                w, h = round(ratio * w), round(ratio * h)
                ratio = max_side / max(h, w)
                input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
                w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
                h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
            input_image = input_image.resize([w_resize_new, h_resize_new], mode)
            
            if pad_to_max_side:
                res = self.np.ones([max_side, max_side, 3], dtype=self.np.uint8) * 255
                offset_x = (max_side - w_resize_new) // 2
                offset_y = (max_side - h_resize_new) // 2
                res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = self.np.array(input_image)
                input_image = Image.fromarray(res)
            return input_image
        
        # 调整面部参考图像大小（保持宽高比，不拉伸）
        face_image = resize_img(face_image, max_side=1280, min_side=1024)
        
        # 提取面部信息
        face_info_list = self.face_analysis.get(self.cv2.cvtColor(self.np.array(face_image), self.cv2.COLOR_RGB2BGR))
        if not face_info_list:
            # 如果无法识别人脸，抛出异常，让调用者决定是否回退到文生图
            raise ValueError("在参考图像中未检测到人脸，请使用包含清晰人脸的图像")
        
        # 选择最大的人脸
        face_info = sorted(face_info_list, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        face_emb = face_info['embedding']
        
        # 生成面部关键点图像
        if hasattr(self, 'draw_kps'):
            draw_kps = self.draw_kps
        else:
            # 如果 draw_kps 未保存，尝试导入
            try:
                instantid_repo_path = Path(__file__).parent.parent / "InstantID"
                if instantid_repo_path.exists():
                    import sys
                    if str(instantid_repo_path) not in sys.path:
                        sys.path.insert(0, str(instantid_repo_path))
                    from pipeline_stable_diffusion_xl_instantid import draw_kps
                else:
                    raise ImportError("InstantID 仓库未找到")
            except ImportError:
                raise RuntimeError("无法导入 draw_kps 函数，请确保 InstantID 仓库存在")
        
        face_kps_raw = draw_kps(face_image, face_info['kps'])
        
        # 首先确保单人场景（在所有处理之前，确保单人约束在 prompt 最前面）
        # 检查场景是否包含角色（通过检查 scene 数据或 prompt 中是否包含角色关键词）
        has_character = scene and (scene.get("characters") or any(kw in prompt.lower() for kw in ["han li", "hanli", "cultivator", "character", "韩立", "主角", "person", "figure", "man", "people"]))
        
        # 对所有有角色的场景都在 prompt 最前面添加单人约束
        if has_character:
            prompt_lower_check = prompt.lower()
            single_person_keywords = ["single person", "lone figure", "one person", "only one", "sole character"]
            has_single_keyword = any(kw in prompt_lower_check for kw in single_person_keywords)
            
            if not has_single_keyword:
                # 在 prompt 最前面强调单人，使用最高权重（2.0，用户反馈场景5和7生成了多个人物）
                prompt = "(single person, lone figure, only one character, one person only, sole character, single individual:2.0), " + prompt
                print(f"  ✓ 已添加单人场景约束：在 prompt 最前面强调单人（权重2.0，防止重复人物）")
        
        # 检查 prompt 中是否已有明确的镜头描述（来自 camera 字段或其他来源）
        prompt_lower = prompt.lower()
        wide_shot_keywords = [
            'extreme wide shot', 'very long shot', 'distant view', 'far away',
            'wide shot', 'establishing shot', 'wide angle', 'landscape',
            '远景', '全景', '广角', 'long shot', 'extreme long shot'
        ]
        # 检查是否有任何镜头类型的关键词（包括远景、中景、近景、特写等）
        shot_type_keywords = [
            'extreme wide shot', 'very long shot', 'long shot', 'wide shot',
            'medium shot', 'mid shot', '中景',
            'close-up', 'closeup', 'close up', 'portrait', 'headshot', '特写', '近景',
            'full body', 'full figure', '全身',
            'aerial view', 'low angle', 'side view', 'back view', 'front view'
        ]
        has_any_shot_keyword = any(keyword in prompt_lower for keyword in shot_type_keywords)
        has_wide_keyword = any(keyword in prompt_lower for keyword in wide_shot_keywords)
        
        # 检查 camera 字段是否明确指定了特写（特别是眼睛特写）
        camera_desc = scene.get("camera") if scene else ""
        camera_desc_lower = (camera_desc or "").lower()
        is_eye_closeup_in_camera = any(kw in camera_desc_lower for kw in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔', 'extreme close', 'close-up', 'closeup'])
        has_closeup_in_camera = any(kw in camera_desc_lower for kw in ['close-up', 'closeup', 'close up', '特写', '近景'])
        
        # 如果已经有明确的镜头描述（来自 camera 字段），尊重用户意图，不强制添加远景
        if has_any_shot_keyword and not has_wide_keyword:
            # 有镜头描述但不是远景，说明用户想要中景或近景，不强制添加远景
            print(f"  ✓ 检测到明确的镜头描述（非远景），尊重用户意图，不强制添加远景")
        elif has_closeup_in_camera and not has_any_shot_keyword:
            # camera 字段明确指定了特写，但 prompt 中没有镜头关键词
            # 特别是眼睛特写场景，不应该强制添加远景
            if is_eye_closeup_in_camera:
                # 眼睛特写场景：添加特写描述，而不是远景
                prompt = "(extreme close-up on eyes:2.0), (eye close-up:1.8), " + prompt
                print(f"  ✓ camera 字段指定了眼睛特写，已添加眼睛特写描述（不添加远景）")
            else:
                # 其他特写场景：添加中景描述（避免身体过宽）
                prompt = "(medium shot:1.8), " + prompt
                print(f"  ✓ camera 字段指定了特写，已添加中景描述（不添加远景）")
        elif not has_any_shot_keyword:
            # 如果没有任何镜头关键词，且 camera 字段也没有指定特写
            # 对于人物场景，默认使用中景而非远景，避免人物太小和背影
            # 检查场景是否包含角色（通过检查 scene 数据或 prompt 中是否包含角色关键词）
            has_character_in_scene = scene and (scene.get("characters") or any(kw in prompt.lower() for kw in ["han li", "hanli", "cultivator", "character", "韩立", "主角", "person", "figure", "man", "people"]))
            if has_character_in_scene:
                # 人物场景：默认中景，确保人物清晰可见且正面
                prompt = "(medium shot, character clearly visible, front view, facing camera:2.0), " + prompt
                print(f"  ⚠ 未检测到任何镜头关键词，人物场景默认添加中景描述（高权重2.0），避免远景和背影")
            else:
                # 无人物场景：可以使用远景
                prompt = "(extreme wide shot:2.0), (distant view:1.8), " + prompt
            print(f"  ⚠ 未检测到任何镜头关键词，已强制在 prompt 开头添加远景描述（高权重）")
        else:
            # 检查远景关键词是否在 prompt 开头（前 100 个字符）
            prompt_start = prompt_lower[:100]
            has_wide_at_start = any(keyword in prompt_start for keyword in wide_shot_keywords)
            if not has_wide_at_start:
                # 如果有远景关键词但不在开头，在开头添加带更高权重的远景关键词
                found_keywords = [kw for kw in wide_shot_keywords if kw in prompt_lower]
                if found_keywords:
                    primary_keyword = found_keywords[0]
                    # 使用更高权重标记增强效果（2.0 表示 2 倍权重）
                    prompt = f"({primary_keyword}:2.0), " + prompt
                    print(f"  ✓ 检测到远景关键词但不在开头，已移至开头并增强权重（2.0倍）")
        
        # 检查并精简 prompt，确保不超过 77 tokens
        # 在添加镜头描述后，重新计算 token 数
        # 使用 token_estimator 进行更准确的估算（如果可用）
        token_checker = None
        if hasattr(self, 'token_estimator'):
            token_checker = self.token_estimator
        elif hasattr(self, '_clip_tokenizer') and self._clip_tokenizer is not None:
            # 使用 _clip_tokenizer 作为备选
            token_checker = self._clip_tokenizer
        
        if token_checker:
            try:
                # 使用 token_estimator 或 _clip_tokenizer 计算 token 数
                if hasattr(token_checker, 'estimate'):
                    # 使用 token_estimator
                    actual_tokens = token_checker.estimate(prompt)
                else:
                    # 使用 _clip_tokenizer
                    tokens_obj = token_checker(prompt, truncation=False, return_tensors="pt")
                    actual_tokens = tokens_obj.input_ids.shape[1]
                
                if actual_tokens > 77:
                    print(f"  ⚠ 警告: Prompt 长度 ({actual_tokens} tokens) 超过 77 tokens 限制，开始智能精简...")
                    # 智能精简策略：优先保留关键信息（角色名、动作、场景）
                    import re
                    
                    # 1. 提取所有部分
                    parts = [p.strip() for p in prompt.split(',')]
                    
                    # 2. 识别关键部分（包含角色名、关键动作等）
                    character_keywords = ['han li', 'hanli', '韩立', 'character', 'cultivator', 'male']
                    action_keywords = ['lying', 'motionless', 'facing', 'view', 'top-down', 'back']
                    style_keywords = ['xianxia', 'fantasy']
                    
                    key_parts = []  # 必须保留的关键部分
                    other_parts = []  # 其他部分
                    
                    for part in parts:
                        part_lower = part.lower()
                        is_key = (
                            any(kw in part_lower for kw in character_keywords) or
                            any(kw in part_lower for kw in action_keywords) or
                            any(kw in part_lower for kw in style_keywords)
                        )
                        if is_key:
                            key_parts.append(part)
                        else:
                            other_parts.append(part)
                    
                    # 3. 优先保留关键部分，然后添加其他部分直到达到限制
                    selected_parts = []
                    current_tokens = 0
                    
                    # 先添加关键部分
                    for part in key_parts:
                        test_prompt = ', '.join(selected_parts + [part])
                        if hasattr(token_checker, 'estimate'):
                            test_tokens = token_checker.estimate(test_prompt)
                        else:
                            test_tokens_obj = token_checker(test_prompt, truncation=False, return_tensors="pt")
                            test_tokens = test_tokens_obj.input_ids.shape[1]
                        
                        if test_tokens <= 77:
                            selected_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            # 如果添加这个关键部分会超过，尝试精简它
                            # 简化权重：将高权重降低
                            simplified = re.sub(r':\d+\.\d+', ':1.3', part)  # 降低权重
                            simplified = re.sub(r':\d+', ':1.3', simplified)  # 处理整数权重
                            test_prompt = ', '.join(selected_parts + [simplified])
                            if hasattr(token_checker, 'estimate'):
                                test_tokens = token_checker.estimate(test_prompt)
                            else:
                                test_tokens_obj = token_checker(test_prompt, truncation=False, return_tensors="pt")
                                test_tokens = test_tokens_obj.input_ids.shape[1]
                            
                            if test_tokens <= 77:
                                selected_parts.append(simplified)
                                current_tokens = test_tokens
                            # 如果精简后还是超过，跳过这个部分（但关键部分应该尽量保留）
                    
                    # 4. 添加其他部分（如果还有空间）
                    for part in other_parts:
                        test_prompt = ', '.join(selected_parts + [part])
                        if hasattr(token_checker, 'estimate'):
                            test_tokens = token_checker.estimate(test_prompt)
                        else:
                            test_tokens_obj = token_checker(test_prompt, truncation=False, return_tensors="pt")
                            test_tokens = test_tokens_obj.input_ids.shape[1]
                        
                        if test_tokens <= 77:
                            selected_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            break  # 没有更多空间
                    
                    prompt = ', '.join(selected_parts)
                    
                    # 最终验证
                    if hasattr(token_checker, 'estimate'):
                        final_tokens = token_checker.estimate(prompt)
                    else:
                        final_tokens_obj = token_checker(prompt, truncation=False, return_tensors="pt")
                        final_tokens = final_tokens_obj.input_ids.shape[1]
                    
                    print(f"  ✓ 智能精简完成: {len(selected_parts)} 个部分，{final_tokens} tokens")
                    if final_tokens > 77:
                        print(f"  ⚠ 警告: 精简后仍超过 77 tokens ({final_tokens} tokens)，可能会被截断")
            except Exception as e:
                print(f"  ⚠ Token 检查失败: {e}")
        
        # 根据场景自动调整参数（不同焦距）
        # 优先使用 camera 字段判断镜头类型，更准确
        prompt_lower = prompt.lower()
        camera_desc_lower = ""
        if scene and scene.get("camera"):
            camera_desc_lower = str(scene.get("camera", "")).lower()
        
        # 检测场景类型（优先级：camera 字段 > prompt 关键词）
        # 先检查 camera 字段
        is_wide_shot = False
        is_full_body = False
        is_close_up = False
        is_medium_shot = False
        
        if camera_desc_lower:
            # 优先使用 camera 字段判断
            if any(kw in camera_desc_lower for kw in ['wide', 'long', 'establish', '远景', '全景']):
                is_wide_shot = True
            elif any(kw in camera_desc_lower for kw in ['close', 'closeup', 'portrait', 'headshot', '特写', '近景']):
                # 检查是否是眼睛特写场景（需要保持特写）
                is_eye_closeup = any(kw in camera_desc_lower for kw in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔'])
                if is_eye_closeup:
                    # 眼睛特写场景：保持特写，不转换为中景
                    is_close_up = True
                    is_medium_shot = False
                    print(f"  ✓ 检测到眼睛特写场景，保持特写镜头（不转换为中景）")
                else:
                    # 其他特写场景：避免太近的镜头，将特写转换为中景
                    print(f"  ⚠ 检测到特写/近景镜头，为避免身体过宽和模糊，转换为中景")
                is_medium_shot = True  # 转换为中景，而不是特写
                is_close_up = False
            elif any(kw in camera_desc_lower for kw in ['medium', 'mid', '中景']):
                is_medium_shot = True
            elif any(kw in camera_desc_lower for kw in ['full', '全身']):
                is_full_body = True
        
        # 如果 camera 字段没有明确判断，再从 prompt 中检测
        if not (is_wide_shot or is_full_body or is_close_up or is_medium_shot):
            is_wide_shot = any(keyword in prompt_lower for keyword in wide_shot_keywords)
            is_full_body = any(keyword in prompt_lower for keyword in ['full body', 'full figure', '全身'])
            # 避免太近的镜头，如果检测到特写关键词，转换为中景
            # 但眼睛特写场景需要保持特写
            close_keywords_found = any(keyword in prompt_lower for keyword in ['close-up', 'closeup', 'close up', 'portrait', 'headshot', '特写', '近景'])
            eye_closeup_keywords = any(keyword in prompt_lower for keyword in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔', 'extreme close'])
            if close_keywords_found:
                if eye_closeup_keywords:
                    # 眼睛特写场景：保持特写，不转换为中景
                    is_close_up = True
                    is_medium_shot = False
                    print(f"  ✓ 检测到眼睛特写关键词，保持特写镜头（不转换为中景）")
                else:
                    print(f"  ⚠ 检测到特写/近景关键词，为避免身体过宽和模糊，转换为中景")
                    is_medium_shot = True  # 转换为中景
                is_close_up = False
            else:
                is_close_up = False
            is_medium_shot = any(keyword in prompt_lower for keyword in ['medium shot', 'mid shot', '中景'])
        
        # 检查是否是躺着姿势（lying, top-down view等），这些姿势可能影响人脸相似度
        is_lying_pose = False
        if scene:
            action = str(scene.get("action", "")).lower()
            camera_desc = str(scene.get("camera", "")).lower()
            prompt_lower_check = prompt.lower()
            is_lying_pose = (
                "lying" in action or 
                "lying" in prompt_lower_check or
                "top-down" in camera_desc or 
                "top down" in prompt_lower_check or
                "俯视" in camera_desc or
                "俯视" in prompt_lower_check
            )
        
        # 根据场景调整面部权重和 ControlNet 强度（参考官方建议）
        # 官方默认值: controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8
        # 官方建议: For higher similarity, increase both scales
        #           For over-saturation, decrease ip_adapter_scale
        #           For higher text control, decrease ip_adapter_scale
        # 注意: 用户反馈相似度不够（发型差、脸部不像），需要进一步提高权重
        # 注意: 躺着姿势可能影响人脸相似度，需要提高权重
        min_ip_adapter_scale = 0.35
        min_controlnet_scale = 0.4
        if is_wide_shot or is_full_body:
            # 远景/全身：适度降低面部权重，但确保人脸完整且相似，同时避免瘦长脸
            # 平衡权重，确保人脸完整且相似，但不过度控制导致瘦长脸
            # 提高基础权重，从0.80提高到0.85，确保人脸完整且相似（用户反馈效果不好）
            ip_adapter_scale = max(self.face_emb_scale * 0.85, 0.6)  # 从0.80提高到0.85，从0.5提高到0.6
            controlnet_scale = 0.60  # 保持0.60，平衡控制强度
            min_ip_adapter_scale = 0.6  # 从0.5提高到0.6，确保人脸完整且相似
            min_controlnet_scale = 0.45  # 保持0.45，给更多自由度，避免瘦长脸
            print(f"  检测到远景/全身场景，面部权重: {ip_adapter_scale:.2f}, ControlNet: {controlnet_scale:.2f} (确保人脸完整且比例自然)")
        elif is_close_up:
            # 近景/特写：降低ControlNet权重，避免身体过宽、模糊和横向压缩变形
            # 但保持较高的面部权重，确保人脸相似度
            ip_adapter_scale = self.face_emb_scale * 1.3  # 从1.2提高到1.3，进一步增强相似度（用户反馈站姿也不太像）
            controlnet_scale = 0.28  # 从0.30降到0.28，进一步避免身体过宽、模糊和横向压缩变形
            print(f"  检测到近景/特写场景，面部权重: {ip_adapter_scale:.2f}, ControlNet: {controlnet_scale:.2f} (降低ControlNet避免横向压缩变形)")
        elif is_medium_shot:
            # 中景/半身像：优化参数使其更自然，避免横向压缩变形
            # 1. 适度提高IP-Adapter权重确保人脸相似度，但不过高
            ip_adapter_scale = self.face_emb_scale * 1.35  # 从1.3提高到1.35，进一步增强人脸相似度（用户反馈效果不好）
            # 2. 进一步降低ControlNet权重，避免身体过宽、僵硬、横向压缩变形和瘦长脸
            controlnet_scale = 0.22  # 保持0.22，平衡控制强度
            min_ip_adapter_scale = 0.5  # 从0.4提高到0.5，确保最小权重足够高，保证人脸相似度
            min_controlnet_scale = 0.20  # 保持0.20，给更多自由度，减少压缩变形和瘦长脸
            print(f"  检测到中景/半身像场景，面部权重: {ip_adapter_scale:.2f}, ControlNet: {controlnet_scale:.2f} (优化参数避免横向压缩变形和瘦长脸)")
        else:
            # 默认：使用中等权重，确保角色一致性
            ip_adapter_scale = self.face_emb_scale * 1.2  # 从1.1提高到1.2，进一步增强角色一致性和人脸相似度（用户反馈站姿也不太像）
            controlnet_scale = 0.75
            print(f"  使用默认场景参数，面部权重: {ip_adapter_scale:.2f} (确保角色一致性和人脸相似度)")

        # 应用 face_style_auto 参数调整（如果存在）
        face_style = scene.get("face_style_auto") or {} if scene else {}
        if isinstance(face_style, dict) and face_style:
            try:
                from face_style_auto_generator import to_instantid_params
                instantid_params = to_instantid_params(face_style)
                
                # 应用强度调整到 ip_adapter_scale
                style_multiplier = instantid_params.get("ip_adapter_scale_multiplier", 1.0)
                original_scale = ip_adapter_scale
                
                # 如果 multiplier 太低，限制最小值，避免过度降低权重
                # 对于躺着姿势，需要更高的最小值（0.95），因为躺着姿势本身就需要更高权重
                # 对于远景/全身场景，也需要更高的最小值（0.9），确保人脸完整且相似
                # 对于中景/近景场景，最小值设为 0.9，确保人脸相似度
                if is_lying_pose:
                    min_multiplier = 0.95
                elif is_wide_shot or is_full_body:
                    min_multiplier = 0.9  # 从0.85提高到0.9，确保远景场景人脸完整且相似
                else:
                    min_multiplier = 0.9  # 从0.85提高到0.9，确保中景/近景场景人脸相似度
                if style_multiplier < min_multiplier:
                    print(f"    ⚠ face_style_auto multiplier ({style_multiplier:.2f}) 过低，限制为 {min_multiplier:.2f} 以保持人脸相似度")
                    style_multiplier = min_multiplier
                
                ip_adapter_scale = ip_adapter_scale * style_multiplier
                
                print(f"  应用 face_style_auto: {face_style.get('expression')}/{face_style.get('lighting')}/{face_style.get('detail')}")
                print(f"    面部权重调整: {original_scale:.2f} -> {ip_adapter_scale:.2f} (x{style_multiplier:.2f})")
            except ImportError:
                # 如果没有 face_style_auto_generator，使用简单逻辑
                detail = (face_style.get("detail") or "").lower()
                if detail in {"detailed", "cinematic"}:
                    ip_adapter_scale *= 1.1
                    controlnet_scale *= 1.05
                    print("  face_style_auto.detail=detailed/cinematic，提升面部权重")
                elif detail in {"subtle"}:
                    ip_adapter_scale *= 0.9
                    controlnet_scale *= 0.95
                    print("  face_style_auto.detail=subtle，降低面部权重")
                expression = (face_style.get("expression") or "").lower()
                if expression in {"focused", "serious", "alert"}:
                    ip_adapter_scale *= 1.05
                lighting = (face_style.get("lighting") or "").lower()
                if lighting in {"dramatic", "rim_light"}:
                    controlnet_scale *= 1.05
            except Exception as e:
                print(f"  ⚠ face_style_auto 参数应用失败: {e}")
        
        # 确保参数在合理范围内
        # 提高最小权重，确保角色一致性（发型、服饰、脸部），远景场景也要确保人脸完整
        # 用户反馈效果不好，进一步提高最小权重限制
        ip_adapter_scale = max(min_ip_adapter_scale, min(1.0, ip_adapter_scale))
        # 对于非远景场景，确保最小权重至少为 0.6（从0.5提高到0.6，提高人脸相似度）
        if not (is_wide_shot or is_full_body):
            ip_adapter_scale = max(ip_adapter_scale, 0.6)
        controlnet_scale = max(min_controlnet_scale, min(1.0, controlnet_scale))
        
        # 躺着姿势特殊处理：提高面部权重，因为躺着姿势可能影响人脸相似度
        if is_lying_pose:
            # 躺着姿势需要更高的权重来保持人脸相似度
            # 由于躺着姿势时面部角度与参考图不同，需要更高的权重来补偿
            # 提高25%，最小0.85（比之前的0.7更高，确保人脸相似度）
            ip_adapter_scale = max(ip_adapter_scale * 1.25, 0.85)
            print(f"  ⚠ 检测到躺着姿势，大幅提高面部权重至 {ip_adapter_scale:.2f}，确保人脸相似度（躺着姿势需要更高权重）")
        
        # 远景场景额外检查：确保面部权重不会太低，导致人脸不像或不完整
        if is_wide_shot or is_full_body:
            if ip_adapter_scale < 0.6:  # 从0.5提高到0.6，确保人脸完整且相似
                ip_adapter_scale = 0.6
                print(f"  ⚠ 远景场景：强制提高面部权重至 {ip_adapter_scale:.2f}，确保人脸完整且相似")
            if controlnet_scale < 0.5:
                controlnet_scale = 0.5
                print(f"  ⚠ 远景场景：强制提高ControlNet权重至 {controlnet_scale:.2f}，确保人脸完整")
        
        # 改进 negative_prompt，避免人脸过大、拉伸、颜色和面部比例问题
        enhanced_negative = negative_prompt or self.negative_prompt or ""
        
        # 对于 lying_still 动作，添加"standing"到 negative prompt，确保生成"躺着"而不是"站着"的图像
        # 同时加强多腿排除，因为躺着姿势容易出现多腿问题
        if scene and scene.get("action"):
            action = str(scene.get("action", "")).lower()
            if action == "lying_still" or "lying" in action:
                if "standing" not in enhanced_negative.lower():
                    enhanced_negative += ", standing, standing up, upright, vertical pose"
                    print(f"  ✓ 检测到 lying_still 动作，已添加 'standing' 到 negative_prompt")
                # 躺着姿势容易出现多腿问题，加强排除
                if "three legs" not in enhanced_negative.lower():
                    enhanced_negative += ", three legs, four legs, multiple legs, extra legs, duplicate legs, broken legs, severed legs, cut off legs, (three legs:1.8), (four legs:1.8), (multiple legs:1.8), (extra legs:1.8), (duplicate legs:1.8)"
                    print(f"  ✓ 检测到 lying_still 动作，已加强多腿排除（高权重1.8）")
        
        # 基于意图分析添加负面提示（通用处理）
        if scene:
            # 使用意图分析结果
            intent = self.intent_analyzer.analyze(scene)
            
            # 基于意图分析添加负面提示（通用方式，不硬编码）
            # 如果主要实体是角色，且视角不是背面，添加防止背影的负面提示（增强权重）
            if intent['primary_entity'] and intent['primary_entity'].get('type') == 'character':
                viewpoint = intent.get('viewpoint', {})
                viewpoint_type_check = viewpoint.get('type', 'front')
                viewpoint_explicit_check = viewpoint.get('explicit', False)
                if viewpoint_type_check != 'back':
                    # 对于所有非背面场景，都增强负面提示权重，确保排除背影
                    if "back view" not in enhanced_negative.lower() and "from behind" not in enhanced_negative.lower():
                        # 如果明确要求正面，使用更高权重；否则使用中等权重
                        if viewpoint_explicit_check and viewpoint_type_check == 'front':
                            enhanced_negative += ", back view:1.8, from behind:1.8, character back:1.8, rear view:1.8, turned away:1.8, facing away:1.8, back of head:1.8, back of character:1.8, back facing:1.8"
                            print(f"  ✓ 基于意图分析：角色场景（明确要求正面），已增强防止背影的负面提示（权重1.8）")
                        else:
                            enhanced_negative += ", back view:1.6, from behind:1.6, character back:1.6, rear view:1.6, turned away:1.6, facing away:1.6, back of head:1.6, back of character:1.6, back facing:1.6"
                            print(f"  ✓ 基于意图分析：角色场景（默认正面），已增强防止背影的负面提示（权重1.6）")
            
            # 基于意图分析添加镜头类型相关的负面提示
            viewpoint = intent.get('viewpoint', {})
            viewpoint_type = viewpoint.get('type', 'front')
            
            if viewpoint_type == 'wide':
                # 远景场景：排除特写和中景
                if "close-up" not in enhanced_negative.lower() and "特写" not in enhanced_negative.lower():
                    enhanced_negative += ", close-up, extreme close-up, medium shot, mid shot, 特写, 近景, 中景"
                    print(f"  ✓ 基于意图分析：远景场景，已添加排除特写/中景的负面提示")
            elif viewpoint_type == 'close':
                # 特写场景：排除远景
                if "wide shot" not in enhanced_negative.lower() and "远景" not in enhanced_negative.lower():
                    enhanced_negative += ", wide shot, distant view, long shot, very long shot, 远景, 远距离, 全景"
                    print(f"  ✓ 基于意图分析：特写场景，已添加排除远景的负面提示")
            
            # 添加排除项的负面提示
            if intent['exclusions']:
                for exclusion in intent['exclusions']:
                    if exclusion not in enhanced_negative.lower():
                        enhanced_negative += f", {exclusion}"
                print(f"  ✓ 基于意图分析添加排除项负面提示: {', '.join(intent['exclusions'])}")
        
        # 添加动作/姿态控制负面提示（防止画面飘、不连贯、晃动）
        motion_control_negative = [
            "shaky camera, camera shake, unstable camera",
            "floating, drifting, unsteady movement",
            "inconsistent motion, discontinuous action",
            "jittery, flickering, unstable frame",
            "rapid movement, fast motion, sudden movement",
            "disconnected frames, frame jumping"
        ]
        for neg_term in motion_control_negative:
            if neg_term not in enhanced_negative.lower():
                enhanced_negative += f", {neg_term}"
        
        # 添加背景一致性负面提示（防止场景跳帧、风格漂移）
        if not self.ascii_only_prompt:
            background_consistency_negative = [
                "背景不一致，背景变化",
                "背景跳帧，场景跳跃",
                "风格漂移，风格不一致",
                "环境不稳定，场景变换"
            ]
        else:
            background_consistency_negative = [
                "inconsistent background, changing background",
                "background jumping, scene jumping",
                "style drift, style inconsistency",
                "unstable environment, shifting scene"
            ]
        for neg_term in background_consistency_negative:
            if neg_term not in enhanced_negative.lower():
                enhanced_negative += f", {neg_term}"
        
        # 添加禁止现代物品的负面提示（防止出现摩托车、汽车等现代物品）
        if not self.ascii_only_prompt:
            modern_items_negative = [
                "摩托车，汽车，自行车，电动车",
                "现代交通工具，现代车辆",
                "现代科技，电子产品，电器",
                "现代建筑，高楼大厦，现代城市",
                "现代服装，现代装饰",
                "手机，电脑，电视，相机",
                "现代武器，枪械，现代装备"
            ]
        else:
            modern_items_negative = [
                "motorcycle, bike, bicycle, car, vehicle, automobile",
                "modern vehicle, modern transportation",
                "modern technology, electronic device, electric appliance",
                "modern building, skyscraper, modern city, modern architecture",
                "modern clothing, modern decoration, modern fashion",
                "phone, mobile phone, computer, TV, television, camera",
                "modern weapon, gun, firearm, modern equipment"
            ]
        for neg_term in modern_items_negative:
            # 检查负面提示词中是否已包含这个负面提示词
            if neg_term.lower() not in enhanced_negative.lower():
                enhanced_negative += f", {neg_term}"
        
        # 中景和近景场景：增加身体宽度、模糊、拉伸的负面描述
        if is_medium_shot or is_close_up:
            medium_negative_additions = [
                "wide body, broad shoulders, thick torso",
                "wide chest, bulky build, heavy body",
                "body too wide, horizontally stretched body",
                "exaggerated body width, disproportionate body",
                "blurry, blurred, out of focus, unfocused, fuzzy, hazy, unclear, indistinct",
                "stretched, distorted proportions, bad aspect ratio, horizontal stretch",
                "wide stretched, aspect ratio distortion, face stretched horizontally",
                "body stretched, width distortion, horizontally distorted",
                "wide face, stretched face width, horizontally elongated",
                "face too wide, body too wide, stretched horizontally",
                "horizontally deformed, wide distortion, horizontal deformation",
                "face width distortion, body width distortion",
                "horizontally stretched face, horizontally stretched body",
                "unnatural width, excessive width, distorted width, horizontal elongation",
                "unnatural pose, stiff pose, rigid posture, awkward stance",  # 半身像：避免僵硬姿势
                "unnatural body proportions, distorted body shape, bad body anatomy"  # 半身像：避免身体比例不自然
            ]
            for neg_term in medium_negative_additions:
                if neg_term not in enhanced_negative.lower():
                    enhanced_negative += f", {neg_term}"
        
        # 确保单人场景（所有场景都应该是单人，避免重复人物）
        # 检查场景是否包含角色（通过检查 scene 数据或 prompt 中是否包含角色关键词）
        has_character = scene and (scene.get("characters") or any(kw in prompt.lower() for kw in ["han li", "hanli", "cultivator", "character", "韩立", "主角", "person", "figure"]))
        
        # 检查是否为纯背景场景（无人物场景）
        # 通过检查 scene 的 visual.composition 或 prompt 中是否明确表示无人物
        is_background_only = False
        if scene:
            visual = scene.get("visual") or {}
            composition = str(visual.get("composition", "")).lower()
            prompt_lower = prompt.lower()
            # 检查是否明确表示无人物
            no_character_keywords = [
                "only environment", "environment only", "no character", "no person", "no figure",
                "pure environment", "landscape only", "scene only", "background only",
                "只有环境", "纯环境", "无人物", "仅环境", "纯背景"
            ]
            # 检查 composition 或 prompt 中是否包含无人物关键词
            if any(kw in composition or kw in prompt_lower for kw in no_character_keywords):
                is_background_only = True
            # 检查 character_pose 是否为空（通常表示无人物）
            elif not visual.get("character_pose") and not scene.get("characters"):
                # 进一步检查 prompt 中是否明确没有人物相关关键词
                character_keywords = ["han li", "hanli", "cultivator", "character", "韩立", "主角", "person", "figure", "man", "woman", "people"]
                if not any(kw in prompt_lower for kw in character_keywords):
                    is_background_only = True
        
        # 对所有有角色的场景都加强单人约束（不只是远景）- 在 prompt 构建完成后统一处理
        # 注意：这部分代码在 prompt 构建之后执行，确保单人约束在 prompt 最前面
        
        # 加强负面提示：排除多人、重复人物、多腿、多手等（对所有场景都适用）
        # 用户反馈：场景5和7生成了多个人物，需要大幅提高权重
        # 提高多人排除权重从1.2到1.8，确保不会生成多个人物
        multiple_people_negative = ", multiple people, two people, crowd, group of people, extra person, duplicate character, second person, additional figure, cloned person, duplicate person, identical person, twin character, repeated character, second identical figure, duplicate figure, mirrored person, copy of person, repeated appearance, same person twice, duplicate appearance, two same people, two identical people, cloned figure, mirrored character, identical duplicate, (duplicate person:1.8), (same person twice:1.8), (two same people:1.8), (multiple people:1.8), (two people:1.8), (crowd:1.8), (group of people:1.8), (extra person:1.8), (second person:1.8), (additional figure:1.8), (cloned person:1.8), (duplicate character:1.8), (twin character:1.8), (repeated character:1.8), (second identical figure:1.8), (duplicate figure:1.8), (mirrored person:1.8), (copy of person:1.8), extra legs, multiple legs, duplicate legs, two sets of legs, three legs, four legs, extra feet, multiple feet, duplicate feet, three feet, four feet, extra limbs, multiple limbs, duplicate limbs, extra body parts, multiple body parts, duplicate body parts, extra hands, multiple hands, duplicate hands, extra arms, multiple arms, duplicate arms, (extra legs:1.5), (multiple legs:1.5), (duplicate legs:1.5), (three legs:1.5), (four legs:1.5), (two sets of legs:1.5), (extra feet:1.5), (multiple feet:1.5), (duplicate feet:1.5), (three feet:1.5), (extra limbs:1.3), (multiple limbs:1.3), (duplicate limbs:1.3), (extra body parts:1.3), (multiple body parts:1.3), (duplicate body parts:1.3), (extra hands:1.2), (multiple hands:1.2), (duplicate hands:1.2), (extra arms:1.2), (multiple arms:1.2), (duplicate arms:1.2), malformed legs, deformed legs, wrong number of legs, incorrect leg count, abnormal leg count, too many legs, leg duplication, leg repetition, leg cloning, leg mirroring, leg copy, leg repeat, leg duplicate, leg twin, leg identical, leg repeated, leg second, leg extra, leg additional, leg cloned, leg mirrored, leg identical duplicate, leg same twice, leg two same, leg two identical, leg cloned figure, leg mirrored character, broken legs, severed legs, cut off legs, missing legs, leg amputation, leg injury, leg damage, leg fracture, leg break, leg cut, leg severed, leg missing, leg incomplete, leg partial, leg fragment, leg piece, leg part, leg section, leg portion, leg segment"
        
        # 对于纯背景场景，明确排除所有人物（高权重），包括仙女等
        if is_background_only:
            no_character_negative = ", person, people, human, character, figure, man, woman, male, female, boy, girl, individual, someone, anybody, anyone, (person:1.8), (people:1.8), (human:1.8), (character:1.8), (figure:1.8), (man:1.8), (woman:1.8), (male:1.8), (female:1.8), (boy:1.8), (girl:1.8), (individual:1.8), (someone:1.8), (anybody:1.8), (anyone:1.8), face, faces, body, bodies, portrait, portraits, person in scene, character in scene, figure in scene, human in scene, man in scene, woman in scene, (face:1.8), (faces:1.8), (body:1.8), (bodies:1.8), (portrait:1.8), (portraits:1.8), (person in scene:1.8), (character in scene:1.8), (figure in scene:1.8), (human in scene:1.8), (man in scene:1.8), (woman in scene:1.8), female character, woman character, girl character, female figure, woman figure, girl figure, (female character:1.8), (woman character:1.8), (girl character:1.8), (female figure:1.8), (woman figure:1.8), (girl figure:1.8), fairy, fairy woman, celestial maiden, immortal woman, fairy maiden, immortal maiden, xianzi, fairy girl, (fairy:1.8), (fairy woman:1.8), (celestial maiden:1.8), (immortal woman:1.8), (fairy maiden:1.8), (immortal maiden:1.8), (xianzi:1.8), (fairy girl:1.8), goddess, goddess figure, female deity, (goddess:1.8), (goddess figure:1.8), (female deity:1.8)"
            if "person" not in enhanced_negative.lower() or "people" not in enhanced_negative.lower():
                enhanced_negative += no_character_negative
                print(f"  ✓ 纯背景场景：已添加排除所有人物（包括仙女、仙子）到 negative prompt（高权重1.8）")
            
            # 对于包含卷轴的场景，在prompt中额外强调卷轴
            if scene and prompt:
                prompt_lower = prompt.lower()
                if "scroll" in prompt_lower or "卷轴" in prompt_lower:
                    # 在prompt最前面添加卷轴强调（如果还没有）
                    # 注意：这里不能直接修改 prompt，因为 prompt 已经构建完成
                    # 但可以在 negative prompt 中排除人物，确保卷轴可见
                    print(f"  ✓ 纯背景场景包含卷轴：已确保排除人物，卷轴应可见且突出")
        elif has_character:
            if "multiple people" not in enhanced_negative.lower():
                enhanced_negative += multiple_people_negative
                print(f"  ✓ 已添加多人、多腿排除约束到 negative prompt")
            
            # 对于有角色的场景，检查是否是韩立角色，明确排除女性特征
            if scene and prompt:
                # 检查是否是韩立角色
                is_hanli = any(kw in prompt.lower() for kw in ["han li", "hanli", "韩立", "主角"]) or \
                          (scene.get("characters") and "hanli" in str(scene.get("characters", "")).lower())
                if is_hanli:
                    # 明确排除女性特征（高权重）
                    female_negative = ", female, woman, girl, feminine, female character, woman character, girl character, female figure, woman figure, girl figure, female appearance, woman appearance, girl appearance, (female:1.5), (woman:1.5), (girl:1.5), (feminine:1.5), (female character:1.5), (woman character:1.5), (girl character:1.5), (female figure:1.5), (woman figure:1.5), (girl figure:1.5), (female appearance:1.5), (woman appearance:1.5), (girl appearance:1.5), female face, woman face, girl face, female body, woman body, girl body, (female face:1.5), (woman face:1.5), (girl face:1.5), (female body:1.5), (woman body:1.5), (girl body:1.5)"
                    if "female" not in enhanced_negative.lower() or "woman" not in enhanced_negative.lower():
                        enhanced_negative += female_negative
                        print(f"  ✓ 韩立角色：已添加排除女性特征到 negative prompt（高权重）")
            
            # 对于所有人物场景，强制添加多人排除项（防止出现多个相同的人）
            if scene and prompt and not is_background_only:
                # 检查是否已经添加了多人排除
                if "multiple people" not in enhanced_negative.lower() or "duplicate person" not in enhanced_negative.lower():
                    # 添加更强烈的多人排除项
                    strong_multiple_negative = ", multiple people, two people, three people, four people, five people, crowd, group of people, extra person, duplicate character, second person, additional figure, cloned person, duplicate person, identical person, twin character, repeated character, second identical figure, duplicate figure, mirrored person, copy of person, repeated appearance, same person twice, duplicate appearance, two same people, two identical people, cloned figure, mirrored character, identical duplicate, ten people, many people, (duplicate person:2.0), (same person twice:2.0), (two same people:2.0), (multiple people:2.0), (two people:2.0), (crowd:2.0), (group of people:2.0), (extra person:2.0), (second person:2.0), (ten people:2.0), (many people:2.0)"
                    enhanced_negative += strong_multiple_negative
                    print(f"  ✓ 人物场景：已添加强化多人排除项到 negative prompt（高权重2.0，防止出现多个相同的人）")
            
            # 对于所有场景，排除现代交通工具和飞船（防止出现不符合仙侠风格的现代元素）
            if scene and prompt:
                if "vehicle" not in enhanced_negative.lower() or "spaceship" not in enhanced_negative.lower() or "tank" not in enhanced_negative.lower() or "car" not in enhanced_negative.lower():
                    modern_vehicle_negative = ", vehicle, vehicles, car, cars, truck, trucks, tank, tanks, military vehicle, military vehicles, spaceship, spaceships, spacecraft, aircraft, airplane, airplanes, helicopter, helicopters, modern technology, modern equipment, military equipment, weapons, gun, guns, automobile, automobiles, (vehicle:2.0), (car:2.0), (tank:2.0), (spaceship:2.0), (military vehicle:2.0), (modern technology:2.0), (automobile:2.0)"
                    enhanced_negative += modern_vehicle_negative
                    print(f"  ✓ 已添加现代交通工具和飞船排除项到 negative prompt（高权重2.0，防止出现不符合仙侠风格的现代元素）")
        
        if "oversized face" not in enhanced_negative.lower():
            enhanced_negative += ", oversized face, face too large, distorted face, wrong hairstyle, deformed face, character too large, person too large, figure too large, oversized character, oversized person, oversized figure, character too big, person too big, figure too big, character size wrong, person size wrong, figure size wrong, character scale wrong, person scale wrong, figure scale wrong, character proportions wrong, person proportions wrong, figure proportions wrong, character too prominent, person too prominent, figure too prominent, character dominates frame, person dominates frame, figure dominates frame, character fills frame, person fills frame, figure fills frame, character takes up too much space, person takes up too much space, figure takes up too much space"
        
        # 远景场景：添加负面提示，避免人脸不完整、模糊或缺失
        if is_wide_shot or is_full_body:
            if "incomplete face" not in enhanced_negative.lower():
                enhanced_negative += ", incomplete face, missing face parts, cut off face, face cut off, face partially hidden, face obscured, face blurred, face unclear, face not visible, face missing, face incomplete, face cut, face cropped, face out of frame, face partially visible, face barely visible, face too small, face too tiny, face unrecognizable, face unclear, face indistinct"
        if "stretched" not in enhanced_negative.lower():
            enhanced_negative += ", stretched, distorted proportions, bad aspect ratio, horizontal stretch, wide stretched, aspect ratio distortion, face stretched horizontally, body stretched, width distortion, elongated face horizontally, horizontally distorted, wide face, stretched face width, horizontally elongated, face too wide, body too wide, stretched horizontally, horizontally deformed, wide distortion, horizontal deformation, face width distortion, body width distortion, horizontally stretched face, horizontally stretched body, unnatural width, excessive width, distorted width, horizontal elongation, wide body, broad shoulders, bulky physique, thick body, wide torso, broad chest, heavy build"
        
        # 排除现代物品（坦克、车辆等）- 高权重，确保不被生成
        if "tank" not in enhanced_negative.lower():
            enhanced_negative += ", tank, military vehicle, armored vehicle, modern vehicle, car, truck, bus, modern technology, modern machinery, modern equipment, modern weapon, gun, firearm, modern building, modern architecture, skyscraper, modern city, modern infrastructure, modern object, contemporary object, 21st century, 20th century, modern era, contemporary era, industrial, mechanical, engine, motor, vehicle, automobile, transportation vehicle"
        if "head only" not in enhanced_negative.lower():
            enhanced_negative += ", head only, body hidden, missing torso, buried in sand, body submerged, only face visible"
        if "horse" not in enhanced_negative.lower():
            enhanced_negative += ", horse, camel, riding animal, mounted, riding, on horseback, saddle, beast mount"
        # 排除铠甲、盔甲，强调修仙服饰
        if "armor" not in enhanced_negative.lower():
            enhanced_negative += ", armor, plate armor, metal armor, heavy armor, knight armor, warrior armor, military uniform, combat gear, protective gear, metal plates, chainmail"
        if "wide body" not in enhanced_negative.lower():
            enhanced_negative += ", wide body, broad shoulders, bulky physique, muscular build, thick body, wide torso, broad chest, heavy build"
        # 强化绿色抑制和颜色平衡
        if "green" not in enhanced_negative.lower():
            enhanced_negative += ", oversaturated green, green tint, green cast, unnatural green color, excessive green, greenish tint, green color cast, green filter"
        if "color" not in enhanced_negative.lower():
            enhanced_negative += ", color cast, color shift, unnatural colors, color imbalance, monochrome green, oversaturated colors, undersaturated colors, wrong colors, bad color grading"
        # 添加面部比例问题抑制（防止瘦长脸、不协调）
        if "thin face" not in enhanced_negative.lower():
            enhanced_negative += ", thin face, narrow face, elongated face, face too thin, skinny face, gaunt face, long narrow face, vertically stretched face, face too long, face too narrow, disproportionate face, unnatural face proportions, face aspect ratio distortion, vertically elongated face, stretched vertically, face height distortion, unnatural face shape, face too tall, face too skinny, uncoordinated face proportions, face width too narrow, face length too long, face ratio imbalance, face proportions distorted, face shape distorted, face geometry wrong, face dimensions wrong, face scale wrong, face too elongated, face too stretched vertically, face vertical stretch, face height too long, face width too short, narrow elongated face, long thin face, face not proportional, face proportions off, face shape off, face dimensions off"
        
        # LoRA 适配（InstantID）
        # 注意：InstantID 模式下，use_ip_adapter 通常是 False（InstantID 有自己的 IP-Adapter）
        # 所以 LoRA 不会降低 InstantID 的 IP-Adapter 权重，这是正确的
        if self.use_lora and hasattr(self.pipeline, "set_adapters"):
            self.pipeline.set_adapters(
                [self.lora_adapter_name],
                adapter_weights=[self.lora_alpha],
            )
            print(f"  ✓ 已加载 LoRA: {self.lora_adapter_name} (alpha={self.lora_alpha})")
            # InstantID 模式下，use_ip_adapter 通常是 False，所以不会降低权重
            # 这是正确的，因为 InstantID 的 IP-Adapter 是必需的，不应该被 LoRA 影响
            if self.use_ip_adapter:
                ip_adapter_scale = max(0.1, ip_adapter_scale * self.lora_ip_scale_multiplier)
                print(f"  ⚠ 使用额外的 IP-Adapter（非 InstantID），调节面部权重至: {ip_adapter_scale:.2f}")
            else:
                print(f"  ℹ InstantID 模式：LoRA 不影响 InstantID 的 IP-Adapter 权重（保持 {ip_adapter_scale:.2f}）")

        # 根据镜头自动调整面部关键点缩放
        face_kps_scale_cfg = float(self.instantid_config.get("face_kps_scale", 1.0))
        face_kps_offset_y = int(self.instantid_config.get("face_kps_offset_y", 0))
        face_kps_scale = face_kps_scale_cfg
        if is_wide_shot or is_full_body:
            # 远景场景：适度提高关键点缩放，确保人脸完整且清晰
            # 从0.65提高到0.75，提高清晰度，但不过度控制避免瘦长脸
            face_kps_scale = face_kps_scale_cfg * 0.75  # 从0.65提高到0.75，提高远景场景清晰度
            print(f"  远景场景：面部关键点缩放至 {face_kps_scale:.2f}，确保人脸完整且清晰")
        elif is_medium_shot:
            # 中景/半身像：进一步降低面部关键点缩放，使半身像更自然，避免横向压缩变形和瘦长脸
            # 由于基准值已降低到0.70，进一步降低乘数以减少变形
            face_kps_scale = face_kps_scale_cfg * 0.50  # 从0.55降到0.50，进一步减少横向压缩变形和瘦长脸
            print(f"  中景/半身像场景：降低面部关键点缩放至 {face_kps_scale:.2f}，避免横向压缩变形和瘦长脸")
        elif is_close_up:
            # 降低关键点缩放，避免身体过宽、模糊和横向压缩变形
            face_kps_scale = face_kps_scale_cfg * 0.60  # 从0.65降到0.60，进一步减少横向压缩变形和瘦长脸
            print(f"  近景场景：降低面部关键点缩放至 {face_kps_scale:.2f}，避免身体过宽、模糊和横向压缩变形")
        
        # 应用 face_style_auto 的 face_kps_scale 调整
        if isinstance(face_style, dict) and face_style:
            try:
                from face_style_auto_generator import to_instantid_params
                instantid_params = to_instantid_params(face_style)
                kps_multiplier = instantid_params.get("face_kps_scale_multiplier", 1.0)
                face_kps_scale = face_kps_scale * kps_multiplier
            except (ImportError, Exception):
                # 如果没有 face_style_auto_generator，使用简单逻辑
                detail = (face_style.get("detail") or "").lower()
                if detail in {"detailed", "cinematic"}:
                    face_kps_scale *= 1.1
                elif detail in {"subtle"}:
                    face_kps_scale *= 0.9
        
        # 确保关键点缩放不会太小，远景场景至少0.45，确保人脸完整
        # 同时限制最大值，避免过度控制导致横向压缩变形和瘦长脸
        if is_wide_shot or is_full_body:
            face_kps_scale = max(0.45, min(face_kps_scale, 0.85))  # 远景至少0.45，最大0.85，避免过度控制和瘦长脸
        else:
            face_kps_scale = max(0.3, min(face_kps_scale, 0.75))  # 其他场景最大0.75，避免横向压缩变形和瘦长脸
        face_kps = self._adjust_face_kps_canvas(face_kps_raw, face_kps_scale, face_kps_offset_y)
        if abs(face_kps_scale - face_kps_scale_cfg) > 1e-3:
            print(f"  面部关键点缩放: {face_kps_scale:.2f} (camera + face_style_auto 控制)")
        
        # 根据官方建议设置 IP-Adapter 强度
        # 官方推荐：先 set_ip_adapter_scale，再调用 pipeline
        if hasattr(self.pipeline, 'set_ip_adapter_scale'):
            self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
        
        # InstantID 生成参数（按照官方用法）
        kwargs = {
            "prompt": prompt,
            "negative_prompt": enhanced_negative.strip() if enhanced_negative.strip() else None,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "height": self.height,
            "width": self.width,
            "generator": generator,
            "image_embeds": face_emb,  # 面部嵌入（官方用法）
            "image": face_kps,  # 面部关键点图像（官方用法）
            "controlnet_conditioning_scale": controlnet_scale,  # ControlNet 条件强度（根据场景调整）
            "ip_adapter_scale": ip_adapter_scale,  # IP-Adapter 强度（根据场景调整，官方默认 0.8）
        }
        
        # 添加 CFG Rescale（如果配置了，主流做法：减少颜色过度饱和）
        # 注意：guidance_rescale 是 diffusers 0.20.0+ 引入的参数
        # 如果 pipeline 不支持，会在调用时忽略
        if guidance_rescale is not None:
            kwargs["guidance_rescale"] = guidance_rescale
        
        # 在调用 pipeline 之前，确保 InstantID 的 IP-Adapter 状态正确
        if not hasattr(self.pipeline, "image_proj_model_in_features"):
            if self._ensure_instantid_ip_adapter_ready():
                print("  ✓ InstantID IP-Adapter 状态已修复")
            else:
                print("  ⚠ InstantID IP-Adapter 状态异常，尝试重新加载...")
                self._reload_instantid_ip_adapter()

        # 调用 InstantID pipeline（按照官方用法）
        # 注意：如果 guidance_rescale 不被支持，会自动忽略（不会报错）
        try:
            result = self.pipeline(**kwargs)
        except (TypeError, AttributeError, NameError) as e:
            error_str = str(e)
            # 检查是否是 image_proj_model_in_features 错误
            if "image_proj_model_in_features" in error_str:
                print(f"  ✗ InstantID IP-Adapter 未正确加载: {e}")
                raise RuntimeError(f"InstantID IP-Adapter 未正确加载，请检查 IP-Adapter 文件是否正确: {e}") from e
            # 如果是因为不支持 guidance_rescale 参数，移除它后重试
            elif "guidance_rescale" in error_str or "unexpected keyword" in error_str.lower():
                print(f"  ⚠ 当前 pipeline 不支持 guidance_rescale 参数，已忽略")
                kwargs.pop("guidance_rescale", None)
                result = self.pipeline(**kwargs)
            else:
                raise
        image = result.images[0] if hasattr(result, "images") else result[0]

        # 保存图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        # 清理GPU缓存，释放内存
        import torch
        import gc
        if torch.cuda.is_available():
            del result
            torch.cuda.empty_cache()
            gc.collect()

        return output_path

    def _ensure_instantid_ip_adapter_ready(self) -> bool:
        """
        确保 image_proj_model_in_features 可用。
        Returns:
            True  -> 已具备或已修复
            False -> 需要重新加载
        """
        if hasattr(self.pipeline, "image_proj_model_in_features"):
            return True
        if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
            try:
                self.pipeline.image_proj_model_in_features = 512
                return True
            except Exception as exc:
                print(f"  ⚠ 无法手动设置 image_proj_model_in_features: {exc}")
        return False
        
    def _reload_instantid_ip_adapter(self):
        """重新加载 InstantID IP-Adapter"""
        # 首先尝试快速修复（如果 image_proj_model 已存在）
        if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
            print("  ⚠ 检测到 image_proj_model 已存在，尝试直接设置 image_proj_model_in_features...")
            try:
                self.pipeline.image_proj_model_in_features = 512
                print("  ✓ 已手动设置 image_proj_model_in_features = 512，无需重新加载")
                return
            except Exception as e:
                print(f"  ⚠ 无法手动设置: {e}，尝试重新加载...")
        
        # 如果快速修复失败，尝试重新加载
        instantid_config = self.instantid_config
        ip_adapter_path = instantid_config.get("ip_adapter_path")
        if not ip_adapter_path:
            raise ValueError("InstantID 需要 ip_adapter_path 配置")

        ip_adapter_path_obj = Path(ip_adapter_path)
        if not ip_adapter_path_obj.is_absolute():
            config_dir = Path(self.config_path).parent if hasattr(self, 'config_path') else Path.cwd()
            ip_adapter_path_obj = config_dir / ip_adapter_path
            if not ip_adapter_path_obj.exists():
                ip_adapter_path_obj = Path.cwd() / ip_adapter_path

        ip_adapter_file = None
        if ip_adapter_path_obj.is_file():
            ip_adapter_file = ip_adapter_path_obj
        elif ip_adapter_path_obj.is_dir():
            for ext in ['.bin', '.safetensors']:
                found = list(ip_adapter_path_obj.glob(f'*{ext}'))
                if found:
                    ip_adapter_file = found[0]
                    break
        if not ip_adapter_file or not ip_adapter_file.exists():
            raise FileNotFoundError(f"未找到 InstantID IP-Adapter 文件: {ip_adapter_path_obj} (查找路径: {ip_adapter_path})")

        print(f"  重新加载 InstantID IP-Adapter: {ip_adapter_file}")
        
        # 在重新加载前，先尝试卸载已加载的 IP-Adapter（如果存在）
        if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
            print("  ℹ 检测到已加载的 IP-Adapter，先尝试卸载...")
            try:
                # InstantID pipeline 可能没有 unload 方法，尝试手动清理
                if hasattr(self.pipeline, "unload_ip_adapter"):
                    self.pipeline.unload_ip_adapter()
                    print("  ✓ 已卸载旧的 IP-Adapter")
                else:
                    # 手动清理
                    self.pipeline.image_proj_model = None
                    if hasattr(self.pipeline, "image_proj_model_in_features"):
                        delattr(self.pipeline, "image_proj_model_in_features")
                    print("  ✓ 已手动清理旧的 IP-Adapter 状态")
            except Exception as e:
                print(f"  ⚠ 卸载旧 IP-Adapter 失败: {e}，继续尝试重新加载...")
        
        # 规范化 IP-Adapter 权重文件的键顺序，避免 diffusers 的严格检查出错
        ip_adapter_file = self._normalize_ip_adapter_weights(ip_adapter_file)

        try:
            if hasattr(self.pipeline, "load_ip_adapter_instantid"):
                # InstantID 管线优先使用官方 API，避免 diffusers 额外的权重结构校验
                self._load_ip_adapter_via_instantid_api(ip_adapter_file)
            else:
                self._load_ip_adapter_from_file(ip_adapter_file)
        except Exception as err:
            print(f"  ✗ InstantID IP-Adapter 重新加载失败: {err}")
            # 最后尝试：如果 image_proj_model 还存在，直接设置
            if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                print("  ⚠ 重新加载失败，但检测到 image_proj_model，尝试直接设置 image_proj_model_in_features = 512...")
                try:
                    self.pipeline.image_proj_model_in_features = 512
                    print("  ✓ 已手动设置 image_proj_model_in_features = 512，继续执行")
                    return
                except Exception as err2:
                    raise RuntimeError(
                        f"InstantID IP-Adapter 未正确加载，且无法手动修复: {err2}"
                    ) from err
            raise RuntimeError(f"InstantID IP-Adapter 未正确加载，无法生成图像: {err}") from err

    def _load_ip_adapter_from_file(self, ip_adapter_file: Path) -> None:
        """根据不同 Pipeline API 将 IP-Adapter 加载进 pipeline"""
        if hasattr(self.pipeline, "load_ip_adapter_instantid"):
            self._load_ip_adapter_via_instantid_api(ip_adapter_file)
        elif hasattr(self.pipeline, "load_ip_adapter"):
            self._load_ip_adapter_via_diffusers_api(ip_adapter_file)
        else:
            raise AttributeError("pipeline 既不支持 load_ip_adapter_instantid 也不支持 load_ip_adapter 方法")

    def _load_ip_adapter_via_instantid_api(self, ip_adapter_file: Path) -> None:
        """
        使用 InstantID API 加载 IP-Adapter
        
        注意：如果 IP-Adapter 已经部分加载（image_proj_model 存在），
        直接设置 image_proj_model_in_features，避免重新加载导致冲突
        """
        # 如果 image_proj_model 已经存在，说明 IP-Adapter 已经部分加载
        # 直接设置 image_proj_model_in_features，避免重新加载
        if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
            print("  ℹ 检测到 image_proj_model 已存在，跳过重新加载，直接设置 image_proj_model_in_features...")
            try:
                self.pipeline.image_proj_model_in_features = 512
                print("  ✓ 已手动设置 image_proj_model_in_features = 512")
                return
            except Exception as e:
                print(f"  ⚠ 无法手动设置: {e}，尝试重新加载...")
                # 如果手动设置失败，尝试卸载后重新加载
                try:
                    if hasattr(self.pipeline, "unload_ip_adapter"):
                        self.pipeline.unload_ip_adapter()
                    else:
                        # 手动清理
                        self.pipeline.image_proj_model = None
                        if hasattr(self.pipeline, "image_proj_model_in_features"):
                            delattr(self.pipeline, "image_proj_model_in_features")
                except Exception as e2:
                    print(f"  ⚠ 卸载失败: {e2}，继续尝试重新加载...")
        
        # 如果 image_proj_model 不存在，或者卸载成功，尝试重新加载
        try:
            self.pipeline.load_ip_adapter_instantid(str(ip_adapter_file))
        except Exception as err:
            error_str = str(err).lower()
            # 如果是 state dict 相关的错误，可能是文件格式问题或已加载冲突
            if "state dict" in error_str or "missing" in error_str:
                if ip_adapter_file.is_file():
                    print(f"  ⚠ 使用文件路径加载失败（可能是已加载冲突），尝试使用目录路径: {err}")
                    try:
                        self.pipeline.load_ip_adapter_instantid(str(ip_adapter_file.parent))
                    except Exception as e2:
                        # 如果目录路径也失败，检查是否已经部分加载
                        if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                            print(f"  ⚠ 重新加载失败，但检测到 image_proj_model，尝试直接设置 image_proj_model_in_features...")
                            try:
                                self.pipeline.image_proj_model_in_features = 512
                                print("  ✓ 已手动设置 image_proj_model_in_features = 512")
                                return
                            except Exception as e3:
                                raise RuntimeError(f"无法重新加载 IP-Adapter，且无法手动修复: {e3}") from err
                        raise RuntimeError(f"无法重新加载 IP-Adapter: {e2}") from err
                else:
                    raise
            else:
                raise

        if not self._ensure_instantid_ip_adapter_ready():
            raise RuntimeError("load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置")
        print("  ✓ InstantID IP-Adapter 重新加载成功（使用 load_ip_adapter_instantid）")

    def _load_ip_adapter_via_diffusers_api(self, ip_adapter_file: Path) -> None:
        import inspect
        sig = inspect.signature(self.pipeline.load_ip_adapter)
        params = list(sig.parameters.keys())

        # 对于单个文件，优先使用文件路径 + subfolder/weight_name 的方式
        if ip_adapter_file.is_file():
            if "pretrained_model_name_or_path_or_dict" in params:
                # 使用文件所在目录作为路径，文件名作为 weight_name
                pretrained_path = str(ip_adapter_file.parent)
                weight_name = ip_adapter_file.name
                if "subfolder" in params:
                    self.pipeline.load_ip_adapter(pretrained_path, subfolder="", weight_name=weight_name)
                else:
                    self.pipeline.load_ip_adapter(pretrained_path, weight_name=weight_name)
            elif "subfolder" in params and "weight_name" in params:
                # 必需参数方式：使用父目录作为 subfolder，文件名作为 weight_name
                subfolder = str(ip_adapter_file.parent)
                weight_name = ip_adapter_file.name
                self.pipeline.load_ip_adapter(subfolder=subfolder, weight_name=weight_name)
            else:
                # 回退：直接传递文件路径（可能不支持，但尝试一下）
                self.pipeline.load_ip_adapter(str(ip_adapter_file))
        elif ip_adapter_file.is_dir():
            # 目录情况：作为 HuggingFace 格式的模型目录
            if "pretrained_model_name_or_path_or_dict" in params:
                pretrained_path = str(ip_adapter_file)
                weight_name = "ip-adapter.bin"
                if "subfolder" in params:
                    self.pipeline.load_ip_adapter(pretrained_path, subfolder="", weight_name=weight_name)
                else:
                    self.pipeline.load_ip_adapter(pretrained_path, weight_name=weight_name)
            elif "subfolder" in params and "weight_name" in params:
                subfolder = str(ip_adapter_file)
                weight_name = "ip-adapter.bin"
                self.pipeline.load_ip_adapter(subfolder=subfolder, weight_name=weight_name)
            else:
                self.pipeline.load_ip_adapter(str(ip_adapter_file))
        else:
            raise FileNotFoundError(f"IP-Adapter 路径不存在: {ip_adapter_file}")

        if not self._ensure_instantid_ip_adapter_ready():
            raise RuntimeError("load_ip_adapter 调用后仍缺少 image_proj_model_in_features")
        print("  ✓ InstantID IP-Adapter 重新加载成功（使用 load_ip_adapter）")

    def _normalize_ip_adapter_weights(self, ip_adapter_file: Path) -> Path:
        """
        diffusers 官方的 load_ip_adapter 要求 state_dict 的顶层键顺序必须是
        ["image_proj", "ip_adapter"]（严格按顺序比较）。部分开源权重文件保存时
        先写入了 "ip_adapter" 再写 "image_proj"，导致后续加载报
        "Required keys are (`image_proj` and `ip_adapter`) missing..."。

        这里在加载前检测权重文件，如果只有这两个键但顺序不对，则按期望顺序
        重新写回，确保 diffusers 能顺利读取。
        """
        try:
            state = torch.load(ip_adapter_file, map_location="cpu")
        except Exception as exc:
            print(f"  ⚠ 无法读取 IP-Adapter 权重进行校验: {exc}")
            return ip_adapter_file

        if not isinstance(state, dict):
            return ip_adapter_file

        keys = list(state.keys())
        if keys == ["image_proj", "ip_adapter"]:
            return ip_adapter_file

        if set(keys) == {"image_proj", "ip_adapter"}:
            print("  ℹ 调整 IP-Adapter 权重键顺序以兼容 diffusers 载入逻辑")
            ordered = {"image_proj": state["image_proj"], "ip_adapter": state["ip_adapter"]}
            torch.save(ordered, ip_adapter_file)
        return ip_adapter_file
    
    def _generate_image_sdxl(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        reference_image_path: Optional[Path] = None,
        face_reference_image_path: Optional[Path] = None,
        use_lora: Optional[bool] = None,
        scene: Optional[Dict[str, Any]] = None,
        use_ip_adapter_override: Optional[bool] = None,
    ) -> Path:
        """使用 SDXL 生成图像（旧方案）"""
        if self.pipeline is None:
            raise RuntimeError("pipeline 未加载，请先调用 load_pipeline()")
        
        # 使用 override 值（如果提供），否则使用配置值
        use_ip_adapter = use_ip_adapter_override if use_ip_adapter_override is not None else self.use_ip_adapter
        
        # 检查 self.pipeline 是否是 InstantID pipeline
        # 如果是，需要使用普通的 SDXL pipeline
        is_instantid_pipeline = False
        try:
            from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
            is_instantid_pipeline = isinstance(self.pipeline, StableDiffusionXLInstantIDPipeline)
        except ImportError:
            # 如果不能导入，检查 pipeline 的类型名称
            pipeline_type_name = type(self.pipeline).__name__
            is_instantid_pipeline = "InstantID" in pipeline_type_name or "instantid" in pipeline_type_name.lower()
        
        if is_instantid_pipeline:
            # 如果还没有加载普通的 SDXL pipeline，加载它
            if self.sdxl_pipeline is None:
                print(f"  ℹ 检测到 InstantID pipeline，加载普通的 SDXL pipeline 用于文生图")
                try:
                    self._load_sdxl_pipeline()
                    # 保存 InstantID pipeline
                    instantid_pipeline = self.pipeline
                    # 加载普通的 SDXL pipeline 会替换 self.pipeline，所以我们需要保存它
                    # 但实际上，_load_sdxl_pipeline 会设置 self.pipeline，所以我们先保存
                    # 然后恢复 InstantID pipeline，并将 SDXL pipeline 保存到 self.sdxl_pipeline
                    self.sdxl_pipeline = self.pipeline
                    self.pipeline = instantid_pipeline
                    print(f"  ✓ 普通的 SDXL pipeline 已加载并保存")
                except Exception as e:
                    print(f"  ⚠ 无法加载普通的 SDXL pipeline: {e}")
                    print(f"  ⚠ 将尝试使用 InstantID pipeline（可能需要提供参考图像）")
                    self.sdxl_pipeline = None
            
            # 使用普通的 SDXL pipeline
            if self.sdxl_pipeline is not None:
                pipeline_to_use = self.sdxl_pipeline
                print(f"  ℹ 使用普通的 SDXL pipeline（非 InstantID）")
            else:
                pipeline_to_use = self.pipeline
                print(f"  ⚠ 使用 InstantID pipeline（可能失败）")
        else:
            # 使用普通的 pipeline
            pipeline_to_use = self.pipeline
        
        # 初始化 init_image（用于检查是否有有效的 init_image）
        init_image = None
        
        # 如果启用 IP-Adapter，确保它已经被加载
        if use_ip_adapter:
            target_pipe = pipeline_to_use or self.pipeline
            # 检查 IP-Adapter 是否已加载（检查是否有 ip_adapter_image_processor 属性且不为 None）
            ip_adapter_loaded = False
            if hasattr(target_pipe, "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
                ip_adapter_loaded = True
            elif hasattr(target_pipe, "prepare_ip_adapter_image_embeds"):
                # 尝试调用 prepare_ip_adapter_image_embeds 来检查是否已加载
                # 如果 IP-Adapter 未加载，这个方法可能会失败
                try:
                    # 创建一个测试图像来检查
                    from PIL import Image
                    test_img = Image.new("RGB", (224, 224), color="black")
                    # 不实际调用，只检查方法是否存在
                    ip_adapter_loaded = True
                except:
                    ip_adapter_loaded = False
            
            if not ip_adapter_loaded:
                # IP-Adapter 未加载，尝试加载
                print(f"  ⚠ IP-Adapter 未加载，尝试加载...")
                try:
                    # 每次生成前都重新加载 IP-Adapter，确保使用正确的适配器
                    # 先卸载旧的 IP-Adapter（如果存在）
                    target_pipe = pipeline_to_use or self.pipeline
                    if hasattr(target_pipe, "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
                        print(f"  ℹ 检测到已加载的 IP-Adapter，先卸载...")
                        try:
                            if hasattr(target_pipe, "disable_ip_adapter"):
                                target_pipe.disable_ip_adapter()
                                print(f"  ✓ 已卸载旧的 IP-Adapter")
                            elif hasattr(target_pipe, "unload_ip_adapter"):
                                target_pipe.unload_ip_adapter()
                                print(f"  ✓ 已卸载旧的 IP-Adapter")
                        except Exception as e:
                            print(f"  ⚠ 无法卸载旧的 IP-Adapter: {e}，尝试直接加载")
                    
                    # 检查是否使用了 InstantID（可能已经加载了 InstantID 的 IP-Adapter）
                    # 无论 engine 是什么，只要检测到 InstantID 的 IP-Adapter，都需要卸载
                    if hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                        # 检查是否是 InstantID 的 IP-Adapter（InstantID 的 IP-Adapter 没有 prepare_ip_adapter_image_embeds 方法）
                        is_instantid_adapter = not hasattr(self.pipeline, "prepare_ip_adapter_image_embeds")
                        if is_instantid_adapter or self.engine == "instantid":
                            print(f"  ℹ 检测到 InstantID 的 IP-Adapter，卸载 InstantID 的 IP-Adapter")
                            # InstantID 和 SDXL 的 IP-Adapter 不兼容，需要先卸载
                            try:
                                if hasattr(self.pipeline, "disable_ip_adapter"):
                                    self.pipeline.disable_ip_adapter()
                                    print(f"  ✓ 已卸载 InstantID 的 IP-Adapter")
                                elif hasattr(self.pipeline, "unload_ip_adapter"):
                                    self.pipeline.unload_ip_adapter()
                                    print(f"  ✓ 已卸载 InstantID 的 IP-Adapter")
                            except Exception as e:
                                print(f"  ⚠ 无法卸载 InstantID 的 IP-Adapter: {e}")
                    
                    # 加载新的 IP-Adapter
                    print(f"  ℹ 加载 SDXL IP-Adapter...")
                    self._load_ip_adapter()
                    # 重新加载后，确保 pipeline_to_use 也加载了 IP-Adapter
                    if pipeline_to_use is self.img2img_pipeline and self.img2img_pipeline is not None:
                        # img2img_pipeline 的 IP-Adapter 会在 _load_ip_adapter 中自动加载
                        # 但为了确保，我们再次检查
                        if not (hasattr(self.img2img_pipeline, "ip_adapter_image_processor") and self.img2img_pipeline.ip_adapter_image_processor is not None):
                            print(f"  ℹ img2img_pipeline 的 IP-Adapter 未加载，尝试加载...")
                            try:
                                self._load_ip_adapter()  # 这会同时加载到 img2img_pipeline
                            except Exception as e:
                                print(f"  ⚠ 无法为 img2img_pipeline 加载 IP-Adapter: {e}")
                    
                    # 验证 IP-Adapter 是否已正确加载
                    target_pipe = pipeline_to_use or self.pipeline
                    if hasattr(target_pipe, "prepare_ip_adapter_image_embeds"):
                        # 检查 image_projection_layers 是否存在
                        try:
                            # 尝试访问 image_projection_layers 来验证 IP-Adapter 是否已加载
                            if hasattr(target_pipe, "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
                                # 检查是否有 image_projection_layers（通过检查 processor 的内部属性）
                                if hasattr(target_pipe.ip_adapter_image_processor, "image_encoder") or hasattr(target_pipe, "image_projection_layers"):
                                    print(f"  ✓ IP-Adapter 已正确加载并验证")
                                else:
                                    print(f"  ⚠ IP-Adapter 可能未完全加载，但继续尝试使用")
                        except Exception as e:
                            print(f"  ⚠ 验证 IP-Adapter 时出错: {e}，但继续尝试使用")
                    
                    print(f"  ✓ IP-Adapter 加载完成")
                except Exception as e:
                    print(f"  ✗ IP-Adapter 加载失败: {e}")
                    print(f"  ⚠ 禁用 IP-Adapter，使用纯文生图")
                    use_ip_adapter = False
                    if ip_adapter_image is not None:
                        ip_adapter_image = None
                    # 当 IP-Adapter 不可用时，确保不使用 img2img pipeline（除非有有效的 init_image）
                    # 因为 img2img pipeline 需要 image 参数，而我们现在没有参考图像
                    if pipeline_to_use is self.img2img_pipeline and init_image is None:
                        pipeline_to_use = self.pipeline
                        print(f"  ℹ IP-Adapter 不可用，切换到普通文生图 pipeline")
        
        # 动态控制 LoRA 权重
        # 如果使用 LoRA，降低 IP-Adapter 权重以避免冲突导致脸部变形
        ip_adapter_scale_override = None
        if self.use_lora and hasattr(self.pipeline, "set_adapters"):
            if use_lora is None:
                use_lora = True  # 默认使用
            lora_weight = self.lora_alpha if use_lora else 0.0
            self.pipeline.set_adapters([self.lora_adapter_name], adapter_weights=[lora_weight])
            if self.img2img_pipeline is not None and hasattr(self.img2img_pipeline, "set_adapters"):
                self.img2img_pipeline.set_adapters([self.lora_adapter_name], adapter_weights=[lora_weight])
            
            # 使用 LoRA 时，降低 IP-Adapter 权重（从 0.7 降到 0.3-0.4）以避免冲突
            if use_lora and use_ip_adapter:
                ip_adapter_scale_override = 0.35  # 降低 IP-Adapter 权重

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        guidance = guidance_scale or self.sdxl_config.get("guidance_scale", 8.5) or self.image_config.get("guidance_scale", 7.5)
        steps = num_inference_steps or self.sdxl_config.get("num_inference_steps", 40) or self.image_config.get("num_inference_steps", 40)  # 统一为 40 步（与 InstantID 保持一致）

        negative_prompt = (
            negative_prompt
            if negative_prompt is not None
            else self.negative_prompt
        )

        init_image = None
        ip_adapter_image = None
        pipeline_to_use = self.pipeline

        # 如果 IP-Adapter 不可用（纯文生图），强制使用普通 pipeline，不使用 img2img
        # 只有当 IP-Adapter 可用且有有效的 reference_image_path 时，才考虑使用 img2img
        if use_ip_adapter and reference_image_path and Path(reference_image_path).exists() and self.img2img_pipeline is not None and self.use_img2img:
            from PIL import Image
            from PIL import ImageOps

        try:
            init_image = Image.open(reference_image_path).convert("RGB")
            init_image = ImageOps.fit(
                init_image,
                (self.width, self.height),
                method=Image.Resampling.LANCZOS,
            )
            pipeline_to_use = self.img2img_pipeline
            print(f"  ✓ 使用 reference_image_path 进行 img2img: {reference_image_path}")
        except Exception as e:
                print(f"  ⚠ 无法加载 reference_image_path: {e}，使用普通文生图")
                init_image = None
                pipeline_to_use = self.pipeline
        else:
            # IP-Adapter 不可用或没有有效的 reference_image_path，确保使用普通 pipeline
            pipeline_to_use = self.pipeline
            init_image = None  # 确保 init_image 为 None
            if not use_ip_adapter:
                print(f"  ℹ IP-Adapter 不可用，使用普通文生图 pipeline（不使用 img2img）")
            elif reference_image_path:
                print(f"  ⚠ reference_image_path 不存在或无效: {reference_image_path}，使用普通文生图")

        if use_ip_adapter:
            from PIL import Image, ImageOps

            clip_source = None
            face_source = None

            # 只使用明确传入的参考图像路径，不使用自动选择的参考图像
            if reference_image_path and Path(reference_image_path).exists():
                clip_source = Image.open(reference_image_path).convert("RGB")
                print(f"  ℹ 使用传入的 reference_image_path: {reference_image_path}")
            if face_reference_image_path and Path(face_reference_image_path).exists():
                face_source = Image.open(face_reference_image_path).convert("RGB")
                print(f"  ℹ 使用传入的 face_reference_image_path: {face_reference_image_path}")

            if clip_source is None and face_source is not None:
                clip_source = face_source
            if face_source is None and clip_source is not None:
                face_source = clip_source

            # 如果没有参考图像，禁用 IP-Adapter（避免错误）
            if clip_source is None:
                print(f"  ⚠ 未提供参考图像，禁用 IP-Adapter（使用纯文生图）")
                use_ip_adapter = False
                ip_adapter_image = None
            else:
                # 有参考图像，继续处理 IP-Adapter
                print(f"  ✓ 使用参考图像进行 IP-Adapter 生成")
                clip_size = int(self.ip_adapter_config.get("clip_image_size", 224))
                face_size = int(self.ip_adapter_config.get("face_image_size", 128))
                face_crop_ratio = float(self.ip_adapter_config.get("face_crop_ratio", 1.0))
                face_only = bool(self.ip_adapter_config.get("face_reference_only", False))

                prepared_images: List[Any] = []
                weight_names = self.ip_adapter_weight_names or [""]

                for name in weight_names:
                    name_l = str(name).lower()
                    if "face" in name_l:
                        source_img = face_source
                        if source_img is None:
                            source_img = clip_source

                        if (
                            source_img is not None
                            and face_crop_ratio > 0.0
                            and face_crop_ratio < 1.0
                        ):
                            w, h = source_img.size
                            crop_w = int(w * face_crop_ratio)
                            crop_h = int(h * face_crop_ratio)
                            crop_w = max(1, min(w, crop_w))
                            crop_h = max(1, min(h, crop_h))
                            left = max(0, (w - crop_w) // 2)
                            top = max(0, (h - crop_h) // 2)
                            right = min(w, left + crop_w)
                            bottom = min(h, top + crop_h)
                            source_img = source_img.crop((left, top, right, bottom))

                        prepared_images.append(
                            ImageOps.fit(
                                source_img,
                                (face_size, face_size),
                                method=Image.Resampling.LANCZOS,
                            )
                        )
                    else:
                        source_img = face_source if face_only and face_source is not None else clip_source
                        prepared_images.append(
                            ImageOps.fit(
                                source_img,
                                (clip_size, clip_size),
                                method=Image.Resampling.LANCZOS,
                            )
                        )

                if len(prepared_images) == 1:
                    ip_adapter_image = prepared_images[0]
                else:
                    ip_adapter_image = prepared_images

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": generator,
        }

        if ip_adapter_image is not None and use_ip_adapter:
            target_pipe = pipeline_to_use or self.pipeline
            if hasattr(target_pipe, "set_ip_adapter_scale"):
                # 如果设置了 override，使用 override 值（使用 LoRA 时降低 IP-Adapter 权重）
                if ip_adapter_scale_override is not None:
                    scale = ip_adapter_scale_override
                else:
                    scale = (
                        self.ip_adapter_scales
                        if len(self.ip_adapter_scales) > 1
                        else self.ip_adapter_scales[0]
                    )
                target_pipe.set_ip_adapter_scale(scale)
            # 检查是否是 InstantID pipeline（InstantID 不使用 prepare_ip_adapter_image_embeds）
            is_instantid_pipeline = (
                self.engine == "instantid" or 
                hasattr(target_pipe, "image_proj_model") or
                "InstantID" in type(target_pipe).__name__
            )
            
            if is_instantid_pipeline:
                # InstantID 不使用 prepare_ip_adapter_image_embeds，直接传递 image_embeds
                # InstantID 的 IP-Adapter 在 _generate_image_instantid 中处理
                print(f"  ℹ InstantID pipeline：跳过 prepare_ip_adapter_image_embeds（InstantID 使用 image_embeds 参数）")
                prepared = None
                use_ip_adapter = False  # InstantID 有自己的 IP-Adapter 处理方式
            elif hasattr(target_pipe, "prepare_ip_adapter_image_embeds"):
                # 在调用之前，先验证 IP-Adapter 是否已正确加载
                # 检查是否有 image_projection_layers 属性（这是 IP-Adapter 的核心组件）
                ip_adapter_ready = False
                try:
                    # 检查 IP-Adapter 的关键属性是否存在
                    if hasattr(target_pipe, "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
                        # 进一步检查 image_projection_layers（如果可以直接访问）
                        # 注意：某些版本的 diffusers 可能不直接暴露这个属性
                        # 但我们可以通过尝试调用 prepare_ip_adapter_image_embeds 来验证
                        ip_adapter_ready = True
                    else:
                        print(f"  ⚠ ip_adapter_image_processor 未加载，尝试加载...")
                        try:
                            self._load_ip_adapter()
                            ip_adapter_ready = True
                        except Exception as e:
                            print(f"  ✗ IP-Adapter 加载失败: {e}")
                            ip_adapter_ready = False
                except Exception as e:
                    print(f"  ⚠ 检查 IP-Adapter 状态时出错: {e}")
                    ip_adapter_ready = False
                
                if not ip_adapter_ready:
                    print(f"  ⚠ IP-Adapter 未就绪，禁用 IP-Adapter，使用纯文生图")
                    use_ip_adapter = False
                    prepared = None
                else:
                    prepared = None
                    try:
                        prepared = target_pipe.prepare_ip_adapter_image_embeds(
                            ip_adapter_image=ip_adapter_image,
                            ip_adapter_image_embeds=None,
                            device=self.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                        )
                    except (AttributeError, TypeError, NameError) as e:
                        error_str = str(e)
                        # 检查是否是 InstantID pipeline 相关的错误
                        if "image_proj_model_in_features" in error_str or ("InstantID" in error_str and "image_proj" in error_str):
                            print(f"  ✗ 检测到 InstantID pipeline 错误: {e}")
                            print(f"  ℹ InstantID 不使用 prepare_ip_adapter_image_embeds，跳过 IP-Adapter 处理")
                            prepared = None
                            use_ip_adapter = False
                        elif "image_projection_layers" in error_str or "NoneType" in error_str:
                            print(f"  ✗ IP-Adapter 未正确加载: {e}")
                            print(f"  ⚠ 尝试重新加载 IP-Adapter...")
                            try:
                                # 先卸载旧的 IP-Adapter
                                if hasattr(target_pipe, "disable_ip_adapter"):
                                    try:
                                        target_pipe.disable_ip_adapter()
                                    except:
                                        pass
                                elif hasattr(target_pipe, "unload_ip_adapter"):
                                    try:
                                        target_pipe.unload_ip_adapter()
                                    except:
                                        pass
                                
                                # 重新加载
                                self._load_ip_adapter()
                                # 重新尝试
                                prepared = target_pipe.prepare_ip_adapter_image_embeds(
                                    ip_adapter_image=ip_adapter_image,
                                    ip_adapter_image_embeds=None,
                                    device=self.device,
                                    num_images_per_prompt=1,
                                    do_classifier_free_guidance=True,
                                )
                                print(f"  ✓ IP-Adapter 重新加载成功")
                            except Exception as e2:
                                print(f"  ✗ IP-Adapter 重新加载失败: {e2}")
                                print(f"  ⚠ 禁用 IP-Adapter，使用纯文生图")
                                prepared = None
                                use_ip_adapter = False
                        else:
                            raise
                
                if prepared is not None:
                    if isinstance(prepared, tuple) and len(prepared) == 2:
                        image_embeds, uncond_embeds = prepared
                else:
                    image_embeds, uncond_embeds = prepared, None
                
                if image_embeds is not None:
                    if isinstance(image_embeds, (list, tuple)):
                        image_embeds = list(image_embeds)
                    else:
                        image_embeds = [image_embeds]
                if uncond_embeds is not None:
                    if isinstance(uncond_embeds, (list, tuple)):
                        uncond_embeds = list(uncond_embeds)
                    else:
                        uncond_embeds = [uncond_embeds]

                kwargs["ip_adapter_image_embeds"] = image_embeds
                if uncond_embeds is not None:
                    kwargs["ip_adapter_uncond_image_embeds"] = uncond_embeds
                    print(f"  ✓ IP-Adapter 参数已添加到生成参数")
            else:
                # IP-Adapter 加载失败，移除相关参数，使用纯文生图
                print(f"  ⚠ IP-Adapter 不可用，使用纯文生图（不传递参考图像）")
                use_ip_adapter = False
            # 如果 pipeline 不支持 prepare_ip_adapter_image_embeds，使用直接传递图像的方式
            # 只有当 ip_adapter_image 不为 None 时才添加到 kwargs
            if ip_adapter_image is not None:
                kwargs["ip_adapter_image"] = ip_adapter_image
                print(f"  ✓ 使用直接传递图像的方式（IP-Adapter）")
            else:
                print(f"  ⚠ ip_adapter_image 为 None，不添加到 kwargs")

        # 当 IP-Adapter 不可用时，确保不使用 img2img pipeline（除非有有效的 init_image）
        # 因为 img2img pipeline 需要 image 参数
        if not use_ip_adapter:
            if pipeline_to_use is self.img2img_pipeline:
                # IP-Adapter 不可用，强制切换到普通 pipeline
                pipeline_to_use = self.pipeline
                init_image = None  # 确保 init_image 为 None
                print(f"  ℹ IP-Adapter 不可用，强制切换到普通文生图 pipeline（不使用 img2img）")
        
        # 在调用 pipeline 之前，最终检查：如果 pipeline_to_use 是 img2img_pipeline，必须有有效的 init_image
        # 否则强制切换到普通 pipeline
        if pipeline_to_use is self.img2img_pipeline:
            from PIL import Image as PILImage
            has_valid_init_image = (
                init_image is not None and 
                isinstance(init_image, PILImage.Image)
            )
            if not has_valid_init_image:
                pipeline_to_use = self.pipeline
                init_image = None
                print(f"  ⚠ 最终检查：pipeline_to_use 是 img2img_pipeline，但 init_image 无效，强制切换到普通 pipeline")

        # 如果提供了init_image（用于场景连贯性），使用img2img
        # 注意：init_image 可能是 Path 对象（从 scene 参数传入）或 PIL Image（从 reference_image_path 加载）
        init_image_path = None
        if scene and scene.get("init_image"):
            init_image_path = scene.get("init_image")
        
        if init_image_path is not None and Path(init_image_path).exists():
            from PIL import Image
            try:
                init_img = Image.open(init_image_path).convert("RGB")
            except Exception as e:
                print(f"  ⚠ 无法加载 init_image: {e}，使用普通文生图")
                init_img = None
            
            if init_img is not None:
                # 场景连贯性使用较低的strength（0.2-0.3），只影响背景和整体风格，不影响角色和构图
                continuity_strength = self.img2img_strength * 0.4  # 降低到约0.07（如果默认0.18）
                print(f"  ✓ 使用场景连贯性img2img，strength={continuity_strength:.2f}")
                
                # 确保有img2img pipeline
                if self.img2img_pipeline is None and self.pipeline:
                    try:
                        from diffusers import StableDiffusionXLImg2ImgPipeline
                        self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**self.pipeline.components)
                        if hasattr(self.img2img_pipeline, "enable_model_cpu_offload"):
                            self.img2img_pipeline.enable_model_cpu_offload()
                        else:
                            self.img2img_pipeline = self.img2img_pipeline.to(self.device)
                    except Exception as e:
                        print(f"  ⚠ 无法创建img2img pipeline: {e}，将使用普通生成")
                        init_img = None
                
                if self.img2img_pipeline and init_img is not None:
                    result = self.img2img_pipeline(
                        image=init_img,
                        strength=continuity_strength,
                        **kwargs,
                    )
                else:
                    # 如果没有img2img pipeline 或 init_img 无效，使用普通生成
                    # 确保 pipeline_to_use 是普通 pipeline
                    pipeline_to_use = self.pipeline
                    print(f"  ⚠ 场景连贯性 img2img 不可用，使用普通文生图")
                    # 清理 kwargs 中的 image 相关参数
                    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image']}
                    result = pipeline_to_use(
                        width=self.width,
                        height=self.height,
                        **kwargs_clean,
                    )
                print(f"  ⚠ 无法加载场景连贯性图像: {e}，使用普通文生图")
                # 确保 pipeline_to_use 是普通 pipeline
                pipeline_to_use = self.pipeline
                # 清理 kwargs 中的 image 相关参数
                kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image']}
                result = pipeline_to_use(
                    width=self.width,
                    height=self.height,
                    **kwargs_clean,
                )
        # 在检查 elif 之前，先确保如果 pipeline_to_use 是 img2img_pipeline，必须有有效的 init_image
        # 如果没有，强制切换到普通 pipeline
        if pipeline_to_use is self.img2img_pipeline:
            from PIL import Image as PILImage
            has_valid_init_image = (
                init_image is not None and 
                isinstance(init_image, PILImage.Image)
            )
            if not has_valid_init_image:
                print(f"  ⚠ pipeline_to_use 是 img2img_pipeline，但 init_image 无效，强制切换到普通 pipeline")
                pipeline_to_use = self.pipeline
                init_image = None
                # 清理 kwargs 中的 image 相关参数
                kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image']}
                kwargs = kwargs_clean
        
        if pipeline_to_use is self.img2img_pipeline and init_image is not None:
            # 检查 init_image 是否是 PIL Image（从 reference_image_path 加载的）
            from PIL import Image as PILImage
            if isinstance(init_image, PILImage.Image):
                # 如果 pipeline_to_use 是 img2img_pipeline 且 init_image 是 PIL Image（从 reference_image_path 加载）
                result = pipeline_to_use(
                    image=init_image,
                    strength=self.img2img_strength,
                    **kwargs,
                )
            else:
                # init_image 不是有效的 PIL Image，使用普通文生图
                pipeline_to_use = self.pipeline
                print(f"  ℹ init_image 不是有效的 PIL Image，使用普通文生图 pipeline")
                # 清理 kwargs 中的 image 相关参数
                kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image']}
                result = pipeline_to_use(
                    width=self.width,
                    height=self.height,
                    **kwargs_clean,
                )
        else:
            # 普通文生图（不使用 img2img）
            # 在调用 pipeline 之前，强制检查并修复 pipeline_to_use
            # 如果 pipeline_to_use 是 img2img_pipeline，但 init_image 是 None 或无效，强制切换到普通 pipeline
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                # 检查 init_image 是否有效
                has_valid_init_image = (
                    init_image is not None and 
                    isinstance(init_image, PILImage.Image)
                )
                
                if not has_valid_init_image:
                    pipeline_to_use = self.pipeline
                    print(f"  ⚠ 强制修复：pipeline_to_use 是 img2img_pipeline，但 init_image 无效，切换到普通 pipeline")
                elif not use_ip_adapter:
                    # IP-Adapter 不可用，即使有 init_image 也不使用 img2img
                    pipeline_to_use = self.pipeline
                    print(f"  ⚠ 强制修复：IP-Adapter 不可用，切换到普通 pipeline（不使用 img2img）")
            
            # 最终安全检查：确保不会传递 None 作为 image 参数
            # 如果 pipeline_to_use 仍然是 img2img_pipeline，但 init_image 是 None，强制切换
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                if init_image is None or not isinstance(init_image, PILImage.Image):
                    print(f"  ⚠ 最终安全检查：pipeline_to_use 仍然是 img2img_pipeline，但 init_image 无效，强制切换到普通 pipeline")
                    pipeline_to_use = self.pipeline
                elif not use_ip_adapter:
                    print(f"  ⚠ 最终安全检查：IP-Adapter 不可用，强制切换到普通 pipeline")
                    pipeline_to_use = self.pipeline
            
            # 确保 kwargs 中不包含 image 参数（如果使用普通 pipeline）
            if pipeline_to_use is self.pipeline:
                # 移除 kwargs 中可能存在的 image 相关参数，避免冲突
                kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image']}
                kwargs = kwargs_clean
            else:
                # 即使使用 img2img_pipeline，也确保 kwargs 中没有 None 的 image 参数
                if 'image' in kwargs and kwargs['image'] is None:
                    del kwargs['image']
                    print(f"  ⚠ 移除 kwargs 中的 None image 参数")
            
            # 最终检查：确保 pipeline_to_use 不是 img2img_pipeline（除非有有效的 init_image）
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                # 检查是否有有效的 image 参数（在 kwargs 中或作为 init_image）
                has_image_in_kwargs = 'image' in kwargs and kwargs['image'] is not None
                has_valid_init_image = (
                    init_image is not None and 
                    isinstance(init_image, PILImage.Image)
                )
                if not (has_image_in_kwargs or has_valid_init_image):
                    print(f"  ⚠ 最终修复：pipeline_to_use 是 img2img_pipeline，但没有任何有效的 image，强制切换到普通 pipeline")
                    pipeline_to_use = self.pipeline
                    # 清理 kwargs
                    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image']}
                    kwargs = kwargs_clean
            
            # 绝对最终检查：在调用 pipeline 之前，再次确保不会使用 img2img_pipeline 且 image 为 None
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                # 检查 kwargs 中是否有有效的 image
                has_valid_image = 'image' in kwargs and kwargs['image'] is not None
                # 检查 init_image 是否有效
                has_valid_init = init_image is not None and isinstance(init_image, PILImage.Image)
                
                if not (has_valid_image or has_valid_init):
                    print(f"  ⚠ 绝对最终检查：强制切换到普通 pipeline（img2img_pipeline 需要有效的 image）")
                    pipeline_to_use = self.pipeline
                    # 彻底清理 kwargs
                    kwargs = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image', 'strength']}
            
            # 打印调试信息
            if pipeline_to_use is self.img2img_pipeline:
                print(f"  ℹ 使用 img2img_pipeline（有有效的 image）")
            else:
                print(f"  ℹ 使用普通 pipeline（text-to-image）")
                # 使用普通 pipeline 时，彻底清理 kwargs，移除所有可能冲突的参数
                # 包括 image, ip_adapter_image, strength 等
                # 首先打印清理前的 kwargs 键，用于调试
                keys_before = set(kwargs.keys())
                kwargs_final = {k: v for k, v in kwargs.items() if k not in ['image', 'ip_adapter_image', 'strength']}
                kwargs = kwargs_final
                # 再次确保没有 None 值（除了 negative_prompt，因为它可以为 None）
                kwargs = {k: v for k, v in kwargs.items() if v is not None or k in ['negative_prompt']}
                # 打印清理后的 kwargs 键，用于调试
                keys_after = set(kwargs.keys())
                removed_keys = keys_before - keys_after
                if removed_keys:
                    print(f"  ℹ 已从 kwargs 中移除: {', '.join(removed_keys)}")
                # 最终检查：确保 kwargs 中不包含 image 相关参数
                if 'image' in kwargs:
                    print(f"  ⚠ 警告：kwargs 中仍然包含 'image' 参数，强制移除")
                    del kwargs['image']
                if 'ip_adapter_image' in kwargs:
                    print(f"  ⚠ 警告：kwargs 中仍然包含 'ip_adapter_image' 参数，强制移除")
                    del kwargs['ip_adapter_image']
                if 'strength' in kwargs:
                    print(f"  ⚠ 警告：kwargs 中仍然包含 'strength' 参数，强制移除")
                    del kwargs['strength']
            
            # 最终安全检查：在调用 pipeline 之前，再次确保 kwargs 中没有 image 相关参数
            # 这对于普通 pipeline 是必需的
            if pipeline_to_use is self.pipeline:
                forbidden_keys = ['image', 'ip_adapter_image', 'strength']
                for key in forbidden_keys:
                    if key in kwargs:
                        print(f"  ⚠ 最终清理：移除 kwargs 中的 '{key}' 参数")
                        del kwargs[key]
                # 打印最终的 kwargs 键，用于调试
                final_keys = list(kwargs.keys())
                print(f"  ℹ 最终 kwargs 键: {', '.join(final_keys)}")
                # 确保 kwargs 中绝对不包含 image 相关参数
                if 'image' in kwargs:
                    raise ValueError(f"严重错误：kwargs 中仍然包含 'image' 参数，这不应该发生！kwargs 键: {final_keys}")
            
            # 调用 pipeline
            try:
                result = pipeline_to_use(
                width=self.width,
                height=self.height,
                **kwargs,
            )
            except TypeError as e:
                if "image must be passed" in str(e) or "image" in str(e).lower():
                    print(f"  ✗ Pipeline 调用失败，错误信息: {e}")
                    print(f"  ℹ kwargs 内容: {list(kwargs.keys())}")
                    print(f"  ℹ pipeline_to_use 类型: {type(pipeline_to_use)}")
                    print(f"  ℹ pipeline_to_use 是否为 img2img_pipeline: {pipeline_to_use is self.img2img_pipeline}")
                    # 最后一次尝试：创建一个全新的、干净的 kwargs
                    clean_kwargs = {
                        "prompt": kwargs.get("prompt"),
                        "negative_prompt": kwargs.get("negative_prompt"),
                        "guidance_scale": kwargs.get("guidance_scale"),
                        "num_inference_steps": kwargs.get("num_inference_steps"),
                        "generator": kwargs.get("generator"),
                    }
                    # 移除所有 None 值（除了 negative_prompt）
                    clean_kwargs = {k: v for k, v in clean_kwargs.items() if v is not None or k == 'negative_prompt'}
                    print(f"  ℹ 使用干净的 kwargs 重试: {list(clean_kwargs.keys())}")
                    result = pipeline_to_use(
                        width=self.width,
                        height=self.height,
                        **clean_kwargs,
                    )
                else:
                    raise

        image = result.images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # 入口：批量生成
    # ------------------------------------------------------------------
    def generate_from_script(
        self,
        script_json_path: str,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        update_script: bool = True,
    ) -> List[Path]:
        """根据脚本 JSON 生成全量场景图像"""

        script_json_path = Path(script_json_path)
        if not script_json_path.exists():
            raise FileNotFoundError(f"脚本 JSON 未找到: {script_json_path}")

        output_dir = Path(output_dir or self.paths_config.get("image_output", "outputs/images"))
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(script_json_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        scenes = script_data.get("scenes", [])
        
        # 在每个场景生成后清理GPU缓存
        import torch
        import gc
        
        if not scenes:
            print("⚠ 未在脚本中找到场景数据，跳过图像生成")
            return []

        # 自动生成 face_style_auto（如果不存在）
        try:
            from face_style_auto_generator import generate_face_styles_for_episode
            print("\n[自动生成 face_style_auto]")
            generate_face_styles_for_episode(scenes, smooth=True, overwrite_existing=False)
            print(f"✓ 已为 {len(scenes)} 个场景生成/更新 face_style_auto")
        except ImportError:
            print("⚠ face_style_auto_generator 未找到，跳过自动生成")
        except Exception as e:
            print(f"⚠ face_style_auto 生成失败: {e}")

        # 显示角色和场景模板加载状态
        if self.character_profiles:
            print(f"\n[角色模板] 已加载 {len(self.character_profiles)} 个角色模板")
        if self.scene_profiles:
            print(f"[场景模板] 已加载 {len(self.scene_profiles)} 个场景模板")
            # 尝试匹配当前脚本的场景模板
            episode = script_data.get("episode")
            title = script_data.get("title", "")
            matched_profile = self._get_scene_profile(title, episode)
            if matched_profile:
                print(f"  ✓ 匹配到场景模板: {matched_profile.get('scene_name', '未知')}")

        if self.pipeline is None:
            self.load_pipeline()

        saved_paths: List[Path] = []
        previous_scene = None  # 跟踪前一个场景，用于连贯性控制
        previous_image_path = None  # 前一个场景的图像路径，用于img2img
        
        for idx, scene in enumerate(scenes, start=1):
            # 在每个场景生成前清理GPU缓存（除了第一个场景）
            if idx > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # 判断是否需要主角（使用 LoRA）
            needs_character = self._needs_character(scene)
            # 传递前一个场景信息，用于连贯性控制
            prompt = self.build_prompt(scene, include_character=needs_character, script_data=script_data, previous_scene=previous_scene)
            filename = f"scene_{idx:03d}.png"
            output_path = output_dir / filename

            reference_image = self._select_reference_image(scene, idx)
            
            # 识别场景中的角色，自动选择对应的参考图像
            identified_characters = self._identify_characters_in_scene(scene)
            primary_character = identified_characters[0] if identified_characters else None
            
            # 根据角色ID选择对应的参考图像
            # 如果识别到角色且有对应的参考图像，使用该参考图像；否则使用默认的循环选择
            face_reference = self._select_face_reference_image(idx, character_id=primary_character)
            
            if primary_character and face_reference:
                print(f"  ✓ 识别到角色: {primary_character}，使用对应参考图像: {face_reference.name}")

            if output_path.exists() and not overwrite:
                print(f"跳过 {output_path}（文件已存在）")
                scene["image_path"] = str(output_path)
                saved_paths.append(output_path)
                continue

            print(f"生成场景图像 {idx}/{len(scenes)}: {prompt[:80]}...")
            if needs_character:
                print(f"  ✓ 使用 LoRA（主角场景）")
            else:
                print(f"  - 纯背景场景（不使用 LoRA）")
            if reference_image:
                print(f"  使用参考图像: {reference_image}")
            if face_reference:
                print(f"  使用面部参考图: {face_reference}")
            try:
                # 如果启用场景连贯性，且不是第一个场景，使用前一个场景作为img2img参考
                init_image = None
                if idx > 1 and previous_image_path and Path(previous_image_path).exists():
                    # 检查是否应该使用img2img（同一环境或连续场景）
                    current_env = scene.get("scene_name") or script_data.get("title", "") if script_data else ""
                    prev_env = previous_scene.get("scene_name") or "" if previous_scene else ""
                    
                    # 如果环境相同或相似，使用img2img保持连贯性
                    use_continuity = False
                    if current_env and prev_env:
                        # 检查环境关键词匹配
                        env_keywords = ["desert", "chamber", "corridor", "遗迹", "沙漠", "地下", "underground"]
                        if any(kw in current_env.lower() and kw in prev_env.lower() for kw in env_keywords):
                            use_continuity = True
                    elif idx <= 5:  # 前5个场景通常在同一环境（更宽松的条件）
                        use_continuity = True
                    
                    if use_continuity and self.use_img2img:
                        init_image = Path(previous_image_path)
                        print(f"  ✓ 使用前一个场景作为参考（场景连贯性），strength={self.img2img_strength * 0.4:.2f}")
                    elif idx <= 5 and self.use_img2img and previous_image_path:
                        # 即使环境关键词不匹配，前5个场景也尝试使用img2img保持连贯性
                        init_image = Path(previous_image_path)
                        print(f"  ✓ 使用前一个场景作为参考（相邻场景连贯性），strength={self.img2img_strength * 0.4:.2f}")
                
                path = self.generate_image(
                    prompt,
                    output_path,
                    reference_image_path=reference_image,
                    face_reference_image_path=face_reference,
                    use_lora=needs_character,  # 仅在需要主角时使用 LoRA
                    scene=scene,
                    init_image=init_image,  # 传递前一个场景图像用于连贯性
                )
                if not path or not Path(path).exists():
                    raise RuntimeError(f"图像生成函数未返回有效文件: {path}")
                print(f"  ✓ 场景 {idx} 图像已保存: {path}")
                scene["image_path"] = str(path)
                saved_paths.append(path)
                
                # 更新前一个场景信息
                previous_scene = scene
                previous_image_path = str(path)
                
                # 每个场景生成后清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as exc:
                print(f"✗ 场景 {idx} 生成失败: {exc}")
                scene.setdefault("image_path", str(output_path))
                # 即使失败也清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        if update_script:
            with open(script_json_path, "w", encoding="utf-8") as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)

        return saved_paths

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _adjust_face_kps_canvas(self, face_kps, scale: float, offset_y: int):
        from PIL import Image
        is_np = isinstance(face_kps, self.np.ndarray)
        face_img = Image.fromarray(face_kps) if is_np else face_kps
        w, h = face_img.size
        scale = max(0.1, min(scale, 1.5))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = face_img.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2 + offset_y
        paste_y = max(-new_h, min(paste_y, h))
        canvas.paste(resized, (paste_x, paste_y))
        return self.np.array(canvas) if is_np else canvas

    def _as_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if value is None or value == "":
            return []
        return [value]

    def _clean_prompt_text(self, text: str) -> str:
        """清理 prompt 文本，支持中文
        
        Args:
            text: 原始文本（可以是中文或英文）
            
        Returns:
            清理后的文本
        """
        text = (text or "").strip().strip('"')
        if not text:
            return ""

        # 如果配置为仅 ASCII，过滤掉所有非 ASCII 字符（包括中文）
        # 否则保留中文，因为 CLIPTokenizer 支持中文
        if self.ascii_only_prompt:
            text = "".join(ch if ord(ch) < 128 else " " for ch in text)
            # 处理空格
            text = " ".join(t for t in text.split() if t)
        else:
            # 支持中文：保留所有字符，只清理多余空格
            # 中文不需要按空格分词，可以直接使用
            import re
            # 保留中英文、数字、标点和括号，清理多余空格
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _load_character_profiles(self) -> Dict[str, Any]:
        """加载角色模板配置文件"""
        import yaml
        profile_path = Path(__file__).parent / "character_profiles.yaml"
        if not profile_path.exists():
            print(f"⚠ 角色模板文件不存在: {profile_path}，将使用默认配置")
            return {}
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get("characters", {})
        except Exception as e:
            print(f"⚠ 加载角色模板失败: {e}，将使用默认配置")
            return {}

    def _load_scene_profiles(self) -> Dict[str, Any]:
        """加载场景模板配置文件"""
        import yaml
        profile_path = Path(__file__).parent / "scene_profiles.yaml"
        if not profile_path.exists():
            print(f"⚠ 场景模板文件不存在: {profile_path}，将使用默认配置")
            return {}
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get("scenes", {})
        except Exception as e:
            print(f"⚠ 加载场景模板失败: {e}，将使用默认配置")
            return {}

    def _get_character_profile(self, character_id: str = "hanli") -> Dict[str, Any]:
        """获取角色模板"""
        return self.character_profiles.get(character_id, {})

    def _get_scene_profile(
        self,
        scene_name: str = None,
        episode: int = None,
        profile_key: str = None,
    ) -> Dict[str, Any]:
        """根据场景 key、名称或集数获取场景模板"""

        # 1. 若显式指定模板 key，直接精确匹配
        if profile_key:
            profile = self.scene_profiles.get(profile_key)
            if profile:
                return profile
        
        # 2. 使用场景名称模糊匹配（改进：同时匹配scene_name字段和key）
        if scene_name:
            scene_name_lower = scene_name.lower()
            
            # 先检查精确匹配（scene_name字段和key）
            for key, profile in self.scene_profiles.items():
                profile_scene_name = profile.get("scene_name", "").lower()
                # 检查scene_name是否匹配
                if profile_scene_name and (profile_scene_name in scene_name_lower or scene_name_lower in profile_scene_name):
                    return profile
                # 检查key是否匹配
                if key.lower() in scene_name_lower or scene_name_lower in key.lower():
                    return profile
            
            # 然后按优先级检查关键词匹配
            # 优先级1: 沙漠关键词（最高优先级）
            if "沙漠" in scene_name or "沙地" in scene_name or "沙砾" in scene_name:
                for key, profile in self.scene_profiles.items():
                    profile_scene_name = profile.get("scene_name", "").lower()
                    if "沙漠" in profile_scene_name or "desert" in key.lower():
                        return profile
            
            # 优先级2: 草原关键词
            if "草原" in scene_name:
                for key, profile in self.scene_profiles.items():
                    profile_scene_name = profile.get("scene_name", "").lower()
                    if "草原" in profile_scene_name or "prairie" in key.lower():
                        return profile
            
            # 优先级3: 城市/商号关键词（只有当没有沙漠/草原关键词时才匹配）
            # 注意：商号可能出现在沙漠中，所以只有在明确有城市/市集关键词时才匹配城市模板
            if ("城市" in scene_name or "市集" in scene_name) and ("沙漠" not in scene_name and "沙地" not in scene_name):
                for key, profile in self.scene_profiles.items():
                    profile_scene_name = profile.get("scene_name", "").lower()
                    if "城市" in profile_scene_name or "市集" in profile_scene_name or "city" in key.lower() or "market" in key.lower():
                        return profile
        
        # 3. 其次使用集数匹配（为了兼容历史脚本）
        if episode:
            for key, profile in self.scene_profiles.items():
                if profile.get("episode") == episode:
                    return profile
        
        # 默认返回第一个场景模板（如果有）
        if self.scene_profiles:
            return list(self.scene_profiles.values())[0]
        
        return {}

