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
from typing import Dict, Any, List, Optional, Tuple
import warnings

import torch
from pathlib import Path
import huggingface_hub
from scene_intent_analyzer import SceneIntentAnalyzer
from prompt import TokenEstimator, PromptParser, PromptOptimizer, PromptBuilder
from model_selector import ModelSelector, TaskType
import re

# ⚡ 抑制 CLIP tokenizer 的 77 token 警告（Flux 使用 T5 作为主编码器，支持 512 tokens，CLIP 只是辅助编码器）
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than the specified maximum sequence length.*")
warnings.filterwarnings("ignore", message=".*The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens.*")


class ImageGenerator:
    """场景图像生成主类"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        import yaml
        
        # 保存 config_path 供后续使用（如 ModelManager）
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            # 如果传入的是字符串，可能是相对路径，优先使用当前工作目录
            # 如果当前工作目录下没有，再尝试相对于 image_generator.py 的位置
            config_candidate = Path.cwd() / self.config_path
            if config_candidate.exists():
                self.config_path = config_candidate.resolve()
            else:
                # 尝试相对于 image_generator.py 的位置（gen_video 目录）
                self.config_path = (Path(__file__).parent / self.config_path).resolve()

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.image_config: Dict[str, Any] = self.config.get("image", {})
        self.paths_config: Dict[str, Any] = self.config.get("paths", {})
        
        # 获取 models 根目录（用于 LoRA 路径解析）
        models_root = self.image_config.get("models_root")
        if not models_root:
            # 默认使用当前目录下的 models 目录
            models_root = Path(__file__).parent / "models"
        self.models_root = Path(models_root)

        self.device = torch.device(
            self.image_config.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu"))

        self.pipeline = None
        self.img2img_pipeline = None
        self.sdxl_pipeline = None  # 用于存储普通的 SDXL pipeline（当使用 InstantID 引擎时）
        self._instantid_pipeline_class = None
        self._instantid_ip_adapter_ready = False
        
        # 多模型 pipeline 缓存
        self.flux_pipeline = None  # Flux.1 + InstantID（人物生成）
        self.flux1_pipeline = None  # Flux.1（实验室/医学场景）
        self.flux2_pipeline = None  # Flux.2（科学背景图、太空/粒子/量子类）
        self.hunyuan_dit_pipeline = None
        self.kolors_pipeline = None
        self.sd3_turbo_pipeline = None
        
        # 初始化模型选择器
        self.model_selector = ModelSelector(self.config)
        self.pipe_name: str = self.image_config.get("model_name", "")
        self.negative_prompt: str = self.image_config.get(
            "negative_prompt", "")
        self.base_style_prompt: str = self.image_config.get(
            "base_style_prompt", "")
        self.environment_prompt: str = self.image_config.get(
            "environment_prompt", "")
        self.character_prompt: str = self.image_config.get(
            "character_prompt", "")

        # 加载角色和场景模板
        self.character_profiles = self._load_character_profiles()
        self.scene_profiles = self._load_scene_profiles()

        # 初始化通用场景意图分析器
        self.intent_analyzer = SceneIntentAnalyzer()

        # 设置 ascii_only_prompt（需要在 Prompt 模块初始化之前设置）
        self.ascii_only_prompt: bool = bool(
            self.image_config.get(
                "ascii_only_prompt", False))
        
        # 增强模式配置（新方案：PuLID + 解耦融合，v2.2-final直接从JSON读取参数）
        enhanced_config = self.image_config.get("enhanced_mode", {})
        self.use_enhanced_mode = enhanced_config.get("enabled", False)
        self.enhanced_generator = None  # 延迟加载
        
        if self.use_enhanced_mode:
            print("  ℹ️  增强模式已启用（PuLID + 解耦融合，v2.2-final模式）")
            print("     当 scene 参数存在时，将自动使用增强模式生成")
            print("     ⚡ v2.2-final: 直接从JSON读取锁定参数，不使用Planner决策")

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
        self.img2img_strength: float = float(
            self.image_config.get("img2img_strength", 0.6))
        self.use_img2img: bool = bool(
            self.image_config.get(
                "use_img2img", False))
        self.ip_adapter_config: Dict[str, Any] = self.image_config.get(
            "ip_adapter", {}) or {}
        self.use_ip_adapter: bool = bool(
            self.image_config.get(
                "use_ip_adapter", False))
        self.ip_adapter_weight_names: List[str] = self._as_list(
            self.ip_adapter_config.get("weight_name", []))
        if not self.ip_adapter_weight_names:
            self.ip_adapter_weight_names = ["ip-adapter_sdxl.bin"]

        raw_scales = self._as_list(self.ip_adapter_config.get("scale", 0.6))
        if not raw_scales:
            raw_scales = [0.6]
        if len(raw_scales) == 1 and len(self.ip_adapter_weight_names) > 1:
            raw_scales = raw_scales * len(self.ip_adapter_weight_names)
        self.ip_adapter_scales: List[float] = [float(s) for s in raw_scales]
        # 注意：ascii_only_prompt 已在上方初始化 Prompt 模块之前定义

        self.lora_config: Dict[str, Any] = self.image_config.get(
            "lora", {}) or {}
        self.use_lora: bool = bool(self.lora_config.get("enabled", False))
        self.lora_adapter_name: str = str(
            self.lora_config.get(
                "adapter_name",
                "default") or "default")
        self.lora_alpha: float = float(self.lora_config.get("alpha", 1.0))
        self.lora_ip_scale_multiplier: float = float(
            self.lora_config.get("ip_adapter_scale_multiplier", 0.6)
        )

        self.reference_images: List[Path] = self._load_reference_images()
        self.face_reference_dir = self.image_config.get("face_reference_dir")
        self.face_reference_images: List[Path] = self._load_face_reference_images(
        )

        # 角色参考图像目录（用于存储生成的参考图像）
        # ⚡ 修复：去掉多余的 gen_video 层级
        self.character_reference_dir = self.image_config.get(
            "character_reference_dir")
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
                    if any(Path(self.face_reference_dir).glob(
                            "*_reference.png")) or any(Path(self.face_reference_dir).glob("*_reference.jpg")):
                        self.character_reference_dir = self.face_reference_dir
                    else:
                        # ⚡ 修复：默认路径去掉 gen_video 层级
                        self.character_reference_dir = "character_references"
            else:
                # ⚡ 修复：默认路径去掉 gen_video 层级
                self.character_reference_dir = "character_references"

        # 加载角色参考图像映射
        self.character_reference_images: Dict[str, Path] = {}
        if self.character_reference_dir:
            self.character_reference_images = self._load_character_reference_images()

        # 检查使用的引擎
        self.engine = self.image_config.get("engine", "sdxl")  # 默认使用 sdxl
        self.instantid_config = self.image_config.get("instantid", {})
        self.sdxl_config = self.image_config.get("sdxl", {})
        self.camera_config = self.image_config.get("camera", {})
        self.allow_close_up = self.camera_config.get("allow_close_up", False)

        # 场景参考图像选择配置
        self.scene_reference_config = self.image_config.get(
            "scene_reference", {})
        self.use_scene_reference = self.scene_reference_config.get(
            "enabled", False)
        self.scene_reference_index_path = self.scene_reference_config.get(
            "index_path", "processed/global_index.faiss")
        self.scene_reference_metadata_path = self.scene_reference_config.get(
            "metadata_path", "processed/index_metadata.json")
        self.scene_reference_use_as_img2img = self.scene_reference_config.get(
            "use_as_img2img", True)  # 默认启用
        self.scene_reference_top_k = self.scene_reference_config.get(
            "top_k", 3)
        self.scene_reference_method = self.scene_reference_config.get(
            "method", "hybrid")
        self.scene_reference_weight = self.scene_reference_config.get(
            "weight", 0.6)  # IP-Adapter权重
        self.scene_reference_keyframes_base = self.scene_reference_config.get(
            "keyframes_base", "processed")

        # InstantID 特定配置（无论是instantid还是auto模式，都先初始化）
        # 如果是auto模式，这些属性会在_load_instantid_pipeline时再次检查
        instantid_width = self.instantid_config.get("width", 1536)
        instantid_height = self.instantid_config.get("height", 864)
        self.face_image_path = self.instantid_config.get("face_image_path")
        self.face_cache_enabled = self.instantid_config.get("enable_face_cache", True)
        self.face_emb_scale = float(self.instantid_config.get("face_emb_scale", 0.8))
        self.instantid_width_set = False  # 标记是否已设置InstantID的宽高
        
        if self.engine == "instantid":
            self.width = int(instantid_width)
            self.height = int(instantid_height)
            self.instantid_width_set = True
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
    def load_pipeline(self, engine: Optional[str] = None) -> None:
        """按配置加载图像生成模型。

        支持多种引擎：
        - instantid: InstantID-SDXL-1080P（新方案）
        - sdxl: Stable Diffusion XL（旧方案，兼容）
        - flux-instantid: Flux 1-dev + InstantID（人物生成）
        - hunyuan-dit: Hunyuan-DiT（中文场景）
        - kolors: Kolors（真实感场景）
        - sd3-turbo: SD3 Turbo（批量生成）
        - auto: 自动选择（根据任务类型，延迟加载）
        """
        # 如果指定了引擎，使用指定引擎；否则使用配置中的引擎
        target_engine = engine or self.engine
        
        if target_engine == "auto":
            # auto 模式：延迟加载，根据实际任务选择
            print("ℹ️  使用自动模式，将在生成时根据任务类型选择模型")
            return
        
        if target_engine == "instantid":
            self._load_instantid_pipeline()
        elif target_engine == "sdxl":
            self._load_sdxl_pipeline()
        elif target_engine == "flux-instantid":
            self._load_flux_pipeline()  # Flux.1 + InstantID（人物生成）
        elif target_engine == "flux1":
            self._load_flux1_pipeline()  # Flux.1（实验室/医学场景）
        elif target_engine == "flux2":
            self._load_flux2_pipeline()  # Flux.2（科学背景图、太空/粒子/量子类）
        elif target_engine == "hunyuan-dit":
            self._load_hunyuan_dit_pipeline()
        elif target_engine == "kolors":
            self._load_kolors_pipeline()
        elif target_engine == "realistic-vision":
            self._load_kolors_pipeline()  # 使用相同的加载方法（向后兼容）
        elif target_engine == "sd3-turbo":
            self._load_sd3_turbo_pipeline()
        else:
            raise ValueError(
                f"不支持的图像生成引擎: {target_engine}，"
                f"请使用 'instantid', 'sdxl', 'flux-instantid', 'flux1', 'flux2', "
                f"'hunyuan-dit', 'kolors', 'sd3-turbo' 或 'auto'")

    def _load_instantid_pipeline(self) -> None:
        """加载 InstantID-SDXL-1080P 模型"""
        # 确保 InstantID 相关属性已初始化（即使是 auto 模式）
        if not hasattr(self, 'face_cache_enabled'):
            self.face_cache_enabled = self.instantid_config.get("enable_face_cache", True)
        if not hasattr(self, 'face_emb_scale'):
            self.face_emb_scale = float(self.instantid_config.get("face_emb_scale", 0.8))
        if not hasattr(self, 'face_image_path'):
            self.face_image_path = self.instantid_config.get("face_image_path")
        
        # 如果使用 auto 模式切换到 instantid，也需要更新宽高
        if not hasattr(self, 'instantid_width_set') or not self.instantid_width_set:
            instantid_width = self.instantid_config.get("width", 1536)
            instantid_height = self.instantid_config.get("height", 864)
            self.width = int(instantid_width)
            self.height = int(instantid_height)
            self.instantid_width_set = True
        
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
                    instantid_repo_path = Path(
                        __file__).parent.parent / "InstantID"
                    if instantid_repo_path.exists():
                        import sys
                        if str(instantid_repo_path) not in sys.path:
                            sys.path.insert(0, str(instantid_repo_path))
                        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
                        InstantIDPipeline = StableDiffusionXLInstantIDPipeline
                        print(
                            "✓ 从克隆的 InstantID 仓库导入 StableDiffusionXLInstantIDPipeline")
                    else:
                        raise ImportError("未找到 InstantID 仓库")
                except ImportError as e:
                    # 方式3: 提示安装
                    print("⚠ 无法导入 InstantID")
                    print("  请选择以下方式之一：")
                    print("  1. 安装 instantid 包: pip install instantid")
                    print(
                        "  2. 或从 GitHub 安装: pip install git+https://github.com/instantX-research/InstantID.git")
                    print("  3. 或确保 InstantID 仓库在 ../InstantID 目录下")
                    raise ImportError(f"无法导入 InstantID: {e}") from e

            base_model = self.instantid_config.get(
                "base_model", "Juggernaut-XL-v9-anime")
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
                has_unet = unet_dir.exists() and (list(unet_dir.glob("*.safetensors"))
                                                  or list(unet_dir.glob("*.bin")))
                has_vae = vae_dir.exists() and (list(vae_dir.glob("*.safetensors"))
                                                or list(vae_dir.glob("*.bin")))

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
                config_dir = Path(
                    self.config_path).parent if hasattr(
                    self, 'config_path') else Path.cwd()
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
                        print(
                            f"  尝试不使用 variant 参数（模型可能不支持 {variant} variant）...")
                        # 如果 FP8 失败，自动回退到 FP16
                        if "Float8" in str(e) or "float8" in str(e).lower():
                            print(f"  ⚠ FP8 不支持，自动回退到 FP16")
                            try:
                                self.pipeline = InstantIDPipeline.from_pretrained(
                                    base_model, controlnet=controlnet, torch_dtype=torch.float16, )
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
                    print(
                        f"  提示: 如需保存到 {model_path}，请手动复制或使用 snapshot_download")

            # 加载 IP-Adapter（InstantID 必需）
            ip_adapter_path = self.instantid_config.get("ip_adapter_path")
            if not ip_adapter_path:
                raise ValueError("InstantID 需要 ip_adapter_path 配置")

            ip_adapter_path_obj = Path(ip_adapter_path)
            if not ip_adapter_path_obj.is_absolute():
                config_dir = Path(
                    self.config_path).parent if hasattr(
                    self, 'config_path') else Path.cwd()
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
                        self.pipeline.load_ip_adapter_instantid(
                            str(ip_adapter_file))
                        # 验证 IP-Adapter 是否正确加载
                        if not hasattr(
                                self.pipeline,
                                "image_proj_model_in_features"):
                            # 如果 image_proj_model_in_features 未设置，尝试手动设置
                            if hasattr(
                                    self.pipeline,
                                    "image_proj_model") and self.pipeline.image_proj_model is not None:
                                try:
                                    # InstantID 默认使用 512
                                    self.pipeline.image_proj_model_in_features = 512
                                    print(
                                        "  ℹ 已手动设置 image_proj_model_in_features = 512")
                                except Exception as e:
                                    raise RuntimeError(
                                        f"load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置且无法手动设置: {e}")
                            else:
                                raise RuntimeError(
                                    "load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置，IP-Adapter 可能未正确加载")
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
                                # 如果是文件，pretrained_model_name_or_path_or_dict
                                # 是父目录
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
                            print(
                                f"  ✓ IP-Adapter 加载成功（使用 load_ip_adapter，pretrained_path={pretrained_path}, weight_name={weight_name}）")
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
                            print(
                                f"  ✓ IP-Adapter 加载成功（使用 load_ip_adapter，subfolder={subfolder}, weight_name={weight_name}）")
                        else:
                            # 旧版本，只需要文件路径
                            self.pipeline.load_ip_adapter(str(ip_adapter_file))
                            print("  ✓ IP-Adapter 加载成功（使用 load_ip_adapter，旧版本）")
                    elif hasattr(self.pipeline, "load_ip_adapter_instantid"):
                        # 如果存在 InstantID 特定的方法，使用它
                        self.pipeline.load_ip_adapter_instantid(
                            str(ip_adapter_file))
                        # 验证 IP-Adapter 是否正确加载
                        if not hasattr(
                                self.pipeline,
                                "image_proj_model_in_features"):
                            raise RuntimeError(
                                "load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置，IP-Adapter 可能未正确加载")
                        print("  ✓ IP-Adapter 加载成功（使用 load_ip_adapter_instantid）")
                    else:
                        # InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载
                        # 或者需要通过其他方式加载
                        print(
                            "  ℹ InstantID pipeline 可能已经包含 IP-Adapter，或需要重新创建 pipeline")
                        # 验证 IP-Adapter 是否正确加载
                        if not hasattr(
                                self.pipeline,
                                "image_proj_model_in_features"):
                            print(
                                "  ⚠ 警告：未检测到 image_proj_model_in_features，IP-Adapter 可能未正确加载")
                        else:
                            print(
                                "  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model_in_features）")
                except Exception as e:
                    print(f"  ⚠ IP-Adapter 加载失败: {e}")
                    # 检查是否已经加载了 IP-Adapter
                    # 对于 InstantID，优先检查 image_proj_model_in_features
                    if hasattr(self.pipeline, "image_proj_model_in_features"):
                        print(
                            "  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model_in_features）")
                    elif hasattr(self.pipeline, "image_proj_model"):
                        # 如果有 image_proj_model 但没有
                        # image_proj_model_in_features，说明加载不完整
                        print(
                            "  ⚠ 检测到 image_proj_model 但缺少 image_proj_model_in_features，IP-Adapter 加载不完整")
                        print("  ⚠ 尝试手动设置 image_proj_model_in_features...")
                        try:
                            # 尝试从 image_proj_model 推断 image_emb_dim
                            # InstantID 默认使用 512
                            self.pipeline.image_proj_model_in_features = 512
                            print("  ✓ 已手动设置 image_proj_model_in_features = 512")
                        except Exception as e2:
                            print(
                                f"  ✗ 无法手动设置 image_proj_model_in_features: {e2}")
                    elif hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                        print("  ✓ 检测到 IP-Adapter 已加载（通过 ip_adapter_image_processor）")
                    elif hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
                        # 注意：InstantID 不使用
                        # prepare_ip_adapter_image_embeds，这个检测可能不准确
                        print(
                            "  ⚠ 检测到 prepare_ip_adapter_image_embeds 方法，但这可能不是 InstantID 的 IP-Adapter")
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
            scheduler_name = self.instantid_config.get(
                "scheduler", "EulerDiscreteScheduler")
            if scheduler_name != "EulerDiscreteScheduler":
                try:
                    from diffusers import DPMSolverMultistepScheduler

                    # 获取调度器配置（从当前调度器或默认配置）
                    scheduler_config = self.instantid_config.get(
                        "scheduler_config", {})

                    # 使用 DPM++ 采样器
                    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipeline.scheduler.config,
                        algorithm_type=scheduler_config.get(
                            "algorithm_type",
                            "dpmsolver++"),
                        solver_order=scheduler_config.get(
                            "solver_order",
                            2),
                        lower_order_final=scheduler_config.get(
                            "lower_order_final",
                            True),
                        use_karras_sigmas=scheduler_config.get(
                            "use_karras_sigmas",
                            True),
                    )
                    print(f"  ✓ 已切换采样器: {scheduler_name} (DPM++，质量更好)")
                except ImportError:
                    print(f"  ⚠ 无法导入 DPMSolverMultistepScheduler，使用默认采样器")
                    print(
                        f"  提示: 请确保 diffusers >= 0.21.0: pip install --upgrade diffusers")
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
                config_antelopev2 = self.instantid_config.get(
                    "antelopev2_path")
                if config_antelopev2:
                    antelopev2_path = Path(config_antelopev2)
                    if not antelopev2_path.is_absolute():
                        config_dir = Path(
                            self.config_path).parent if hasattr(
                            self, 'config_path') else Path.cwd()
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
                        name='antelopev2', root=antelopev2_root, providers=[
                            'CUDAExecutionProvider', 'CPUExecutionProvider'])
                else:
                    # 使用默认路径（会从网上下载到 ~/.insightface/models/antelopev2）
                    print(f"  ⚠ 未找到本地 antelopev2 模型，将使用默认路径（可能从网上下载）")
                    print(f"    查找过的路径: {antelopev2_path}")
                    self.face_analysis = FaceAnalysis(
                        name='antelopev2',
                        root='./',
                        providers=[
                            'CUDAExecutionProvider',
                            'CPUExecutionProvider'])

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
                        face_images = sorted(face_image_path.glob(
                            "*.png")) + sorted(face_image_path.glob("*.jpg"))
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
                                self.pipeline.set_face_cache(
                                    str(face_image_path))
                            elif hasattr(self.pipeline, "prepare_face_emb"):
                                # 备用方法：准备面部嵌入
                                from PIL import Image
                                face_img = Image.open(
                                    face_image_path).convert("RGB")
                                self.pipeline.prepare_face_emb(face_img)
                        except Exception as e:
                            print(f"⚠ 设置面部缓存失败: {e}")
                            print("  将继续使用单次面部特征提取")

            self.pipe_name = base_model
            print("✓ InstantID 模型加载成功")
            self._instantid_pipeline_class = InstantIDPipeline
            self._instantid_ip_adapter_ready = False
            try:
                self._load_instantid_ip_adapter_once(force=True)
            except Exception as ip_err:
                print(f"  ⚠ 初始加载 InstantID IP-Adapter 失败: {ip_err}")
                print("  ⚠ 后续将尝试在生成时回退到普通 SDXL pipeline")
            if self.use_lora:
                self._load_lora()

        except ImportError as exc:
            raise RuntimeError(
                "未安装 instantid，请先 `pip install instantid`.") from exc
        except Exception as e:
            raise RuntimeError(f"加载 InstantID 模型失败: {e}") from e

    def _load_sdxl_pipeline(self, load_lora: bool = None) -> None:
        """加载 SDXL 模型（旧方案，兼容）
        
        Args:
            load_lora: 是否加载LoRA。None时使用配置中的默认值，False时强制不加载
        """
        model_path = self.sdxl_config.get(
            "model_path") or self.image_config.get("model_path")
        if not model_path:
            raise ValueError(
                "image.sdxl.model_path 或 image.model_path 未配置，无法加载模型")

        try:
            from huggingface_hub import hf_hub_download  # noqa: F401
            if not hasattr(huggingface_hub, "cached_download"):
                def _cached_download(*args, **kwargs):
                    return hf_hub_download(*args, **kwargs)
                huggingface_hub.cached_download = _cached_download  # type: ignore

            from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

            dtype = torch.float16 if self.image_config.get(
                "mixed_precision", "bf16") == "fp16" else torch.float32

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
                if self.image_config.get(
                        "enable_vae_tiling", True) and hasattr(
                        self.pipeline, "enable_vae_tiling"):
                    self.pipeline.enable_vae_tiling()

            if self.use_img2img and self.reference_images:
                try:
                    # 尝试使用 components 创建 img2img pipeline
                    # 注意：如果使用 enable_model_cpu_offload()，components 可能不是标准字典
                    components = self.pipeline.components
                    # 检查 components 是否为字典或类似字典的对象，并包含必要的键
                    if isinstance(components, dict):
                        required_keys = ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']
                        if all(key in components for key in required_keys):
                            self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**components)
                        else:
                            raise KeyError(f"components missing required keys: {set(required_keys) - set(components.keys())}")
                    else:
                        # components 不是字典，使用 from_pretrained
                        raise TypeError("components is not a dictionary")
                except (KeyError, AttributeError, TypeError) as e:
                    # 如果 components 方法失败，使用 from_pretrained
                    print(f"  ℹ 无法使用 pipeline.components 创建 img2img pipeline: {e}，使用 from_pretrained")
                    self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        model_path, **pipe_kwargs, )

                if self.image_config.get("enable_cpu_offload", True):
                    self.img2img_pipeline.enable_model_cpu_offload()
                else:
                    self.img2img_pipeline = self.img2img_pipeline.to(
                        self.device)
                    if self.image_config.get(
                            "enable_vae_tiling", True) and hasattr(
                            self.img2img_pipeline, "enable_vae_tiling"):
                        self.img2img_pipeline.enable_vae_tiling()

            if self.use_ip_adapter:
                self._load_ip_adapter()

            # 只有在明确指定时才加载LoRA（避免自动加载hanli和anime_style）
            should_load_lora = load_lora if load_lora is not None else self.use_lora
            if should_load_lora:
                self._load_lora()
            else:
                print(f"  ℹ 跳过LoRA加载（load_lora={load_lora}, use_lora={self.use_lora}）")

            self.pipe_name = model_path
            print("✓ SDXL 模型加载成功")
        except ImportError as exc:
            raise RuntimeError(
                "未安装 diffusers，请先 `pip install diffusers`.") from exc

    def _load_sd3_turbo_pipeline(self) -> None:
        """加载 SD3.5 Large Turbo 模型（极速批量生成）"""
        if self.sd3_turbo_pipeline is not None:
            print("ℹ️  SD3.5 Large Turbo pipeline 已加载，跳过重复加载")
            return
        
        try:
            from diffusers import StableDiffusion3Pipeline
            from diffusers.utils import pt_to_pil
            
            model_selection = self.image_config.get("model_selection", {})
            scene_config = model_selection.get("scene", {})
            sd3_config = scene_config.get("sd3_turbo", {})
            
            model_path = sd3_config.get("model_path")
            base_model = sd3_config.get("base_model", "calcuis/sd3.5-large-turbo")
            
            if not model_path:
                model_path = base_model
            
            print(f"加载 SD3.5 Large Turbo 模型: {model_path}")
            
            dtype = torch.float16 if sd3_config.get("quantization", "fp16") == "fp16" else torch.float32
            
            self.sd3_turbo_pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
            
            if self.image_config.get("enable_cpu_offload", False):
                self.sd3_turbo_pipeline.enable_model_cpu_offload()
            else:
                self.sd3_turbo_pipeline = self.sd3_turbo_pipeline.to(self.device)
            
            self.pipe_name = "sd3-turbo"
            print("✓ SD3.5 Large Turbo 模型加载成功")
        except ImportError as exc:
            raise RuntimeError(
                "未安装 diffusers 或 SD3 支持，请先 `pip install diffusers>=0.27.0`.") from exc
        except Exception as e:
            raise RuntimeError(f"加载 SD3 Turbo 模型失败: {e}") from e

    def _load_flux_pipeline(self) -> None:
        """加载 Flux.1 + InstantID pipeline（人物生成，主持人脸）"""
        if self.flux_pipeline is not None:
            print("ℹ️  Flux.1 + InstantID pipeline 已加载，跳过重复加载")
            return
        
        try:
            from diffusers import DiffusionPipeline
            
            model_selection = self.image_config.get("model_selection", {})
            character_config = model_selection.get("character", {})
            
            flux1_model_path = character_config.get("flux1_model_path")
            flux1_base_model = character_config.get("flux1_base_model", "black-forest-labs/FLUX.1-dev")
            
            if not flux1_model_path:
                flux1_model_path = flux1_base_model
            
            print(f"加载 Flux.1 + InstantID 模型: {flux1_model_path}")
            print("⚠️  注意: Flux.1 + InstantID 集成需要额外实现，当前回退到普通 Flux.1")
            
            # 回退到普通 Flux.1 pipeline（不使用 InstantID）
            try:
                from diffusers import DiffusionPipeline
                import torch
                
                print("  ℹ 加载普通 Flux.1 pipeline（不使用 InstantID）...")
                # ⚡ 修复：FluxPipeline 不支持 torch_dtype/dtype 参数，使用默认 dtype
                self.flux_pipeline = DiffusionPipeline.from_pretrained(
                    str(flux1_model_path) if isinstance(flux1_model_path, Path) else flux1_model_path,
                    device_map="balanced"
                )
                # 注意：不手动设置 dtype，让 FluxPipeline 使用其默认 dtype（会自动处理组件间的 dtype 一致性）
                self.pipeline = self.flux_pipeline
                self.pipe_name = "flux1"
                print("  ✅ 普通 Flux.1 pipeline 加载成功")
            except Exception as e:
                print(f"  ❌ 加载 Flux.1 pipeline 失败: {e}")
                self.flux_pipeline = None
                raise RuntimeError(
                    f"无法加载 Flux.1 pipeline。错误: {e}\n"
                    f"建议：使用 'kolors' 引擎生成人物图像，或使用 'sdxl' 引擎"
                ) from e
        except Exception as e:
            raise RuntimeError(f"加载 Flux.1 + InstantID pipeline 失败: {e}") from e

    def _load_flux1_pipeline(self) -> None:
        """加载 Flux.1 pipeline（实验室/医学场景，更干净自然）"""
        if self.flux1_pipeline is not None:
            print("ℹ️  Flux.1 pipeline 已加载，跳过重复加载")
            return
        
        try:
            from diffusers import DiffusionPipeline
            
            model_selection = self.image_config.get("model_selection", {})
            scene_config = model_selection.get("scene", {})
            flux1_config = scene_config.get("flux1", {})
            
            model_path = flux1_config.get("model_path")
            base_model = flux1_config.get("base_model", "black-forest-labs/FLUX.1-dev")
            quantization = flux1_config.get("quantization", "bfloat16")
            device_map = flux1_config.get("device_map", "cuda")
            
            if not model_path:
                model_path = base_model
            
            print(f"加载 Flux.1 模型: {model_path}")
            print(f"  用途: 实验室/医学场景（更干净自然）")
            print(f"  量化类型: {quantization}")
            
            # 确定数据类型
            if quantization == "bfloat16":
                dtype = torch.bfloat16
            elif quantization == "float16" or quantization == "fp16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # ⚡ Flux 性能优化：启用加速选项（根据建议）
            enable_cpu_offload = flux1_config.get("enable_model_cpu_offload", True)
            enable_attention_slicing = flux1_config.get("enable_attention_slicing", True)
            
            # 注意：device_map 和 enable_model_cpu_offload 不能同时使用
            # 如果 enable_cpu_offload 为 True，就不使用 device_map
            if enable_cpu_offload:
                # 不使用 device_map，稍后使用 enable_model_cpu_offload
                load_device_map = None
            else:
                # 使用 device_map
                load_device_map = device_map
            
            # 加载 pipeline
            # ⚡ 修复：FluxPipeline 不支持 dtype 参数，使用默认 dtype（它会自动处理组件间的 dtype 一致性）
            load_kwargs = {}
            if load_device_map is not None:
                load_kwargs["device_map"] = load_device_map
            
            self.flux1_pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                **load_kwargs
            )
            # 注意：不手动设置 dtype，因为 FluxPipeline 会自动处理组件间的 dtype 一致性
            # 手动设置可能导致组件间 dtype 不匹配（如 unet 是 bfloat16 但 text_encoder 是 float32）
            
            # 如果未使用 device_map，可以启用 CPU offload
            if enable_cpu_offload and load_device_map is None:
                if hasattr(self.flux1_pipeline, "enable_model_cpu_offload"):
                    self.flux1_pipeline.enable_model_cpu_offload()
                    print("  ⚡ 已启用 CPU offload（减少显存占用）")
            elif load_device_map is not None:
                print(f"  ℹ 使用 device_map={load_device_map}（已自动管理设备分配）")
            
            if enable_attention_slicing and hasattr(self.flux1_pipeline, "enable_attention_slicing"):
                self.flux1_pipeline.enable_attention_slicing()
                print("  ⚡ 已启用 Attention slicing（加速推理）")
            
            self.pipe_name = "flux1"
            print("✓ Flux.1 pipeline 加载成功（已优化）")
        except Exception as e:
            raise RuntimeError(f"加载 Flux.1 pipeline 失败: {e}") from e

    def _load_flux2_pipeline(self) -> None:
        """加载 Flux.2 pipeline（科学背景图、太空/粒子/量子类，冲击力强）"""
        if self.flux2_pipeline is not None:
            print("ℹ️  Flux.2 pipeline 已加载，跳过重复加载")
            return
        
        try:
            from diffusers import DiffusionPipeline
            
            model_selection = self.image_config.get("model_selection", {})
            scene_config = model_selection.get("scene", {})
            flux2_config = scene_config.get("flux2", {})
            
            model_path = flux2_config.get("model_path")
            # 注意：Flux.2 的实际模型ID需要确认，可能是 FLUX.1-schnell 或其他
            base_model = flux2_config.get("base_model", "black-forest-labs/FLUX.1-schnell")
            quantization = flux2_config.get("quantization", "bfloat16")
            device_map = flux2_config.get("device_map", "cuda")
            
            if not model_path:
                model_path = base_model
            
            print(f"加载 Flux.2 模型: {model_path}")
            print(f"  用途: 科学背景图、太空/粒子/量子类（冲击力强）")
            print(f"  量化类型: {quantization}")
            print(f"⚠️  注意: Flux.2 模型ID可能需要确认，当前使用: {base_model}")
            
            # 确定数据类型
            if quantization == "bfloat16":
                dtype = torch.bfloat16
            elif quantization == "float16" or quantization == "fp16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # ⚡ Flux 性能优化：启用加速选项（根据建议）
            enable_cpu_offload = flux2_config.get("enable_model_cpu_offload", True)
            enable_attention_slicing = flux2_config.get("enable_attention_slicing", True)
            
            # 注意：device_map 和 enable_model_cpu_offload 不能同时使用
            # 如果 enable_cpu_offload 为 True，就不使用 device_map
            if enable_cpu_offload:
                # 不使用 device_map，稍后使用 enable_model_cpu_offload
                load_device_map = None
            else:
                # 使用 device_map
                load_device_map = device_map
            
            # 加载 pipeline
            # ⚡ 修复：FluxPipeline 不支持 dtype 参数，使用默认 dtype（它会自动处理组件间的 dtype 一致性）
            load_kwargs = {}
            if load_device_map is not None:
                load_kwargs["device_map"] = load_device_map
            
            self.flux2_pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                **load_kwargs
            )
            # 注意：不手动设置 dtype，因为 FluxPipeline 会自动处理组件间的 dtype 一致性
            # 手动设置可能导致组件间 dtype 不匹配（如 unet 是 bfloat16 但 text_encoder 是 float32）
            
            # 如果未使用 device_map，可以启用 CPU offload
            if enable_cpu_offload and load_device_map is None:
                if hasattr(self.flux2_pipeline, "enable_model_cpu_offload"):
                    self.flux2_pipeline.enable_model_cpu_offload()
                    print("  ⚡ 已启用 CPU offload（减少显存占用）")
            elif load_device_map is not None:
                print(f"  ℹ 使用 device_map={load_device_map}（已自动管理设备分配）")
            
            if enable_attention_slicing and hasattr(self.flux2_pipeline, "enable_attention_slicing"):
                self.flux2_pipeline.enable_attention_slicing()
                print("  ⚡ 已启用 Attention slicing（加速推理）")
            
            self.pipe_name = "flux2"
            print("✓ Flux.2 pipeline 加载成功（已优化）")
        except Exception as e:
            raise RuntimeError(f"加载 Flux.2 pipeline 失败: {e}") from e

    def _load_hunyuan_dit_pipeline(self) -> None:
        """加载 Hunyuan-DiT pipeline（中文场景）"""
        if self.hunyuan_dit_pipeline is not None:
            print("ℹ️  Hunyuan-DiT pipeline 已加载，跳过重复加载")
            return
        
        try:
            model_selection = self.image_config.get("model_selection", {})
            scene_config = model_selection.get("scene", {})
            hunyuan_config = scene_config.get("hunyuan_dit", {})
            
            model_path = hunyuan_config.get("model_path")
            base_model = hunyuan_config.get("base_model", "Tencent-Hunyuan/HunyuanDiT")
            
            if not model_path:
                model_path = base_model
            
            print(f"加载 Hunyuan-DiT 模型: {model_path}")
            print("⚠️  注意: Hunyuan-DiT 集成需要额外实现，当前为占位实现")
            
            # TODO: 实现 Hunyuan-DiT 的完整集成
            # Hunyuan-DiT 可能需要特殊的加载方式，请参考官方文档
            
            self.hunyuan_dit_pipeline = None  # 占位
            self.pipe_name = "hunyuan-dit"
            print("⚠️  Hunyuan-DiT pipeline 加载未完成，需要实现完整集成")
        except Exception as e:
            raise RuntimeError(f"加载 Hunyuan-DiT pipeline 失败: {e}") from e

    def _load_kolors_pipeline(self) -> None:
        """加载 Kolors pipeline（真实感场景，快手可图团队开发）
        
        使用 Kolors-IP-Adapter-FaceID-Plus 版本，可直接用 diffusers 加载
        
        Kolors 特点：
        - 真人质感强
        - 肤色真实
        - 五官清晰
        - 光影自然
        - 色彩稳定，不会脏
        - 中文 prompt 理解优秀
        """
        if self.kolors_pipeline is not None:
            print("ℹ️  Kolors pipeline 已加载，跳过重复加载")
            return
        
        try:
            from diffusers import DiffusionPipeline
            
            model_selection = self.image_config.get("model_selection", {})
            scene_config = model_selection.get("scene", {})
            kolors_config = scene_config.get("kolors", {})
            
            model_path = kolors_config.get("model_path")
            base_model = kolors_config.get("base_model", "Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus")
            quantization = kolors_config.get("quantization", "bfloat16")
            device_map = kolors_config.get("device_map", "cuda")
            
            if not model_path:
                model_path = base_model
            
            print(f"加载 Kolors 模型: {model_path}")
            print(f"  量化类型: {quantization}")
            print(f"  设备映射: {device_map}")
            
            # 确定数据类型
            if quantization == "bfloat16":
                dtype = torch.bfloat16
            elif quantization == "float16" or quantization == "fp16":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # 加载 pipeline（使用官方推荐的方式）
            # 注意: device_map 可能不支持 "cuda"，使用 "balanced" 或直接 to(device)
            try:
                self.kolors_pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    dtype=dtype,
                    device_map="balanced" if device_map == "cuda" else device_map
                )
            except Exception as e:
                # 如果 device_map 失败，尝试不使用 device_map
                if "device_map" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"  ⚠️  device_map 失败，尝试直接加载到设备: {e}")
                    self.kolors_pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        dtype=dtype
                    )
                    # 手动移动到设备
                    if torch.cuda.is_available():
                        self.kolors_pipeline = self.kolors_pipeline.to(self.device)
                else:
                    raise
            
            # 确保 pipeline 在正确的设备上
            if torch.cuda.is_available() and hasattr(self.kolors_pipeline, "to"):
                try:
                    self.kolors_pipeline = self.kolors_pipeline.to(self.device)
                except:
                    # 如果 to() 失败，尝试 enable_model_cpu_offload
                    if hasattr(self.kolors_pipeline, "enable_model_cpu_offload"):
                        self.kolors_pipeline.enable_model_cpu_offload()
            
            self.pipe_name = "kolors"
            print("✓ Kolors pipeline 加载成功")
        except ImportError as e:
            raise RuntimeError(
                f"加载 Kolors pipeline 失败: 缺少依赖。请运行: pip install -U diffusers transformers accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"加载 Kolors pipeline 失败: {e}") from e

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

        face_images = sorted(dir_path.glob("*.png")) + \
            sorted(dir_path.glob("*.jpg"))
        return face_images

    def _select_reference_image(
            self, scene: Dict[str, Any], index: int) -> Optional[Path]:
        """选择参考图像。

        注意：不使用场景中已有的 image_path（可能是输入目录中的旧图），
        只从配置的 reference_image_dir 中选择，确保使用高质量的参考图。
        """
        if not (self.use_img2img or self.use_ip_adapter):
            return None

        # 不从场景的 image_path 选择（可能是输入目录中的旧图，质量不好）
        # 只从配置的 reference_image_dir 中选择
        if self.reference_images:
            return self.reference_images[(
                index - 1) % len(self.reference_images)]
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

    def _select_face_reference_image(
            self,
            index: int,
            character_id: Optional[str] = None) -> Optional[Path]:
        """选择面部参考图像

        Args:
            index: 场景索引（用于循环选择）
            character_id: 角色ID（如果提供，优先使用对应的参考图像）
        """
        # ⚡ 关键修复：韩立角色统一使用 reference_image/hanli_mid.jpg
        if character_id == "hanli":
            # 优先级 1：配置中的 face_image_path
            if hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                ref_path = Path(self.face_image_path)
                print(f"  ✓ 韩立角色：使用配置中的参考图: {ref_path.name}")
                return ref_path
            # 优先级 2：reference_image/hanli_mid.jpg
            base_paths = [
                Path(__file__).parent / "reference_image",
                Path(__file__).parent.parent / "reference_image",
                Path.cwd() / "reference_image",
            ]
            for base in base_paths:
                hanli_mid_path = base / "hanli_mid.jpg"
                if hanli_mid_path.exists():
                    print(f"  ✓ 韩立角色：使用统一参考图: {hanli_mid_path.name}")
                    return hanli_mid_path
            # 如果找不到，尝试 .png
            for base in base_paths:
                hanli_mid_path = base / "hanli_mid.png"
                if hanli_mid_path.exists():
                    print(f"  ✓ 韩立角色：使用统一参考图: {hanli_mid_path.name}")
                    return hanli_mid_path
            print(f"  ⚠ 警告：未找到韩立参考图 hanli_mid.jpg，将尝试使用 character_references")
        
        # 如果提供了角色ID，优先使用对应的参考图像（非韩立角色）
        if character_id and character_id in self.character_reference_images:
            ref_path = self.character_reference_images[character_id]
            print(f"  ✓ 使用角色参考图像: {character_id} -> {ref_path.name}")
            return ref_path

        # 如果没有找到对应的参考图像，使用默认的循环选择方式
        if not self.face_reference_images:
            return None
        return self.face_reference_images[(
            index - 1) % len(self.face_reference_images)]

    def _select_scene_reference_images(
            self, scene: Dict[str, Any]) -> List[Path]:
        """
        根据场景描述，从processed目录中选择最相关的参考图像

        Returns:
            参考图像路径列表，按相似度排序
        """
        if not self.use_scene_reference:
            return []

        try:
            # 动态导入场景参考图像选择器
            import sys
            tools_path = Path(__file__).parent / "tools"
            if str(tools_path) not in sys.path:
                sys.path.insert(0, str(tools_path))

            from select_scene_reference_images import select_reference_images
            from glob import glob

            # 查找所有scene_metadata.json文件
            scene_metadata_pattern = str(
                Path(
                    self.scene_reference_keyframes_base) /
                "episode_*/scene_metadata.json")
            scene_metadata_files = [Path(f)
                                    for f in glob(scene_metadata_pattern)]

            if not scene_metadata_files:
                print(f"  ⚠ 未找到场景metadata文件，跳过场景参考图像选择")
                return []

            # 选择参考图像
            results = select_reference_images(
                scene,
                Path(self.scene_reference_index_path),
                Path(self.scene_reference_metadata_path),
                scene_metadata_files,
                Path(self.scene_reference_keyframes_base),
                top_k=self.scene_reference_top_k,
                method=self.scene_reference_method
            )

            if results:
                reference_paths = [
                    Path(path) for path,
                    score,
                    _ in results if Path(path).exists()]
                if reference_paths:
                    print(f"  ✓ 找到 {len(reference_paths)} 个场景参考图像:")
                    for i, path in enumerate(reference_paths, 1):
                        print(f"    {i}. {path.name}")
                    return reference_paths

            return []
        except Exception as e:
            print(f"  ⚠ 场景参考图像选择失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _load_ip_adapter(self) -> None:
        if self.pipeline is None:
            # 自动加载pipeline（如果未加载）
            print("⚠️  Pipeline未加载，正在自动加载...")
            self.load_pipeline()

        model_path = self.ip_adapter_config.get("model_path")
        if not model_path:
            raise ValueError(
                "image.ip_adapter.model_path 未配置，无法加载 IP-Adapter 权重")

        weight_name = self.ip_adapter_weight_names
        subfolder = self.ip_adapter_config.get("subfolder", "sdxl_models")

        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"未找到 IP-Adapter 权重目录: {path_obj}")

        def _load_into(pipe: Any) -> None:
            # ⚡ 关键修复：在加载 IP-Adapter 之前，验证 pipeline 的 unet 组件
            print(f"  🔍 [DEBUG] 准备加载 IP-Adapter，pipeline 类型: {type(pipe).__name__}")
            try:
                # 验证 pipeline 是否有 unet 属性
                if not hasattr(pipe, 'unet'):
                    raise AttributeError(f"Pipeline {type(pipe).__name__} 缺少 'unet' 属性")
                # 尝试访问 unet（如果使用 CPU offload，可能需要触发加载）
                unet = pipe.unet
                if unet is None:
                    raise AttributeError(f"Pipeline {type(pipe).__name__}.unet 为 None")
                print(f"  ✓ Pipeline unet 验证成功: {type(unet).__name__}")
            except (AttributeError, KeyError) as unet_error:
                print(f"  ⚠ Pipeline unet 验证失败: {unet_error}")
                # 尝试通过 components 检查
                if hasattr(pipe, 'components'):
                    try:
                        components = pipe.components
                        if isinstance(components, dict):
                            print(f"  🔍 [DEBUG] components 键: {list(components.keys())[:10]}")
                            if 'unet' not in components:
                                raise KeyError(f"Pipeline.components 字典中缺少 'unet' 键，现有键: {list(components.keys())[:10]}")
                        else:
                            raise TypeError(f"Pipeline.components 不是字典，类型: {type(components)}")
                    except Exception as comp_error:
                        print(f"  ⚠ Pipeline.components 检查失败: {comp_error}")
                        import traceback
                        print(f"  📋 完整错误堆栈:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Pipeline 组件验证失败，无法加载 IP-Adapter: {unet_error}, components错误: {comp_error}") from unet_error
                else:
                    import traceback
                    print(f"  📋 完整错误堆栈:\n{traceback.format_exc()}")
                    raise RuntimeError(f"Pipeline 缺少 'unet' 属性且没有 'components' 属性: {unet_error}") from unet_error
            
            # 检查 IP-Adapter 是否已经加载
            ip_adapter_already_loaded = False
            if hasattr(
                    pipe,
                    "ip_adapter_image_processor") and pipe.ip_adapter_image_processor is not None:
                ip_adapter_already_loaded = True
                print(f"  ℹ IP-Adapter 已加载，跳过重新加载")
            elif hasattr(pipe, "prepare_ip_adapter_image_embeds"):
                # 尝试检查是否已加载（通过检查是否有 image_projection_layers）
                try:
                    # 如果 IP-Adapter 已加载，prepare_ip_adapter_image_embeds 应该可以调用
                    # 但这里只是检查，不实际调用
                    ip_adapter_already_loaded = False  # 保守起见，假设未加载
                except BaseException:
                    ip_adapter_already_loaded = False

            if not ip_adapter_already_loaded:
                try:
                    # 每次加载前都先尝试卸载旧的 IP-Adapter（如果存在）
                    # 这样可以确保每次都能成功加载新的 IP-Adapter
                    if hasattr(pipe, "disable_ip_adapter"):
                        try:
                            pipe.disable_ip_adapter()
                            print(f"  ℹ 已卸载旧的 IP-Adapter（如果存在）")
                        except BaseException:
                            pass
                    elif hasattr(pipe, "unload_ip_adapter"):
                        try:
                            pipe.unload_ip_adapter()
                            print(f"  ℹ 已卸载旧的 IP-Adapter（如果存在）")
                        except BaseException:
                            pass

                    # 加载新的 IP-Adapter
                    print(f"  🔍 [DEBUG] 准备调用 pipe.load_ip_adapter，pipe类型: {type(pipe).__name__}")
                    pipe.load_ip_adapter(
                        str(path_obj),
                        subfolder=subfolder,
                        weight_name=weight_name,
                    )
                    print(f"  ✓ IP-Adapter 加载成功")
                except AttributeError as exc:
                    raise RuntimeError(
                        "当前 diffusers 版本不支持 IP-Adapter，请升级到 0.21.0 及以上版本。") from exc
                except (KeyError, RuntimeError) as e:
                    # ⚡ 关键修复：捕获 KeyError 'unet' 和其他运行时错误
                    error_str = str(e)
                    if isinstance(e, KeyError) and 'unet' in error_str:
                        import traceback
                        print(f"  ✗ IP-Adapter 加载失败（KeyError 'unet'）: {e}")
                        print(f"  📋 完整错误堆栈:\n{traceback.format_exc()}")
                        raise RuntimeError(f"IP-Adapter 加载失败：pipeline.components 字典中缺少 'unet' 键。这可能是因为使用了 CPU offload。请尝试禁用 CPU offload 或重新加载 pipeline。") from e
                    raise
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
        
        # ⚡ 关键修复：方案2需要使用 sdxl_pipeline，确保 IP-Adapter 也加载到它上面
        if self.sdxl_pipeline is not None and self.sdxl_pipeline is not self.pipeline and self.sdxl_pipeline is not self.img2img_pipeline:
            _load_into(self.sdxl_pipeline)

    def _load_lora(self) -> None:
        if self.pipeline is None:
            # 自动加载pipeline（如果未加载）
            print("⚠️  Pipeline未加载，正在自动加载...")
            self.load_pipeline()

        adapters: List[Tuple[str, float]] = []

        def _load_single(config: Dict[str, Any], default_name: str,
                         default_alpha: float) -> Optional[Tuple[str, float]]:
            enabled = config.get("enabled", True)
            if not enabled:
                return None

            weights_path = config.get("weights_path")
            if not weights_path:
                return None

            path_obj = Path(weights_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"未找到 LoRA 权重文件: {path_obj}")

            adapter_name = config.get("adapter_name") or default_name
            alpha = float(config.get("alpha", default_alpha))
            load_kwargs: Dict[str, Any] = {}

            if bool(config.get("force_cpu", False)):
                load_kwargs["device"] = torch.device("cpu")

            def _load_into(pipe: Any) -> None:
                pipe.load_lora_weights(
                    str(path_obj),
                    adapter_name=adapter_name,
                    **load_kwargs,
                )

                print(
                    f"  加载 LoRA 权重: {path_obj} (adapter={adapter_name}, alpha={alpha})")
            _load_into(self.pipeline)
            if self.img2img_pipeline is not None:
                _load_into(self.img2img_pipeline)
            
            return (adapter_name, alpha)

            return adapter_name, alpha

        base_entry = _load_single(
            self.lora_config,
            self.lora_adapter_name or "default",
            self.lora_alpha,
        )
        if base_entry:
            adapters.append(base_entry)

        style_config = self.lora_config.get("style_lora")
        if isinstance(style_config, dict):
            style_entry = _load_single(
                style_config,
                style_config.get("adapter_name", "style"),
                float(style_config.get("alpha", 0.7)),
            )
            if style_entry:
                adapters.append(style_entry)

        def _apply_adapters(pipe: Any) -> None:
            if hasattr(pipe, "set_adapters") and adapters:
                names = [name for name, _ in adapters]
                weights = [alpha for _, alpha in adapters]
                pipe.set_adapters(names, adapter_weights=weights)

        _apply_adapters(self.pipeline)
        if self.img2img_pipeline is not None:
            _apply_adapters(self.img2img_pipeline)

    def _unload_all_lora_adapters(self, pipeline: Any, pipeline_name: str = "pipeline") -> None:
        """卸载pipeline中所有已加载的LoRA适配器"""
        try:
            # 方法1: 使用disable_adapters（如果可用）
            if hasattr(pipeline, "disable_adapters"):
                pipeline.disable_adapters()
                print(f"  ✓ [{pipeline_name}] 已禁用所有LoRA适配器")
            
            # 方法2: 使用unload_lora_weights卸载每个适配器
            if hasattr(pipeline, "unload_lora_weights"):
                # 获取所有已加载的适配器
                loaded_adapters = []
                if hasattr(pipeline, "peft_config") and pipeline.peft_config:
                    loaded_adapters = list(pipeline.peft_config.keys())
                elif hasattr(pipeline, "get_active_adapters"):
                    try:
                        active = pipeline.get_active_adapters()
                        if active:
                            loaded_adapters = list(active) if isinstance(active, (list, tuple)) else [active]
                    except Exception:
                        pass
                
                # 如果无法自动获取，尝试卸载已知的适配器（hanli和anime_style）
                if not loaded_adapters:
                    # 尝试卸载可能已加载的适配器
                    known_adapters = ["hanli", "anime_style"]
                    if self.lora_adapter_name:
                        known_adapters.append(self.lora_adapter_name)
                    if isinstance(self.lora_config.get("style_lora"), dict):
                        style_adapter = self.lora_config.get("style_lora", {}).get("adapter_name")
                        if style_adapter:
                            known_adapters.append(style_adapter)
                    
                    for adapter_name in known_adapters:
                        if adapter_name:
                            try:
                                pipeline.unload_lora_weights(adapter_name)
                                print(f"  ✓ [{pipeline_name}] 已卸载LoRA适配器: {adapter_name}")
                            except Exception:
                                pass  # 如果适配器未加载，忽略错误
                else:
                    # 卸载所有已加载的适配器
                    for adapter_name in loaded_adapters:
                        try:
                            pipeline.unload_lora_weights(adapter_name)
                            print(f"  ✓ [{pipeline_name}] 已卸载LoRA适配器: {adapter_name}")
                        except Exception as e:
                            print(f"  ⚠ [{pipeline_name}] 卸载LoRA适配器 {adapter_name} 失败: {e}")
            
            # 方法3: 如果pipeline支持fuse_lora，可能需要先unfuse
            if hasattr(pipeline, "unfuse_lora"):
                try:
                    pipeline.unfuse_lora()
                    print(f"  ✓ [{pipeline_name}] 已unfuse LoRA")
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"  ⚠ [{pipeline_name}] 卸载LoRA适配器时出错: {e}，但已禁用适配器")

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------
    def _identify_characters_in_scene(
            self, scene: Dict[str, Any]) -> List[str]:
        """识别场景中的所有角色（不仅仅是韩立）。

        Returns:
            角色ID列表，例如 ['hanli', 'huangliang_lingjun', 'huan_cangqi']
        """
        identified_characters = []

        # ⚡ v2 格式支持：优先检查 character.id 字段
        character = scene.get("character", {}) or {}
        if isinstance(character, dict):
            character_id = character.get("id", "")
            if character_id:
                char_id_lower = character_id.lower()
                if char_id_lower not in identified_characters:
                    identified_characters.append(char_id_lower)
                    print(f"  ✓ v2 格式：从 character.id 识别到角色: {character_id}")

        # 获取场景文本
        text_parts = [
            scene.get("title", ""),
            scene.get("description", ""),
            scene.get("prompt", ""),
            scene.get("narration", ""),
        ]
        combined_text = " ".join(str(p) for p in text_parts).lower()

        # 检查 visual 中的相关字段（composition、character_pose等）
        visual = scene.get("visual", {}) or {}
        if isinstance(visual, dict):
            # 添加 composition（通常包含角色信息，如 "Han Li recalls..."）
            composition = str(visual.get("composition", "")).lower()
            if composition:
                combined_text += " " + composition
            # 添加 character_pose
            character_pose = str(visual.get("character_pose", "")).lower()
            if character_pose:
                combined_text += " " + character_pose

        # 角色关键词映射（中文和英文）
        character_keywords = {
            "hanli": ["韩立", "han li", "hanli", "主角", "main character", "hero"],
            "huangliang_lingjun": ["黄粱灵君", "huangliang", "huangliang spirit lord", "黄粱"],
            "huan_cangqi": ["寰姓少年", "寰天奇", "huan cangqi", "huan tianqi", "huan youth", "寰"],
            "dumu_juren": ["独目巨人", "one-eyed giant", "giant", "巨人"],
            # 科普主持人
            "kepu_gege": ["科普哥哥", "kepu gege", "kepu_gege", "科普", "science presenter", "host"],
            "weilai_jiejie": ["未来姐姐", "weilai jiejie", "weilai_jiejie", "未来", "science presenter", "host"],
        }
        
        # 检查 characters 字段（v1 格式，优先）
        characters = scene.get("characters", [])
        if characters:
            for char in characters:
                if isinstance(char, str):
                    char_name = char.lower()
                else:
                    char_name = str(char.get("name", "")).lower()
                
                # 直接匹配角色ID
                if char_name in ["kepu_gege", "科普哥哥", "kepu gege"]:
                    if "kepu_gege" not in identified_characters:
                        identified_characters.append("kepu_gege")
                elif char_name in ["weilai_jiejie", "未来姐姐", "weilai jiejie"]:
                    if "weilai_jiejie" not in identified_characters:
                        identified_characters.append("weilai_jiejie")

        # 检查每个角色
        for char_id, keywords in character_keywords.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    if char_id not in identified_characters:
                        identified_characters.append(char_id)
                    break

        return identified_characters

    def _needs_character(self, scene: Dict[str, Any]) -> bool:
        """判断场景是否需要主角（韩立）或主持人（科普哥哥/未来姐姐）。

        检查场景描述、标题、提示词中是否包含主角或主持人相关关键词。
        同时识别纯环境场景（只有环境，不需要人物）。
        """
        # ⚡ v2 格式支持：优先检查 character.present 字段
        character = scene.get("character", {}) or {}
        if isinstance(character, dict):
            character_present = character.get("present", None)
            if character_present is True:
                print(f"  ✓ v2 格式：character.present=true，需要角色")
                return True
            elif character_present is False:
                print(f"  ✓ v2 格式：character.present=false，不需要角色")
                return False
        
        # 首先检查是否有明确的"纯环境"标记
        if scene.get("environment_only") is True:
            return False

        # 检查 visual.composition 是否明确表示只有环境
        visual = scene.get("visual") or {}
        if isinstance(visual, dict):
            composition = str(visual.get("composition", "")).lower()
            # 如果 composition 中明确表示只有环境/物体，没有人物
            env_only_keywords = [
                "only environment",
                "environment only",
                "no character",
                "no person",
                "pure environment",
                "landscape only",
                "scene only",
                "只有环境",
                "纯环境",
                "无人物",
                "仅环境"]
            if any(kw in composition for kw in env_only_keywords):
                return False

        keywords = [
            "韩立",
            "han li",
            "主角",
            "hero",
            "cultivator",
            "main character",
            # 科普视频主持人关键词
            "科普哥哥",
            "未来姐姐",
            "kepu gege",
            "weilai jiejie",
            "主持人",
            "host",
            "presenter",
            "science presenter"]
        text_parts = [
            scene.get("title", ""),
            scene.get("description", ""),
            scene.get("prompt", ""),
            scene.get("action", ""),
            scene.get("narration", ""),  # 科普视频的旁白中可能包含主持人信息
        ]
        combined_text = " ".join(str(p) for p in text_parts).lower()

        # 检查是否有主角或主持人关键词
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
                # 支持字符串和字典两种格式
                if isinstance(char, str):
                    char_name = char.lower()
                else:
                    char_name = str(char.get("name", "")).lower()
                
                # 检查是否是主角或主持人
                if any(kw in char_name for kw in ["han", "韩立", "主角", "kepu", "weilai", "科普", "未来", "gege", "jiejie", "主持人", "host", "presenter"]):
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
            "出现",
            "显现",
            "露出",
            "透出",
            "发出",
            "传来",
            "形成",
            "扩大",
            "下坠",
            "appears",
            "revealed",
            "leaks",
            "emerges",
            "forms",
            "expands",
            "collapses"]

        # 人物动作关键词（表示有人物）
        character_action_keywords = [
            "踏入",
            "停下",
            "站起",
            "望向",
            "半蹲",
            "跃下",
            "检查",
            "靠近",
            "准备",
            "walking",
            "stands",
            "gazes",
            "kneels",
            "jumps",
            "examines",
            "approaches",
            "prepares",
            "han li",
            "character",
            "person",
            "figure",
            "cultivator"]

        # 如果描述和 prompt 中有环境指示词，但没有人物动作关键词
        has_env_indicator = any(
            ind in description or ind in prompt for ind in env_only_indicators)
        has_character_action = any(
            kw in description or kw in prompt or kw in action for kw in character_action_keywords)

        # 如果只有环境指示词，没有人物动作，且 action 是观察类或为空
        if has_env_indicator and not has_character_action:
            if not action or action.startswith(
                    "observe") or "observe" in action:
                print(f"  ✓ 检测到纯环境场景（无人物），跳过角色生成")
                return False

        return False

    def _get_camera_string(self, scene: Dict[str, Any]) -> str:
        """安全地获取 camera 字段的字符串表示（支持 v1 字符串和 v2 字典格式）"""
        if not scene:
            return ""
        
        camera = scene.get("camera", "")
        if not camera:
            return ""
        
        # 如果是字典（v2 格式），转换为字符串
        if isinstance(camera, dict):
            parts = []
            
            # shot 字段映射
            shot_map = {
                "wide": "远景",
                "medium": "中景",
                "close_up": "特写",
                "closeup": "特写",
                "extreme_close": "极近特写",
                "full_body": "全身",
                "long": "长镜头",
            }
            shot = camera.get("shot", "")
            if shot:
                shot_str = shot_map.get(shot.lower(), shot)
                parts.append(shot_str)
            
            # angle 字段映射
            angle_map = {
                "eye_level": "平视",
                "top_down": "俯拍",
                "bird_eye": "鸟瞰",
                "low_angle": "仰拍",
                "worm_eye": "极低角度",
                "side": "侧拍",
                "front": "正面",
                "back": "背后",
            }
            angle = camera.get("angle", "")
            if angle:
                angle_str = angle_map.get(angle.lower(), angle)
                parts.append(angle_str)
            
            # movement 字段映射
            movement_map = {
                "static": "静止",
                "pan": "横移",
                "tilt": "上下摇",
                "push_in": "推近",
                "pull_out": "拉远",
                "orbit": "环绕",
                "follow": "跟随",
                "shake": "抖动",
            }
            movement = camera.get("movement", "")
            if movement:
                movement_str = movement_map.get(movement.lower(), movement)
                parts.append(movement_str)
            
            return " ".join(parts) if parts else ""
        
        # 如果是字符串（v1 格式），直接返回
        return str(camera) if camera else ""

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
    # 如需查看实现，请参考 prompt/builder.py, prompt/parser.py, prompt/optimizer.py,
    # prompt/token_estimator.py

    def build_prompt(self,
                     scene: Dict[str,
                                 Any],
                     include_character: Optional[bool] = None,
                     script_data: Dict[str,
                                       Any] = None,
                     previous_scene: Optional[Dict[str,
                                                   Any]] = None,
                     use_semantic_prompt: Optional[bool] = None) -> str:  # ⚡ 新增：是否使用语义化 prompt（FLUX 专用）
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
            previous_scene=previous_scene,
            use_semantic_prompt=use_semantic_prompt  # ⚡ 新增：传递语义化 prompt 标志
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
        character_lora: Optional[str] = None,  # 角色LoRA适配器名称，None时使用配置中的默认值
        style_lora: Optional[str] = None,  # 风格LoRA适配器名称，None时使用配置中的默认值，空字符串表示禁用
        scene: Optional[Dict[str, Any]] = None,
        init_image: Optional[Path] = None,  # 用于场景连贯性的前一个场景图像
        model_engine: Optional[str] = None,  # 手动指定模型引擎（可选）
        task_type: Optional[str] = None,  # 任务类型（character/scene/batch）
    ) -> Path:
        """调用 pipeline 生成单张图像

        Args:
            use_lora: 是否使用 LoRA。None 时使用配置中的默认值（仅 SDXL）
            character_lora: 角色LoRA适配器名称，None时使用配置中的默认值
            style_lora: 风格LoRA适配器名称，None时使用配置中的默认值，空字符串表示禁用
            model_engine: 手动指定模型引擎（可选），如 "flux-instantid", "hunyuan-dit", "kolors", "sd3-turbo", "auto"
            task_type: 任务类型（可选），如 "character", "scene", "batch"
        """
        
        # ⚡ 增强模式：如果启用且提供了 scene 参数，使用增强生成器
        if self.use_enhanced_mode and scene is not None:
            try:
                # 延迟加载增强生成器
                if self.enhanced_generator is None:
                    from enhanced_image_generator import EnhancedImageGenerator
                    print("  🚀 初始化增强模式生成器...")
                    import time
                    start_time = time.time()
                    try:
                        self.enhanced_generator = EnhancedImageGenerator(str(self.config_path))
                        elapsed = time.time() - start_time
                        print(f"  ✓ 增强模式生成器初始化完成 (耗时: {elapsed:.2f}秒)")
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"  ❌ 增强模式生成器初始化失败 (耗时: {elapsed:.2f}秒): {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # 准备参考图像
                face_ref = None
                if face_reference_image_path:
                    from PIL import Image
                    face_ref = Image.open(face_reference_image_path).convert('RGB')
                elif reference_image_path:
                    from PIL import Image
                    face_ref = Image.open(reference_image_path).convert('RGB')
                
                # 使用增强生成器生成
                print("  ✨ 使用增强模式生成（PuLID + 解耦融合，v2.2-final模式）")
                print("  [调试] 准备调用 generate_scene...")
                import time
                call_start = time.time()
                try:
                    image = self.enhanced_generator.generate_scene(
                        scene=scene,
                        face_reference=face_ref,
                        original_prompt=prompt  # ⚡ 传递优化后的 prompt，确保包含完整信息（场景、性别、服饰等）
                    )
                    call_elapsed = time.time() - call_start
                    print(f"  [调试] generate_scene 调用完成 (耗时: {call_elapsed:.2f}秒)")
                except Exception as e:
                    call_elapsed = time.time() - call_start
                    print(f"  [调试] generate_scene 调用失败 (耗时: {call_elapsed:.2f}秒): {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                if image:
                    # ⚡ 关键修复：处理 Result 对象（如果返回的是 Result，提取 images[0]）
                    if hasattr(image, 'images') and isinstance(image.images, list) and len(image.images) > 0:
                        # 这是 Result 对象，提取第一个图像
                        image = image.images[0]
                    elif hasattr(image, 'save'):
                        # 这是 PIL Image，直接使用
                        pass
                    else:
                        # 未知类型，尝试转换
                        print(f"  ⚠️  警告：未知的图像类型: {type(image)}")
                        if isinstance(image, (list, tuple)) and len(image) > 0:
                            image = image[0]
                    
                    # 确保是 PIL Image
                    from PIL import Image as PILImage
                    if not isinstance(image, PILImage.Image):
                        print(f"  ⚠️  错误：返回的不是 PIL Image: {type(image)}")
                        raise TypeError(f"generate_scene 返回的不是 PIL Image: {type(image)}")
                    
                    image.save(output_path)
                    print(f"  ✅ 增强模式生成成功: {output_path}")
                    return output_path
                else:
                    print("  ⚠️  增强模式生成失败，回退到标准模式")
            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️  增强模式生成出错: {e}")
                import traceback
                traceback.print_exc()
                
                # ⚡ 关键修复：如果是内存不足错误，不要回退到标准模式（避免无限循环）
                if "out of memory" in error_msg.lower() or "cuda error" in error_msg.lower():
                    print("  ❌ 检测到内存不足错误，不再回退到标准模式（避免无限循环）")
                    raise RuntimeError(f"内存不足，无法生成图像。请先清理 GPU 内存或减少并发数量。原始错误: {e}") from e
        
        # ⚡ 调试：记录传入的 reference_image_path
        print(f"  🔍 调试：generate_image 接收到的 reference_image_path = {reference_image_path}")
        print(f"  🔍 调试：generate_image 接收到的 face_reference_image_path = {face_reference_image_path}")
        
        # ⚡ 关键修复：保存原始的 reference_image_path，防止后续逻辑覆盖
        original_reference_image_path = reference_image_path
        
        # 多模型选择逻辑（如果启用自动模式）
        if self.engine == "auto" or model_engine:
            # 构建场景上下文
            scene_context = {
                "face_reference_image_path": face_reference_image_path,
                "character_lora": character_lora,
                "reference_image_path": reference_image_path,
            }
            
            # 转换任务类型字符串为枚举
            task_type_enum = None
            if task_type:
                try:
                    task_type_enum = TaskType[task_type.upper()]
                except KeyError:
                    print(f"⚠️  无效的任务类型: {task_type}，将自动检测")
            
            # 选择最适合的引擎
            selected_engine = self.model_selector.select_engine(
                task_type=task_type_enum,
                prompt=prompt,
                scene_context=scene_context,
                manual_engine=model_engine
            )
            
            print(f"🎯 自动选择模型引擎: {selected_engine}")
            
            # 如果选择的引擎与当前不同，加载对应的 pipeline
            if selected_engine != self.engine:
                # 保存原始引擎
                original_engine = self.engine
                # 临时切换到选择的引擎
                self.engine = selected_engine
                # 加载对应的 pipeline
                try:
                    self.load_pipeline(engine=selected_engine)
                except Exception as e:
                    print(f"⚠️  加载 {selected_engine} 失败: {e}")
                    
                    # 智能回退：如果是CHARACTER任务且InstantID失败，尝试使用Flux或其他引擎
                    if selected_engine == "instantid" and task_type_enum == TaskType.CHARACTER:
                        print(f"  ℹ  CHARACTER任务InstantID失败，尝试使用Flux...")
                        try:
                            # 尝试使用Flux.1（支持角色生成）
                            self.engine = "flux1"
                            self.load_pipeline(engine="flux1")
                            print(f"  ✓ 成功回退到 Flux.1")
                        except Exception as flux_err:
                            print(f"  ⚠ Flux.1 也加载失败: {flux_err}，回退到 {original_engine}")
                            self.engine = original_engine
                            if self.pipeline is None:
                                # 如果original_engine是auto，尝试加载sdxl作为最后的备选
                                if original_engine == "auto":
                                    try:
                                        self.engine = "sdxl"
                                        self.load_pipeline(engine="sdxl")
                                        print(f"  ✓ 回退到 SDXL")
                                    except:
                                        self.engine = "auto"
                                        print(f"  ⚠ 所有引擎加载失败，保持auto模式")
                                else:
                                    self.load_pipeline()
                    else:
                        # 非CHARACTER任务或非InstantID失败，正常回退
                        print(f"  回退到 {original_engine}")
                        self.engine = original_engine
                        if self.pipeline is None:
                            # 如果original_engine是auto，尝试加载sdxl作为最后的备选
                            if original_engine == "auto":
                                try:
                                    self.engine = "sdxl"
                                    self.load_pipeline(engine="sdxl")
                                    print(f"  ✓ 回退到 SDXL")
                                except:
                                    self.engine = "auto"
                                    print(f"  ⚠ 所有引擎加载失败，保持auto模式")
                            else:
                                self.load_pipeline()
        
        # 检查场景中的角色，确定使用哪种生成方法
        use_text_to_image = False
        primary_character = None
        has_character_reference = False

        # 如果明确指定了task_type="scene"，跳过角色检测，避免误识别
        skip_character_detection = (task_type == "scene")
        
        if scene and not skip_character_detection:
            identified_characters = self._identify_characters_in_scene(scene)
            if identified_characters:
                primary_character = identified_characters[0]
                
                # 科普主持人：自动加载对应的LoRA，并使用 ModelManager 优化 prompt（MVP 流程）
                if primary_character in ["kepu_gege", "weilai_jiejie"]:
                    # 如果未指定 character_lora，自动映射到 host_person_v2 LoRA
                    if character_lora is None:
                        character_lora = "host_person_v2"  # 使用 host_person_v2 LoRA
                        print(f"  ✓ 检测到科普主持人: {primary_character}，自动加载LoRA: {character_lora}")
                    else:
                        print(f"  ℹ 检测到科普主持人: {primary_character}，使用指定的LoRA: {character_lora}")
                    
                    # MVP 流程：使用 ModelManager 优化 prompt（防止卡通风格）
                    try:
                        from model_manager import ModelManager
                        # 创建临时 ModelManager 实例用于 prompt 优化
                        models_root = self.models_root
                        config_path = str(self.config_path) if hasattr(self, 'config_path') else None
                        temp_manager = ModelManager(models_root=str(models_root), lazy_load=True, config_path=config_path)
                        # 使用 host_face 任务优化 prompt
                        optimized_prompt, optimized_negative = temp_manager._optimize_prompt("host_face", prompt, negative_prompt)
                        if optimized_prompt != prompt or (optimized_negative or "") != (negative_prompt or ""):
                            prompt = optimized_prompt
                            negative_prompt = optimized_negative
                            print(f"  ✅ 已使用 ModelManager 优化 prompt（防止卡通风格）")
                    except Exception as e:
                        print(f"  ⚠ ModelManager prompt 优化失败: {e}，继续使用原始 prompt")
                
                # 重要：韩立角色应该使用原来的高质量参考图（face_image_path），而不是生成的参考图
                # 这样可以确保人脸相似度和发型一致性
                if primary_character == "hanli":
                    # ⚡ 核心修复：如果配置中LoRA被禁用，不要自动加载LoRA
                    if not self.use_lora:
                        character_lora = None  # 配置中LoRA已禁用，不使用LoRA
                        print(f"  ✓ 检测到角色: {primary_character}（韩立），但配置中LoRA已禁用，不使用LoRA（使用HanLi.prompt模板代替）")
                    elif character_lora is None:
                        # 如果未指定 character_lora 且配置中LoRA启用，自动使用 hanli LoRA
                        character_lora = "hanli"  # 使用 hanli LoRA
                        print(f"  ✓ 检测到角色: {primary_character}（韩立），自动加载LoRA: {character_lora}")
                    else:
                        print(f"  ℹ 检测到角色: {primary_character}（韩立），使用指定的LoRA: {character_lora}")
                    # 韩立角色：使用仙侠动漫风格（凡人修仙传风格）
                    # 保持style_lora为None，使用配置中的anime_style LoRA（仙侠动漫风格）
                    print(f"  ✓ 韩立角色：使用仙侠动漫风格（凡人修仙传风格）")
                    # 韩立使用原来的参考图，不使用 character_reference_images
                    # 重要：必须设置 face_reference_image_path 和 has_character_reference，确保使用 InstantID
                    if hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        face_reference_image_path = Path(self.face_image_path)
                        has_character_reference = True
                        print(f"  ✓ 使用韩立的参考图: {face_reference_image_path}")
                    else:
                        print(f"  ⚠ 韩立参考图不存在: {self.face_image_path if hasattr(self, 'face_image_path') else '未设置'}")
                        # 即使参考图不存在，也标记为有角色参考，这样会尝试使用 InstantID
                        has_character_reference = True
                elif primary_character in ["kepu_gege", "weilai_jiejie"]:
                    # 科普主持人：使用指定的参考图
                    if primary_character == "kepu_gege":
                        # 使用 kupu_gege.png 作为参考图
                        # 优先从配置文件读取，否则使用默认路径
                        model_selection = self.image_config.get('model_selection', {})
                        character_config = model_selection.get('character', {})
                        config_face_path = character_config.get('face_image_path')
                        if config_face_path and Path(config_face_path).exists():
                            ref_path = Path(config_face_path)
                        else:
                            # 默认路径：gen_video/reference_image/kupu_gege.png
                            ref_path = Path(__file__).parent.parent / "reference_image" / "kupu_gege.png"
                        
                        if ref_path.exists():
                            has_character_reference = True
                            face_reference_image_path = ref_path
                            print(f"  ✓ 检测到科普哥哥，使用参考图像: {ref_path.name}")
                        else:
                            print(f"  ⚠ 科普哥哥参考图不存在: {ref_path}")
                    elif primary_character == "weilai_jiejie":
                        # 未来姐姐的参考图（如果有）
                        ref_path = Path(__file__).parent.parent / "reference_image" / "weilai_jiejie.png"
                        if ref_path.exists():
                            has_character_reference = True
                            face_reference_image_path = ref_path
                            print(f"  ✓ 检测到未来姐姐，使用参考图像: {ref_path.name}")
                        else:
                            print(f"  ℹ 未来姐姐参考图不存在，将使用默认参考图")
                elif primary_character in self.character_reference_images:
                    # 其他角色使用生成的参考图
                    has_character_reference = True
                    ref_path = self.character_reference_images[primary_character]
                    print(
                        f"  ℹ 检测到角色: {primary_character}，找到参考图像: {
                            ref_path.name}")
                    # 如果有参考图像，使用它作为 face_reference_image_path
                    if face_reference_image_path is None:
                        face_reference_image_path = ref_path
                else:
                    print(f"  ℹ 检测到角色: {primary_character}，未找到参考图像")

        # 场景参考图像选择（如果启用）
        # 注意：如果用户已经提供了参考图像（reference_image_path），就不使用场景查询功能
        # 场景查询功能只用于视频生成场景，不应该在普通图像生成时使用
        scene_reference_images = []
        scene_reference_image_for_img2img = None
        if scene and self.use_scene_reference and reference_image_path is None:
            # 只有在用户没有提供参考图像时，才使用场景查询功能
            # 这样可以避免加载针对特定领域的索引（如"凡人修仙传"的global_index）
            try:
                scene_reference_images = self._select_scene_reference_images(scene)
            except Exception as e:
                # 场景参考图像选择失败不影响主要功能，只记录警告
                print(f"  ⚠ 场景参考图像选择失败（不影响主要功能）: {type(e).__name__}")
                scene_reference_images = []
            if scene_reference_images:
                # 选择最相关的参考图像（第一个，相似度最高）
                best_reference = scene_reference_images[0]
                # 如果启用img2img且配置允许，使用场景参考图像作为init_image
                if self.scene_reference_use_as_img2img and self.use_img2img and self.img2img_pipeline is not None:
                    scene_reference_image_for_img2img = best_reference
                    print(f"  ✓ 使用场景参考图像作为img2img输入: {best_reference.name}")
                # 同时作为IP-Adapter的参考图像
                reference_image_path = best_reference
                print(f"  ✓ 使用场景参考图像作为IP-Adapter输入: {best_reference.name}")
        elif scene and self.use_scene_reference and reference_image_path is not None:
            # 用户已经提供了参考图像，跳过场景查询
            print(f"  ℹ 用户已提供参考图像，跳过场景查询功能（避免加载领域特定索引）")

        # 根据引擎类型选择生成方法
        # 重要：如果检测到hanli角色，必须使用InstantID（SDXL + InstantID），不能使用Flux
        # 原因：SDXL支持风格LoRA，可以保持风格统一；Flux不支持风格LoRA
        
        # 初始化 has_face_reference（必须在检查前初始化）
        has_face_reference = (
            (face_reference_image_path is not None and Path(face_reference_image_path).exists()) or
            has_character_reference or
            (primary_character == "hanli" and hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists())
        )
        
        # ⚡ 关键修复：提前计算 should_disable_instantid（在所有分支中使用）
        # 检测top-down + far away + lying场景，禁用InstantID（InstantID在脸部占比<5%时失效）
        should_disable_instantid = False
        if scene:
            camera_desc = self._get_camera_string(scene).lower() if scene else ""
            prompt_lower_check = prompt.lower()
            visual = scene.get("visual", {}) or {}
            composition_text = str(visual.get("composition", "")).lower() if isinstance(visual, dict) else ""
            environment_text = str(visual.get("environment", "")).lower() if isinstance(visual, dict) else ""
            all_text = f"{camera_desc} {prompt_lower_check} {composition_text} {environment_text}".lower()
            
            is_top_down = any(kw in all_text for kw in ["top-down", "top down", "俯视", "bird's eye", "bird eye", "bird's-eye", "bird-eye", "topdown", "top_down"])
            is_far_away = any(kw in all_text for kw in ["far away", "distant view", "distant", "wide shot", "long shot", "extreme wide", "faraway"])
            action_lower = str(scene.get("action", "")).lower() if scene else ""
            is_lying_check = any(kw in all_text or kw in action_lower for kw in ["lying", "lie", "躺", "lying on", "lie on"])
            
            # top-down + far away + lying是InstantID的死刑组合
            should_disable_instantid = is_top_down and is_far_away and is_lying_check
        
        # ⚡ 关键修复：检查场景是否适合使用InstantID
        # 对于远景+俯拍+低可见度+face_visible=false的场景，InstantID效果很差，应该使用SDXL
        should_use_instantid_for_hanli = True
        if primary_character == "hanli" and scene:
            # 检查v2格式的字段
            character = scene.get("character", {}) or {}
            if isinstance(character, dict):
                face_visible = character.get("face_visible", True)
                visibility = character.get("visibility", "high")
                # 如果face_visible=false 或 visibility=low，不适合使用InstantID
                if face_visible is False or visibility == "low":
                    should_use_instantid_for_hanli = False
                    print(f"  ℹ 场景配置：face_visible={face_visible}, visibility={visibility}，不适合使用InstantID，保持当前引擎")
            
            # 检查camera配置
            camera = scene.get("camera", {}) or {}
            if isinstance(camera, dict):
                shot = camera.get("shot", "").lower()
                angle = camera.get("angle", "").lower()
                # 如果wide+top_down，不适合使用InstantID
                if shot == "wide" and angle in ["top_down", "topdown", "bird_eye"]:
                    should_use_instantid_for_hanli = False
                    print(f"  ℹ 场景配置：shot={shot}, angle={angle}，不适合使用InstantID，保持当前引擎")
            
            # 检查generation_policy.allow_face_lock
            generation_policy = scene.get("generation_policy", {}) or {}
            if isinstance(generation_policy, dict):
                allow_face_lock = generation_policy.get("allow_face_lock", True)
                if allow_face_lock is False:
                    should_use_instantid_for_hanli = False
                    print(f"  ℹ 场景配置：allow_face_lock=false，不适合使用InstantID，保持当前引擎")
            
            # 如果检测到top-down+far away+lying组合，也不适合使用InstantID
            if should_disable_instantid:
                should_use_instantid_for_hanli = False
                print(f"  ℹ 场景配置：检测到top-down+far away+lying组合，不适合使用InstantID，保持当前引擎")
        
        # 重要：如果检测到hanli角色且场景适合，才强制使用InstantID引擎
        if primary_character == "hanli" and self.engine != "instantid" and should_use_instantid_for_hanli:
            print(f"  ⚠ 检测到hanli角色，但当前引擎是{self.engine}，切换到InstantID")
            try:
                self.engine = "instantid"
                self.load_pipeline(engine="instantid")
                print(f"  ✓ 已切换到InstantID引擎")
            except Exception as e:
                print(f"  ⚠ 切换到InstantID失败: {e}，继续使用当前引擎")
        
        if self.engine == "instantid":
            # 重要：所有角色统一使用InstantID（SDXL + InstantID），保证风格统一
            # 原因：
            # 1. SDXL支持风格LoRA（anime_style），可以保持风格统一
            # 2. Flux不支持风格LoRA，会导致风格不统一
            # 3. InstantID可以保证角色人脸一致性
            if primary_character:
                # 所有角色都使用InstantID（特别是hanli角色）
                if not has_face_reference:
                    # 对于韩立角色，从配置文件获取参考图
                    if primary_character == "hanli" and hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        face_reference_image_path = Path(self.face_image_path)
                        has_face_reference = True
                        has_character_reference = True
                        print(f"  ✓ {primary_character}角色：使用InstantID，使用配置文件中的参考图: {self.face_image_path}")
                    else:
                        print(f"  ⚠ {primary_character}角色缺少参考图，但使用InstantID（SDXL支持风格LoRA，保证风格统一）")
                        has_face_reference = True  # 强制使用InstantID
                        has_character_reference = True
                print(f"  ℹ 使用 InstantID 生成（角色: {primary_character}，SDXL + InstantID + 风格LoRA，保证风格统一）")
                # 确保hanli角色和其他角色都使用InstantID，不会切换到Flux
                # has_face_reference已经在上面初始化了，这里只需要确保它被正确设置
                if primary_character == "hanli" and not has_face_reference:
                    # 韩立角色必须使用InstantID，即使没有明确的参考图
                    has_face_reference = True
                    has_character_reference = True
                    if hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        face_reference_image_path = Path(self.face_image_path)
            else:
                # 没有检测到角色的情况
                # 如果没有检测到角色，检查是否有场景参考图
                has_face_reference = (
                    (face_reference_image_path is not None and Path(face_reference_image_path).exists()) or
                    has_character_reference
                )
                
                # 如果只有场景参考图（reference_image_path），没有面部参考图，使用 Flux.1
                if reference_image_path and not has_face_reference:
                    # 只有场景参考图，使用 Flux.1（不使用 InstantID 和 SDXL）
                    print(f"  ℹ 只有场景参考图，使用 Flux.1（不使用 InstantID 和 SDXL）")
                    # 切换到 Flux.1 引擎
                    try:
                        if self.flux1_pipeline is None:
                            self.engine = "flux1"
                            self.load_pipeline(engine="flux1")
                            print(f"  ✓ 已切换到 Flux.1 引擎")
                        else:
                            self.engine = "flux1"
                            print(f"  ✓ 使用已加载的 Flux.1 pipeline")
                    except Exception as e:
                        print(f"  ⚠ Flux.1 加载失败: {e}，回退到 SDXL")
                        use_text_to_image = True
                elif not has_face_reference:
                    # 没有任何参考图像，使用 Flux.1 而不是 SDXL
                    print(f"  ℹ 没有参考图像，使用 Flux.1（不使用 SDXL）")
                    try:
                        if self.flux1_pipeline is None:
                            self.engine = "flux1"
                            self.load_pipeline(engine="flux1")
                            print(f"  ✓ 已切换到 Flux.1 引擎")
                        else:
                            self.engine = "flux1"
                            print(f"  ✓ 使用已加载的 Flux.1 pipeline")
                    except Exception as e:
                        print(f"  ⚠ Flux.1 加载失败: {e}，回退到 SDXL")
                        use_text_to_image = True

        # MVP 流程：如果检测到科普主持人且有参考图，使用 ModelManager 的完整流程（与 mvp_main 一致）
        # 但如果明确指定了task_type="scene"，跳过这个流程，避免生成人物图像
        if (not skip_character_detection and 
            primary_character in ["kepu_gege", "weilai_jiejie"] and 
            face_reference_image_path and Path(face_reference_image_path).exists()):
            try:
                from model_manager import ModelManager
                from PIL import Image
                
                print(f"  🎯 检测到科普主持人 + 参考图，使用 ModelManager 完整流程（与 mvp_main 一致）")
                
                # 创建 ModelManager 实例
                models_root = self.models_root
                config_path = str(self.config_path) if hasattr(self, 'config_path') else None
                manager = ModelManager(models_root=str(models_root), lazy_load=True, config_path=config_path)
                
                # 加载参考图
                face_image = Image.open(face_reference_image_path)
                print(f"  ✅ 已加载参考图: {face_reference_image_path.name}")
                
                # 使用 ModelManager 生成（自动使用 Flux + InstantID）
                image = manager.generate(
                    task="host_face",  # 会自动切换到 host_face_instantid
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width if 'width' in locals() else self.width,
                    height=height if 'height' in locals() else self.height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    face_image=face_image,
                    face_strength=0.8
                )
                
                # 保存图像
                image.save(output_path)
                print(f"  ✅ ModelManager 生成成功: {output_path}")
                return output_path
                
            except Exception as e:
                print(f"  ⚠ ModelManager 生成失败: {e}，回退到普通流程")
                import traceback
                traceback.print_exc()
                # 继续使用普通流程
        
        # 根据引擎类型和是否有参考图像选择生成方法
        if self.engine == "kolors":
            # 使用 Kolors 生成（真实感场景）
            return self._generate_image_kolors(
                prompt, output_path, negative_prompt, guidance_scale,
                num_inference_steps, seed, scene=scene
            )
        elif self.engine == "flux1":
            # 使用 Flux.1 生成（实验室/医学场景）
            # ⚡ 关键修复：传递 reference_image_path 和 face_reference_image_path
            return self._generate_image_flux1(
                prompt, output_path, negative_prompt, guidance_scale,
                num_inference_steps, seed, scene=scene,
                reference_image_path=reference_image_path,  # ⚡ 新增：传递参考图
                face_reference_image_path=face_reference_image_path  # ⚡ 新增：传递面部参考图
            )
        elif self.engine == "flux2":
            # 使用 Flux.2 生成（科学背景图、太空/粒子/量子类）
            return self._generate_image_flux2(
                prompt, output_path, negative_prompt, guidance_scale,
                num_inference_steps, seed, scene=scene
            )
        elif self.engine == "instantid":
            # InstantID引擎：hanli角色必须使用InstantID生成
            # 如果没有参考图，尝试从配置文件获取
            if primary_character == "hanli" and not face_reference_image_path:
                if hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                    face_reference_image_path = Path(self.face_image_path)
                    has_face_reference = True
                    has_character_reference = True
                    print(f"  ✓ hanli角色：从配置文件获取参考图: {self.face_image_path}")
            
            # ⚡ 关键修复：should_disable_instantid已在函数开始处计算，这里直接使用
            # ⚡ 调试信息：仅在需要时打印（减少日志）
            if should_disable_instantid:
                print(f"  ⚠ 检测到top-down + far away + lying场景，禁用InstantID（脸部占比<5%，InstantID失效）")
                print(f"  ℹ 将使用SDXL/img2img代替InstantID，或后续使用I2V方式生成")
                use_instantid_for_this_scene = False
            else:
                use_instantid_for_this_scene = True
            
            # hanli角色通常使用InstantID，但在top-down+far away+lying场景除外
            if use_instantid_for_this_scene and (primary_character == "hanli" or (not use_text_to_image and (face_reference_image_path and Path(face_reference_image_path).exists()))):
                # 韩立角色使用 InstantID（需要参考照片）
                # 如果之前使用了 SDXL 的 IP-Adapter，需要先卸载它
                # 因为 InstantID 和 SDXL 使用不同的 IP-Adapter
                if hasattr(
                        self.pipeline,
                        "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                    # 检查是否是 SDXL 的 IP-Adapter（通过检查是否有 prepare_ip_adapter_image_embeds 方法）
                    # InstantID 的 IP-Adapter 使用不同的加载方式
                    try:
                        # 尝试卸载 SDXL 的 IP-Adapter
                        if hasattr(self.pipeline, "disable_ip_adapter"):
                            self.pipeline.disable_ip_adapter()
                            print(
                                f"  ℹ 已卸载 SDXL 的 IP-Adapter，准备加载 InstantID 的 IP-Adapter")
                        elif hasattr(self.pipeline, "unload_ip_adapter"):
                            self.pipeline.unload_ip_adapter()
                            print(
                                f"  ℹ 已卸载 SDXL 的 IP-Adapter，准备加载 InstantID 的 IP-Adapter")
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
                    print("  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model_in_features）")
                elif hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                    # 如果有 image_proj_model 但没有 image_proj_model_in_features，尝试手动设置
                    try:
                        # InstantID 默认使用 512
                        self.pipeline.image_proj_model_in_features = 512
                        ip_adapter_loaded = True
                        print("  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model），已手动设置 image_proj_model_in_features = 512")
                    except Exception as e:
                        print(f"  ⚠ 无法手动设置 image_proj_model_in_features: {e}")
                        # 即使无法设置，如果 image_proj_model 存在，也认为已加载
                        ip_adapter_loaded = True
                        print("  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model）")
                elif hasattr(self.pipeline, "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                    ip_adapter_loaded = True
                    print("  ✓ 检测到 IP-Adapter 已加载（通过 ip_adapter_image_processor）")
                elif hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
                    # 注意：InstantID 不使用 prepare_ip_adapter_image_embeds，这个检测可能不准确
                    # 但为了兼容性，仍然检查
                    ip_adapter_loaded = True
                    print("  ⚠ 检测到 prepare_ip_adapter_image_embeds（InstantID 可能不使用此方法，但假设已加载）")

                # ⚡ 重要：如果 IP-Adapter 已加载，跳过加载步骤，避免错误
                if ip_adapter_loaded:
                    print("  ℹ InstantID 的 IP-Adapter 已在 pipeline 创建时加载，跳过重新加载")
                else:
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
                            print(
                                f"  加载 InstantID IP-Adapter: {ip_adapter_file}")
                            # InstantID 的 IP-Adapter 使用标准的 load_ip_adapter 方法
                            try:
                                if hasattr(self.pipeline, "load_ip_adapter"):
                                    # 检查 load_ip_adapter 的签名，确定需要哪些参数
                                    import inspect
                                    sig = inspect.signature(
                                        self.pipeline.load_ip_adapter)
                                    params = list(sig.parameters.keys())

                                    # 如果方法需要
                                    # pretrained_model_name_or_path_or_dict
                                    # 作为第一个参数
                                    if "pretrained_model_name_or_path_or_dict" in params:
                                        # 新版本：第一个参数是路径或字典，然后是 subfolder 和
                                        # weight_name
                                        if ip_adapter_file.is_file():
                                            pretrained_path = str(
                                                ip_adapter_file.parent)
                                            weight_name = ip_adapter_file.name
                                        else:
                                            pretrained_path = str(
                                                ip_adapter_file)
                                            weight_name = "ip-adapter.bin"

                                        # 检查是否需要 subfolder 参数
                                        if "subfolder" in params:
                                            # ⚡ 修复：subfolder 不能是 None，应该传递空字符串或不传递
                                            # InstantID 的 IP-Adapter 通常在根目录，所以使用空字符串
                                            self.pipeline.load_ip_adapter(
                                                pretrained_path,
                                                subfolder="",  # InstantID 的 IP-Adapter 通常在根目录，使用空字符串而不是 None
                                                weight_name=weight_name
                                            )
                                        else:
                                            self.pipeline.load_ip_adapter(
                                                pretrained_path,
                                                weight_name=weight_name
                                            )
                                        print(
                                            f"  ✓ InstantID IP-Adapter 加载成功（使用 load_ip_adapter，pretrained_path={pretrained_path}, weight_name={weight_name}）")
                                        ip_adapter_loaded = True
                                    elif "subfolder" in params and "weight_name" in params:
                                        # 中等版本：需要 subfolder 和 weight_name
                                        if ip_adapter_file.is_file():
                                            subfolder = str(
                                                ip_adapter_file.parent)
                                            weight_name = ip_adapter_file.name
                                        else:
                                            subfolder = str(ip_adapter_file)
                                            weight_name = "ip-adapter.bin"

                                        self.pipeline.load_ip_adapter(
                                            subfolder=subfolder,
                                            weight_name=weight_name
                                        )
                                        print(
                                            f"  ✓ InstantID IP-Adapter 加载成功（使用 load_ip_adapter，subfolder={subfolder}, weight_name={weight_name}）")
                                        ip_adapter_loaded = True
                                    else:
                                        # 旧版本，只需要文件路径
                                        self.pipeline.load_ip_adapter(
                                            str(ip_adapter_file))
                                        print(
                                            "  ✓ InstantID IP-Adapter 加载成功（使用 load_ip_adapter，旧版本）")
                                        ip_adapter_loaded = True
                                else:
                                    print(
                                        "  ⚠ InstantID pipeline 不支持 load_ip_adapter 方法")
                                    print(
                                        "  ℹ InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载，继续...")
                            except AttributeError as e:
                                print(f"  ⚠ 无法加载 InstantID IP-Adapter: {e}")
                                print(
                                    "  ℹ InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载，继续...")
                            except Exception as e2:
                                print(f"  ⚠ 无法加载 InstantID IP-Adapter: {e2}")
                                print(
                                    "  ℹ InstantID 的 IP-Adapter 可能在创建 pipeline 时已经加载，继续...")
                                # 检查是否已经加载了 IP-Adapter
                                # ⚡ 重要：对于 InstantID，优先检查 image_proj_model 和 image_proj_model_in_features
                                if hasattr(self.pipeline, "image_proj_model_in_features"):
                                    print(
                                        "  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model_in_features）")
                                    ip_adapter_loaded = True
                                elif hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                                    print(
                                        "  ✓ 检测到 IP-Adapter 已加载（通过 image_proj_model）")
                                    # 尝试手动设置 image_proj_model_in_features
                                    try:
                                        self.pipeline.image_proj_model_in_features = 512
                                        print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                                    except Exception as e:
                                        print(f"  ⚠ 无法手动设置 image_proj_model_in_features: {e}")
                                    ip_adapter_loaded = True
                                elif hasattr(
                                        self.pipeline,
                                        "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                                    print(
                                        "  ✓ 检测到 IP-Adapter 已加载（通过 ip_adapter_image_processor）")
                                    ip_adapter_loaded = True
                                elif hasattr(self.pipeline, "prepare_ip_adapter_image_embeds"):
                                    print(
                                        "  ⚠ 检测到 prepare_ip_adapter_image_embeds（InstantID 可能不使用此方法，但假设已加载）")
                                    ip_adapter_loaded = True
                        else:
                            print(
                                f"  ⚠ 未找到 InstantID IP-Adapter 文件，假设已在创建 pipeline 时加载")
                    else:
                        print(
                            f"  ⚠ InstantID IP-Adapter 路径未配置或不存在，假设已在创建 pipeline 时加载")
                except Exception as e:
                    print(f"  ⚠ 无法加载 InstantID 的 IP-Adapter: {e}")
                    print("  ℹ 假设 InstantID 的 IP-Adapter 已在创建 pipeline 时加载，继续...")
            else:
                print(f"  ✓ InstantID 的 IP-Adapter 已加载")

            # ⚡ 关键修复：如果should_disable_instantid为True，使用两阶段法（方案2）
            if should_disable_instantid:
                # 不调用InstantID，让代码继续执行到else分支使用两阶段法
                pass
            else:
                try:
                    return self._generate_image_instantid(
                        prompt, output_path, negative_prompt, guidance_scale,
                        num_inference_steps, seed, face_reference_image_path, scene=scene,
                        init_image=init_image,  # 传递前一个场景图像用于连贯性
                        character_lora=character_lora, style_lora=style_lora
                    )
                except ValueError as e:
                    # 如果 InstantID 无法识别人脸，自动回退到 SDXL 文生图
                    error_msg = str(e)
                    if "未检测到人脸" in error_msg or "no face" in error_msg.lower(
                    ) or "face not detected" in error_msg.lower():
                        print(f"  ⚠ InstantID 无法识别人脸: {error_msg}")
                        print(f"  ℹ 自动回退到 SDXL 文生图（不使用参考图像）")
                        # 回退到 SDXL 文生图，禁用 IP-Adapter
                        return self._generate_image_sdxl(
                            prompt, output_path, negative_prompt, guidance_scale,
                            num_inference_steps, seed, reference_image_path=None,
                            face_reference_image_path=None, use_lora=use_lora, scene=scene,
                            use_ip_adapter_override=False,  # 禁用 IP-Adapter，使用纯文生图
                            scene_reference_images=scene_reference_images,  # 传递已选择的场景参考图像
                            scene_reference_image_for_img2img=scene_reference_image_for_img2img,  # 传递用于img2img的参考图像
                            character_lora=character_lora, style_lora=style_lora
                        )
                    else:
                        # 其他类型的 ValueError，继续抛出
                        raise
            
            # ⚡ 关键修复：如果should_disable_instantid为True，直接执行两阶段法（方案2）
            if should_disable_instantid and primary_character == "hanli":
                print(f"  ⚠ 已跳过InstantID生成，直接执行两阶段法（方案2）")
                print(f"  📋 Stage A: 查找或生成人设图 -> Stage B: 使用人设图生成场景（SDXL+IP-Adapter）")
                # 直接执行两阶段法的完整逻辑
                # ⚡ 方案2：两阶段生成
                # Stage A: 生成人设图（InstantID，中景/半身，脸优先）
                # Stage B: 使用人设图作为IP-Adapter输入生成场景（SDXL+IP-Adapter）
                
                # 查找或生成人设图
                hanli_character_image = None
                
                # 1. 检查是否已有缓存的人设图
                character_cache_dir = Path(output_path).parent / "character_cache"
                character_cache_dir.mkdir(parents=True, exist_ok=True)
                cached_character_path = character_cache_dir / "hanli_character.png"
                
                if cached_character_path.exists():
                    hanli_character_image = cached_character_path
                    print(f"  ✓ Stage A: 找到缓存的人设图: {hanli_character_image}")
                else:
                    # 2. 使用现有的素材图（如果有）
                    if face_reference_image_path and Path(face_reference_image_path).exists():
                        hanli_character_image = face_reference_image_path
                        print(f"  ✓ Stage A: 使用传入的韩立素材图作为人设图: {face_reference_image_path}")
                    elif hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        hanli_character_image = Path(self.face_image_path)
                        print(f"  ✓ Stage A: 从配置文件获取韩立素材图作为人设图: {self.face_image_path}")
                    elif hasattr(self, 'face_reference_dir') and self.face_reference_dir:
                        face_ref_dir = Path(self.face_reference_dir)
                        if face_ref_dir.exists():
                            # 查找中景素材图（优先）
                            mid_images = list(face_ref_dir.glob("hanli_mid*.png")) + list(face_ref_dir.glob("hanli_mid*.jpg"))
                            if mid_images:
                                hanli_character_image = mid_images[0]
                                print(f"  ✓ Stage A: 从face_reference_dir找到中景素材图作为人设图: {hanli_character_image}")
                            else:
                                # 查找其他韩立素材图
                                other_images = list(face_ref_dir.glob("hanli*.png")) + list(face_ref_dir.glob("hanli*.jpg"))
                                if other_images:
                                    hanli_character_image = other_images[0]
                                    print(f"  ✓ Stage A: 从face_reference_dir找到素材图作为人设图: {hanli_character_image}")
                    
                    # 3. 如果没有素材图，生成人设图（Stage A）
                    if not hanli_character_image or not Path(hanli_character_image).exists():
                        print(f"  📋 Stage A: 未找到素材图，生成人设图（InstantID，中景/半身，脸优先）")
                        # 生成人设图的prompt（方案2推荐）
                        character_prompt = "Han Li, young male cultivator, dark green robe, black hair tied back, calm and serious expression, front view, upper body, Chinese xianxia anime illustration, clean background"
                        character_negative = "multiple people, extra limbs, deformed face, wrong clothes, modern clothing, armor, helmet, standing, walking, extreme long shot, tiny character, text, watermark"
                        
                        # 确保使用InstantID pipeline（此时self.engine已经是instantid）
                        if self.engine != "instantid":
                            original_engine = self.engine
                            self.engine = "instantid"
                            self.load_pipeline(engine="instantid")
                            print(f"  ✓ 已切换到InstantID引擎（从{original_engine}）用于生成人设图")
                        
                        # 确保有参考图
                        stage_a_face_ref = face_reference_image_path
                        if not stage_a_face_ref and hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                            stage_a_face_ref = Path(self.face_image_path)
                        
                        try:
                            # 生成人设图（使用InstantID，中景，脸优先）
                            print(f"  🎨 Stage A: 生成人设图: {character_prompt[:60]}...")
                            hanli_character_image = self._generate_image_instantid(
                                character_prompt,
                                cached_character_path,
                                character_negative,
                                guidance_scale=6.5,  # 方案2推荐值
                                num_inference_steps=40,  # 方案2推荐值
                                seed=seed,
                                face_reference_image_path=stage_a_face_ref,
                                scene=None,  # 人设图不需要场景信息
                                character_lora=None,  # 人设图不使用LoRA
                                style_lora=None
                            )
                            print(f"  ✓ Stage A完成: 人设图已生成并缓存: {hanli_character_image}")
                        except Exception as e:
                            print(f"  ⚠ Stage A失败: 无法生成人设图: {e}")
                            print(f"  ℹ 将使用现有素材图或纯文生图")
                            # 如果生成失败，尝试使用现有素材图
                            if face_reference_image_path and Path(face_reference_image_path).exists():
                                hanli_character_image = face_reference_image_path
                                print(f"  ✓ 使用现有素材图: {hanli_character_image}")
                
                # Stage B: 使用人设图作为IP-Adapter输入生成场景
                # ⚡ 关键修复：检测是否是 wide + top_down + lying 场景
                # 如果是，则禁用 LoRA 和 IP-Adapter（当作"剪影+氛围"镜头处理）
                is_wide_topdown_lying = False
                if scene:
                    camera = scene.get("camera", {}) or {}
                    character_data = scene.get("character", {}) or {}
                    camera_shot = camera.get("shot", "medium")
                    camera_angle = camera.get("angle", "eye_level")
                    character_pose = character_data.get("pose", "")
                    is_wide_topdown_lying = (
                        camera_shot == "wide" and 
                        camera_angle == "top_down" and 
                        character_pose in ["lying_motionless", "lying"]
                    )
                
                if hanli_character_image and Path(hanli_character_image).exists():
                    if is_wide_topdown_lying:
                        # ⚡ 关键修复：wide + top_down + lying 场景，禁用 LoRA 和 IP-Adapter
                        # 原因：人物只有 10-15% 像素，LoRA 和 IP-Adapter 会导致脸崩、比例怪
                        # 这类镜头当作"剪影+氛围"处理，不追求人脸一致性
                        print(f"  📋 Stage B: wide + top_down + lying 场景，使用纯文生图（禁用 LoRA 和 IP-Adapter）")
                        print(f"  ⚠ 这类镜头当作'剪影+氛围'处理，不追求人脸一致性")
                        reference_image_path = None  # 不使用人设图
                        stage_b_character_lora = None  # 禁用角色 LoRA
                        stage_b_style_lora = ""  # 禁用风格 LoRA
                        stage_b_use_lora = False  # 禁用 LoRA
                        stage_b_use_ip_adapter = False  # 禁用 IP-Adapter
                        print(f"  ✓ Stage B: 已禁用所有 LoRA 和 IP-Adapter（避免姿态冲突和脸崩）")
                    else:
                        # 非 wide + top_down + lying 场景，使用正常的两阶段法
                        print(f"  📋 Stage B: 使用人设图生成场景（SDXL+IP-Adapter）")
                        print(f"  📸 人设图: {hanli_character_image}")
                        # 将人设图作为IP-Adapter的参考图像（方案2推荐：使用IP-Adapter而不是img2img）
                        reference_image_path = hanli_character_image
                        # ⚡ 修复：提高IP-Adapter scale到0.85，确保人设图影响足够强
                        self._two_stage_ip_adapter_scale = 0.85
                        print(f"  ✓ 将使用人设图作为IP-Adapter输入（ip_adapter_scale=0.85，确保人设图影响足够强）")
                        
                        # ⚡ 修复：Stage B必须使用LoRA，确保角色和风格正确
                        # 即使配置中禁用了LoRA，两阶段法也需要使用LoRA来保证角色一致性
                        stage_b_character_lora = character_lora if character_lora is not None else "hanli"
                        # ⚡ 修复：强制使用anime_style LoRA（仙侠风格），即使配置中禁用了
                        if style_lora is not None:
                            stage_b_style_lora = style_lora
                        else:
                            # 从配置中读取anime_style LoRA的adapter_name
                            style_lora_config = self.lora_config.get("style_lora", {})
                            if isinstance(style_lora_config, dict) and style_lora_config.get("adapter_name"):
                                stage_b_style_lora = style_lora_config.get("adapter_name")  # 通常是"anime_style"
                            else:
                                stage_b_style_lora = "anime_style"  # 默认值
                        stage_b_use_lora = True  # 强制启用LoRA
                        stage_b_use_ip_adapter = True  # 使用 IP-Adapter
                        print(f"  ✓ Stage B强制使用LoRA: character_lora={stage_b_character_lora}, style_lora={stage_b_style_lora}（确保修仙风格）")
                    
                    # ⚡ 修复：强制在prompt最前面添加高权重的lying描述，确保人物躺着
                    import re
                    # 检查prompt中是否已有lying描述
                    has_lying = bool(re.search(r'lying|lie|躺', prompt, re.IGNORECASE))
                    
                    if has_lying:
                        # 如果已有lying，检查权重是否足够高（至少3.0）
                        lying_weight_match = re.search(r'\([^)]*lying[^)]*:([\d.]+)\)', prompt, re.IGNORECASE)
                        if lying_weight_match:
                            lying_weight = float(lying_weight_match.group(1))
                            if lying_weight < 3.0:
                                # 权重不够，在开头添加更高权重的lying描述
                                # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting（SDXL 对 NOT 不敏感）
                                enhanced_lying = "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)"
                                prompt = f"{enhanced_lying}, {prompt}"
                                print(f"  ✓ 已提升lying描述权重到3.0（最高优先级），确保人物躺着")
                        else:
                            # 没有权重标记，在开头添加高权重描述
                            enhanced_lying = "(lying on desert sand, lying on ground, NOT standing, NOT sitting, horizontal position, prone, supine:3.0)"
                            prompt = f"{enhanced_lying}, {prompt}"
                            print(f"  ✓ 已在prompt最前面添加lying描述（权重3.0），确保人物躺着")
                    else:
                        # 完全没有lying描述，在开头添加
                        enhanced_lying = "(lying on desert sand, lying on ground, NOT standing, NOT sitting, horizontal position, prone, supine:3.0)"
                        prompt = f"{enhanced_lying}, {prompt}"
                        print(f"  ✓ 已在prompt最前面添加lying描述（权重3.0），确保人物躺着")
                    
                    # ⚡ 方案2：不需要切换pipeline，_generate_image_sdxl会自动处理InstantID pipeline
                    # 调用SDXL生成方法（根据场景决定是否使用IP-Adapter）
                    return self._generate_image_sdxl(
                        prompt, output_path, negative_prompt, guidance_scale,
                        num_inference_steps, seed, reference_image_path=reference_image_path,
                        face_reference_image_path=None, use_lora=stage_b_use_lora, scene=scene,
                        use_ip_adapter_override=stage_b_use_ip_adapter,  # ⚡ 关键修复：根据场景决定是否使用IP-Adapter
                        scene_reference_images=scene_reference_images,
                        scene_reference_image_for_img2img=None,  # 方案2不使用img2img
                        character_lora=stage_b_character_lora, style_lora=stage_b_style_lora
                    )
                else:
                    print(f"  ⚠ 未找到人设图，将使用纯文生图（建议配置face_image_path或face_reference_dir）")
                    # 如果没有人设图，回退到纯文生图
                    return self._generate_image_sdxl(
                        prompt, output_path, negative_prompt, guidance_scale,
                        num_inference_steps, seed, reference_image_path=None,
                        face_reference_image_path=None, use_lora=use_lora, scene=scene,
                        use_ip_adapter_override=False,  # 禁用IP-Adapter
                        scene_reference_images=scene_reference_images,
                        scene_reference_image_for_img2img=scene_reference_image_for_img2img,
                        character_lora=character_lora, style_lora=style_lora
                    )
        else:
            # 检查是否是hanli角色但使用了其他引擎（应该使用InstantID）
            # ⚡ 关键修复：但如果场景不适合使用InstantID，不强制切换
            # 需要同时检查 should_use_instantid_for_hanli 和 should_disable_instantid
            if primary_character == "hanli" and should_use_instantid_for_hanli and not should_disable_instantid:
                print(f"  ⚠ hanli角色但当前引擎是{self.engine}，应该使用InstantID，切换到InstantID...")
                try:
                    # 切换到InstantID
                    original_engine = self.engine
                    original_pipeline = self.pipeline
                    self.engine = "instantid"
                    self.load_pipeline(engine="instantid")
                    print(f"  ✓ 已切换到InstantID引擎（从{original_engine}）")
                    # 确保有参考图
                    if not face_reference_image_path and hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        face_reference_image_path = Path(self.face_image_path)
                        print(f"  ✓ hanli角色：从配置文件获取参考图: {self.face_image_path}")
                    # 直接调用InstantID生成方法（避免递归）
                    return self._generate_image_instantid(
                        prompt, output_path, negative_prompt, guidance_scale,
                        num_inference_steps, seed, face_reference_image_path, scene=scene,
                        init_image=init_image,
                        character_lora=character_lora, style_lora=style_lora
                    )
                except Exception as e:
                    print(f"  ⚠ 切换到InstantID失败: {e}，继续使用当前引擎{self.engine}")
                    # 如果切换失败，恢复原来的pipeline
                    if original_pipeline:
                        self.pipeline = original_pipeline
                    # 继续使用当前引擎（但这不是理想情况）
            elif primary_character == "hanli" and (not should_use_instantid_for_hanli or should_disable_instantid):
                print(f"  ✓ hanli角色但检测到top-down+far away+lying场景，已禁用InstantID，使用两阶段法（方案2）")
                
                # ⚡ 关键修复：检测是否是 wide + top_down + lying 场景
                # 如果是，则禁用所有 LoRA 和 IP-Adapter（当作"剪影+氛围"镜头处理）
                is_wide_topdown_lying = False
                if scene:
                    camera = scene.get("camera", {}) or {}
                    character_data = scene.get("character", {}) or {}
                    camera_shot = camera.get("shot", "medium")
                    camera_angle = camera.get("angle", "eye_level")
                    character_pose = character_data.get("pose", "")
                    is_wide_topdown_lying = (
                        camera_shot == "wide" and 
                        camera_angle == "top_down" and 
                        character_pose in ["lying_motionless", "lying"]
                    )
                
                # ⚡ 关键修复：只有在非 wide+top_down+lying 场景才启用风格锚点
                # wide + top_down + lying 场景：禁用所有 LoRA 和 IP-Adapter
                if is_wide_topdown_lying:
                    print(f"  ⚠ wide + top_down + lying 场景：禁用所有 LoRA 和 IP-Adapter（当作'剪影+氛围'处理）")
                    style_lora = ""  # 禁用风格 LoRA
                    character_lora = None  # 禁用角色 LoRA
                    # ⚡ 关键修复：增强 negative prompt，添加多人排除词
                    if negative_prompt is None:
                        negative_prompt = self.negative_prompt
                    # 添加多人排除词（确保只生成一个人物）
                    multiple_people_negative = "multiple people, two people, crowd, group of people, several people, many people, extra people, additional person, duplicate person"
                    if negative_prompt:
                        negative_prompt = f"{negative_prompt}, {multiple_people_negative}"
                    else:
                        negative_prompt = multiple_people_negative
                    print(f"  ✓ 增强 negative prompt：添加多人排除词（确保只生成一个人物）")
                elif style_lora is None or style_lora == "":
                    # 检查 Execution Planner 是否返回了 style_anchor 配置
                    style_anchor = None
                    if scene:
                        # 尝试从 Execution Planner 决策中获取 style_anchor
                        # 如果 Execution Planner 已经返回了 style_anchor，使用它
                        # 否则，使用默认的凡人修仙传风格 LoRA
                        style_anchor = {
                            "type": "lora",
                            "name": "fanren_style",  # 默认使用 anime_style（凡人修仙传风格）
                            "weight": 0.35,  # 低权重，只绑定风格，不抢戏
                            "enabled": True
                        }
                    
                    if style_anchor and style_anchor.get("enabled", False):
                        # 使用 Execution Planner 指定的风格 LoRA
                        style_lora_name = style_anchor.get("name", "anime_style")
                        style_lora_weight = style_anchor.get("weight", 0.35)
                        # 映射 fanren_style 到实际的 LoRA 名称
                        if style_lora_name == "fanren_style":
                            style_lora_name = "anime_style"  # 使用配置中的 anime_style LoRA
                        style_lora = style_lora_name
                        print(f"  ✓ 启用风格锚点（风格 LoRA）: {style_lora_name}，权重: {style_lora_weight}")
                        # 注意：权重会在 _load_lora 中应用
                # ⚡ 方案2：两阶段生成
                # Stage A: 生成人设图（InstantID，中景/半身，脸优先）
                # Stage B: 使用人设图作为IP-Adapter输入生成场景（SDXL+IP-Adapter）
                
                # 查找或生成人设图
                hanli_character_image = None
                
                # 1. 检查是否已有缓存的人设图
                character_cache_dir = Path(output_path).parent / "character_cache"
                character_cache_dir.mkdir(parents=True, exist_ok=True)
                cached_character_path = character_cache_dir / "hanli_character.png"
                
                if cached_character_path.exists():
                    hanli_character_image = cached_character_path
                    print(f"  ✓ 找到缓存的人设图: {hanli_character_image}")
                else:
                    # 2. 使用现有的素材图（如果有）
                    if face_reference_image_path and Path(face_reference_image_path).exists():
                        hanli_character_image = face_reference_image_path
                        print(f"  ✓ 使用传入的韩立素材图作为人设图: {face_reference_image_path}")
                    elif hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        hanli_character_image = Path(self.face_image_path)
                        print(f"  ✓ 从配置文件获取韩立素材图作为人设图: {self.face_image_path}")
                    elif hasattr(self, 'face_reference_dir') and self.face_reference_dir:
                        face_ref_dir = Path(self.face_reference_dir)
                        if face_ref_dir.exists():
                            # 查找中景素材图（优先）
                            mid_images = list(face_ref_dir.glob("hanli_mid*.png")) + list(face_ref_dir.glob("hanli_mid*.jpg"))
                            if mid_images:
                                hanli_character_image = mid_images[0]
                                print(f"  ✓ 从face_reference_dir找到中景素材图作为人设图: {hanli_character_image}")
                            else:
                                # 查找其他韩立素材图
                                other_images = list(face_ref_dir.glob("hanli*.png")) + list(face_ref_dir.glob("hanli*.jpg"))
                                if other_images:
                                    hanli_character_image = other_images[0]
                                    print(f"  ✓ 从face_reference_dir找到素材图作为人设图: {hanli_character_image}")
                    
                    # 3. 如果没有素材图，生成人设图（Stage A）
                    if not hanli_character_image or not Path(hanli_character_image).exists():
                        print(f"  📋 Stage A: 生成人设图（InstantID，中景/半身，脸优先）")
                        # 生成人设图的prompt（方案2推荐）
                        character_prompt = "Han Li, young male cultivator, dark green robe, black hair tied back, calm and serious expression, front view, upper body, Chinese xianxia anime illustration, clean background"
                        character_negative = "multiple people, extra limbs, deformed face, wrong clothes, modern clothing, armor, helmet, standing, walking, extreme long shot, tiny character, text, watermark"
                        
                        # 确保使用InstantID pipeline
                        if self.engine != "instantid":
                            original_engine = self.engine
                            self.engine = "instantid"
                            self.load_pipeline(engine="instantid")
                            print(f"  ✓ 已切换到InstantID引擎（从{original_engine}）用于生成人设图")
                        
                        # 确保有参考图
                        stage_a_face_ref = face_reference_image_path
                        if not stage_a_face_ref and hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                            stage_a_face_ref = Path(self.face_image_path)
                        
                        try:
                            # 生成人设图（使用InstantID，中景，脸优先）
                            print(f"  🎨 生成人设图: {character_prompt[:60]}...")
                            hanli_character_image = self._generate_image_instantid(
                                character_prompt,
                                cached_character_path,
                                character_negative,
                                guidance_scale=6.5,  # 方案2推荐值
                                num_inference_steps=40,  # 方案2推荐值
                                seed=seed,
                                face_reference_image_path=stage_a_face_ref,
                                scene=None,  # 人设图不需要场景信息
                                character_lora=None,  # 人设图不使用LoRA
                                style_lora=None
                            )
                            print(f"  ✓ Stage A完成: 人设图已生成并缓存: {hanli_character_image}")
                        except Exception as e:
                            print(f"  ⚠ Stage A失败: 无法生成人设图: {e}")
                            print(f"  ℹ 将使用现有素材图或纯文生图")
                            # 如果生成失败，尝试使用现有素材图
                            if face_reference_image_path and Path(face_reference_image_path).exists():
                                hanli_character_image = face_reference_image_path
                                print(f"  ✓ 使用现有素材图: {hanli_character_image}")
                
                # Stage B: 使用人设图作为IP-Adapter输入生成场景
                if hanli_character_image and Path(hanli_character_image).exists():
                    if is_wide_topdown_lying:
                        # ⚡ 关键修复：wide + top_down + lying 场景，禁用 LoRA 和 IP-Adapter
                        # 原因：人物只有 10-15% 像素，LoRA 和 IP-Adapter 会导致脸崩、比例怪
                        # 这类镜头当作"剪影+氛围"处理，不追求人脸一致性
                        print(f"  📋 Stage B: wide + top_down + lying 场景，使用纯文生图（禁用 LoRA 和 IP-Adapter）")
                        print(f"  ⚠ 这类镜头当作'剪影+氛围'处理，不追求人脸一致性")
                        reference_image_path = None  # 不使用人设图
                        use_ip_adapter_for_this = False  # 禁用 IP-Adapter
                        stage_b_character_lora = None  # 禁用角色 LoRA
                        stage_b_style_lora = ""  # 禁用风格 LoRA
                        stage_b_use_lora = False  # 禁用 LoRA
                        print(f"  ✓ Stage B: 已禁用所有 LoRA 和 IP-Adapter（避免姿态冲突和脸崩）")
                    else:
                        # 非 wide + top_down + lying 场景，使用正常的两阶段法
                        print(f"  📋 Stage B: 使用人设图生成场景（SDXL+IP-Adapter）")
                        print(f"  📸 人设图: {hanli_character_image}")
                        # 将人设图作为IP-Adapter的参考图像（方案2推荐：使用IP-Adapter而不是img2img）
                        reference_image_path = hanli_character_image
                        use_ip_adapter_for_this = True
                        # ⚡ 修复：提高IP-Adapter scale到0.85，确保人设图影响足够强
                        self._two_stage_ip_adapter_scale = 0.85
                        print(f"  ✓ 将使用人设图作为IP-Adapter输入（ip_adapter_scale=0.85，确保人设图影响足够强）")
                        
                        # ⚡ 修复：Stage B必须使用LoRA，确保角色和风格正确
                        # 即使配置中禁用了LoRA，两阶段法也需要使用LoRA来保证角色一致性
                        stage_b_character_lora = character_lora if character_lora is not None else "hanli"
                        # ⚡ 修复：强制使用anime_style LoRA（仙侠风格），即使配置中禁用了
                        if style_lora is not None and style_lora != "":
                            stage_b_style_lora = style_lora
                        else:
                            # 从配置中读取anime_style LoRA的adapter_name
                            style_lora_config = self.lora_config.get("style_lora", {})
                            if isinstance(style_lora_config, dict) and style_lora_config.get("adapter_name"):
                                stage_b_style_lora = style_lora_config.get("adapter_name")  # 通常是"anime_style"
                            else:
                                stage_b_style_lora = "anime_style"  # 默认值
                        stage_b_use_lora = True  # 强制启用LoRA
                        print(f"  ✓ Stage B强制使用LoRA: character_lora={stage_b_character_lora}, style_lora={stage_b_style_lora}（确保修仙风格）")
                    
                    # ⚡ 修复：强制在prompt最前面添加高权重的lying描述，确保人物躺着
                    import re
                    # 检查prompt中是否已有lying描述
                    has_lying = bool(re.search(r'lying|lie|躺', prompt, re.IGNORECASE))
                    
                    if has_lying:
                        # 如果已有lying，检查权重是否足够高（至少3.0）
                        lying_weight_match = re.search(r'\([^)]*lying[^)]*:([\d.]+)\)', prompt, re.IGNORECASE)
                        if lying_weight_match:
                            lying_weight = float(lying_weight_match.group(1))
                            if lying_weight < 3.0:
                                # 权重不够，在开头添加更高权重的lying描述
                                # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting（SDXL 对 NOT 不敏感）
                                enhanced_lying = "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)"
                                prompt = f"{enhanced_lying}, {prompt}"
                                print(f"  ✓ 已提升lying描述权重到3.0（最高优先级），确保人物躺着")
                        else:
                            # 没有权重标记，在开头添加高权重描述
                            # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting（SDXL 对 NOT 不敏感）
                            enhanced_lying = "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)"
                            prompt = f"{enhanced_lying}, {prompt}"
                            print(f"  ✓ 已在prompt最前面添加lying描述（权重3.0），确保人物躺着")
                    else:
                        # 完全没有lying描述，在开头添加
                        # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting（SDXL 对 NOT 不敏感）
                        enhanced_lying = "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)"
                        prompt = f"{enhanced_lying}, {prompt}"
                        print(f"  ✓ 已在prompt最前面添加lying描述（权重3.0），确保人物躺着")
                else:
                    print(f"  ⚠ 未找到人设图，将使用纯文生图（建议配置face_image_path或face_reference_dir）")
                    # ⚡ 关键修复：如果是 wide + top_down + lying 场景，即使没有人设图也要禁用 LoRA 和 IP-Adapter
                    # ⚡ 但是，如果是 FLUX pipeline，需要保留传入的 reference_image_path（用于 IP-Adapter）
                    if is_wide_topdown_lying:
                        # 检查是否是 FLUX pipeline（通过检查 engine 或 model_engine）
                        is_flux = (model_engine and "flux" in str(model_engine).lower()) or (self.engine and "flux" in str(self.engine).lower())
                        if is_flux:
                            # FLUX pipeline：保留传入的 reference_image_path（用于 IP-Adapter）
                            # 但禁用 LoRA
                            print(f"  ✓ wide + top_down + lying 场景 + FLUX：保留 reference_image_path 用于 IP-Adapter，但禁用 LoRA")
                            # reference_image_path 保持不变（使用传入的值）
                            use_ip_adapter_for_this = True  # FLUX 需要使用 IP-Adapter
                        else:
                            # SDXL pipeline：禁用所有
                            reference_image_path = None
                            use_ip_adapter_for_this = False
                        stage_b_character_lora = None
                        stage_b_style_lora = ""
                        stage_b_use_lora = False
                        print(f"  ✓ wide + top_down + lying 场景：禁用所有 LoRA")
                    else:
                        # 非 wide + top_down + lying 场景：检查是否是 FLUX pipeline
                        is_flux = (model_engine and "flux" in str(model_engine).lower()) or (self.engine and "flux" in str(self.engine).lower())
                        if is_flux:
                            # FLUX pipeline：保留传入的 reference_image_path（用于 IP-Adapter）
                            print(f"  ✓ FLUX pipeline：保留 reference_image_path 用于 IP-Adapter")
                            use_ip_adapter_for_this = True
                        else:
                            # SDXL pipeline：不使用参考图
                            reference_image_path = None
                            use_ip_adapter_for_this = False
            
            # 使用 SDXL/Flux（包括：只有场景参考图、没有参考图、非韩立角色等情况）
            # ⚡ 方案2：如果should_disable_instantid且是hanli角色，use_ip_adapter_for_this和reference_image_path已在上面设置
            # 对于普通场景（只有场景参考图，没有面部参考图）：
            # - 如果有场景参考图像，使用 IP-Adapter（使用参考图像）
            # - 如果没有参考图像，禁用 IP-Adapter（纯文生图）
            if not (primary_character == "hanli" and should_disable_instantid):
                # 如果不是方案2的情况，使用原有逻辑
                use_ip_adapter_for_this = self.use_ip_adapter
                if use_text_to_image:
                    if reference_image_path or has_character_reference:
                        # 有参考图像（场景参考图或角色参考图），使用 IP-Adapter
                        use_ip_adapter_for_this = True
                        print(f"  ℹ 有参考图像，启用 IP-Adapter（使用参考图像）")
                    else:
                        # 没有参考图像，禁用 IP-Adapter（纯文生图）
                        use_ip_adapter_for_this = False
                        print(f"  ℹ 无参考图像，禁用 IP-Adapter（纯文生图）")

            # 检查是否是 Flux pipeline（Flux 使用不同的架构，需要特殊处理）
            # ⚡ 关键修复：检查 flux1_pipeline 或 flux_pipeline，而不仅仅是 self.pipeline
            print(f"  🔍 调试：检查 FLUX pipeline，当前 reference_image_path = {reference_image_path}")
            print(f"  🔍 调试：检查 FLUX pipeline，flux1_pipeline = {self.flux1_pipeline is not None}")
            print(f"  🔍 调试：检查 FLUX pipeline，flux_pipeline = {self.flux_pipeline is not None}")
            print(f"  🔍 调试：检查 FLUX pipeline，self.pipeline = {self.pipeline is not None}")
            
            is_flux_pipeline = False
            if self.flux1_pipeline is not None:
                is_flux_pipeline = True
                print(f"  ✓ 检测到 flux1_pipeline，将使用 FLUX 生成方法")
            elif self.flux_pipeline is not None:
                is_flux_pipeline = True
                print(f"  ✓ 检测到 flux_pipeline，将使用 FLUX 生成方法")
            elif self.pipeline is not None:
                pipeline_type = type(self.pipeline).__name__
                is_flux_pipeline = "Flux" in pipeline_type or "flux" in pipeline_type.lower()
                if is_flux_pipeline:
                    print(f"  ✓ 检测到 self.pipeline 是 FLUX 类型: {pipeline_type}")
            
            print(f"  🔍 调试：is_flux_pipeline = {is_flux_pipeline}")
            
            if is_flux_pipeline:
                # ⚡ 调试：记录检测到 FLUX pipeline 时的 reference_image_path
                print(f"  🔍 调试：检测到 FLUX pipeline，当前 reference_image_path = {reference_image_path}")
                print(f"  🔍 调试：检测到 FLUX pipeline，当前 face_reference_image_path = {face_reference_image_path}")
                print(f"  🔍 调试：检测到 FLUX pipeline，原始 reference_image_path = {original_reference_image_path}")
                
                # ⚡ 关键修复：如果 reference_image_path 被设置为 None，尝试恢复原始值或从其他来源获取
                if reference_image_path is None:
                    print(f"  🔍 调试：reference_image_path 为 None，尝试从其他来源获取")
                    # 优先级 0：恢复原始传入的 reference_image_path
                    if original_reference_image_path is not None:
                        reference_image_path = original_reference_image_path
                        print(f"  ✓ FLUX pipeline：恢复原始 reference_image_path: {reference_image_path.name if hasattr(reference_image_path, 'name') else reference_image_path}")
                    # 优先级 1：使用 face_reference_image_path
                    elif face_reference_image_path is not None:
                        reference_image_path = face_reference_image_path
                        print(f"  ✓ FLUX pipeline：使用 face_reference_image_path 作为 reference_image_path: {reference_image_path.name if hasattr(reference_image_path, 'name') else reference_image_path}")
                    # 优先级 2：从 scene 中获取 reference_image（如果存在）
                    elif scene and scene.get("reference_image"):
                        reference_image_path = Path(scene["reference_image"])
                        print(f"  ✓ FLUX pipeline：从 scene 获取 reference_image_path: {reference_image_path.name if hasattr(reference_image_path, 'name') else reference_image_path}")
                    # 优先级 3：从配置中获取 face_image_path
                    elif hasattr(self, 'face_image_path') and self.face_image_path and Path(self.face_image_path).exists():
                        reference_image_path = Path(self.face_image_path)
                        print(f"  ✓ FLUX pipeline：从配置获取 face_image_path 作为 reference_image_path: {reference_image_path.name}")
                    else:
                        print(f"  ⚠ FLUX pipeline：无法从任何来源获取 reference_image_path")
                else:
                    print(f"  ✓ FLUX pipeline：使用传入的 reference_image_path: {reference_image_path.name if hasattr(reference_image_path, 'name') else reference_image_path}")
                
                # 使用 Flux 专用的生成方法（支持 LoRA 和 IP-Adapter）
                # ⚡ 关键修复：传递 reference_image_path，让 FLUX 使用参考图
                print(f"  🔍 调试：调用 _generate_image_flux_simple，reference_image_path = {reference_image_path}")
                return self._generate_image_flux_simple(
                    prompt, output_path, negative_prompt, guidance_scale,
                    num_inference_steps, seed, scene=scene,
                    character_lora=character_lora, style_lora=style_lora,
                    reference_image_path=reference_image_path  # ⚡ 新增：传递参考图
                )
            else:
                # 使用 SDXL 生成方法
                # ⚡ 方案2：如果禁用了InstantID且是韩立角色，reference_image_path已在上面设置为人设图
                # 方案2使用IP-Adapter而不是img2img，所以不需要设置img2img_ref_image
                img2img_ref_image = scene_reference_image_for_img2img
                if primary_character == "hanli" and should_disable_instantid:
                    # 方案2：使用人设图作为IP-Adapter输入，不使用img2img
                    # reference_image_path已在上面设置为人设图
                    print(f"  ✓ 方案2：将使用人设图作为IP-Adapter输入（不使用img2img）")
                    # 不设置img2img_ref_image，因为方案2使用IP-Adapter
                    img2img_ref_image = None
                
                # ⚡ 修复：如果是方案2，使用Stage B设置的LoRA参数
                final_character_lora = character_lora
                final_style_lora = style_lora
                final_use_lora = use_lora
                
                if primary_character == "hanli" and should_disable_instantid:
                    # ⚡ 修复：使用Stage B设置的LoRA参数（如果已设置）
                    if 'stage_b_character_lora' in locals():
                        final_character_lora = stage_b_character_lora
                        final_style_lora = stage_b_style_lora
                        final_use_lora = stage_b_use_lora
                        print(f"  ✓ 方案2：使用Stage B的LoRA参数")
                    elif 'is_wide_topdown_lying' in locals() and is_wide_topdown_lying:
                        # ⚡ 关键修复：如果是 wide + top_down + lying 场景，即使没有进入 Stage B，也要禁用 LoRA
                        final_character_lora = None
                        final_style_lora = ""
                        final_use_lora = False
                        print(f"  ✓ wide + top_down + lying 场景：禁用所有 LoRA（即使没有进入 Stage B）")
                
                return self._generate_image_sdxl(
                    prompt, output_path, negative_prompt, guidance_scale,
                    num_inference_steps, seed, reference_image_path,
                    face_reference_image_path, final_use_lora, scene=scene,
                    use_ip_adapter_override=use_ip_adapter_for_this,  # 传递 IP-Adapter 使用标志
                    scene_reference_images=scene_reference_images,  # 传递已选择的场景参考图像
                    scene_reference_image_for_img2img=img2img_ref_image,  # 传递用于img2img的参考图像（优先使用韩立素材图）
                    character_lora=final_character_lora, style_lora=final_style_lora
                )

    def _generate_image_flux_simple(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        scene: Optional[Dict[str, Any]] = None,
        character_lora: Optional[str] = None,  # 角色LoRA适配器名称
        style_lora: Optional[str] = None,  # 风格LoRA适配器名称
        reference_image_path: Optional[Path] = None,  # ⚡ 新增：参考图路径（用于 IP-Adapter）
    ) -> Path:
        """使用 Flux pipeline 生成图像（支持 LoRA 和 IP-Adapter）"""
        # 确保使用 Flux pipeline，而不是 InstantID 或其他 pipeline
        flux_pipeline = None
        
        # 优先使用 flux1_pipeline（如果已加载）
        if self.flux1_pipeline is not None:
            flux_pipeline = self.flux1_pipeline
            print(f"  ℹ 使用 flux1_pipeline")
        # 其次检查 self.pipeline 是否为 Flux pipeline
        elif self.pipeline is not None:
            pipeline_type = type(self.pipeline).__name__
            if "Flux" in pipeline_type or "flux" in pipeline_type.lower():
                flux_pipeline = self.pipeline
                print(f"  ℹ 使用 self.pipeline (Flux)")
            else:
                # self.pipeline 不是 Flux，尝试加载 Flux.1
                print(f"  ⚠ self.pipeline 是 {pipeline_type}，不是 Flux pipeline，尝试加载 Flux.1...")
                try:
                    if self.flux1_pipeline is None:
                        self._load_flux1_pipeline()
                    flux_pipeline = self.flux1_pipeline
                    print(f"  ✓ 已加载 flux1_pipeline")
                except Exception as e:
                    print(f"  ❌ 无法加载 Flux.1 pipeline: {e}")
                    raise RuntimeError(f"Flux pipeline 未加载且无法加载: {e}")
        else:
            # 没有可用的 pipeline，尝试加载 Flux.1
            print(f"  ℹ 没有可用的 pipeline，尝试加载 Flux.1...")
            try:
                if self.flux1_pipeline is None:
                    self._load_flux1_pipeline()
                flux_pipeline = self.flux1_pipeline
                print(f"  ✓ 已加载 flux1_pipeline")
            except Exception as e:
                print(f"  ❌ 无法加载 Flux.1 pipeline: {e}")
                raise RuntimeError(f"Flux pipeline 未加载且无法加载: {e}")
        
        if flux_pipeline is None:
            raise RuntimeError("Flux pipeline 未加载")
        
        import torch
        from PIL import Image
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 使用配置的默认值（从 config.yaml 读取，如果没有则使用 Flux 默认值）
        # 优先使用传入的参数，其次使用配置文件中的值，最后使用 Flux 默认值
        if guidance_scale is None:
            # 尝试从配置中读取 Flux 的引导强度
            flux_config = self.image_config.get('model_selection', {}).get('character', {})
            guidance = flux_config.get('guidance_scale', 7.5)  # 默认 7.5（与配置一致）
        else:
            guidance = guidance_scale
        
        if num_inference_steps is None:
            # 尝试从配置中读取 Flux 的推理步数
            flux_config = self.image_config.get('model_selection', {}).get('character', {})
            steps = flux_config.get('num_inference_steps', 40)  # 默认 40 步（与配置一致）
        else:
            steps = num_inference_steps
        
        # ⚡ 关键修复：对于 wide shot 和包含复杂场景的情况，提高推理步数以提升质量
        camera = scene.get("camera", {}) if scene else {}
        camera_shot = camera.get("shot", "medium") if isinstance(camera, dict) else "medium"
        if camera_shot == "wide":
            # wide shot 需要更多步数来保证细节和质量
            if steps < 40:
                steps = 40
                print(f"  🔧 提高推理步数（wide shot）: {num_inference_steps} -> {steps}")
        
        print(f"  🎨 使用 Flux pipeline 生成图像")
        print(f"  提示词: {prompt[:50]}...")
        print(f"  完整提示词: {prompt}")
        print(f"  引导强度: {guidance}")
        print(f"  推理步数: {steps}")
        if character_lora:
            print(f"  ⚠ 警告: 检测到 character_lora={character_lora}，这可能导致生成人物图像而非场景图像")
        if negative_prompt:
            print(f"  负面提示词: {negative_prompt[:100]}...")
        
        # 从 scene 获取尺寸（如果有）
        # ⚡ 关键修复：优先从 generation_params 读取分辨率（v2.2-final格式）
        width = self.width
        height = self.height
        if scene and isinstance(scene, dict):
            # 优先从 generation_params 读取（v2.2-final格式）
            gen_params = scene.get("generation_params", {})
            if gen_params and isinstance(gen_params, dict):
                width = gen_params.get("width", width)
                height = gen_params.get("height", height)
            else:
                # 向后兼容：直接从 scene 读取（旧格式）
                width = scene.get("width", width)
                height = scene.get("height", height)
        
        # ⚡ 关键修复：对于 wide shot 和 top_down 场景，确保分辨率足够高
        # 低分辨率会导致图像模糊，特别是对于包含人物和环境的场景
        camera = scene.get("camera", {}) if scene else {}
        camera_shot = camera.get("shot", "medium") if isinstance(camera, dict) else "medium"
        
        # 对于 wide shot，至少需要 1024x1024，推荐 1536x1536 或更高
        if camera_shot == "wide":
            min_width = 1536
            min_height = 1536
            if width < min_width:
                width = min_width
                print(f"  🔧 提高分辨率（wide shot）: width -> {width}")
            if height < min_height:
                height = min_height
                print(f"  🔧 提高分辨率（wide shot）: height -> {height}")
        
        # 确保分辨率是 64 的倍数（FLUX 的要求）
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        print(f"  📐 图像分辨率: {width}x{height}")
        
        # MVP 流程：将 character_lora 名称解析为 lora_path
        lora_path = None
        # 对于科普主持人，优先从 model_selection.character.lora.alpha 读取
        lora_alpha = self.lora_alpha  # 默认使用全局配置
        if character_lora in ["kepu_gege", "weilai_jiejie", "host_person_v2"]:
            # 从 model_selection.character.lora.alpha 读取（如果存在）
            model_selection = self.image_config.get('model_selection', {})
            character_config = model_selection.get('character', {})
            character_lora_config = character_config.get('lora', {})
            if 'alpha' in character_lora_config:
                lora_alpha = float(character_lora_config['alpha'])
                print(f"  ℹ 使用科普主持人专用 LoRA alpha: {lora_alpha}（从 model_selection.character.lora.alpha 读取）")
        
        if character_lora:
            # 将 character_lora 名称解析为实际文件路径
            if character_lora == "host_person_v2":
                lora_path = str(self.models_root / "lora" / "host_person_v2" / "pytorch_lora_weights.safetensors")
            elif character_lora == "host_person":
                lora_path = str(self.models_root / "lora" / "host_person" / "pytorch_lora_weights.safetensors")
            elif character_lora in ["kepu_gege", "weilai_jiejie"]:
                # 科普主持人映射到 host_person_v2
                lora_path = str(self.models_root / "lora" / "host_person_v2" / "pytorch_lora_weights.safetensors")
            else:
                # 尝试作为目录名或路径
                lora_path_obj = Path(character_lora)
                if not lora_path_obj.is_absolute():
                    lora_path_obj = self.models_root / "lora" / character_lora / "pytorch_lora_weights.safetensors"
                if lora_path_obj.exists():
                    lora_path = str(lora_path_obj)
            
            if lora_path and Path(lora_path).exists():
                print(f"  🔧 将使用 LoRA: {Path(lora_path).name} (alpha={lora_alpha})")
            else:
                print(f"  ⚠ LoRA 路径不存在: {character_lora}，将不使用 LoRA")
                lora_path = None
        
        # ⚡ 关键修复：处理参考图（用于 IP-Adapter）
        ip_adapter_image = None
        print(f"  🔍 调试：_generate_image_flux_simple 接收到的 reference_image_path = {reference_image_path}")
        print(f"  🔍 调试：_generate_image_flux_simple，reference_image_path 类型 = {type(reference_image_path)}")
        if reference_image_path:
            reference_path = Path(reference_image_path)
            print(f"  🔍 调试：reference_path = {reference_path}, exists() = {reference_path.exists()}")
            if reference_path.exists():
                try:
                    from PIL import Image
                    ip_adapter_image = Image.open(reference_path).convert("RGB")
                    print(f"  ✓ 已加载参考图用于 IP-Adapter: {reference_path.name}")
                    # 调整图像尺寸（FLUX IP-Adapter 推荐 1024x1024）
                    w, h = ip_adapter_image.size
                    min_size = 1024
                    if min(w, h) < min_size:
                        scale = min_size / min(w, h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        # 确保是 64 的倍数（Flux 的要求）
                        new_w = (new_w // 64) * 64
                        new_h = (new_h // 64) * 64
                        ip_adapter_image = ip_adapter_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        print(f"  ✓ 已调整参考图尺寸: {w}x{h} -> {new_w}x{new_h}")
                    else:
                        # 确保是 64 的倍数
                        new_w = (w // 64) * 64
                        new_h = (h // 64) * 64
                        if new_w != w or new_h != h:
                            ip_adapter_image = ip_adapter_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            print(f"  ✓ 已调整参考图尺寸到 64 的倍数: {w}x{h} -> {new_w}x{new_h}")
                except Exception as e:
                    print(f"  ⚠ 加载参考图失败: {e}，将不使用 IP-Adapter")
                    import traceback
                    traceback.print_exc()
                    ip_adapter_image = None
            else:
                print(f"  ⚠ 参考图路径不存在: {reference_path}")
        else:
            print(f"  ⚠ reference_image_path 为 None，将不使用 IP-Adapter")
        
        try:
            # 检查 pipeline 是否有 generate 方法（FluxPipeline）
            if hasattr(flux_pipeline, 'generate'):
                # 使用 FluxPipeline.generate 方法（支持 LoRA 和 IP-Adapter）
                generate_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "seed": seed,
                    "lora_path": lora_path,
                    "lora_alpha": lora_alpha,
                }
                # ⚡ 关键修复：如果提供了参考图，使用 IP-Adapter
                if ip_adapter_image is not None:
                    # 检查 pipeline 是否已加载 IP-Adapter
                    ip_adapter_loaded = False
                    if hasattr(flux_pipeline, 'ip_adapter_image_encoder') or hasattr(flux_pipeline, '_ip_adapter'):
                        ip_adapter_loaded = True
                        print(f"  ✓ 检测到 IP-Adapter 已加载")
                    else:
                        # 尝试加载 IP-Adapter
                        try:
                            print(f"  🔧 尝试加载 FLUX IP-Adapter...")
                            # FLUX 使用标准的 load_ip_adapter 方法
                            # 需要指定 IP-Adapter 路径（从配置中读取或使用默认路径）
                            ip_adapter_path = self.image_config.get("model_selection", {}).get("flux", {}).get("ip_adapter_path")
                            
                            # ⚡ 关键修复：优先使用本地路径，避免从 HuggingFace 下载
                            if not ip_adapter_path:
                                # 尝试多个可能的本地路径（按优先级排序）
                                possible_paths = [
                                    Path(self.models_root) / "instantid" / "ip-adapter-flux",  # 优先级 1：已找到的路径
                                    Path(self.models_root) / "ip-adapter" / "flux-ip-adapter",
                                    Path(self.models_root) / "ip-adapter",
                                    Path(__file__).parent / "models" / "instantid" / "ip-adapter-flux",
                                    Path(__file__).parent / "models" / "ip-adapter" / "flux-ip-adapter",
                                ]
                                for possible_path in possible_paths:
                                    if possible_path.exists():
                                        # 检查是否是目录（需要包含 ip_adapter.safetensors 文件）
                                        if possible_path.is_dir():
                                            # 检查目录中是否有 ip_adapter.safetensors 文件
                                            ip_adapter_file = possible_path / "ip_adapter.safetensors"
                                            if ip_adapter_file.exists():
                                                ip_adapter_path = str(possible_path)
                                                print(f"  ✓ 找到本地 IP-Adapter 路径: {ip_adapter_path}")
                                                break
                                        elif possible_path.is_file():
                                            # 如果是文件，使用父目录
                                            ip_adapter_path = str(possible_path.parent)
                                            print(f"  ✓ 找到本地 IP-Adapter 路径: {ip_adapter_path}")
                                            break
                            
                            if not ip_adapter_path:
                                # 如果都没有找到，使用 HuggingFace 模型 ID（会下载）
                                ip_adapter_path = "XLabs-AI/flux-ip-adapter"
                                print(f"  ⚠ 未找到本地 IP-Adapter，将尝试从 HuggingFace 下载: {ip_adapter_path}")
                                print(f"  ⚠ 注意：首次下载可能需要较长时间，请耐心等待...")
                            
                            # 检查是否是本地路径
                            ip_adapter_path_obj = Path(ip_adapter_path) if ip_adapter_path else None
                            if ip_adapter_path_obj and ip_adapter_path_obj.exists():
                                # 本地路径
                                print(f"  📂 使用本地 IP-Adapter 路径: {ip_adapter_path}")
                                print(f"  📂 路径详情: {ip_adapter_path_obj.absolute()}")
                                # 检查文件是否存在
                                ip_adapter_file = ip_adapter_path_obj / "ip_adapter.safetensors"
                                if not ip_adapter_file.exists():
                                    print(f"  ⚠ 警告：未找到 ip_adapter.safetensors 文件，尝试直接使用目录...")
                                flux_pipeline.load_ip_adapter(
                                    str(ip_adapter_path_obj),
                                    weight_name="ip_adapter.safetensors",
                                    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
                                )
                            else:
                                # HuggingFace 模型 ID（会下载）
                                print(f"  🌐 从 HuggingFace 加载 IP-Adapter: {ip_adapter_path}")
                                print(f"  ⚠ 注意：如果卡住，可能正在下载模型（首次使用需要下载，约 937MB）...")
                                print(f"  ⚠ 建议：如果下载太慢，可以手动下载到: {self.models_root}/instantid/ip-adapter-flux/")
                                flux_pipeline.load_ip_adapter(
                                    ip_adapter_path,
                                    weight_name="ip_adapter.safetensors",
                                    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
                                )
                            ip_adapter_loaded = True
                            print(f"  ✅ FLUX IP-Adapter 加载成功")
                            
                            # ⚡ 关键修复：确保 image encoder 在正确的设备上
                            if hasattr(flux_pipeline, 'image_encoder') and flux_pipeline.image_encoder is not None:
                                import torch
                                # 获取 pipeline 的设备（从 transformer 或其他组件）
                                if hasattr(flux_pipeline, '_execution_device'):
                                    device = flux_pipeline._execution_device
                                elif hasattr(flux_pipeline, 'device'):
                                    device = flux_pipeline.device
                                elif hasattr(flux_pipeline, 'transformer') and hasattr(flux_pipeline.transformer, 'device'):
                                    device = flux_pipeline.transformer.device
                                else:
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                
                                # 获取 pipeline 的 dtype（从 transformer 获取）
                                if hasattr(flux_pipeline, 'transformer') and next(flux_pipeline.transformer.parameters(), None) is not None:
                                    pipeline_dtype = next(flux_pipeline.transformer.parameters()).dtype
                                else:
                                    pipeline_dtype = torch.float16
                                
                                # 确保 image encoder 在正确的设备和 dtype
                                current_device = next(flux_pipeline.image_encoder.parameters()).device
                                current_dtype = next(flux_pipeline.image_encoder.parameters()).dtype
                                
                                if current_device != device or current_dtype != pipeline_dtype:
                                    print(f"  🔧 移动 image encoder: {current_device}/{current_dtype} -> {device}/{pipeline_dtype}")
                                    flux_pipeline.image_encoder = flux_pipeline.image_encoder.to(device=device, dtype=pipeline_dtype)
                                    print(f"  ✓ 已确保 image encoder 在设备: {device}, dtype: {pipeline_dtype}")
                                else:
                                    print(f"  ✓ image encoder 已在正确的设备: {device}, dtype: {pipeline_dtype}")
                        except Exception as e:
                            print(f"  ⚠ FLUX IP-Adapter 加载失败: {e}")
                            import traceback
                            traceback.print_exc()
                            print(f"  ℹ 将不使用 IP-Adapter，仅使用 prompt 生成")
                    
                    if ip_adapter_loaded:
                        # ⚡ 关键修复：对于 wide shot，降低 IP-Adapter scale 以避免过度融合导致模糊
                        # 对于 wide shot，参考图主要用于风格和形象参考，不需要过强的融合
                        camera = scene.get("camera", {}) if scene else {}
                        camera_shot = camera.get("shot", "medium") if isinstance(camera, dict) else "medium"
                        
                        if camera_shot == "wide":
                            ip_adapter_scale = 0.9  # wide shot 使用较低的 scale，避免过度融合
                            print(f"  🔧 wide shot：使用较低的 IP-Adapter scale: {ip_adapter_scale}（避免过度融合导致模糊）")
                        else:
                            ip_adapter_scale = 1.2  # 其他场景使用较高的 scale
                        
                        if hasattr(flux_pipeline, 'set_ip_adapter_scale'):
                            flux_pipeline.set_ip_adapter_scale(ip_adapter_scale)
                            print(f"  ✓ 已设置 IP-Adapter scale: {ip_adapter_scale}")
                        
                        # ⚡ 关键修复：在调用 pipeline 之前，再次确保 image_encoder 在正确的设备上
                        if hasattr(flux_pipeline, 'image_encoder') and flux_pipeline.image_encoder is not None:
                            import torch
                            # 获取 pipeline 的执行设备
                            if hasattr(flux_pipeline, '_execution_device'):
                                target_device = flux_pipeline._execution_device
                            elif hasattr(flux_pipeline, 'device'):
                                target_device = flux_pipeline.device
                            else:
                                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            
                            # 获取 pipeline 的 dtype
                            if hasattr(flux_pipeline, 'transformer') and next(flux_pipeline.transformer.parameters(), None) is not None:
                                target_dtype = next(flux_pipeline.transformer.parameters()).dtype
                            else:
                                target_dtype = torch.float16
                            
                            # 检查 image_encoder 的当前设备
                            encoder_device = next(flux_pipeline.image_encoder.parameters()).device
                            encoder_dtype = next(flux_pipeline.image_encoder.parameters()).dtype
                            
                            if encoder_device != target_device or encoder_dtype != target_dtype:
                                print(f"  🔧 在生成前移动 image encoder: {encoder_device}/{encoder_dtype} -> {target_device}/{target_dtype}")
                                flux_pipeline.image_encoder = flux_pipeline.image_encoder.to(device=target_device, dtype=target_dtype)
                                print(f"  ✓ image encoder 已移动到: {target_device}, dtype: {target_dtype}")
                            else:
                                print(f"  ✓ image encoder 已在正确设备: {target_device}, dtype: {target_dtype}")
                        
                        generate_kwargs["ip_adapter_image"] = ip_adapter_image
                        print(f"  🎯 使用 IP-Adapter 生成图像（参考图: {reference_image_path.name if hasattr(reference_image_path, 'name') else reference_image_path}）")
                    else:
                        print(f"  ⚠ IP-Adapter 未加载，将不使用参考图")
                
                image = flux_pipeline.generate(**generate_kwargs)
            else:
                # 使用标准的 pipeline 调用（支持 IP-Adapter）
                pipeline_kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "generator": generator,
                }
                # ⚡ 关键修复：如果提供了参考图，使用 IP-Adapter（与上面相同的逻辑）
                if ip_adapter_image is not None:
                    # 检查 pipeline 是否已加载 IP-Adapter
                    ip_adapter_loaded = False
                    if hasattr(flux_pipeline, 'ip_adapter_image_encoder') or hasattr(flux_pipeline, '_ip_adapter'):
                        ip_adapter_loaded = True
                        print(f"  ✓ 检测到 IP-Adapter 已加载")
                    else:
                        # 尝试加载 IP-Adapter（与上面相同的逻辑）
                        try:
                            print(f"  🔧 尝试加载 FLUX IP-Adapter...")
                            ip_adapter_path = self.image_config.get("model_selection", {}).get("flux", {}).get("ip_adapter_path")
                            if not ip_adapter_path:
                                ip_adapter_path = "XLabs-AI/flux-ip-adapter"
                            
                            if isinstance(ip_adapter_path, str) and Path(ip_adapter_path).exists():
                                flux_pipeline.load_ip_adapter(
                                    ip_adapter_path,
                                    weight_name="ip_adapter.safetensors",
                                    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
                                )
                            else:
                                flux_pipeline.load_ip_adapter(
                                    ip_adapter_path,
                                    weight_name="ip_adapter.safetensors",
                                    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
                                )
                            ip_adapter_loaded = True
                            print(f"  ✅ FLUX IP-Adapter 加载成功")
                        except Exception as e:
                            print(f"  ⚠ FLUX IP-Adapter 加载失败: {e}")
                            print(f"  ℹ 将不使用 IP-Adapter，仅使用 prompt 生成")
                    
                    if ip_adapter_loaded:
                        # ⚡ 关键修复：对于 wide shot，降低 IP-Adapter scale 以避免过度融合导致模糊
                        camera = scene.get("camera", {}) if scene else {}
                        camera_shot = camera.get("shot", "medium") if isinstance(camera, dict) else "medium"
                        
                        if camera_shot == "wide":
                            ip_adapter_scale = 0.9  # wide shot 使用较低的 scale，避免过度融合
                            print(f"  🔧 wide shot：使用较低的 IP-Adapter scale: {ip_adapter_scale}（避免过度融合导致模糊）")
                        else:
                            ip_adapter_scale = 1.2  # 其他场景使用较高的 scale
                        
                        if hasattr(flux_pipeline, 'set_ip_adapter_scale'):
                            flux_pipeline.set_ip_adapter_scale(ip_adapter_scale)
                            print(f"  ✓ 已设置 IP-Adapter scale: {ip_adapter_scale}")
                        
                        # ⚡ 关键修复：在调用 pipeline 之前，再次确保 image_encoder 在正确的设备上
                        if hasattr(flux_pipeline, 'image_encoder') and flux_pipeline.image_encoder is not None:
                            import torch
                            # 获取 pipeline 的执行设备
                            if hasattr(flux_pipeline, '_execution_device'):
                                target_device = flux_pipeline._execution_device
                            elif hasattr(flux_pipeline, 'device'):
                                target_device = flux_pipeline.device
                            else:
                                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            
                            # 获取 pipeline 的 dtype
                            if hasattr(flux_pipeline, 'transformer') and next(flux_pipeline.transformer.parameters(), None) is not None:
                                target_dtype = next(flux_pipeline.transformer.parameters()).dtype
                            else:
                                target_dtype = torch.float16
                            
                            # 检查 image_encoder 的当前设备
                            encoder_device = next(flux_pipeline.image_encoder.parameters()).device
                            encoder_dtype = next(flux_pipeline.image_encoder.parameters()).dtype
                            
                            if encoder_device != target_device or encoder_dtype != target_dtype:
                                print(f"  🔧 在生成前移动 image encoder: {encoder_device}/{encoder_dtype} -> {target_device}/{target_dtype}")
                                flux_pipeline.image_encoder = flux_pipeline.image_encoder.to(device=target_device, dtype=target_dtype)
                                print(f"  ✓ image encoder 已移动到: {target_device}, dtype: {target_dtype}")
                            else:
                                print(f"  ✓ image encoder 已在正确设备: {target_device}, dtype: {target_dtype}")
                        
                        pipeline_kwargs["ip_adapter_image"] = ip_adapter_image
                        print(f"  🎯 使用 IP-Adapter 生成图像（参考图: {reference_image_path.name if hasattr(reference_image_path, 'name') else reference_image_path}）")
                    else:
                        print(f"  ⚠ IP-Adapter 未加载，将不使用参考图")
                
                result = flux_pipeline(**pipeline_kwargs)
                image = result.images[0]
            
            image.save(output_path)
            print(f"  ✅ Flux 图像生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  ❌ Flux 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_image_flux1(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        scene: Optional[Dict[str, Any]] = None,
        character_lora: Optional[str] = None,  # 角色LoRA适配器名称
        style_lora: Optional[str] = None,  # 风格LoRA适配器名称
        reference_image_path: Optional[Path] = None,  # ⚡ 新增：参考图路径（用于 IP-Adapter）
        face_reference_image_path: Optional[Path] = None,  # ⚡ 新增：面部参考图路径
    ) -> Path:
        """使用 Flux.1 pipeline 生成图像（实验室/医学场景）"""
        # 如果 flux1_pipeline 未加载，尝试加载
        if self.flux1_pipeline is None:
            print("  ℹ️  Flux.1 pipeline 未加载，正在加载...")
            try:
                self._load_flux1_pipeline()
            except Exception as e:
                print(f"  ⚠️  加载 Flux.1 pipeline 失败: {e}")
                # 如果加载失败，尝试使用主 pipeline
                if self.pipeline is None:
                    raise RuntimeError("Flux.1 pipeline 未加载，且主 pipeline 也未加载")
                print("  ℹ️  使用主 pipeline 作为备选")
                pipeline = self.pipeline
            else:
                pipeline = self.flux1_pipeline
        else:
            pipeline = self.flux1_pipeline
        
        # 临时切换到 flux1_pipeline
        original_pipeline = self.pipeline
        self.pipeline = pipeline
        
        try:
            # ⚡ 关键修复：如果 reference_image_path 为 None，尝试使用 face_reference_image_path
            final_reference_image_path = reference_image_path
            if final_reference_image_path is None and face_reference_image_path is not None:
                final_reference_image_path = face_reference_image_path
                print(f"  ✓ _generate_image_flux1：使用 face_reference_image_path 作为 reference_image_path: {final_reference_image_path.name if hasattr(final_reference_image_path, 'name') else final_reference_image_path}")
            
            return self._generate_image_flux_simple(
                prompt=prompt,
                output_path=output_path,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                scene=scene,
                character_lora=character_lora,  # 传递 character_lora
                style_lora=style_lora,  # 传递 style_lora
                reference_image_path=final_reference_image_path,  # ⚡ 关键修复：传递参考图
            )
        finally:
            # 恢复原始 pipeline
            self.pipeline = original_pipeline
    
    def _generate_image_flux2(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        scene: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """使用 Flux.2 pipeline 生成图像（科学背景图、太空/粒子/量子类）"""
        # 如果 flux2_pipeline 未加载，尝试加载
        if self.flux2_pipeline is None:
            print("  ℹ️  Flux.2 pipeline 未加载，正在加载...")
            try:
                self._load_flux2_pipeline()
            except Exception as e:
                print(f"  ⚠️  加载 Flux.2 pipeline 失败: {e}")
                # 如果加载失败，尝试使用主 pipeline
                if self.pipeline is None:
                    raise RuntimeError("Flux.2 pipeline 未加载，且主 pipeline 也未加载")
                print("  ℹ️  使用主 pipeline 作为备选")
                pipeline = self.pipeline
            else:
                pipeline = self.flux2_pipeline
        else:
            pipeline = self.flux2_pipeline
        
        # 临时切换到 flux2_pipeline
        original_pipeline = self.pipeline
        self.pipeline = pipeline
        
        try:
            return self._generate_image_flux_simple(
                prompt=prompt,
                output_path=output_path,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                scene=scene
            )
        finally:
            # 恢复原始 pipeline
            self.pipeline = original_pipeline

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
        character_lora: Optional[str] = None,  # 角色LoRA适配器名称
        style_lora: Optional[str] = None,  # 风格LoRA适配器名称
    ) -> Path:
        """InstantID 图像生成

        注意：如果 prompt 中没有明确的远景关键词，会自动在开头添加，
        确保镜头足够远，避免只看到头部。
        """
        """使用 InstantID 生成图像"""
        if self.pipeline is None:
            # 自动加载pipeline（如果未加载）
            print("⚠️  Pipeline未加载，正在自动加载...")
            self.load_pipeline()

        from PIL import Image

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # 使用 InstantID 配置
        guidance = guidance_scale or self.instantid_config.get(
            "guidance_scale", 7.5)
        base_steps = num_inference_steps or self.instantid_config.get(
            "num_inference_steps", 40)  # 基础步数（使用 DPM++ 采样器时）

        # 检测是否是远景场景（提前检测，用于调整推理步数）
        prompt_lower = prompt.lower() if prompt else ""
        is_wide_shot_early = any(
            kw in prompt_lower for kw in [
                "wide shot",
                "full body",
                "full figure",
                "全身",
                "wide view",
                "full view",
                "long shot",
                "extreme wide",
                "distant view",
                "far away",
                "establishing shot"])
        is_full_body_early = any(
            kw in prompt_lower for kw in [
                "full body", "full figure", "全身"])
        # 确保 scene 是字典类型，避免字符串或其他类型导致的错误
        if scene and isinstance(scene, dict):
            visual = scene.get("visual", {})
            if isinstance(visual, dict):
                composition = visual.get("composition", {})
                if isinstance(composition, dict):
                    scene_camera = composition.get(
                        "camera", "").lower() if isinstance(
                        composition.get("camera"), str) else ""
                    if not is_wide_shot_early:
                        is_wide_shot_early = any(
                            kw in scene_camera for kw in [
                                "wide shot",
                                "full body",
                                "full figure",
                                "全身",
                                "wide view",
                                "full view",
                                "long shot",
                                "extreme wide",
                                "distant view"])
                    if not is_full_body_early:
                        is_full_body_early = any(
                            kw in scene_camera for kw in [
                                "full body", "full figure", "全身"])

        # 如果是远景场景，增加推理步数以提高清晰度
        if is_wide_shot_early or is_full_body_early:
            # 增加25%的步数（40步 -> 50步，50步 -> 62.5步约63步）
            steps = int(base_steps * 1.25)
            print(f"  ℹ 远景场景：推理步数从 {base_steps} 增加到 {steps}，提高清晰度")
        else:
            steps = base_steps

        guidance_rescale = self.instantid_config.get(
            "guidance_rescale", None)  # CFG Rescale（可选）
        print(
            f"  图像生成参数: num_inference_steps={steps}, guidance_scale={guidance}",
            end="")
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
        if face_reference_image_path and Path(
                face_reference_image_path).exists():
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
                            face_image = Image.open(
                                preferred_path).convert("RGB")
                            used_reference_path = preferred_path
                            print(f"  ✓ 使用优先面部参考图像: {preferred_path}")
                            if '半身' in name or 'half_body' in name.lower():
                                print(f"    (半身照：包含发型和姿势信息)")
                            break

                    # 如果没找到优先图片，使用目录中的第一张
                    if not face_image:
                        face_images = sorted(face_path.glob(
                            "*.png")) + sorted(face_path.glob("*.jpg"))
                        if face_images:
                            face_image = Image.open(
                                face_images[0]).convert("RGB")
                            used_reference_path = face_images[0]
                            print(f"  ✓ 使用目录中的第一张图片: {face_images[0]}")
                elif face_path.is_file():
                    face_image = Image.open(face_path).convert("RGB")
                    used_reference_path = face_path
                    print(f"  ✓ 使用配置文件中的参考图像: {face_path}")

        if not face_image:
            raise ValueError(
                "InstantID 需要面部参考图像，请提供 face_reference_image_path 或配置 face_image_path")

        # 打印使用的参考图像信息（用于调试）
        if used_reference_path:
            print(f"  📸 InstantID 参考图像: {used_reference_path}")
            print(f"     图像尺寸: {face_image.size[0]}x{face_image.size[1]}")

        # 提取面部特征和关键点（InstantID 必需）
        if not hasattr(self, 'face_analysis') or self.face_analysis is None:
            raise RuntimeError("FaceAnalysis 未初始化，无法提取面部特征")

        # 调整图像大小（使用 InstantID 官方的 resize_img 函数，保持宽高比）
        def resize_img(
                input_image,
                max_side=1280,
                min_side=1024,
                size=None,
                pad_to_max_side=False,
                mode=Image.BILINEAR,
                base_pixel_number=64):
            """InstantID 官方的 resize_img 函数，保持宽高比"""
            w, h = input_image.size
            if size is not None:
                w_resize_new, h_resize_new = size
            else:
                ratio = min_side / min(h, w)
                w, h = round(ratio * w), round(ratio * h)
                ratio = max_side / max(h, w)
                input_image = input_image.resize(
                    [round(ratio * w), round(ratio * h)], mode)
                w_resize_new = (round(ratio * w) //
                                base_pixel_number) * base_pixel_number
                h_resize_new = (round(ratio * h) //
                                base_pixel_number) * base_pixel_number
            input_image = input_image.resize(
                [w_resize_new, h_resize_new], mode)

            if pad_to_max_side:
                res = self.np.ones([max_side, max_side, 3],
                                   dtype=self.np.uint8) * 255
                offset_x = (max_side - w_resize_new) // 2
                offset_y = (max_side - h_resize_new) // 2
                res[offset_y:offset_y +
                    h_resize_new, offset_x:offset_x +
                    w_resize_new] = self.np.array(input_image)
                input_image = Image.fromarray(res)
            return input_image

        # 调整面部参考图像大小（保持宽高比，不拉伸）
        face_image = resize_img(face_image, max_side=1280, min_side=1024)

        # 提取面部信息
        face_info_list = self.face_analysis.get(
            self.cv2.cvtColor(
                self.np.array(face_image),
                self.cv2.COLOR_RGB2BGR))
        if not face_info_list:
            # 如果无法识别人脸，抛出异常，让调用者决定是否回退到文生图
            raise ValueError("在参考图像中未检测到人脸，请使用包含清晰人脸的图像")

        # 选择最大的人脸
        face_info = sorted(face_info_list, key=lambda x: (
            x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        face_emb = face_info['embedding']

        # 生成面部关键点图像
        if hasattr(self, 'draw_kps'):
            draw_kps = self.draw_kps
        else:
            # 如果 draw_kps 未保存，尝试导入
            try:
                instantid_repo_path = Path(
                    __file__).parent.parent / "InstantID"
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
        # 检查场景是否包含角色（只检查明确的角色关键词，不包括通用词如"person"、"people"）
        # 注意：对于通用场景（如街景），不应该添加角色约束
        has_character = False
        if scene:
            # 检查scene中是否有characters字段
            if scene.get("characters"):
                has_character = True
            else:
                # 只检查明确的角色关键词，不包括通用词
                character_keywords = [
                    "han li", "hanli", "cultivator", "character", "韩立", "主角",
                    # 不包括 "person", "figure", "man", "people" 等通用词
                ]
                prompt_lower = prompt.lower()
                has_character = any(kw in prompt_lower for kw in character_keywords)
        
        # 只有当明确检测到角色时才添加单人约束（避免对通用场景添加不必要的约束）
        if has_character:
            prompt_lower_check = prompt.lower()
            single_person_keywords = [
                "single person",
                "lone figure",
                "one person",
                "only one",
                "sole character"]
            has_single_keyword = any(
                kw in prompt_lower_check for kw in single_person_keywords)
            
            if not has_single_keyword:
                # 在 prompt 最前面强调单人，使用最高权重（2.0，用户反馈场景5和7生成了多个人物）
                # 但先检查token数，避免过长
                test_prompt = "(single person, lone figure, only one character, one person only, sole character, single individual:2.0), " + prompt
                try:
                    from transformers import CLIPTokenizer
                    from pathlib import Path
                    # ⚡ 关键修复：优先使用本地SDXL模型，避免网络下载
                    local_sdxl_path = Path(__file__).parent / "models" / "sdxl-base"
                    tokenizer = None
                    
                    # 尝试从本地SDXL模型加载
                    if local_sdxl_path.exists() and (local_sdxl_path / "tokenizer").exists():
                        try:
                            tokenizer = CLIPTokenizer.from_pretrained(
                                str(local_sdxl_path),
                                subfolder="tokenizer",
                                local_files_only=True
                            )
                        except Exception:
                            pass
                    
                    # 如果本地加载失败，尝试使用缓存
                    if tokenizer is None:
                        try:
                            tokenizer = CLIPTokenizer.from_pretrained(
                                "openai/clip-vit-large-patch14",
                                local_files_only=True  # 只使用本地缓存，不联网下载
                            )
                        except Exception:
                            pass
                    
                    if tokenizer:
                        test_tokens = len(tokenizer(test_prompt, truncation=False, return_tensors="pt").input_ids[0])
                    else:
                        # 如果无法加载tokenizer，保守处理：不添加
                        print(f"  ⚠ 无法加载tokenizer，跳过添加单人约束以避免截断")
                        return prompt
                    if test_tokens <= 70:  # 留出安全边界
                        prompt = test_prompt
                        print(f"  ✓ 已添加单人场景约束：在 prompt 最前面强调单人（权重2.0，防止重复人物）")
                    else:
                        print(f"  ⚠ 提示词已较长，跳过添加单人约束以避免超过token限制")
                except Exception:
                    # 如果无法计算token，保守处理：不添加
                    print(f"  ⚠ 无法计算token数，跳过添加单人约束以避免截断")

        # 检查 prompt 中是否已有明确的镜头描述（来自 camera 字段或其他来源）
        prompt_lower = prompt.lower()
        wide_shot_keywords = [
            'extreme wide shot', 'very long shot', 'distant view', 'far away',
            'wide shot', 'establishing shot', 'wide angle', 'landscape',
            '远景', '全景', '广角', 'long shot', 'extreme long shot'
        ]
        # 检查是否有任何镜头类型的关键词（包括远景、中景、近景、特写等）
        shot_type_keywords = [
            'extreme wide shot',
            'very long shot',
            'long shot',
            'wide shot',
            'medium shot',
            'mid shot',
            '中景',
            'close-up',
            'closeup',
            'close up',
            'portrait',
            'headshot',
            '特写',
            '近景',
            'full body',
            'full figure',
            '全身',
            'aerial view',
            'low angle',
            'side view',
            'back view',
            'front view']
        has_any_shot_keyword = any(
            keyword in prompt_lower for keyword in shot_type_keywords)
        has_wide_keyword = any(
            keyword in prompt_lower for keyword in wide_shot_keywords)

        # 检查 camera 字段是否明确指定了特写（特别是眼睛特写）
        camera_desc = self._get_camera_string(scene) if scene else ""
        camera_desc_lower = (camera_desc or "").lower()
        is_eye_closeup_in_camera = any(
            kw in camera_desc_lower for kw in [
                'eye',
                'eyes',
                'pupil',
                'pupils',
                '眼睛',
                '瞳孔',
                'extreme close',
                'close-up',
                'closeup'])
        has_closeup_in_camera = any(
            kw in camera_desc_lower for kw in [
                'close-up', 'closeup', 'close up', '特写', '近景'])

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
            has_character_in_scene = scene and (
                scene.get("characters") or any(
                    kw in prompt.lower() for kw in [
                        "han li",
                        "hanli",
                        "cultivator",
                        "character",
                        "韩立",
                        "主角",
                        "person",
                        "figure",
                        "man",
                        "people"]))
            if has_character_in_scene:
                # ⚡ 重要：检查是否是躺着姿势或Top-down场景，如果是则使用远景而不是中景
                is_lying_or_topdown = (
                    "lying" in prompt_lower or
                    "top-down" in prompt_lower or
                    "top down" in prompt_lower or
                    "俯视" in prompt_lower or
                    (scene and (
                        "lying" in str(scene.get("action", "")).lower() or
                        "top-down" in self._get_camera_string(scene).lower() or
                        "top down" in self._get_camera_string(scene).lower() or
                        "俯视" in self._get_camera_string(scene).lower()
                    ))
                )
                if is_lying_or_topdown:
                    # 躺着姿势或Top-down场景：使用远景，确保能看到全身和背景
                    # ⚡ 使用通用的prompt增强方法（基于语义分析，而不是硬编码）
                    camera_prompt = "(wide shot, top-down view, bird's eye view, distant view, full body visible, lying on ground:2.8)"
                    # 使用optimizer的通用增强方法自动添加排除词等
                    enhanced_camera = self.prompt_optimizer.enhance_prompt_part(camera_prompt, "camera")
                    prompt = f"{enhanced_camera}, {prompt}"
                    print(f"  ✓ 检测到躺着姿势或Top-down场景，使用远景描述（已增强），确保能看到全身和背景")
                else:
                    # 其他人物场景：默认中景，确保人物清晰可见且正面
                    prompt = "(medium shot, character clearly visible, front view, facing camera:2.0), " + prompt
                    print(f"  ⚠ 未检测到任何镜头关键词，人物场景默认添加中景描述（高权重2.0），避免远景和背影")
            else:
                # 无人物场景：可以使用远景
                prompt = "(extreme wide shot:2.0), (distant view:1.8), " + prompt
                is_lying_or_topdown = False  # 无人物场景不需要检查 lying/topdown
            if not is_lying_or_topdown:
                print(f"  ⚠ 未检测到任何镜头关键词，已强制在 prompt 开头添加远景描述（高权重）")
        else:
            # 检查远景关键词是否在 prompt 开头（前 100 个字符）
            prompt_start = prompt_lower[:100]
            has_wide_at_start = any(
                keyword in prompt_start for keyword in wide_shot_keywords)
            if not has_wide_at_start:
                # 如果有远景关键词但不在开头，在开头添加带更高权重的远景关键词
                found_keywords = [
                    kw for kw in wide_shot_keywords if kw in prompt_lower]
                if found_keywords:
                    primary_keyword = found_keywords[0]
                    # 使用更高权重标记增强效果（2.0 表示 2 倍权重）
                    prompt = f"({primary_keyword}:2.0), " + prompt
                    print(f"  ✓ 检测到远景关键词但不在开头，已移至开头并增强权重（2.0倍）")

        # ⚡ 关键修复：根据引擎类型选择正确的 tokenizer 和限制
        # - SDXL/InstantID 使用 T5 tokenizer，支持 512 tokens
        # - Flux 使用 T5 tokenizer，支持 512 tokens
        # - 只有 CLIP-based 模型才需要 77 tokens 限制
        token_limit = 77  # 默认 CLIP 限制
        use_clip_limit = True
        
        if self.engine in ["instantid", "sdxl"]:
            # SDXL 和 InstantID 使用 T5 tokenizer，支持 512 tokens
            token_limit = 512
            use_clip_limit = False
            print(f"  ℹ SDXL/InstantID 引擎：使用 T5 tokenizer，支持 {token_limit} tokens")
        elif self.engine in ["flux1", "flux2", "flux-instantid"]:
            # Flux 使用 T5 tokenizer，支持 512 tokens
            token_limit = 512
            use_clip_limit = False
            print(f"  ℹ Flux 引擎：使用 T5 tokenizer，支持 {token_limit} tokens")
        
        # 检查并精简 prompt，确保不超过 token 限制
        # 在添加镜头描述后，重新计算 token 数
        # 使用 token_estimator 进行更准确的估算（如果可用）
        token_checker = None
        if hasattr(self, 'token_estimator'):
            token_checker = self.token_estimator
        elif use_clip_limit and hasattr(self, '_clip_tokenizer') and self._clip_tokenizer is not None:
            # 只有 CLIP-based 模型才使用 _clip_tokenizer
            token_checker = self._clip_tokenizer

        if token_checker and use_clip_limit:
            try:
                # 使用 token_estimator 或 _clip_tokenizer 计算 token 数
                if hasattr(token_checker, 'estimate'):
                    # 使用 token_estimator
                    actual_tokens = token_checker.estimate(prompt)
                else:
                    # 使用 _clip_tokenizer
                    tokens_obj = token_checker(
                        prompt, truncation=False, return_tensors="pt")
                    actual_tokens = tokens_obj.input_ids.shape[1]

                if actual_tokens > token_limit:
                    print(
                        f"  ⚠ 警告: Prompt 长度 ({actual_tokens} tokens) 超过 {token_limit} tokens 限制，开始智能精简...")
                    # 智能精简策略：优先保留关键信息（角色名、动作、场景）
                    import re

                    # 1. 提取所有部分
                    parts = [p.strip() for p in prompt.split(',')]

                    # 2. 识别关键部分（包含角色名、关键动作等）
                    character_keywords = [
                        'han li', 'hanli', '韩立', 'character', 'cultivator', 'male']
                    action_keywords = [
                        'lying',
                        'motionless',
                        'facing',
                        'view',
                        'top-down',
                        'back']
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
                            test_tokens_obj = token_checker(
                                test_prompt, truncation=False, return_tensors="pt")
                            test_tokens = test_tokens_obj.input_ids.shape[1]

                        if test_tokens <= token_limit:
                            selected_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            # 如果添加这个关键部分会超过，尝试精简它
                            # 简化权重：将高权重降低
                            simplified = re.sub(
                                r':\d+\.\d+', ':1.3', part)  # 降低权重
                            simplified = re.sub(
                                r':\d+', ':1.3', simplified)  # 处理整数权重
                            test_prompt = ', '.join(
                                selected_parts + [simplified])
                            if hasattr(token_checker, 'estimate'):
                                test_tokens = token_checker.estimate(
                                    test_prompt)
                            else:
                                test_tokens_obj = token_checker(
                                    test_prompt, truncation=False, return_tensors="pt")
                                test_tokens = test_tokens_obj.input_ids.shape[1]

                            if test_tokens <= token_limit:
                                selected_parts.append(simplified)
                                current_tokens = test_tokens
                            # 如果精简后还是超过，跳过这个部分（但关键部分应该尽量保留）

                    # 4. 添加其他部分（如果还有空间）
                    for part in other_parts:
                        test_prompt = ', '.join(selected_parts + [part])
                        if hasattr(token_checker, 'estimate'):
                            test_tokens = token_checker.estimate(test_prompt)
                        else:
                            test_tokens_obj = token_checker(
                                test_prompt, truncation=False, return_tensors="pt")
                            test_tokens = test_tokens_obj.input_ids.shape[1]

                        if test_tokens <= token_limit:
                            selected_parts.append(part)
                            current_tokens = test_tokens
                        else:
                            break  # 没有更多空间

                    prompt = ', '.join(selected_parts)

                    # 最终验证
                    if hasattr(token_checker, 'estimate'):
                        final_tokens = token_checker.estimate(prompt)
                    else:
                        final_tokens_obj = token_checker(
                            prompt, truncation=False, return_tensors="pt")
                        final_tokens = final_tokens_obj.input_ids.shape[1]

                    print(
                        f"  ✓ 智能精简完成: {
                            len(selected_parts)} 个部分，{final_tokens} tokens")
                    if final_tokens > token_limit:
                        print(
                            f"  ⚠ 警告: 精简后仍超过 {token_limit} tokens ({final_tokens} tokens)，可能会被截断")
            except Exception as e:
                print(f"  ⚠ Token 检查失败: {e}")

        # 根据场景自动调整参数（不同焦距）
        # 优先使用 camera 字段判断镜头类型，更准确
        prompt_lower = prompt.lower()
        camera_desc_lower = ""
        if scene:
            camera_desc = self._get_camera_string(scene)
            if camera_desc:
                camera_desc_lower = camera_desc.lower()

        # 检测场景类型（优先级：camera 字段 > prompt 关键词）
        # 先检查 camera 字段
        is_wide_shot = False
        is_full_body = False
        is_close_up = False
        is_medium_shot = False

        if camera_desc_lower:
            # 优先使用 camera 字段判断
            # ⚡ 重要：Top-down 应该被识别为远景（俯视远景）
            if any(
                kw in camera_desc_lower for kw in [
                    'wide',
                    'long',
                    'establish',
                    'top-down',
                    'top down',
                    '俯视',
                    '远景',
                    '全景']):
                is_wide_shot = True
            elif any(kw in camera_desc_lower for kw in ['close', 'closeup', 'portrait', 'headshot', '特写', '近景']):
                # 检查是否是眼睛特写场景（需要保持特写）
                is_eye_closeup = any(
                    kw in camera_desc_lower for kw in [
                        'eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔'])
                if is_eye_closeup:
                    # 眼睛特写场景：保持特写，不转换为中景
                    is_close_up = True
                    is_medium_shot = False
                    print(f"  ✓ 检测到眼睛特写场景，保持特写镜头（不转换为中景）")
                elif self.allow_close_up:
                    # 配置允许特写，保持特写
                    is_close_up = True
                    is_medium_shot = False
                    print(f"  ✓ 配置允许特写场景，保持特写镜头（allow_close_up=true）")
                else:
                    # 其他特写场景：避免太近的镜头，将特写转换为中景
                    print(
                        f"  ⚠ 检测到特写/近景镜头，为避免身体过宽和模糊，转换为中景（allow_close_up=false）")
                is_medium_shot = True  # 转换为中景，而不是特写
                is_close_up = False
            elif any(kw in camera_desc_lower for kw in ['medium', 'mid', '中景']):
                is_medium_shot = True
            elif any(kw in camera_desc_lower for kw in ['full', '全身']):
                is_full_body = True

        # 如果 camera 字段没有明确判断，再从 prompt 中检测
        if not (is_wide_shot or is_full_body or is_close_up or is_medium_shot):
            # ⚡ 重要：Top-down 应该被识别为远景（俯视远景）
            wide_shot_keywords_extended = wide_shot_keywords + ['top-down', 'top down', '俯视', 'bird\'s eye', 'bird eye']
            is_wide_shot = any(
                keyword in prompt_lower for keyword in wide_shot_keywords_extended)
            is_full_body = any(
                keyword in prompt_lower for keyword in [
                    'full body', 'full figure', '全身'])
            # 避免太近的镜头，如果检测到特写关键词，转换为中景
            # 但眼睛特写场景需要保持特写
            close_keywords_found = any(
                keyword in prompt_lower for keyword in [
                    'close-up',
                    'closeup',
                    'close up',
                    'portrait',
                    'headshot',
                    '特写',
                    '近景'])
            eye_closeup_keywords = any(
                keyword in prompt_lower for keyword in [
                    'eye',
                    'eyes',
                    'pupil',
                    'pupils',
                    '眼睛',
                    '瞳孔',
                    'extreme close'])
            if close_keywords_found:
                if eye_closeup_keywords:
                    # 眼睛特写场景：保持特写，不转换为中景
                    is_close_up = True
                elif self.allow_close_up:
                    # 配置允许特写，保持特写
                    is_close_up = True
                    print(f"  ✓ 配置允许特写场景，保持特写镜头（allow_close_up=true）")
                    is_medium_shot = False
                    print(f"  ✓ 检测到眼睛特写关键词，保持特写镜头（不转换为中景）")
                else:
                    print(f"  ⚠ 检测到特写/近景关键词，为避免身体过宽和模糊，转换为中景")
                    is_medium_shot = True  # 转换为中景
                is_close_up = False
            else:
                is_close_up = False
            is_medium_shot = any(
                keyword in prompt_lower for keyword in [
                    'medium shot', 'mid shot', '中景'])

        # 检查是否是躺着姿势（lying, top-down view等），这些姿势可能影响人脸相似度
        # ⚡ 关键修复：确保 prompt_lower_check 在所有情况下都被定义
        prompt_lower_check = prompt.lower() if prompt else ""
        is_lying_pose = False
        camera_desc = ""
        if scene:
            action = str(scene.get("action", "")).lower()
            camera_desc = self._get_camera_string(scene).lower()
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
        # ⚡ 关键修复：根据专业分析，top-down + far away + lying是InstantID的死刑组合
        # 检测是否是top-down + far away + lying场景（脸部占比<5%，InstantID失效）
        is_top_down_for_weights = (
            "top-down" in camera_desc or
            "top down" in camera_desc or
            "俯视" in camera_desc or
            "top-down" in prompt_lower_check or
            "top down" in prompt_lower_check or
            "bird's eye" in prompt_lower_check.lower() or
            "bird eye" in prompt_lower_check.lower()
        )
        is_far_away_for_weights = any(kw in prompt_lower_check for kw in ["far away", "distant view", "distant", "wide shot", "long shot", "extreme wide"])
        is_instantid_death_combo = is_top_down_for_weights and is_far_away_for_weights and is_lying_pose
        
        # ⚡ 重要：对于top-down + far away + lying场景，权重应该降低（0.65-0.72），而不是提高
        # 因为脸部占比<5%时，权重越高越假，InstantID几乎失效
        if is_instantid_death_combo:
            # ⚡ 关键修复：这是InstantID的死刑组合，应该禁用InstantID（已在上面处理）
            # 但如果仍然使用InstantID（兜底情况），权重应该降到最低
            print(f"  ⚠ 警告：检测到top-down + far away + lying场景（InstantID死刑组合），如果使用InstantID，权重将降至最低")
            ip_adapter_scale = 0.65  # 降到最低，避免硬拽
            controlnet_scale = 0.70  # 适度降低
            min_ip_adapter_scale = 0.60  # 最低阈值
            min_controlnet_scale = 0.65
            print(
                f"  检测到InstantID死刑组合（top-down + far away + lying），面部权重: {
                    ip_adapter_scale:.2f}, ControlNet: {
                    controlnet_scale:.2f} (使用最低权重避免硬拽)")
        elif is_lying_pose:
            # 躺着姿势+远景：对于非死刑组合，使用中等权重
            if is_wide_shot or is_full_body:
                # 躺着姿势+远景：使用中等权重（不是超高权重，避免硬拽）
                ip_adapter_scale = 0.75  # 从0.98降到0.75，避免硬拽
                controlnet_scale = 0.75  # 保持中等权重
                min_ip_adapter_scale = 0.70  # 降低最低阈值
                min_controlnet_scale = 0.70
                print(
                    f"  检测到躺着姿势+远景/全身场景，面部权重: {
                        ip_adapter_scale:.2f}, ControlNet: {
                        controlnet_scale:.2f} (使用中等权重)")
            else:
                # 躺着姿势+其他场景：使用正常权重
                ip_adapter_scale = self.face_emb_scale * 1.2  # 从1.6降到1.2，避免硬拽
                controlnet_scale = 0.75  # 从0.85降到0.75
                min_ip_adapter_scale = 0.75  # 从0.95降到0.75
                min_controlnet_scale = 0.70  # 从0.80降到0.70
                print(
                    f"  检测到躺着姿势场景，面部权重: {
                        ip_adapter_scale:.2f}, ControlNet: {
                        controlnet_scale:.2f} (使用正常权重)")
        # ⚡ 关键修复：检测是否是人设锚点图生成（需要最高相似度）
        is_character_anchor = scene and scene.get("is_character_anchor", False)
        if is_character_anchor:
            # 人设锚点图生成：使用最高权重，确保人脸相似度最高
            # ⚡ 关键修复：进一步提高权重，确保与参考图高度相似
            ip_adapter_scale = 1.30  # 从 1.20 提高到 1.30（最高权重，确保人脸相似度）
            controlnet_scale = 0.95  # 从 0.90 提高到 0.95（高 ControlNet 权重）
            min_ip_adapter_scale = 1.25  # 从 1.15 提高到 1.25
            min_controlnet_scale = 0.90  # 从 0.85 提高到 0.90
            print(f"  🎯 人设锚点图生成模式：使用最高权重，面部权重: {ip_adapter_scale:.2f}, ControlNet: {controlnet_scale:.2f}")
            print(f"  ⚠ 注意：已禁用 LoRA，只使用 InstantID（确保最高人脸相似度）")
        elif is_wide_shot or is_full_body:
            # 远景/全身：适度降低面部权重，但确保人脸完整且相似，同时避免瘦长脸
            # 平衡权重，确保人脸完整且相似，但不过度控制导致瘦长脸
            # 提高基础权重，从0.80提高到0.85，确保人脸完整且相似（用户反馈效果不好）
            ip_adapter_scale = max(
                self.face_emb_scale * 0.85,
                0.6)  # 从0.80提高到0.85，从0.5提高到0.6
            controlnet_scale = 0.60  # 保持0.60，平衡控制强度
            min_ip_adapter_scale = 0.6  # 从0.5提高到0.6，确保人脸完整且相似
            min_controlnet_scale = 0.45  # 保持0.45，给更多自由度，避免瘦长脸
            print(
                f"  检测到远景/全身场景，面部权重: {
                    ip_adapter_scale:.2f}, ControlNet: {
                    controlnet_scale:.2f} (确保人脸完整且比例自然)")
        elif is_close_up:
            # 近景/特写：降低ControlNet权重，避免身体过宽、模糊和横向压缩变形
            # 但保持较高的面部权重，确保人脸相似度
            ip_adapter_scale = self.face_emb_scale * 1.3  # 从1.2提高到1.3，进一步增强相似度（用户反馈站姿也不太像）
            controlnet_scale = 0.28  # 从0.30降到0.28，进一步避免身体过宽、模糊和横向压缩变形
            print(
                f"  检测到近景/特写场景，面部权重: {
                    ip_adapter_scale:.2f}, ControlNet: {
                    controlnet_scale:.2f} (降低ControlNet避免横向压缩变形)")
        elif is_medium_shot:
            # 中景/半身像：优化参数使其更自然，避免横向压缩变形
            # 1. 适度提高IP-Adapter权重确保人脸相似度，但不过高
            ip_adapter_scale = self.face_emb_scale * \
                1.35  # 从1.3提高到1.35，进一步增强人脸相似度（用户反馈效果不好）
            # 2. 进一步降低ControlNet权重，避免身体过宽、僵硬、横向压缩变形和瘦长脸
            controlnet_scale = 0.22  # 保持0.22，平衡控制强度
            min_ip_adapter_scale = 0.5  # 从0.4提高到0.5，确保最小权重足够高，保证人脸相似度
            min_controlnet_scale = 0.20  # 保持0.20，给更多自由度，减少压缩变形和瘦长脸
            print(
                f"  检测到中景/半身像场景，面部权重: {
                    ip_adapter_scale:.2f}, ControlNet: {
                    controlnet_scale:.2f} (优化参数避免横向压缩变形和瘦长脸)")
        else:
            # 默认：使用中等权重，确保角色一致性
            # 从1.1提高到1.2，进一步增强角色一致性和人脸相似度（用户反馈站姿也不太像）
            ip_adapter_scale = self.face_emb_scale * 1.2
            controlnet_scale = 0.75
            print(f"  使用默认场景参数，面部权重: {ip_adapter_scale:.2f} (确保角色一致性和人脸相似度)")

        # 应用 face_style_auto 参数调整（如果存在）
        face_style = scene.get("face_style_auto") or {} if scene else {}
        if isinstance(face_style, dict) and face_style:
            try:
                from face_style_auto_generator import to_instantid_params
                instantid_params = to_instantid_params(face_style)

                # 应用强度调整到 ip_adapter_scale
                style_multiplier = instantid_params.get(
                    "ip_adapter_scale_multiplier", 1.0)
                original_scale = ip_adapter_scale

                # 如果 multiplier 太低，限制最小值，避免过度降低权重
                # ⚡ 重要：对于躺着姿势，不允许降低权重（min_multiplier = 1.0），因为躺着姿势本身就需要更高权重
                # 对于远景/全身场景，也需要更高的最小值（0.95），确保人脸完整且相似
                # 对于中景/近景场景，最小值设为 0.95，确保人脸相似度
                if is_lying_pose:
                    min_multiplier = 1.0  # ⚡ 修复：躺着姿势不允许降低权重，必须 >= 1.0
                elif is_wide_shot or is_full_body:
                    min_multiplier = 0.95  # 从0.9提高到0.95，确保远景场景人脸完整且相似
                else:
                    min_multiplier = 0.95  # 从0.9提高到0.95，确保中景/近景场景人脸相似度
                if style_multiplier < min_multiplier:
                    print(
                        f"    ⚠ face_style_auto multiplier ({
                            style_multiplier:.2f}) 过低，限制为 {
                            min_multiplier:.2f} 以保持人脸相似度")
                    style_multiplier = min_multiplier

                ip_adapter_scale = ip_adapter_scale * style_multiplier

                print(
                    f"  应用 face_style_auto: {
                        face_style.get('expression')}/{
                        face_style.get('lighting')}/{
                        face_style.get('detail')}")
                print(
                    f"    面部权重调整: {
                        original_scale:.2f} -> {
                        ip_adapter_scale:.2f} (x{
                        style_multiplier:.2f})")
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
        ip_adapter_scale = max(
            min_ip_adapter_scale, min(
                1.0, ip_adapter_scale))
        # 对于非远景场景，确保最小权重至少为 0.6（从0.5提高到0.6，提高人脸相似度）
        if not (is_wide_shot or is_full_body):
            ip_adapter_scale = max(ip_adapter_scale, 0.6)
        controlnet_scale = max(
            min_controlnet_scale, min(
                1.0, controlnet_scale))

        # 躺着姿势特殊处理：提高面部权重，因为躺着姿势可能影响人脸相似度
        # ⚡ 注意：如果已经在上面设置了躺着姿势的基础值，这里只需要微调（避免重复提高）
        if is_lying_pose:
            # 如果基础值已经很高（>= 0.90），只需要小幅提高
            # 如果基础值较低，需要大幅提高
            if ip_adapter_scale < 0.90:
                # 基础值较低，大幅提高
                ip_adapter_scale = max(ip_adapter_scale * 1.30, 0.90)
                controlnet_scale = max(controlnet_scale * 1.15, 0.70)
                print(
                    f"  ⚠ 检测到躺着姿势，大幅提高面部权重至 {
                        ip_adapter_scale:.2f}，ControlNet权重至 {
                        controlnet_scale:.2f}，确保人脸相似度（躺着姿势需要更高权重）")
            else:
                # 基础值已经很高，只需要小幅提高 ControlNet
                controlnet_scale = max(controlnet_scale * 1.10, 0.70)
                print(
                    f"  ✓ 检测到躺着姿势，面部权重已足够高 ({
                        ip_adapter_scale:.2f})，仅提高ControlNet权重至 {
                        controlnet_scale:.2f}")

        # 远景场景额外检查：确保面部权重不会太低，导致人脸不像或不完整
        if is_wide_shot or is_full_body:
            if ip_adapter_scale < 0.6:  # 从0.5提高到0.6，确保人脸完整且相似
                ip_adapter_scale = 0.6
                print(f"  ⚠ 远景场景：强制提高面部权重至 {ip_adapter_scale:.2f}，确保人脸完整且相似")
            if controlnet_scale < 0.5:
                controlnet_scale = 0.5
                print(
                    f"  ⚠ 远景场景：强制提高ControlNet权重至 {
                        controlnet_scale:.2f}，确保人脸完整")

        # 改进 negative_prompt，避免人脸过大、拉伸、颜色和面部比例问题
        enhanced_negative = negative_prompt or self.negative_prompt or ""
        
        # ⚡ 修复风格问题：检测是否为仙侠/动漫场景，如果是则强化排除写实风格
        # 检查prompt中是否包含仙侠/动漫关键词
        is_xianxia_or_anime = False
        if prompt:
            prompt_lower = prompt.lower()
            xianxia_keywords = ['xianxia', 'anime', 'cultivator', 'fantasy', '仙侠', '修仙', '动漫', 'animation', 'han li', '韩立']
            if any(kw in prompt_lower for kw in xianxia_keywords):
                is_xianxia_or_anime = True
        
        # 如果是仙侠/动漫场景，强化排除写实风格
        if is_xianxia_or_anime:
            photorealistic_exclusion = ", photorealistic, hyperrealistic, realistic, real photo, photograph, photography, photoreal, photo-real, photo real, (photorealistic:1.8), (hyperrealistic:1.8), (realistic:1.8), (real photo:1.8), (photograph:1.8), western style, european style, modern style"
            if "photorealistic" not in enhanced_negative.lower() or "realistic" not in enhanced_negative.lower():
                enhanced_negative += photorealistic_exclusion
                print(f"  ✓ 仙侠/动漫场景：已添加强化写实风格排除（权重1.8）到 negative prompt")

        # 对于 lying_still 动作，添加"standing"到 negative prompt，确保生成"躺着"而不是"站着"的图像
        # 同时加强多腿排除，因为躺着姿势容易出现多腿问题
        if scene and scene.get("action"):
            action = str(scene.get("action", "")).lower()
            if action == "lying_still" or "lying" in action:
                if "standing" not in enhanced_negative.lower():
                    enhanced_negative += ", standing, standing up, upright, vertical pose"
                    print(
                        f"  ✓ 检测到 lying_still 动作，已添加 'standing' 到 negative_prompt")
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
            if intent['primary_entity'] and intent['primary_entity'].get(
                    'type') == 'character':
                viewpoint = intent.get('viewpoint', {})
                viewpoint_type_check = viewpoint.get('type', 'front')
                viewpoint_explicit_check = viewpoint.get('explicit', False)
                if viewpoint_type_check != 'back':
                    # 对于所有非背面场景，都增强负面提示权重，确保排除背影
                    if "back view" not in enhanced_negative.lower(
                    ) and "from behind" not in enhanced_negative.lower():
                        # 如果明确要求正面，使用更高权重；否则使用中等权重
                        if viewpoint_explicit_check and viewpoint_type_check == 'front':
                            enhanced_negative += ", back view:1.8, from behind:1.8, character back:1.8, rear view:1.8, turned away:1.8, facing away:1.8, back of head:1.8, back of character:1.8, back facing:1.8"
                            print(
                                f"  ✓ 基于意图分析：角色场景（明确要求正面），已增强防止背影的负面提示（权重1.8）")
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
                if "wide shot" not in enhanced_negative.lower(
                ) and "远景" not in enhanced_negative.lower():
                    enhanced_negative += ", wide shot, distant view, long shot, very long shot, 远景, 远距离, 全景"
                    print(f"  ✓ 基于意图分析：特写场景，已添加排除远景的负面提示")

            # 添加排除项的负面提示
            if intent['exclusions']:
                for exclusion in intent['exclusions']:
                    if exclusion not in enhanced_negative.lower():
                        enhanced_negative += f", {exclusion}"
                print(
                    f"  ✓ 基于意图分析添加排除项负面提示: {
                        ', '.join(
                            intent['exclusions'])}")

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
                "modern weapon, gun, firearm, modern equipment"]
        for neg_term in modern_items_negative:
            # 检查负面提示词中是否已包含这个负面提示词
            if neg_term.lower() not in enhanced_negative.lower():
                enhanced_negative += f", {neg_term}"

        # 中景和近景场景：增加身体宽度、模糊、拉伸的负面描述
        # ⚡ 竖屏模式优化：明确排除过近的镜头
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
                "unnatural body proportions, distorted body shape, bad body anatomy",  # 半身像：避免身体比例不自然
                # ⚡ 竖屏模式优化：明确排除过近的镜头
                "extreme close-up, too close, camera too close, face too close, head too close",
                "macro shot, extreme proximity, uncomfortably close, overly close",
                "face fills frame, head fills frame, character too close to camera",
                "camera distance too short, shot distance too short, distance too close"
            ]
            for neg_term in medium_negative_additions:
                if neg_term not in enhanced_negative.lower():
                    enhanced_negative += f", {neg_term}"
        
        # ⚡ 竖屏模式优化：对所有场景都添加过近镜头的排除（除非明确要求特写）
        # 检查是否是眼睛特写或面部特写场景（这些场景需要保持特写）
        is_eye_closeup = False
        is_face_closeup = False
        if scene:
            camera_desc = self._get_camera_string(scene)
            camera_desc_lower = (camera_desc or "").lower()
            is_eye_closeup = any(kw in camera_desc_lower for kw in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔', 'extreme close'])
            is_face_closeup = any(kw in camera_desc_lower for kw in ['face', 'facial', 'portrait', 'headshot', '面部', '脸部', '头像', 'close-up on face', 'closeup on face'])
        
        # 如果不是眼睛特写或面部特写场景，添加过近镜头的排除
        if not is_eye_closeup and not is_face_closeup:
            too_close_negative = ", extreme close-up, too close, camera too close, face too close, head too close, macro shot, extreme proximity, uncomfortably close, overly close, face fills frame, head fills frame, character too close to camera, camera distance too short, shot distance too short, distance too close, (extreme close-up:1.5), (too close:1.5), (camera too close:1.5)"
            if "too close" not in enhanced_negative.lower() and "extreme close-up" not in enhanced_negative.lower():
                enhanced_negative += too_close_negative
                print(f"  ✓ 竖屏模式优化：已添加排除过近镜头的负面提示（避免镜头太近）")

        # 确保单人场景（所有场景都应该是单人，避免重复人物）
        # 检查场景是否包含角色（通过检查 scene 数据或 prompt 中是否包含角色关键词）
        has_character = scene and (
            scene.get("characters") or any(
                kw in prompt.lower() for kw in [
                    "han li",
                    "hanli",
                    "cultivator",
                    "character",
                    "韩立",
                    "主角",
                    "person",
                    "figure"]))

        # 检查是否为纯背景场景（无人物场景）
        # 通过检查 scene 的 visual.composition 或 prompt 中是否明确表示无人物
        is_background_only = False
        if scene:
            visual = scene.get("visual") or {}
            composition = str(visual.get("composition", "")).lower()
            prompt_lower = prompt.lower()
            # 检查是否明确表示无人物
            no_character_keywords = [
                "only environment",
                "environment only",
                "no character",
                "no person",
                "no figure",
                "pure environment",
                "landscape only",
                "scene only",
                "background only",
                "只有环境",
                "纯环境",
                "无人物",
                "仅环境",
                "纯背景"]
            # 检查 composition 或 prompt 中是否包含无人物关键词
            if any(
                    kw in composition or kw in prompt_lower for kw in no_character_keywords):
                is_background_only = True
            # 检查 character_pose 是否为空（通常表示无人物）
            elif not visual.get("character_pose") and not scene.get("characters"):
                # 进一步检查 prompt 中是否明确没有人物相关关键词
                character_keywords = [
                    "han li",
                    "hanli",
                    "cultivator",
                    "character",
                    "韩立",
                    "主角",
                    "person",
                    "figure",
                    "man",
                    "woman",
                    "people"]
                if not any(kw in prompt_lower for kw in character_keywords):
                    is_background_only = True

        # 对所有有角色的场景都加强单人约束（不只是远景）- 在 prompt 构建完成后统一处理
        # 注意：这部分代码在 prompt 构建之后执行，确保单人约束在 prompt 最前面

        # 加强负面提示：排除多人、重复人物、多腿、多手等（对所有场景都适用）
        # ⚡ 关键修复：用户反馈生成了两个人像，分开了，需要大幅提高多人排除权重到3.0
        # 提高多人排除权重到3.0，确保绝对不会生成多个人物
        multiple_people_negative = ", multiple people, two people, three people, four people, crowd, group of people, extra person, duplicate character, second person, additional figure, cloned person, duplicate person, identical person, twin character, repeated character, second identical figure, duplicate figure, mirrored person, copy of person, repeated appearance, same person twice, duplicate appearance, two same people, two identical people, cloned figure, mirrored character, identical duplicate, separated people, split person, divided person, (duplicate person:3.0), (same person twice:3.0), (two same people:3.0), (multiple people:3.0), (two people:3.0), (crowd:3.0), (group of people:3.0), (extra person:3.0), (second person:3.0), (additional figure:3.0), (cloned person:3.0), (duplicate character:3.0), (twin character:3.0), (repeated character:3.0), (second identical figure:3.0), (duplicate figure:3.0), (mirrored person:3.0), (two heads:3.0), (multiple heads:3.0), (duplicate head:3.0), (second head:3.0), (copy of person:3.0), (separated people:3.0), (split person:3.0), (divided person:3.0), extra legs, multiple legs, duplicate legs, two sets of legs, three legs, four legs, extra feet, multiple feet, duplicate feet, three feet, four feet, extra limbs, multiple limbs, duplicate limbs, extra body parts, multiple body parts, duplicate body parts, extra hands, multiple hands, duplicate hands, extra arms, multiple arms, duplicate arms, (extra legs:2.0), (multiple legs:2.0), (duplicate legs:2.0), (three legs:2.0), (four legs:2.0), (two sets of legs:2.0), (extra feet:2.0), (multiple feet:2.0), (duplicate feet:2.0), (three feet:2.0), (extra limbs:2.0), (multiple limbs:2.0), (duplicate limbs:2.0), (extra body parts:2.0), (multiple body parts:2.0), (duplicate body parts:2.0), (extra hands:2.0), (multiple hands:2.0), (duplicate hands:2.0), (extra arms:2.0), (multiple arms:2.0), (duplicate arms:2.0)"

        # 对于纯背景场景，明确排除所有人物（高权重），包括仙女等
        if is_background_only:
            no_character_negative = ", person, people, human, character, figure, man, woman, male, female, boy, girl, individual, someone, anybody, anyone, (person:1.8), (people:1.8), (human:1.8), (character:1.8), (figure:1.8), (man:1.8), (woman:1.8), (male:1.8), (female:1.8), (boy:1.8), (girl:1.8), (individual:1.8), (someone:1.8), (anybody:1.8), (anyone:1.8), face, faces, body, bodies, portrait, portraits, person in scene, character in scene, figure in scene, human in scene, man in scene, woman in scene, (face:1.8), (faces:1.8), (body:1.8), (bodies:1.8), (portrait:1.8), (portraits:1.8), (person in scene:1.8), (character in scene:1.8), (figure in scene:1.8), (human in scene:1.8), (man in scene:1.8), (woman in scene:1.8), female character, woman character, girl character, female figure, woman figure, girl figure, (female character:1.8), (woman character:1.8), (girl character:1.8), (female figure:1.8), (woman figure:1.8), (girl figure:1.8), fairy, fairy woman, celestial maiden, immortal woman, fairy maiden, immortal maiden, xianzi, fairy girl, (fairy:1.8), (fairy woman:1.8), (celestial maiden:1.8), (immortal woman:1.8), (fairy maiden:1.8), (immortal maiden:1.8), (xianzi:1.8), (fairy girl:1.8), goddess, goddess figure, female deity, (goddess:1.8), (goddess figure:1.8), (female deity:1.8)"
            if "person" not in enhanced_negative.lower(
            ) or "people" not in enhanced_negative.lower():
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
                is_hanli = any(
                    kw in prompt.lower() for kw in [
                        "han li", "hanli", "韩立", "主角"]) or (
                    scene.get("characters") and "hanli" in str(
                        scene.get(
                            "characters", "")).lower())
                if is_hanli:
                    # ⚡ 修复性别错误：明确排除女性特征（超高权重2.5，防止生成女性）
                    # 用户反馈：韩立变成了女性，大幅提高权重到2.5
                    female_negative = ", female, woman, girl, feminine, female character, woman character, girl character, female figure, woman figure, girl figure, female appearance, woman appearance, girl appearance, (female:2.5), (woman:2.5), (girl:2.5), (feminine:2.5), (female character:2.5), (woman character:2.5), (girl character:2.5), (female figure:2.5), (woman figure:2.5), (girl figure:2.5), (female appearance:2.5), (woman appearance:2.5), (girl appearance:2.5), female face, woman face, girl face, female body, woman body, girl body, (female face:2.5), (woman face:2.5), (girl face:2.5), (female body:2.5), (woman body:2.5), (girl body:2.5), breasts, female breasts, (breasts:2.5), (female breasts:2.5), long hair flowing, feminine hair, (feminine hair:2.5), feminine features, womanly features, (feminine features:2.5), (womanly features:2.5)"
                    if "female" not in enhanced_negative.lower(
                    ) or "woman" not in enhanced_negative.lower():
                        enhanced_negative += female_negative
                        print(f"  ✓ 韩立角色：已添加排除女性特征到 negative prompt（高权重）")
                    
                    # ⚡ 关键修复：用户反馈光着上身，必须排除裸露、无上衣等
                    if "naked" not in enhanced_negative.lower() or "bare" not in enhanced_negative.lower() or "topless" not in enhanced_negative.lower():
                        naked_negative = ", naked, bare, topless, shirtless, bare chest, bare torso, exposed chest, exposed torso, no shirt, no clothing, no robe, no garment, without clothing, without robe, without garment, (naked:3.0), (bare:3.0), (topless:3.0), (shirtless:3.0), (bare chest:3.0), (bare torso:3.0), (exposed chest:3.0), (exposed torso:3.0), (no shirt:3.0), (no clothing:3.0), (no robe:3.0), (no garment:3.0), (without clothing:3.0), (without robe:3.0), (without garment:3.0)"
                        enhanced_negative += naked_negative
                        print(f"  ✓ 韩立角色：已添加排除裸露/光着上身到 negative prompt（超高权重3.0，防止生成光着上身的图像）")

            # 对于所有人物场景，强制添加多人排除项（防止出现多个相同的人）
            if scene and prompt and not is_background_only:
                # 检查是否已经添加了多人排除
                if "multiple people" not in enhanced_negative.lower(
                ) or "duplicate person" not in enhanced_negative.lower():
                    # 添加更强烈的多人排除项
                    strong_multiple_negative = ", multiple people, two people, three people, four people, five people, crowd, group of people, extra person, duplicate character, second person, additional figure, cloned person, duplicate person, identical person, twin character, repeated character, second identical figure, duplicate figure, mirrored person, copy of person, repeated appearance, same person twice, duplicate appearance, two same people, two identical people, cloned figure, mirrored character, identical duplicate, ten people, many people, (duplicate person:2.0), (same person twice:2.0), (two same people:2.0), (multiple people:2.0), (two people:2.0), (crowd:2.0), (group of people:2.0), (extra person:2.0), (second person:2.0), (ten people:2.0), (many people:2.0)"
                    enhanced_negative += strong_multiple_negative
                    print(
                        f"  ✓ 人物场景：已添加强化多人排除项到 negative prompt（高权重2.0，防止出现多个相同的人）")

            # 对于所有场景，排除现代交通工具和飞船（防止出现不符合仙侠风格的现代元素）
            if scene and prompt:
                # 检查是否已经包含现代交通工具排除项（使用and而不是or，确保所有关键词都检查）
                has_vehicle_negative = (
                    "aircraft" in enhanced_negative.lower() or
                    "airplane" in enhanced_negative.lower() or
                    "airplane" in enhanced_negative.lower() or
                    "aircraft" in enhanced_negative.lower()
                )
                if not has_vehicle_negative:
                    modern_vehicle_negative = ", vehicle, vehicles, car, cars, truck, trucks, tank, tanks, military vehicle, military vehicles, spaceship, spaceships, spacecraft, aircraft, airplane, airplanes, plane, planes, jet, jets, fighter jet, fighter jets, helicopter, helicopters, drone, drones, modern technology, modern equipment, military equipment, weapons, gun, guns, automobile, automobiles, (vehicle:2.5), (car:2.5), (tank:2.5), (spaceship:2.5), (aircraft:2.5), (airplane:2.5), (plane:2.5), (military vehicle:2.5), (modern technology:2.5), (automobile:2.5)"
                    enhanced_negative += modern_vehicle_negative
                    print(
                        f"  ✓ 已添加现代交通工具和飞机排除项到 negative prompt（高权重2.5，防止出现不符合仙侠风格的现代元素）")

        if "oversized face" not in enhanced_negative.lower():
            enhanced_negative += ", oversized face, face too large, distorted face, wrong hairstyle, deformed face, character too large, person too large, figure too large, oversized character, oversized person, oversized figure, character too big, person too big, figure too big, character size wrong, person size wrong, figure size wrong, character scale wrong, person scale wrong, figure scale wrong, character proportions wrong, person proportions wrong, figure proportions wrong, character too prominent, person too prominent, figure too prominent, character dominates frame, person dominates frame, figure dominates frame, character fills frame, person fills frame, figure fills frame, character takes up too much space, person takes up too much space, figure takes up too much space"

        # 排除画中画、画框、文字等（防止生成画中画或带文字的画面）
        # ⚡ 增强：大幅增强文字排除，防止生成任何文字
        has_art_negative = (
            "painting" in enhanced_negative.lower() and
            "frame" in enhanced_negative.lower() and
            "text" in enhanced_negative.lower()
        )
        if not has_art_negative:
            art_negative = ", painting, picture frame, framed painting, artwork frame, picture in picture, painting in painting, framed picture, picture on wall, artwork on wall, text, letters, words, characters, writing, inscription, calligraphy, text overlay, text on image, watermark text, subtitle, caption, label, sign, (painting:2.5), (picture frame:2.5), (framed painting:2.5), (text:3.0), (letters:3.0), (words:3.0), (writing:3.0), (inscription:3.0), (text overlay:3.0), (text on image:3.0), (watermark:3.0), (subtitle:3.0), (caption:3.0), (label:3.0), (sign:3.0), (chinese text:3.0), (english text:3.0), (any text:3.0), (visible text:3.0), (readable text:3.0), (printed text:3.0), (handwritten text:3.0)"
            enhanced_negative += art_negative
            print(f"  ✓ 已添加画中画、画框、文字排除项到 negative prompt（超高权重3.0，防止生成任何文字）")

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

        # LoRA 适配（InstantID）- 包括角色LoRA和风格LoRA
        # 注意：InstantID 模式下，use_ip_adapter 通常是 False（InstantID 有自己的 IP-Adapter）
        # 所以 LoRA 不会降低 InstantID 的 IP-Adapter 权重，这是正确的
        # 重要：如果character_lora和style_lora都是None，表示不使用LoRA，只使用参考图生成正常图像
        if hasattr(self.pipeline, "set_adapters"):
            # 构建LoRA适配器列表（角色LoRA + 风格LoRA）
            adapter_names = []
            adapter_weights = []
            
            # 只有当用户明确指定了character_lora时才使用角色LoRA
            # 如果character_lora是None，表示不使用角色LoRA（不使用默认的hanli）
            if character_lora is not None:
                if character_lora == "":
                    print(f"  ℹ 已禁用角色LoRA（用户指定）")
                else:
                    # MVP 流程：character_lora 是 adapter_name，需要先检查是否已加载
                    # 如果未加载，需要先加载 LoRA 文件
                    # 将 character_lora 名称解析为实际文件路径
                    lora_path = None
                    if character_lora == "host_person_v2":
                        lora_path = str(self.models_root / "lora" / "host_person_v2" / "pytorch_lora_weights.safetensors")
                    elif character_lora == "host_person":
                        lora_path = str(self.models_root / "lora" / "host_person" / "pytorch_lora_weights.safetensors")
                    elif character_lora in ["kepu_gege", "weilai_jiejie"]:
                        # 科普主持人映射到 host_person_v2
                        lora_path = str(self.models_root / "lora" / "host_person_v2" / "pytorch_lora_weights.safetensors")
                    else:
                        # 尝试作为目录名或路径
                        lora_path_obj = Path(character_lora)
                        if not lora_path_obj.is_absolute():
                            lora_path_obj = self.models_root / "lora" / character_lora / "pytorch_lora_weights.safetensors"
                        if lora_path_obj.exists():
                            lora_path = str(lora_path_obj)
                    
                    # 检查 adapter 是否已加载
                    loaded_adapters = []
                    if hasattr(self.pipeline, "get_active_adapters"):
                        try:
                            loaded_adapters = list(self.pipeline.get_active_adapters()) if self.pipeline.get_active_adapters() else []
                        except:
                            pass
                    elif hasattr(self.pipeline, "peft_config") and self.pipeline.peft_config:
                        loaded_adapters = list(self.pipeline.peft_config.keys())
                    
                    # 如果 adapter 未加载且提供了路径，先加载 LoRA
                    if character_lora not in loaded_adapters and lora_path and Path(lora_path).exists():
                        try:
                            print(f"  🔧 加载 LoRA 文件: {Path(lora_path).name} (adapter={character_lora})")
                            self.pipeline.load_lora_weights(lora_path, adapter_name=character_lora)
                            print(f"  ✅ LoRA 已加载: {character_lora}")
                        except Exception as e:
                            print(f"  ⚠ LoRA 加载失败: {e}，尝试使用已加载的 adapter")
                    
                    # 使用指定的角色LoRA（无论是否刚加载）
                    adapter_names.append(character_lora)
                    adapter_weights.append(self.lora_alpha)
                    print(f"  ✓ 使用用户指定的角色LoRA: {character_lora} (alpha={self.lora_alpha:.2f})")
            else:
                # character_lora是None，不使用角色LoRA（不使用默认的hanli）
                print(f"  ℹ 未指定角色LoRA，禁用角色LoRA（仅使用参考图）")

            # 添加风格LoRA（如果启用）
            if style_lora is not None:
                # 如果 style_lora 是空字符串，禁用风格LoRA
                if style_lora == "":
                    print(f"  ℹ 已禁用风格LoRA（用户指定）")
                else:
                    # ⚡ 关键修复：在使用 style_lora 之前，确保它已被加载
                    style_config = self.lora_config.get("style_lora", {})
                    style_lora_path = None
                    if isinstance(style_config, dict):
                        style_lora_path = style_config.get("weights_path")
                    
                    # 如果提供了路径，尝试加载 style_lora
                    if style_lora_path and Path(style_lora_path).exists():
                        # 检查 adapter 是否已加载
                        loaded_adapters = []
                        if hasattr(self.pipeline, "get_active_adapters"):
                            try:
                                loaded_adapters = list(self.pipeline.get_active_adapters()) if self.pipeline.get_active_adapters() else []
                            except:
                                pass
                        elif hasattr(self.pipeline, "peft_config") and self.pipeline.peft_config:
                            loaded_adapters = list(self.pipeline.peft_config.keys())
                        
                        # 如果 adapter 未加载，先加载 LoRA
                        if style_lora not in loaded_adapters:
                            try:
                                print(f"  🔧 加载风格 LoRA 文件: {Path(style_lora_path).name} (adapter={style_lora})")
                                self.pipeline.load_lora_weights(style_lora_path, adapter_name=style_lora)
                                print(f"  ✅ 风格 LoRA 已加载: {style_lora}")
                            except Exception as e:
                                print(f"  ⚠ 风格 LoRA 加载失败: {e}，跳过风格 LoRA")
                                style_lora = None  # 加载失败，不使用风格 LoRA
                    
                    # 使用用户指定的风格LoRA（如果已加载或不需要加载）
                    if style_lora:
                        adapter_names.append(style_lora)
                        style_alpha = float(style_config.get("alpha", 0.7)) if isinstance(style_config, dict) else 0.7
                        adapter_weights.append(style_alpha)
                        print(f"  ✓ 使用用户指定的风格LoRA: {style_lora} (alpha={style_alpha:.2f})")
                    else:
                        print(f"  ⚠ 风格LoRA未加载，跳过使用风格LoRA")
            else:
                # style_lora是None，不使用风格LoRA（不使用默认的anime_style）
                print(f"  ℹ 未指定风格LoRA，禁用风格LoRA（仅使用参考图）")

            # 确保在生成前应用LoRA（每次生成都重新应用，避免被覆盖）
            if adapter_names:
                try:
                    self.pipeline.set_adapters(
                        adapter_names, adapter_weights=adapter_weights)
                    print(f"  ✓ 已应用LoRA适配器: {adapter_names} (权重: {adapter_weights})")
                except Exception as adapter_err:
                    import traceback
                    error_str = str(adapter_err)
                    print(f"  ✗ 应用LoRA适配器时出错: {adapter_err}")
                    print(f"  📋 错误堆栈:\n{traceback.format_exc()}")
                    if 'unet' in error_str.lower():
                        print(f"  ✗ 检测到 'unet' 相关错误，尝试重新加载 pipeline...")
                        try:
                            self.engine = "instantid"
                            self._load_instantid_pipeline()
                            print(f"  ✓ 已重新加载 InstantID pipeline，重试应用LoRA...")
                            self.pipeline.set_adapters(
                                adapter_names, adapter_weights=adapter_weights)
                            print(f"  ✓ 已应用LoRA适配器: {adapter_names} (权重: {adapter_weights})")
                        except Exception as reload_err:
                            print(f"  ✗ 重新加载失败: {reload_err}")
                            raise RuntimeError(f"无法修复 pipeline 组件错误: {adapter_err}") from adapter_err
                    else:
                        raise
            else:
                # 如果没有适配器，完全禁用并卸载所有LoRA（只使用参考图生成正常图像）
                # ⚡ 重要：如果配置中 LoRA 已禁用，直接跳过 set_adapters 调用
                # 因为 InstantID pipeline 可能没有正确初始化 LoRA 适配器系统，调用 set_adapters([]) 会抛出 KeyError 'unet'
                lora_enabled = self.config.get("image", {}).get("lora", {}).get("enabled", True)
                if not lora_enabled:
                    # LoRA 已禁用，直接跳过 set_adapters 调用
                    print(f"  ℹ LoRA 已禁用，跳过 set_adapters 调用（仅使用参考图生成正常图像）")
                else:
                    # LoRA 未禁用，尝试卸载（但可能没有加载过）
                    try:
                        # 先检查是否有已加载的适配器
                        has_adapters = False
                        if hasattr(self.pipeline, "get_active_adapters"):
                            try:
                                active = self.pipeline.get_active_adapters()
                                if active and len(active) > 0:
                                    has_adapters = True
                            except:
                                pass
                        elif hasattr(self.pipeline, "peft_config") and self.pipeline.peft_config:
                            if len(self.pipeline.peft_config) > 0:
                                has_adapters = True
                        
                        if has_adapters:
                            # 有已加载的适配器，尝试卸载
                            self.pipeline.set_adapters([])
                            self._unload_all_lora_adapters(self.pipeline, "InstantID pipeline")
                            print(f"  ℹ 已禁用所有LoRA适配器，仅使用参考图生成正常图像")
                        else:
                            # 没有已加载的适配器，直接跳过
                            print(f"  ℹ 没有已加载的LoRA适配器，跳过卸载（仅使用参考图生成正常图像）")
                    except Exception as adapter_err:
                        import traceback
                        error_str = str(adapter_err)
                        print(f"  ⚠ 禁用LoRA适配器时出错: {adapter_err}")
                        # 如果是 'unet' 相关错误，说明 pipeline 组件不完整，尝试重新加载
                        if 'unet' in error_str.lower():
                            print(f"  ✗ 检测到 'unet' 相关错误，尝试重新加载 pipeline...")
                            try:
                                self.engine = "instantid"
                                self._load_instantid_pipeline()
                                print(f"  ✓ 已重新加载 InstantID pipeline")
                                # 重新加载后，不需要再次调用 set_adapters，因为新加载的 pipeline 应该没有 LoRA
                                print(f"  ℹ 重新加载后，跳过 LoRA 卸载（新 pipeline 应该没有 LoRA）")
                            except Exception as reload_err:
                                print(f"  ✗ 重新加载失败: {reload_err}")
                                # 即使重新加载失败，也继续执行（因为 LoRA 可能本来就没有加载）
                                print(f"  ⚠ 继续执行，假设 LoRA 未加载")
                        else:
                            # 如果错误不是 'unet' 相关的，继续执行（可能只是警告）
                            print(f"  ⚠ 继续执行，但LoRA可能未完全卸载")

            # InstantID 模式下，use_ip_adapter 通常是 False，所以不会降低权重
            # 这是正确的，因为 InstantID 的 IP-Adapter 是必需的，不应该被 LoRA 影响
            if self.use_ip_adapter:
                ip_adapter_scale = max(
                    0.1, ip_adapter_scale * self.lora_ip_scale_multiplier)
                print(
                    f"  ⚠ 使用额外的 IP-Adapter（非 InstantID），调节面部权重至: {ip_adapter_scale:.2f}")
            else:
                print(
                    f"  ℹ InstantID 模式：LoRA 不影响 InstantID 的 IP-Adapter 权重（保持 {ip_adapter_scale:.2f}）")

        # 根据镜头自动调整面部关键点缩放
        face_kps_scale_cfg = float(
            self.instantid_config.get(
                "face_kps_scale", 1.0))
        face_kps_offset_y = int(
            self.instantid_config.get(
                "face_kps_offset_y", 0))
        face_kps_scale = face_kps_scale_cfg

        # 躺着姿势特殊处理：需要更高的face_kps_scale来保持人物比例
        if is_lying_pose:
            # 躺着姿势时，人物是横向的，需要更高的关键点缩放来保持正确的身体比例
            if is_wide_shot or is_full_body:
                face_kps_scale = face_kps_scale_cfg * 0.85  # 躺着+远景：提高到0.85，保持身体比例
                print(f"  躺着姿势+远景场景：面部关键点缩放至 {face_kps_scale:.2f}，保持身体比例")
            else:
                face_kps_scale = face_kps_scale_cfg * 0.75  # 躺着+其他：提高到0.75，保持身体比例
                print(f"  躺着姿势场景：面部关键点缩放至 {face_kps_scale:.2f}，保持身体比例")
        elif is_wide_shot or is_full_body:
            # 远景场景：适度提高关键点缩放，确保人脸完整且清晰
            # 从0.65提高到0.75，提高清晰度，但不过度控制避免瘦长脸
            face_kps_scale = face_kps_scale_cfg * 0.75  # 从0.65提高到0.75，提高远景场景清晰度
            print(f"  远景场景：面部关键点缩放至 {face_kps_scale:.2f}，确保人脸完整且清晰")
        elif is_medium_shot:
            # 中景/半身像：适度降低面部关键点缩放，但保持足够的身体和背景可见性
            # 从0.50提高到0.65，确保有足够的身体和背景，避免只有头像
            face_kps_scale = face_kps_scale_cfg * 0.65  # 提高以确保身体和背景可见
            print(f"  中景/半身像场景：面部关键点缩放至 {face_kps_scale:.2f}，确保身体和背景可见")
        elif is_close_up:
            # 降低关键点缩放，避免身体过宽、模糊和横向压缩变形
            face_kps_scale = face_kps_scale_cfg * 0.60  # 从0.65降到0.60，进一步减少横向压缩变形和瘦长脸
            print(f"  近景场景：降低面部关键点缩放至 {face_kps_scale:.2f}，避免身体过宽、模糊和横向压缩变形")

        # 应用 face_style_auto 的 face_kps_scale 调整
        if isinstance(face_style, dict) and face_style:
            try:
                from face_style_auto_generator import to_instantid_params
                instantid_params = to_instantid_params(face_style)
                kps_multiplier = instantid_params.get(
                    "face_kps_scale_multiplier", 1.0)
                face_kps_scale = face_kps_scale * kps_multiplier
            except (ImportError, Exception):
                # 如果没有 face_style_auto_generator，使用简单逻辑
                detail = (face_style.get("detail") or "").lower()
                if detail in {"detailed", "cinematic"}:
                    face_kps_scale *= 1.1
                elif detail in {"subtle"}:
                    face_kps_scale *= 0.9

        # 确保关键点缩放不会太小，同时限制最大值
        # 躺着姿势需要更高的最小值来保持身体比例
        if is_lying_pose:
            # 躺着姿势：至少0.60，最大0.90，保持身体比例
            face_kps_scale = max(0.60, min(face_kps_scale, 0.90))
        elif is_wide_shot or is_full_body:
            # 远景场景：至少0.45，最大0.85，确保人脸完整
            face_kps_scale = max(0.45, min(face_kps_scale, 0.85))
        else:
            # 其他场景（中景/特写）：最小0.45确保有身体和背景，最大0.75避免横向压缩变形和瘦长脸
            # 提高最小值从0.3到0.45，确保有足够的身体和背景可见性
            face_kps_scale = max(0.45, min(face_kps_scale, 0.75))
        face_kps = self._adjust_face_kps_canvas(
            face_kps_raw, face_kps_scale, face_kps_offset_y)
        if abs(face_kps_scale - face_kps_scale_cfg) > 1e-3:
            print(
                f"  面部关键点缩放: {
                    face_kps_scale:.2f} (camera + face_style_auto 控制)")

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

        # 在调用 pipeline 之前，确保 InstantID 的 IP-Adapter 状态正确，否则回退到 SDXL 模式
        if self.engine == "instantid":
            try:
                if not self._ensure_instantid_adapter_active():
                    print("  ✗ InstantID IP-Adapter 无法加载，回退到普通 SDXL pipeline")
                    # InstantID 不使用 reference_image_path，传 None
                    return self._generate_image_sdxl(
                        prompt,
                        output_path,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        seed=seed,
                        reference_image_path=None,  # InstantID 不使用此参数
                        face_reference_image_path=face_reference_image_path,
                        use_lora=self.use_lora,  # 使用实例变量
                        scene=scene,
                        character_lora=character_lora,
                        style_lora=style_lora,
                    )
            except (KeyError, RuntimeError) as e:
                error_str = str(e)
                import traceback
                print(f"  ✗ _ensure_instantid_adapter_active 时出错: {e}")
                print(f"  📋 错误堆栈:\n{traceback.format_exc()}")
                if 'unet' in error_str.lower():
                    print(f"  ✗ 检测到 'unet' 相关错误，尝试重新加载 pipeline...")
                    try:
                        self.engine = "instantid"
                        self._load_instantid_pipeline()
                        print(f"  ✓ 已重新加载 InstantID pipeline")
                        # 重试 _ensure_instantid_adapter_active
                        if not self._ensure_instantid_adapter_active():
                            print("  ✗ InstantID IP-Adapter 仍无法加载，回退到普通 SDXL pipeline")
                            return self._generate_image_sdxl(
                                prompt,
                                output_path,
                                negative_prompt=negative_prompt,
                                guidance_scale=guidance,
                                num_inference_steps=steps,
                                seed=seed,
                                reference_image_path=None,
                                face_reference_image_path=face_reference_image_path,
                                use_lora=self.use_lora,
                                scene=scene,
                                character_lora=character_lora,
                                style_lora=style_lora,
                            )
                    except Exception as reload_err:
                        print(f"  ✗ 重新加载失败: {reload_err}")
                        raise RuntimeError(f"无法修复 pipeline 组件错误: {e}") from e
                else:
                    raise

        # 调用 InstantID pipeline（按照官方用法）
        # 注意：如果 guidance_rescale 不被支持，会自动忽略（不会报错）
        # ⚡ 调试：在调用前检查 pipeline 状态
        if self.pipeline is None:
            print(f"  ✗ 错误：pipeline 为 None，无法生成图像")
            raise RuntimeError("InstantID pipeline 未初始化")
        
        # ⚡ 重要：强制刷新输出，确保日志能及时显示
        import sys
        sys.stdout.flush()
        
        # 检查 pipeline 是否有必要的组件
        # ⚡ 重要：在调用 pipeline 前检查组件完整性，避免运行时 KeyError
        print(f"  🔍 检查 pipeline 组件完整性...")
        sys.stdout.flush()
        
        if hasattr(self.pipeline, 'components'):
            try:
                components = self.pipeline.components
                # 检查 components 是否为字典或类似字典的对象
                if isinstance(components, dict):
                    if 'unet' not in components:
                        print(f"  ⚠ 警告：pipeline.components 中缺少 'unet'，尝试重新加载 pipeline...")
                        # 尝试重新加载
                        self.engine = "instantid"
                        self._load_instantid_pipeline()
                    else:
                        print(f"  ✓ pipeline.components 检查通过，包含 'unet'")
                else:
                    # components 不是字典，可能是属性访问器或其他对象
                    # 尝试直接访问 unet 属性来验证
                    if hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
                        print(f"  ✓ pipeline 包含 unet 组件（通过属性访问）")
                    else:
                        print(f"  ⚠ pipeline 可能缺少 unet 组件，尝试重新加载...")
                        self.engine = "instantid"
                        self._load_instantid_pipeline()
            except KeyError as comp_check_err:
                error_str = str(comp_check_err)
                import traceback
                print(f"  ✗ 检查 pipeline.components 时发现 KeyError: {comp_check_err}")
                print(f"  📋 错误堆栈:\n{traceback.format_exc()}")
                if 'unet' in error_str:
                    print(f"  ✗ 检查 pipeline.components 时发现 KeyError 'unet': {comp_check_err}")
                    print(f"  ℹ 尝试重新加载 InstantID pipeline...")
                    try:
                        self.engine = "instantid"
                        self._load_instantid_pipeline()
                        print(f"  ✓ 已重新加载 InstantID pipeline")
                    except Exception as reload_err:
                        print(f"  ✗ 重新加载失败: {reload_err}")
                        raise RuntimeError(f"无法修复 pipeline 组件错误: {comp_check_err}") from comp_check_err
                else:
                    print(f"  ⚠ 检查 pipeline.components 时出错: {comp_check_err}，继续尝试生成...")
            except (AttributeError, TypeError) as comp_check_err:
                error_str = str(comp_check_err)
                print(f"  ⚠ 检查 pipeline.components 时出错: {comp_check_err}，继续尝试生成...")
            except Exception as comp_check_err:
                import traceback
                print(f"  ✗ 检查 pipeline.components 时发生未知错误: {comp_check_err}")
                print(f"  📋 错误堆栈:\n{traceback.format_exc()}")
                print(f"  ⚠ 继续尝试生成...")
        
        print(f"  🚀 开始调用 InstantID pipeline...")
        sys.stdout.flush()
        
        # ⚡ 最后检查：确保 pipeline 有 unet 组件（通过属性访问）
        if not hasattr(self.pipeline, 'unet') or self.pipeline.unet is None:
            print(f"  ✗ 错误：pipeline 缺少 unet 组件（通过属性检查）")
            print(f"  ℹ 尝试重新加载 InstantID pipeline...")
            try:
                self.engine = "instantid"
                self._load_instantid_pipeline()
                print(f"  ✓ 已重新加载 InstantID pipeline")
            except Exception as reload_err:
                print(f"  ✗ 重新加载失败: {reload_err}")
                raise RuntimeError(f"无法修复 pipeline unet 组件缺失: {reload_err}") from reload_err
        
        try:
            result = self.pipeline(**kwargs)
        except Exception as e:  # 捕获所有异常，确保 KeyError 被正确捕获
            error_str = str(e)
            error_type = type(e).__name__
            import traceback
            # ⚡ 重要：立即打印详细错误信息，确保即使被外层捕获也能看到
            # 强制刷新输出，确保日志能及时显示
            print(f"  ✗ InstantID pipeline 调用失败: {error_type}: {error_str}")
            print(f"  📋 错误堆栈:\n{traceback.format_exc()}")
            sys.stdout.flush()
            
            # ⚡ 修复：捕获 KeyError 'unet'，这可能发生在 pipeline 内部访问 components 时
            # 检查是否是 KeyError 且错误信息包含 'unet'
            is_unet_keyerror = (
                isinstance(e, KeyError) or 
                error_type == 'KeyError' or
                ('unet' in error_str and ('KeyError' in error_str or 'key' in error_str.lower()))
            )
            if is_unet_keyerror and 'unet' in error_str:
                print(f"  ✗ InstantID pipeline 内部错误（KeyError 'unet'）: {e}")
                print(f"  ℹ 这通常是因为 pipeline 组件不完整")
                # 检查 pipeline 状态
                if self.pipeline is None:
                    print(f"  ⚠ Pipeline 为 None，尝试重新加载...")
                    try:
                        self.engine = "instantid"
                        self._load_instantid_pipeline()
                        print(f"  ✓ 已重新加载 InstantID pipeline，重试生成...")
                        result = self.pipeline(**kwargs)
                    except Exception as reload_err:
                        print(f"  ✗ 重新加载 pipeline 失败: {reload_err}")
                        raise RuntimeError(
                            f"InstantID pipeline 重新加载失败: {reload_err}") from reload_err
                else:
                    # Pipeline 存在但组件不完整，尝试检查 components
                    print(f"  ⚠ Pipeline 存在但组件可能不完整，尝试重新初始化...")
                    try:
                        # 保存当前配置
                        instantid_config_backup = self.instantid_config.copy() if hasattr(self, 'instantid_config') else {}
                        # 清理当前 pipeline
                        del self.pipeline
                        self.pipeline = None
                        # 重新加载
                        self.engine = "instantid"
                        self._load_instantid_pipeline()
                        print(f"  ✓ 已重新初始化 InstantID pipeline，重试生成...")
                        result = self.pipeline(**kwargs)
                    except Exception as reload_err:
                        print(f"  ✗ 重新初始化 pipeline 失败: {reload_err}")
                        import traceback
                        print(f"  📋 详细错误信息:\n{traceback.format_exc()}")
                        raise RuntimeError(
                            f"InstantID pipeline 重新初始化失败（KeyError 'unet'）: {reload_err}") from reload_err
            # 检查是否是 image_proj_model_in_features 错误
            elif "image_proj_model_in_features" in error_str:
                print(f"  ✗ InstantID IP-Adapter 未正确加载: {e}")
                raise RuntimeError(
                    f"InstantID IP-Adapter 未正确加载，请检查 IP-Adapter 文件是否正确: {e}") from e
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
        """兼容旧逻辑，转调新方法"""
        return self._ensure_instantid_adapter_active()

    def _reload_instantid_ip_adapter(self):
        """兼容旧逻辑，强制重新加载 InstantID IP-Adapter"""
        try:
            self._load_instantid_ip_adapter_once(force=True)
            return True
        except Exception as err:
            print(f"  ✗ InstantID IP-Adapter 重新加载失败: {err}")
            self._instantid_ip_adapter_ready = False
            return False

    def _load_ip_adapter_from_file(self, ip_adapter_file: Path) -> None:
        """根据不同 Pipeline API 将 IP-Adapter 加载进 pipeline"""
        if hasattr(self.pipeline, "load_ip_adapter_instantid"):
            self._load_ip_adapter_via_instantid_api(ip_adapter_file)
        elif hasattr(self.pipeline, "load_ip_adapter"):
            self._load_ip_adapter_via_diffusers_api(ip_adapter_file)
        else:
            raise AttributeError(
                "pipeline 既不支持 load_ip_adapter_instantid 也不支持 load_ip_adapter 方法")

    def _load_ip_adapter_via_instantid_api(
            self, ip_adapter_file: Path) -> None:
        """
        使用 InstantID API 加载 IP-Adapter

        注意：如果 IP-Adapter 已经部分加载（image_proj_model 存在），
        直接设置 image_proj_model_in_features，避免重新加载导致冲突
        """
        # 如果 image_proj_model 已经存在，说明 IP-Adapter 已经部分加载
        # 直接设置 image_proj_model_in_features，避免重新加载
        if hasattr(
                self.pipeline,
                "image_proj_model") and self.pipeline.image_proj_model is not None:
            print(
                "  ℹ 检测到 image_proj_model 已存在，跳过重新加载，直接设置 image_proj_model_in_features...")
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
                        if hasattr(
                                self.pipeline,
                                "image_proj_model_in_features"):
                            delattr(
                                self.pipeline, "image_proj_model_in_features")
                except Exception as e2:
                    print(f"  ⚠ 卸载失败: {e2}，继续尝试重新加载...")
        # 如果 image_proj_model 不存在，或者卸载成功，尝试重新加载
        try:
            self.pipeline.load_ip_adapter_instantid(str(ip_adapter_file))
        except Exception as err:
            error_str = str(err).lower()
            # 如果是 state dict 相关的错误，可能是文件格式问题或已加载冲突
            if "state dict" in error_str or "missing" in error_str or "no file" in error_str:
                if ip_adapter_file.is_file():
                    print(
                        f"  ⚠ 使用文件路径加载失败（可能原因: 已加载冲突/缺少HF文件），尝试复制为pytorch_model.bin后重新加载: {err}")
                    try:
                        # 将权重复制为 pytorch_model.bin 以兼容目录加载
                        pytorch_path = ip_adapter_file.parent / "pytorch_model.bin"
                        if pytorch_path != ip_adapter_file:
                            import shutil
                            shutil.copy2(ip_adapter_file, pytorch_path)
                            print(
                                f"  ✓ 已复制 {ip_adapter_file.name} -> {pytorch_path.name}")
                        # 如果存在 model_index.json，则尝试目录加载
                        model_index = ip_adapter_file.parent / "model_index.json"
                        if model_index.exists():
                            try:
                                self.pipeline.load_ip_adapter_instantid(
                                    str(ip_adapter_file.parent))
                                # 验证并确保 image_proj_model_in_features 已设置
                                if not hasattr(self.pipeline, "image_proj_model_in_features"):
                                    if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                                        try:
                                            self.pipeline.image_proj_model_in_features = 512
                                            print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                                        except Exception as e:
                                            raise RuntimeError(
                                                f"目录加载后无法设置 image_proj_model_in_features: {e}")
                                    else:
                                        raise RuntimeError(
                                            "目录加载后仍缺少 image_proj_model_in_features 和 image_proj_model")
                                print("  ✓ InstantID IP-Adapter 重新加载成功（目录路径）")
                                return
                            except Exception as dir_err:
                                print(f"  ⚠ 使用目录路径重新加载失败: {dir_err}")
                    except Exception as copy_err:
                        print(f"  ⚠ 复制权重为 pytorch_model.bin 失败: {copy_err}")
                # 如果目录加载也失败，最后尝试直接设置特征维度
                if hasattr(
                        self.pipeline,
                        "image_proj_model") and self.pipeline.image_proj_model is not None:
                    print(
                        f"  ⚠ 重新加载失败，但检测到 image_proj_model，尝试直接设置 image_proj_model_in_features...")
                    try:
                        self.pipeline.image_proj_model_in_features = 512
                        print("  ✓ 已手动设置 image_proj_model_in_features = 512")
                        return
                    except Exception as e3:
                        raise RuntimeError(
                            f"无法重新加载 IP-Adapter，且无法手动修复: {e3}") from err
                raise RuntimeError(f"无法重新加载 IP-Adapter: {err}") from err
            else:
                raise

        # 验证并确保 image_proj_model_in_features 已设置
        if not hasattr(self.pipeline, "image_proj_model_in_features"):
            # 如果缺少 image_proj_model_in_features，但存在 image_proj_model，尝试手动设置
            if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                try:
                    self.pipeline.image_proj_model_in_features = 512
                    print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                except Exception as e:
                    raise RuntimeError(
                        f"load_ip_adapter_instantid 调用成功，但无法设置 image_proj_model_in_features: {e}")
            else:
                raise RuntimeError(
                    "load_ip_adapter_instantid 调用成功，但 image_proj_model_in_features 未设置且 image_proj_model 不存在")
        print("  ✓ InstantID IP-Adapter 重新加载成功（使用 load_ip_adapter_instantid）")

    def _load_ip_adapter_via_diffusers_api(
            self, ip_adapter_file: Path) -> None:
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
                    self.pipeline.load_ip_adapter(
                        pretrained_path, subfolder="", weight_name=weight_name)
                else:
                    self.pipeline.load_ip_adapter(
                        pretrained_path, weight_name=weight_name)
            elif "subfolder" in params and "weight_name" in params:
                # 必需参数方式：使用父目录作为 subfolder，文件名作为 weight_name
                subfolder = str(ip_adapter_file.parent)
                weight_name = ip_adapter_file.name
                self.pipeline.load_ip_adapter(
                    subfolder=subfolder, weight_name=weight_name)
            else:
                # 回退：直接传递文件路径（可能不支持，但尝试一下）
                self.pipeline.load_ip_adapter(str(ip_adapter_file))
        elif ip_adapter_file.is_dir():
            # 目录情况：作为 HuggingFace 格式的模型目录
            if "pretrained_model_name_or_path_or_dict" in params:
                pretrained_path = str(ip_adapter_file)
                weight_name = "ip-adapter.bin"
                if "subfolder" in params:
                    self.pipeline.load_ip_adapter(
                        pretrained_path, subfolder="", weight_name=weight_name)
                else:
                    self.pipeline.load_ip_adapter(
                        pretrained_path, weight_name=weight_name)
            elif "subfolder" in params and "weight_name" in params:
                subfolder = str(ip_adapter_file)
                weight_name = "ip-adapter.bin"
                self.pipeline.load_ip_adapter(
                    subfolder=subfolder, weight_name=weight_name)
            else:
                self.pipeline.load_ip_adapter(str(ip_adapter_file))
        else:
            raise FileNotFoundError(f"IP-Adapter 路径不存在: {ip_adapter_file}")

        # 验证并确保 image_proj_model_in_features 已设置
        if not hasattr(self.pipeline, "image_proj_model_in_features"):
            # 如果缺少 image_proj_model_in_features，但存在 image_proj_model，尝试手动设置
            if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
                try:
                    self.pipeline.image_proj_model_in_features = 512
                    print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                except Exception as e:
                    raise RuntimeError(
                        f"load_ip_adapter 调用后无法设置 image_proj_model_in_features: {e}")
            else:
                raise RuntimeError(
                    "load_ip_adapter 调用后仍缺少 image_proj_model_in_features 和 image_proj_model")
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
            ordered = {
                "image_proj": state["image_proj"],
                "ip_adapter": state["ip_adapter"]}
            torch.save(ordered, ip_adapter_file)
        return ip_adapter_file

    def _load_instantid_ip_adapter_once(self, force: bool = False) -> None:
        if self.engine != "instantid":
            return
        if self.pipeline is None:
            # 自动加载pipeline（如果未加载）
            print("⚠️  Pipeline未加载，正在自动加载...")
            self.load_pipeline()
        if not force and getattr(self, "_instantid_ip_adapter_ready", False):
            return

        ip_adapter_path = self.instantid_config.get("ip_adapter_path")
        if not ip_adapter_path:
            raise ValueError(
                "instantid.ip_adapter_path 未配置，无法加载 IP-Adapter 权重")

        path_obj = Path(ip_adapter_path)
        if not path_obj.is_absolute():
            config_dir = Path(
                self.config_path).parent if hasattr(
                self, "config_path") else Path.cwd()
            candidate = config_dir / path_obj
            if candidate.exists():
                path_obj = candidate
            else:
                candidate = Path.cwd() / path_obj
                if candidate.exists():
                    path_obj = candidate
        if not path_obj.exists():
            raise FileNotFoundError(f"未找到 IP-Adapter 路径: {path_obj}")

        weight_file = self._resolve_ip_adapter_weight_file(path_obj)
        weight_file = self._normalize_ip_adapter_weights(weight_file)

        self._manual_load_instantid_ip_adapter(weight_file)
        self._instantid_ip_adapter_ready = True

    def _ensure_instantid_adapter_active(self) -> bool:
        if self.engine != "instantid":
            return True
        if self.pipeline is None:
            print("  ⚠ InstantID pipeline 未初始化")
            return False
        try:
            self._load_instantid_ip_adapter_once()
        except Exception as err:
            print(f"  ✗ InstantID IP-Adapter 加载失败: {err}")
            self._instantid_ip_adapter_ready = False
            return False
        
        # 检查 image_proj_model_in_features 是否存在
        if hasattr(self.pipeline, "image_proj_model_in_features"):
            return True
        
        # 如果缺少 image_proj_model_in_features，但存在 image_proj_model，尝试手动设置
        if hasattr(self.pipeline, "image_proj_model") and self.pipeline.image_proj_model is not None:
            try:
                # InstantID 默认使用 512
                self.pipeline.image_proj_model_in_features = 512
                print("  ℹ 已手动设置 image_proj_model_in_features = 512")
                return True
            except Exception as e:
                print(f"  ⚠ 无法手动设置 image_proj_model_in_features: {e}")
                # 即使设置失败，如果 image_proj_model 存在，仍然认为 IP-Adapter 已加载
                # 因为某些版本的 InstantID 可能不需要这个属性
                print("  ℹ 检测到 image_proj_model，假设 IP-Adapter 已加载")
                return True
        
        print("  ⚠ InstantID pipeline 缺少 image_proj_model_in_features 和 image_proj_model 属性")
        return False

    def _resolve_ip_adapter_weight_file(self, path_obj: Path) -> Path:
        """
        根据给定路径（文件或目录）解析出 IP-Adapter 权重文件
        """
        if path_obj.is_file():
            return path_obj

        if path_obj.is_dir():
            candidate_names = [
                "ip-adapter.bin",
                "ip_adapter.bin",
                "pytorch_model.bin",
                "model.safetensors",
                "model.bin",
            ]
            for name in candidate_names:
                cand = path_obj / name
                if cand.exists():
                    return cand

            for ext in (".bin", ".safetensors", ".pt"):
                found = sorted(path_obj.glob(f"*{ext}"))
                if found:
                    return found[0]

        raise FileNotFoundError(f"未找到可用的 IP-Adapter 权重文件: {path_obj}")

    def _manual_load_instantid_ip_adapter(self, ip_adapter_file: Path) -> None:
        """
        使用 InstantID Pipeline 的 set_image_proj_model/set_ip_adapter 方法手动加载 IP-Adapter，
        避免依赖 HuggingFace 目录结构。
        """
        if not hasattr(
                self.pipeline,
                "set_image_proj_model") or not hasattr(
                self.pipeline,
                "set_ip_adapter"):
            raise RuntimeError(
                f"当前 pipeline ({
                    type(
                        self.pipeline).__name__}) 不支持 InstantID 的手动 IP-Adapter 加载")

        image_emb_dim = int(
            self.instantid_config.get(
                "ip_adapter_image_emb_dim", 512))
        num_tokens = int(self.instantid_config.get("ip_adapter_tokens", 16))
        if num_tokens <= 0:
            num_tokens = 16

        if self.ip_adapter_scales:
            default_scale = float(self.ip_adapter_scales[0])
        else:
            default_scale = float(
                self.instantid_config.get(
                    "ip_adapter_scale", 0.8))

        self.pipeline.set_image_proj_model(
            str(ip_adapter_file),
            image_emb_dim=image_emb_dim,
            num_tokens=num_tokens)
        self.pipeline.set_ip_adapter(
            str(ip_adapter_file),
            num_tokens=num_tokens,
            scale=default_scale)
        self.pipeline.image_proj_model_in_features = image_emb_dim

    def _generate_image_kolors(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        scene: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """使用 Kolors 生成图像（真实感场景）
        
        使用 Kolors-IP-Adapter-FaceID-Plus 版本，可直接用 diffusers 加载
        """
        if self.kolors_pipeline is None:
            # 自动加载 pipeline（如果未加载）
            print("⚠️  Kolors pipeline 未加载，正在自动加载...")
            self._load_kolors_pipeline()
        
        if self.kolors_pipeline is None:
            raise RuntimeError("Kolors pipeline 加载失败")
        
        # 获取配置
        model_selection = self.image_config.get("model_selection", {})
        scene_config = model_selection.get("scene", {})
        kolors_config = scene_config.get("kolors", {})
        
        # 使用配置中的参数，如果没有则使用传入的参数
        guidance = guidance_scale or kolors_config.get("guidance_scale", 7.0)
        steps = num_inference_steps or kolors_config.get("num_inference_steps", 40)
        width = kolors_config.get("width", 1536)
        height = kolors_config.get("height", 864)
        
        # 生成器（用于随机种子）
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"🎨 使用 Kolors 生成图像")
        print(f"   提示词: {prompt[:50]}...")
        print(f"   分辨率: {width}x{height}")
        print(f"   推理步数: {steps}")
        print(f"   引导尺度: {guidance}")
        
        # Kolors tokenizer 有严重 bug，需要严格限制提示词长度，避免 OverflowError
        # Kolors tokenizer 在处理长文本时会出现整数溢出
        # 使用非常保守的长度限制（100字符），确保不会溢出
        max_chars = 100  # 字符长度限制（非常保守，避免 tokenizer 溢出）
        
        # 按字符长度限制
        safe_prompt = prompt[:max_chars] if len(prompt) > max_chars else prompt
        safe_negative = negative_prompt[:max_chars] if negative_prompt and len(negative_prompt) > max_chars else (negative_prompt or "")
        
        if len(prompt) > max_chars:
            print(f"  ⚠ Kolors tokenizer 限制：提示词长度从 {len(prompt)} 字符截断到 {len(safe_prompt)} 字符（避免溢出）")
        if negative_prompt and len(negative_prompt) > max_chars:
            print(f"  ⚠ Kolors tokenizer 限制：负面提示词长度从 {len(negative_prompt)} 字符截断到 {len(safe_negative)} 字符（避免溢出）")
        
        # 调用 Kolors pipeline（按照官方用法）
        try:
            result = self.kolors_pipeline(
                prompt=safe_prompt,
                negative_prompt=safe_negative or "",
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            
            # 保存图像
            image = result.images[0]
            image.save(output_path)
            print(f"✓ Kolors 图像生成完成: {output_path}")
            return output_path
            
        except OverflowError as e:
            # Kolors tokenizer 溢出错误，尝试进一步缩短提示词到非常短的长度
            print(f"  ⚠ Kolors tokenizer 溢出错误，尝试进一步缩短提示词: {e}")
            # 使用非常短的长度（50字符），确保不会溢出
            shorter_prompt = prompt[:50] if len(prompt) > 50 else prompt
            shorter_negative = negative_prompt[:50] if negative_prompt and len(negative_prompt) > 50 else (negative_prompt or "")
            print(f"  ⚠ 提示词已缩短到 {len(shorter_prompt)} 字符，负面提示词已缩短到 {len(shorter_negative)} 字符")
            
            result = self.kolors_pipeline(
                prompt=shorter_prompt,
                negative_prompt=shorter_negative or "",
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            
            image = result.images[0]
            image.save(output_path)
            print(f"  ✅ Kolors 图像生成成功（使用缩短的提示词）: {output_path}")
            return output_path
        except Exception as e:
            print(f"  ❌ Kolors 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Kolors 图像生成失败: {e}") from e

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
        # 传入已选择的场景参考图像，避免重复调用
        scene_reference_images: Optional[List[Path]] = None,
        # 用于img2img的场景参考图像
        scene_reference_image_for_img2img: Optional[Path] = None,
        character_lora: Optional[str] = None,  # 角色LoRA适配器名称
        style_lora: Optional[str] = None,  # 风格LoRA适配器名称
    ) -> Path:
        """使用 SDXL 生成图像（旧方案）"""
        if self.pipeline is None:
            # 自动加载pipeline（如果未加载）
            print("⚠️  Pipeline未加载，正在自动加载...")
            self.load_pipeline()

        # 使用 override 值（如果提供），否则使用配置值
        use_ip_adapter = use_ip_adapter_override if use_ip_adapter_override is not None else self.use_ip_adapter

        # 检查 self.pipeline 是否是 InstantID pipeline
        # 如果是，需要使用普通的 SDXL pipeline
        is_instantid_pipeline = False
        try:
            from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
            is_instantid_pipeline = isinstance(
                self.pipeline, StableDiffusionXLInstantIDPipeline)
        except ImportError:
            # 如果不能导入，检查 pipeline 的类型名称
            pipeline_type_name = type(self.pipeline).__name__
            is_instantid_pipeline = "InstantID" in pipeline_type_name or "instantid" in pipeline_type_name.lower()

        if is_instantid_pipeline:
            # ⚡ 关键修复：方案2需要使用 SDXL + IP-Adapter，而不是 Flux.1
            # 如果使用 IP-Adapter（方案2），必须使用 SDXL pipeline
            if use_ip_adapter:
                # 方案2：强制使用 SDXL pipeline（支持 IP-Adapter）
                if self.sdxl_pipeline is None:
                    try:
                        instantid_pipeline = self.pipeline
                        self._load_sdxl_pipeline(load_lora=False)
                        # ⚡ 关键修复：验证pipeline组件完整性，避免KeyError 'unet'
                        if self.pipeline is not None:
                            if hasattr(self.pipeline, 'components'):
                                components = self.pipeline.components
                                if isinstance(components, dict) and 'unet' not in components:
                                    # 如果components缺少unet，重新加载
                                    self._load_sdxl_pipeline(load_lora=False)
                            elif not hasattr(self.pipeline, 'unet') or self.pipeline.unet is None:
                                # 如果pipeline没有unet属性，重新加载
                                self._load_sdxl_pipeline(load_lora=False)
                        self.sdxl_pipeline = self.pipeline
                        self.pipeline = instantid_pipeline
                    except Exception as e:
                        import traceback
                        print(f"  ✗ 无法加载 SDXL pipeline（方案2需要）: {e}")
                        print(f"  📋 详细错误:\n{traceback.format_exc()}")
                        raise RuntimeError(f"无法加载 SDXL pipeline用于方案2: {e}") from e
                pipeline_to_use = self.sdxl_pipeline
            else:
                # 如果不使用 IP-Adapter，可以使用 Flux.1（质量更好）
                if self.flux1_pipeline is None:
                    try:
                        instantid_pipeline = self.pipeline
                        original_engine = self.engine
                        self.engine = "flux1"
                        self._load_flux1_pipeline()
                        self.sdxl_pipeline = self.flux1_pipeline  # 复用变量名，实际是Flux.1
                        self.pipeline = instantid_pipeline
                        self.engine = original_engine
                    except Exception as e:
                        # 如果 Flux.1 失败，回退到 SDXL
                        instantid_pipeline = self.pipeline
                        self._load_sdxl_pipeline(load_lora=False)
                        self.sdxl_pipeline = self.pipeline
                        self.pipeline = instantid_pipeline
                pipeline_to_use = self.sdxl_pipeline if self.sdxl_pipeline is not None else self.pipeline
        else:
            # 使用普通的 pipeline
            pipeline_to_use = self.pipeline
        
        # 检查 pipeline 是否为 None
        if pipeline_to_use is None:
            # 尝试重新加载 pipeline
            print("  ⚠️  Pipeline 为 None，尝试重新加载...")
            self.load_pipeline()
            pipeline_to_use = self.pipeline
            
            # 如果仍然为 None，抛出错误
            if pipeline_to_use is None:
                raise RuntimeError(
                    f"Pipeline 加载失败。当前引擎: {self.engine}\n"
                    f"请检查：\n"
                    f"1. 模型文件是否存在\n"
                    f"2. 模型路径是否正确\n"
                    f"3. 显存是否足够\n"
                    f"4. 依赖是否已安装\n"
                    f"5. 尝试使用其他引擎（如 'sdxl' 或 'kolors'）"
                )

        # 初始化 init_image（用于检查是否有有效的 init_image）
        init_image = None

        # 如果启用 IP-Adapter，确保它已经被加载
        if use_ip_adapter:
            target_pipe = pipeline_to_use or self.pipeline
            print(f"  🔍 [DEBUG] 检查 IP-Adapter，target_pipe: {type(target_pipe).__name__ if target_pipe else 'None'}")
            
            # ⚡ 关键修复：在加载 IP-Adapter 之前，验证 pipeline 的 unet 组件
            if target_pipe is not None:
                try:
                    if not hasattr(target_pipe, 'unet'):
                        raise AttributeError(f"Pipeline {type(target_pipe).__name__} 缺少 'unet' 属性")
                    unet = target_pipe.unet
                    if unet is None:
                        raise AttributeError(f"Pipeline {type(target_pipe).__name__}.unet 为 None")
                    print(f"  ✓ Pipeline unet 验证成功（IP-Adapter 加载前）: {type(unet).__name__}")
                except (AttributeError, KeyError) as unet_error:
                    print(f"  ⚠ Pipeline unet 验证失败（IP-Adapter 加载前）: {unet_error}")
                    import traceback
                    print(f"  📋 完整错误堆栈:\n{traceback.format_exc()}")
                    raise RuntimeError(f"Pipeline 组件验证失败，无法加载 IP-Adapter: {unet_error}") from unet_error
            
            # 检查 IP-Adapter 是否已加载（检查是否有 ip_adapter_image_processor 属性且不为 None）
            ip_adapter_loaded = False
            if hasattr(
                    target_pipe,
                    "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
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
                except BaseException:
                    ip_adapter_loaded = False

            if not ip_adapter_loaded:
                # IP-Adapter 未加载，尝试加载
                print(f"  ⚠ IP-Adapter 未加载，尝试加载...")
                try:
                    # 每次生成前都重新加载 IP-Adapter，确保使用正确的适配器
                    # 先卸载旧的 IP-Adapter（如果存在）
                    target_pipe = pipeline_to_use or self.pipeline
                    if hasattr(
                            target_pipe,
                            "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
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
                    if hasattr(
                            self.pipeline,
                            "ip_adapter_image_processor") and self.pipeline.ip_adapter_image_processor is not None:
                        # 检查是否是 InstantID 的 IP-Adapter（InstantID 的 IP-Adapter
                        # 没有 prepare_ip_adapter_image_embeds 方法）
                        is_instantid_adapter = not hasattr(
                            self.pipeline, "prepare_ip_adapter_image_embeds")
                        if is_instantid_adapter or self.engine == "instantid":
                            print(
                                f"  ℹ 检测到 InstantID 的 IP-Adapter，卸载 InstantID 的 IP-Adapter")
                            # InstantID 和 SDXL 的 IP-Adapter 不兼容，需要先卸载
                            try:
                                if hasattr(
                                        self.pipeline, "disable_ip_adapter"):
                                    self.pipeline.disable_ip_adapter()
                                    print(f"  ✓ 已卸载 InstantID 的 IP-Adapter")
                                elif hasattr(self.pipeline, "unload_ip_adapter"):
                                    self.pipeline.unload_ip_adapter()
                                    print(f"  ✓ 已卸载 InstantID 的 IP-Adapter")
                            except Exception as e:
                                print(f"  ⚠ 无法卸载 InstantID 的 IP-Adapter: {e}")

                    # 加载新的 IP-Adapter
                    print(f"  ℹ 加载 SDXL IP-Adapter...")
                    # ⚡ 关键修复：确保 IP-Adapter 加载到正确的 pipeline（pipeline_to_use 或 sdxl_pipeline）
                    # 如果 pipeline_to_use 是 sdxl_pipeline，需要确保 IP-Adapter 加载到它上面
                    if pipeline_to_use is self.sdxl_pipeline and self.sdxl_pipeline is not None:
                        # 临时切换到 sdxl_pipeline，加载 IP-Adapter
                        original_pipeline = self.pipeline
                        self.pipeline = self.sdxl_pipeline
                        try:
                            self._load_ip_adapter()
                        finally:
                            self.pipeline = original_pipeline
                    else:
                        self._load_ip_adapter()
                    # 重新加载后，确保 pipeline_to_use 也加载了 IP-Adapter
                    if pipeline_to_use is self.img2img_pipeline and self.img2img_pipeline is not None:
                        # img2img_pipeline 的 IP-Adapter 会在 _load_ip_adapter 中自动加载
                        # 但为了确保，我们再次检查
                        if not (hasattr(self.img2img_pipeline, "ip_adapter_image_processor")
                                and self.img2img_pipeline.ip_adapter_image_processor is not None):
                            print(
                                f"  ℹ img2img_pipeline 的 IP-Adapter 未加载，尝试加载...")
                            try:
                                self._load_ip_adapter()  # 这会同时加载到 img2img_pipeline
                            except Exception as e:
                                print(
                                    f"  ⚠ 无法为 img2img_pipeline 加载 IP-Adapter: {e}")

                    # 验证 IP-Adapter 是否已正确加载
                    target_pipe = pipeline_to_use or self.pipeline
                    if hasattr(target_pipe, "prepare_ip_adapter_image_embeds"):
                        # 检查 image_projection_layers 是否存在
                        try:
                            # 尝试访问 image_projection_layers 来验证 IP-Adapter 是否已加载
                            if hasattr(
                                    target_pipe,
                                    "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
                                # 检查是否有 image_projection_layers（通过检查 processor
                                # 的内部属性）
                                if hasattr(
                                        target_pipe.ip_adapter_image_processor,
                                        "image_encoder") or hasattr(
                                        target_pipe,
                                        "image_projection_layers"):
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

        # 动态控制 LoRA 权重（包括角色LoRA和风格LoRA）
        # 如果使用 LoRA，降低 IP-Adapter 权重以避免冲突导致脸部变形
        # 重要：如果character_lora和style_lora都是None，表示不使用LoRA，只使用参考图生成正常图像
        ip_adapter_scale_override = None
        # ⚡ 关键修复：使用 pipeline_to_use 而不是 self.pipeline（两阶段法中 self.pipeline 是 InstantID，pipeline_to_use 是 SDXL）
        target_pipeline_for_lora = pipeline_to_use if pipeline_to_use is not None else self.pipeline
        if hasattr(target_pipeline_for_lora, "set_adapters"):
            # 构建LoRA适配器列表（角色LoRA + 风格LoRA）
            adapter_names = []
            adapter_weights = []

            # 只有当用户明确指定了character_lora时才使用角色LoRA
            # 如果character_lora是None，表示不使用角色LoRA（不使用默认的hanli）
            if character_lora is not None:
                if character_lora == "":
                    print(f"  ℹ 已禁用角色LoRA（用户指定）")
                else:
                    # MVP 流程：character_lora 是 adapter_name，需要先检查是否已加载
                    # 如果未加载，需要先加载 LoRA 文件
                    # 将 character_lora 名称解析为实际文件路径
                    lora_path = None
                    if character_lora == "host_person_v2":
                        lora_path = str(self.models_root / "lora" / "host_person_v2" / "pytorch_lora_weights.safetensors")
                    elif character_lora == "host_person":
                        lora_path = str(self.models_root / "lora" / "host_person" / "pytorch_lora_weights.safetensors")
                    elif character_lora in ["kepu_gege", "weilai_jiejie"]:
                        # 科普主持人映射到 host_person_v2
                        lora_path = str(self.models_root / "lora" / "host_person_v2" / "pytorch_lora_weights.safetensors")
                    else:
                        # 尝试作为目录名或路径
                        lora_path_obj = Path(character_lora)
                        if not lora_path_obj.is_absolute():
                            lora_path_obj = self.models_root / "lora" / character_lora / "pytorch_lora_weights.safetensors"
                        if lora_path_obj.exists():
                            lora_path = str(lora_path_obj)
                    
                    # 检查 adapter 是否已加载
                    loaded_adapters = []
                    if hasattr(pipeline_to_use, "get_active_adapters"):
                        try:
                            loaded_adapters = list(pipeline_to_use.get_active_adapters()) if pipeline_to_use.get_active_adapters() else []
                        except:
                            pass
                    elif hasattr(pipeline_to_use, "peft_config") and pipeline_to_use.peft_config:
                        loaded_adapters = list(pipeline_to_use.peft_config.keys())
                    
                    # 如果 adapter 未加载且提供了路径，先加载 LoRA
                    if character_lora not in loaded_adapters and lora_path and Path(lora_path).exists():
                        try:
                            print(f"  🔧 加载 LoRA 文件: {Path(lora_path).name} (adapter={character_lora})")
                            pipeline_to_use.load_lora_weights(lora_path, adapter_name=character_lora)
                            print(f"  ✅ LoRA 已加载: {character_lora}")
                        except Exception as e:
                            print(f"  ⚠ LoRA 加载失败: {e}，尝试使用已加载的 adapter")
                    
                    # 使用指定的角色LoRA（无论是否刚加载）
                    adapter_names.append(character_lora)
                    adapter_weights.append(self.lora_alpha)
                    print(f"  ✓ 使用用户指定的角色LoRA: {character_lora} (alpha={self.lora_alpha:.2f})")
            else:
                # character_lora是None，不使用角色LoRA（不使用默认的hanli）
                print(f"  ℹ 未指定角色LoRA，禁用角色LoRA（仅使用参考图）")

            # 添加风格LoRA（如果启用）
            if style_lora is not None:
                # 如果 style_lora 是空字符串，禁用风格LoRA
                if style_lora == "":
                    print(f"  ℹ 已禁用风格LoRA（用户指定）")
                else:
                    # 使用用户指定的风格LoRA
                    # ⚡ 关键修复：优先使用 Execution Planner 指定的权重（0.35，低权重，只绑定风格，不抢戏）
                    style_alpha = 0.35  # Execution Planner 推荐的权重
                    
                    # 检查 adapter 是否已加载
                    loaded_adapters = []
                    if hasattr(pipeline_to_use, "get_active_adapters"):
                        try:
                            loaded_adapters = list(pipeline_to_use.get_active_adapters()) if pipeline_to_use.get_active_adapters() else []
                        except:
                            pass
                    elif hasattr(pipeline_to_use, "peft_config") and pipeline_to_use.peft_config:
                        loaded_adapters = list(pipeline_to_use.peft_config.keys())
                    
                    # 如果 adapter 未加载，先加载 LoRA
                    if style_lora not in loaded_adapters:
                        # 尝试从配置中获取风格 LoRA 路径
                        style_config = self.lora_config.get("style_lora", {})
                        style_lora_path = None
                        if isinstance(style_config, dict):
                            style_lora_path = style_config.get("weights_path")
                        
                        if style_lora_path and Path(style_lora_path).exists():
                            try:
                                print(f"  🔧 加载风格 LoRA 文件: {Path(style_lora_path).name} (adapter={style_lora})")
                                pipeline_to_use.load_lora_weights(style_lora_path, adapter_name=style_lora)
                                print(f"  ✅ 风格 LoRA 已加载: {style_lora}")
                            except Exception as e:
                                print(f"  ⚠ 风格 LoRA 加载失败: {e}，尝试使用已加载的 adapter")
                    
                    adapter_names.append(style_lora)
                    adapter_weights.append(style_alpha)
                    print(f"  ✓ 使用风格锚点（风格 LoRA）: {style_lora} (alpha={style_alpha:.2f}，Execution Planner 推荐权重)")
            else:
                # style_lora是None，不使用风格LoRA（不使用默认的anime_style）
                print(f"  ℹ 未指定风格LoRA，禁用风格LoRA（仅使用参考图）")

            # 应用所有LoRA适配器（每次生成都重新应用，避免被覆盖）
            if adapter_names:
                # ⚡ 关键修复：使用 target_pipeline_for_lora 而不是 self.pipeline
                try:
                    target_pipeline_for_lora.set_adapters(
                        adapter_names, adapter_weights=adapter_weights)
                    print(
                        f"  ✓ 已应用LoRA适配器: {adapter_names} (权重: {adapter_weights})")
                except Exception as e:
                    print(f"  ⚠ 应用LoRA适配器失败: {e}，跳过LoRA操作")
                    adapter_names = []  # 清空适配器列表，继续执行
                # 如果使用 LoRA，降低 IP-Adapter 权重以避免冲突
                if use_lora is not False:  # 只有当明确禁用时才不降低
                    ip_adapter_scale_override = 0.35  # 降低 IP-Adapter 权重
            else:
                # 如果没有适配器，完全禁用并卸载所有LoRA（只使用参考图生成正常图像）
                # 检查是否是 Flux pipeline（Flux 使用 transformer，LoRA 处理不同）
                is_flux = False
                if target_pipeline_for_lora is not None:
                    pipeline_type = type(target_pipeline_for_lora).__name__
                    is_flux = "Flux" in pipeline_type or "flux" in pipeline_type.lower()
                
                if not is_flux:
                    # ⚡ 关键修复：使用 target_pipeline_for_lora 而不是 self.pipeline
                    # 添加安全检查，确保 pipeline 支持 set_adapters
                    try:
                        if hasattr(target_pipeline_for_lora, "set_adapters"):
                            # 检查是否有 _component_adapter_weights 属性（diffusers 内部字典）
                            if hasattr(target_pipeline_for_lora, "_component_adapter_weights"):
                                # 首先禁用适配器
                                target_pipeline_for_lora.set_adapters([])
                            else:
                                print(f"  ℹ Pipeline 不支持 LoRA 适配器操作（缺少 _component_adapter_weights），跳过")
                        else:
                            print(f"  ℹ Pipeline 不支持 set_adapters 方法，跳过 LoRA 操作")
                    except (KeyError, AttributeError) as e:
                        # 如果 set_adapters 失败（例如 KeyError 'unet'），跳过操作
                        print(f"  ⚠ 禁用 LoRA 适配器时出错: {e}，跳过 LoRA 操作")
                    except Exception as e:
                        print(f"  ⚠ 禁用 LoRA 适配器时出错: {e}，跳过 LoRA 操作")
                    # 尝试卸载所有已加载的LoRA权重，确保完全清除影响
                    try:
                        self._unload_all_lora_adapters(target_pipeline_for_lora, "主pipeline")
                    except Exception as e:
                        print(f"  ⚠ 卸载 LoRA 权重时出错: {e}，继续执行")
                else:
                    print(f"  ℹ Flux pipeline 检测到，跳过 LoRA 操作（Flux 架构不同）")
            
            # 同时应用到img2img pipeline
            if self.img2img_pipeline is not None and hasattr(
                    self.img2img_pipeline, "set_adapters"):
                if adapter_names:
                    self.img2img_pipeline.set_adapters(
                        adapter_names, adapter_weights=adapter_weights)
                    print(
                        f"  ✓ 已应用LoRA适配器到img2img pipeline: {adapter_names} (权重: {adapter_weights})")
                else:
                    self.img2img_pipeline.set_adapters([])
                    # 同样尝试卸载img2img pipeline的LoRA
                    self._unload_all_lora_adapters(self.img2img_pipeline, "img2img pipeline")

            # 使用 LoRA 时，降低 IP-Adapter 权重（从 0.7 降到 0.3-0.4）以避免冲突
            if use_lora and use_ip_adapter:
                ip_adapter_scale_override = 0.35  # 降低 IP-Adapter 权重

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        guidance = guidance_scale or self.sdxl_config.get(
            "guidance_scale", 8.5) or self.image_config.get(
            "guidance_scale", 7.5)
        steps = num_inference_steps or self.sdxl_config.get(
            "num_inference_steps", 40) or self.image_config.get(
            "num_inference_steps", 40)  # 统一为 40 步（与 InstantID 保持一致）

        negative_prompt = (
            negative_prompt
            if negative_prompt is not None
            else self.negative_prompt
        )

        init_image = None
        ip_adapter_image = None
        text_pipeline = pipeline_to_use
        img2img_strength = self.sdxl_config.get(
            "img2img_strength", 0.4)  # 默认0.4，平衡结构和细节

        # 优先使用场景参考图像作为img2img输入（如果启用）
        # 这样可以更好地保持参考图像的结构和风格
        if scene_reference_image_for_img2img and Path(
                scene_reference_image_for_img2img).exists():
            from PIL import Image, ImageOps
            try:
                init_image = Image.open(
                    scene_reference_image_for_img2img).convert("RGB")

                # 检测参考图像质量：如果图像主要是纹理/背景（方差低），降低strength或禁用img2img
                import numpy as np
                img_array = np.array(init_image)
                # 计算图像的标准差（衡量图像复杂度）
                img_std = np.std(img_array)
                # 如果标准差很低（<30），说明图像主要是单一颜色/纹理，不适合作为img2img输入
                if img_std < 30:
                    print(
                        f"  ⚠ 场景参考图像质量较低（主要是纹理，std={
                            img_std:.1f}），禁用img2img，仅使用IP-Adapter")
                    init_image = None
                    pipeline_to_use = text_pipeline
                else:
                    init_image = ImageOps.fit(
                        init_image,
                        (self.width, self.height),
                        method=Image.Resampling.LANCZOS,
                    )
                    # 根据场景类型动态调整img2img_strength
                    if scene:
                        # 如果是人物场景，使用更低的strength（0.25-0.3），保留更多细节，避免参考图覆盖人物
                        # 如果是环境场景，使用中等strength（0.35-0.4），保持结构但不过度
                        characters = self._identify_characters_in_scene(scene)
                        has_character = len(characters) > 0
                        if has_character:
                            img2img_strength = 0.28  # 人物场景：更低strength，避免参考图覆盖人物细节
                        else:
                            img2img_strength = 0.38  # 环境场景：中等strength，保持结构但不过度

                    if self.img2img_pipeline is not None:
                        pipeline_to_use = self.img2img_pipeline
                        print(
                            f"  ✓ 使用场景参考图像进行img2img: {
                                scene_reference_image_for_img2img.name} (strength={
                                img2img_strength:.2f}, std={
                                img_std:.1f})")
                    else:
                        print(f"  ⚠ img2img pipeline未加载，使用普通文生图")
                        init_image = None
                        pipeline_to_use = text_pipeline
            except Exception as e:
                print(f"  ⚠ 无法加载场景参考图像: {e}，使用普通文生图")
                init_image = None
                pipeline_to_use = text_pipeline
        # 如果没有场景参考图像，尝试使用传入的reference_image_path
        elif (
            use_ip_adapter
            and reference_image_path
            and Path(reference_image_path).exists()
            and self.img2img_pipeline is not None
            and self.use_img2img
        ):
            from PIL import Image, ImageOps
            try:
                init_image = Image.open(reference_image_path).convert("RGB")
                init_image = ImageOps.fit(
                    init_image,
                    (self.width, self.height),
                    method=Image.Resampling.LANCZOS,
                )
                pipeline_to_use = self.img2img_pipeline
                img2img_strength = self.sdxl_config.get("img2img_strength", 0.4)
                print(
                    f"  ✓ 使用 reference_image_path 进行 img2img: {reference_image_path} (strength={
                        img2img_strength:.2f})")
            except Exception as e:
                print(f"  ⚠ 无法加载 reference_image_path: {e}，使用普通文生图")
                init_image = None
                pipeline_to_use = text_pipeline
        else:
            pipeline_to_use = text_pipeline
            if reference_image_path and not Path(
                    reference_image_path).exists():
                print(
                    f"  ⚠ reference_image_path 不存在或无效: {reference_image_path}，使用普通文生图")
            if not use_ip_adapter:
                print(f"  ℹ IP-Adapter 不可用，使用普通文生图 pipeline（不使用 img2img）")

        if use_ip_adapter:
            from PIL import Image, ImageOps

            clip_source = None
            face_source = None

            # 优先使用场景参考图像（如果启用且存在）
            # 如果已经传入scene_reference_images，直接使用；否则重新选择
            scene_ref_images = scene_reference_images if scene_reference_images is not None else []
            if not scene_ref_images and scene and self.use_scene_reference:
                scene_ref_images = self._select_scene_reference_images(scene)

            if scene_ref_images:
                # 使用多个场景参考图像（如果IP-Adapter支持）
                # 否则只使用第一个
                if len(scene_ref_images) > 1:
                    # 尝试使用多参考图像
                    clip_source = [Image.open(p).convert("RGB")
                                   for p in scene_ref_images[:3]]  # 最多3张
                    print(f"  ℹ 使用 {len(clip_source)} 个场景参考图像进行IP-Adapter")
                else:
                    clip_source = Image.open(
                        scene_ref_images[0]).convert("RGB")
                    print(f"  ℹ 使用场景参考图像: {scene_ref_images[0].name}")

            # 如果没有场景参考图像，使用传入的参考图像
            if clip_source is None:
                if reference_image_path and Path(reference_image_path).exists():
                    clip_source = Image.open(reference_image_path).convert("RGB")
                    print(
                        f"  ℹ 使用传入的 reference_image_path: {reference_image_path}")

            if face_reference_image_path and Path(
                    face_reference_image_path).exists():
                face_source = Image.open(
                    face_reference_image_path).convert("RGB")
                print(
                    f"  ℹ 使用传入的 face_reference_image_path: {face_reference_image_path}")

            # 如果没有场景参考图像且没有传入参考图像，尝试使用face_source
            if clip_source is None and face_source is not None:
                clip_source = face_source
            if face_source is None and clip_source is not None and not isinstance(
                    clip_source, list):
                face_source = clip_source

            # 如果没有参考图像，禁用 IP-Adapter（避免错误）
            if clip_source is None:
                print(f"  ⚠ 未提供参考图像，禁用 IP-Adapter（使用纯文生图）")
                use_ip_adapter = False
                ip_adapter_image = None
            else:
                # 有参考图像，继续处理 IP-Adapter
                print(f"  ✓ 使用参考图像进行 IP-Adapter 生成")
                clip_size = int(
                    self.ip_adapter_config.get(
                        "clip_image_size", 224))
                face_size = int(
                    self.ip_adapter_config.get(
                        "face_image_size", 128))
                face_crop_ratio = float(
                    self.ip_adapter_config.get(
                        "face_crop_ratio", 1.0))
                face_only = bool(
                    self.ip_adapter_config.get(
                        "face_reference_only", False))

                prepared_images: List[Any] = []

                # 处理场景参考图像（可能是列表）
                if isinstance(clip_source, list):
                    # 多参考图像：为每张图像准备
                    for ref_img in clip_source:
                        prepared_images.append(
                            ImageOps.fit(
                                ref_img,
                                (clip_size, clip_size),
                                method=Image.Resampling.LANCZOS,
                            )
                        )
                else:
                    # 单参考图像：使用原有逻辑
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
                                source_img = source_img.crop(
                                    (left, top, right, bottom))

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
                # ⚡ 方案2：如果使用两阶段法，使用0.65（推荐值）
                if hasattr(self, '_two_stage_ip_adapter_scale') and self._two_stage_ip_adapter_scale is not None:
                    scale = self._two_stage_ip_adapter_scale
                    print(f"  ✓ 方案2：使用IP-Adapter scale={scale:.2f}（两阶段法推荐值）")
                # 如果设置了 override，使用 override 值（使用 LoRA 时降低 IP-Adapter 权重）
                elif ip_adapter_scale_override is not None:
                    scale = ip_adapter_scale_override
                else:
                    scale = (
                        self.ip_adapter_scales
                        if len(self.ip_adapter_scales) > 1
                        else self.ip_adapter_scales[0]
                    )
                target_pipe.set_ip_adapter_scale(scale)
            # 检查是否是 InstantID pipeline（InstantID 不使用
            # prepare_ip_adapter_image_embeds）
            is_instantid_pipeline = (
                self.engine == "instantid" or
                hasattr(target_pipe, "image_proj_model") or
                "InstantID" in type(target_pipe).__name__
            )

            if is_instantid_pipeline:
                # InstantID 不使用 prepare_ip_adapter_image_embeds，直接传递 image_embeds
                # InstantID 的 IP-Adapter 在 _generate_image_instantid 中处理
                print(
                    f"  ℹ InstantID pipeline：跳过 prepare_ip_adapter_image_embeds（InstantID 使用 image_embeds 参数）")
                prepared = None
                use_ip_adapter = False  # InstantID 有自己的 IP-Adapter 处理方式
            elif hasattr(target_pipe, "prepare_ip_adapter_image_embeds"):
                # 在调用之前，先验证 IP-Adapter 是否已正确加载
                # 检查是否有 image_projection_layers 属性（这是 IP-Adapter 的核心组件）
                ip_adapter_ready = False
                try:
                    # 检查 IP-Adapter 的关键属性是否存在
                    if hasattr(
                            target_pipe,
                            "ip_adapter_image_processor") and target_pipe.ip_adapter_image_processor is not None:
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
                    except (AttributeError, TypeError, NameError, KeyError) as e:
                        error_str = str(e)
                        # ⚡ 修复：捕获 KeyError（如 'unet'），这可能发生在 pipeline 内部访问 components 时
                        if isinstance(e, KeyError) and 'unet' in str(e):
                            print(f"  ✗ IP-Adapter 准备失败（KeyError 'unet'）: {e}")
                            print(f"  ℹ 这可能是 InstantID pipeline 组件问题，跳过 IP-Adapter 处理")
                            prepared = None
                            use_ip_adapter = False
                        # 检查是否是 InstantID pipeline 相关的错误
                        elif "image_proj_model_in_features" in error_str or (
                                "InstantID" in error_str and "image_proj" in error_str):
                            print(f"  ✗ 检测到 InstantID pipeline 错误: {e}")
                            print(
                                f"  ℹ InstantID 不使用 prepare_ip_adapter_image_embeds，跳过 IP-Adapter 处理")
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
                                    except BaseException:
                                        pass
                                elif hasattr(target_pipe, "unload_ip_adapter"):
                                    try:
                                        target_pipe.unload_ip_adapter()
                                    except BaseException:
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
                print(
                    f"  ⚠ 最终检查：pipeline_to_use 是 img2img_pipeline，但 init_image 无效，强制切换到普通 pipeline")

        # 如果提供了init_image（用于场景连贯性），使用img2img
        # 注意：init_image 可能是 Path 对象（从 scene 参数传入）或 PIL Image（从
        # reference_image_path 加载）
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

                # ⚡ 核心修复：对于 InstantID pipeline，直接使用其 img2img 功能，不创建单独的 img2img pipeline
                # 检查是否为 InstantID pipeline
                is_instantid_pipeline = False
                try:
                    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
                    is_instantid_pipeline = isinstance(self.pipeline, StableDiffusionXLInstantIDPipeline)
                except ImportError:
                    # 如果不能导入，检查 pipeline 的类型名称
                    pipeline_type_name = type(self.pipeline).__name__
                    is_instantid_pipeline = "InstantID" in pipeline_type_name or "instantid" in pipeline_type_name.lower()
                
                # 对于 InstantID pipeline，直接使用原 pipeline 的 img2img 功能（通过 image 参数）
                if is_instantid_pipeline:
                    print(f"  ℹ InstantID pipeline：直接使用原 pipeline 的 img2img 功能（通过 image 参数）")
                    self.img2img_pipeline = None  # 不需要单独的 img2img pipeline
                elif self.img2img_pipeline is None and self.pipeline:
                    # 对于非 InstantID pipeline，创建 img2img pipeline
                    try:
                        from diffusers import StableDiffusionXLImg2ImgPipeline
                        # 尝试使用 components 创建 img2img pipeline
                        # 注意：如果使用 enable_model_cpu_offload()，components 可能不是标准字典
                        try:
                            components = getattr(self.pipeline, 'components', None)
                            if components is None:
                                raise AttributeError("pipeline.components is None")
                            
                            # 检查 components 是否为字典或类似字典的对象
                            if not isinstance(components, dict):
                                raise TypeError(f"components is not a dict, got {type(components)}")
                            
                            # 检查必要的键是否存在
                            required_keys = ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']
                            missing_keys = [key for key in required_keys if key not in components]
                            if missing_keys:
                                raise KeyError(f"components missing required keys: {missing_keys}")
                            
                            # 尝试创建 pipeline
                            self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(**components)
                            print(f"  ✓ 使用 pipeline.components 创建 img2img pipeline 成功")
                            
                        except (KeyError, AttributeError, TypeError) as comp_error:
                            # 如果 components 方法失败，使用 from_pretrained（需要从 pipeline 获取 model_path）
                            print(f"  ℹ 无法使用 components 创建 img2img pipeline: {comp_error}，尝试使用 from_pretrained")
                            model_path = getattr(self.pipeline, '_model_path', None) or getattr(self, 'pipe_name', None)
                            if model_path:
                                dtype = torch.float16 if self.image_config.get("mixed_precision", "bf16") == "fp16" else torch.float32
                                self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                                    model_path, torch_dtype=dtype)
                                print(f"  ✓ 使用 from_pretrained 创建 img2img pipeline 成功")
                            else:
                                raise ValueError(f"无法确定 model_path 用于创建 img2img pipeline（components错误: {comp_error}）")
                        
                        # 配置 img2img pipeline
                        if self.img2img_pipeline:
                            if hasattr(self.img2img_pipeline, "enable_model_cpu_offload"):
                                self.img2img_pipeline.enable_model_cpu_offload()
                            else:
                                self.img2img_pipeline = self.img2img_pipeline.to(self.device)
                    except Exception as e:
                        print(f"  ⚠ 无法创建img2img pipeline: {type(e).__name__}: {e}，将使用普通生成")
                        import traceback
                        traceback.print_exc()
                        init_img = None
                        self.img2img_pipeline = None

                # ⚡ 核心修复：对于 InstantID pipeline，不支持 img2img，使用普通生成
                if is_instantid_pipeline:
                    # InstantID pipeline 不支持标准的 img2img 功能（通过 image 参数）
                    # 强制使用普通文生图，忽略 init_img
                    print(f"  ℹ InstantID pipeline 不支持场景连贯性 img2img，使用普通文生图")
                    init_img = None  # 禁用 img2img
                    pipeline_to_use = self.pipeline
                    # 清理 kwargs 中的 image 相关参数
                    kwargs_clean = {
                        k: v for k, v in kwargs.items() if k not in [
                            'image', 'ip_adapter_image', 'strength']}
                    result = pipeline_to_use(
                        width=self.width,
                        height=self.height,
                        **kwargs_clean,
                    )
                elif self.img2img_pipeline and init_img is not None:
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
                    kwargs_clean = {
                        k: v for k, v in kwargs.items() if k not in [
                            'image', 'ip_adapter_image']}
                    result = pipeline_to_use(
                        width=self.width,
                        height=self.height,
                        **kwargs_clean,
                    )
                print(f"  ⚠ 无法加载场景连贯性图像: {e}，使用普通文生图")
                # 确保 pipeline_to_use 是普通 pipeline
                pipeline_to_use = self.pipeline
                # 清理 kwargs 中的 image 相关参数
                kwargs_clean = {
                    k: v for k, v in kwargs.items() if k not in [
                        'image', 'ip_adapter_image']}
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
                print(
                    f"  ⚠ pipeline_to_use 是 img2img_pipeline，但 init_image 无效，强制切换到普通 pipeline")
                pipeline_to_use = self.pipeline
                init_image = None
                # 清理 kwargs 中的 image 相关参数
                kwargs_clean = {
                    k: v for k, v in kwargs.items() if k not in [
                        'image', 'ip_adapter_image']}
                kwargs = kwargs_clean

        if pipeline_to_use is self.img2img_pipeline and init_image is not None:
            # 检查 init_image 是否是 PIL Image（从 reference_image_path 加载的）
            from PIL import Image as PILImage
            if isinstance(init_image, PILImage.Image):
                # 如果 pipeline_to_use 是 img2img_pipeline 且 init_image 是 PIL
                # Image（从 reference_image_path 加载）
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
                kwargs_clean = {
                    k: v for k, v in kwargs.items() if k not in [
                        'image', 'ip_adapter_image']}
                result = pipeline_to_use(
                    width=self.width,
                    height=self.height,
                    **kwargs_clean,
                )
        else:
            # 普通文生图（不使用 img2img）
            # 在调用 pipeline 之前，强制检查并修复 pipeline_to_use
            # 如果 pipeline_to_use 是 img2img_pipeline，但 init_image 是 None
            # 或无效，强制切换到普通 pipeline
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                # 检查 init_image 是否有效
                has_valid_init_image = (
                    init_image is not None and
                    isinstance(init_image, PILImage.Image)
                )

                if not has_valid_init_image:
                    pipeline_to_use = self.pipeline
                    print(
                        f"  ⚠ 强制修复：pipeline_to_use 是 img2img_pipeline，但 init_image 无效，切换到普通 pipeline")
                elif not use_ip_adapter:
                    # IP-Adapter 不可用，即使有 init_image 也不使用 img2img
                    pipeline_to_use = self.pipeline
                    print(f"  ⚠ 强制修复：IP-Adapter 不可用，切换到普通 pipeline（不使用 img2img）")

            # 最终安全检查：确保不会传递 None 作为 image 参数
            # 如果 pipeline_to_use 仍然是 img2img_pipeline，但 init_image 是 None，强制切换
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                if init_image is None or not isinstance(
                        init_image, PILImage.Image):
                    print(
                        f"  ⚠ 最终安全检查：pipeline_to_use 仍然是 img2img_pipeline，但 init_image 无效，强制切换到普通 pipeline")
                    pipeline_to_use = self.pipeline
                elif not use_ip_adapter:
                    print(f"  ⚠ 最终安全检查：IP-Adapter 不可用，强制切换到普通 pipeline")
                    pipeline_to_use = self.pipeline

            # 确保 kwargs 中不包含 image 参数（如果使用普通 pipeline）
            if pipeline_to_use is self.pipeline:
                # 移除 kwargs 中可能存在的 image 相关参数，避免冲突
                kwargs_clean = {
                    k: v for k, v in kwargs.items() if k not in [
                        'image', 'ip_adapter_image']}
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
                    print(
                        f"  ⚠ 最终修复：pipeline_to_use 是 img2img_pipeline，但没有任何有效的 image，强制切换到普通 pipeline")
                    pipeline_to_use = self.pipeline
                    # 清理 kwargs
                    kwargs_clean = {
                        k: v for k, v in kwargs.items() if k not in [
                            'image', 'ip_adapter_image']}
                    kwargs = kwargs_clean

            # 绝对最终检查：在调用 pipeline 之前，再次确保不会使用 img2img_pipeline 且 image 为 None
            if pipeline_to_use is self.img2img_pipeline:
                from PIL import Image as PILImage
                # 检查 kwargs 中是否有有效的 image
                has_valid_image = 'image' in kwargs and kwargs['image'] is not None
                # 检查 init_image 是否有效
                has_valid_init = init_image is not None and isinstance(
                    init_image, PILImage.Image)

                if not (has_valid_image or has_valid_init):
                    print(
                        f"  ⚠ 绝对最终检查：强制切换到普通 pipeline（img2img_pipeline 需要有效的 image）")
                    pipeline_to_use = self.pipeline
                    # 彻底清理 kwargs
                    kwargs = {
                        k: v for k, v in kwargs.items() if k not in [
                            'image', 'ip_adapter_image', 'strength']}

            # 打印调试信息
            if pipeline_to_use is self.img2img_pipeline:
                print(f"  ℹ 使用 img2img_pipeline（有有效的 image）")
            else:
                print(f"  ℹ 使用普通 pipeline（text-to-image）")
                # 使用普通 pipeline 时，彻底清理 kwargs，移除所有可能冲突的参数
                # 包括 image, ip_adapter_image, strength 等
                # 首先打印清理前的 kwargs 键，用于调试
                keys_before = set(kwargs.keys())
                kwargs_final = {
                    k: v for k, v in kwargs.items() if k not in [
                        'image', 'ip_adapter_image', 'strength']}
                kwargs = kwargs_final
                # 再次确保没有 None 值（除了 negative_prompt，因为它可以为 None）
                kwargs = {k: v for k, v in kwargs.items() if v is not None or k in [
                    'negative_prompt']}
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
                    raise ValueError(
                        f"严重错误：kwargs 中仍然包含 'image' 参数，这不应该发生！kwargs 键: {final_keys}")

            # 调用 pipeline
            # 首先检查 pipeline 是否为 None
            if pipeline_to_use is None:
                raise RuntimeError(
                    "Pipeline 未加载或加载失败。请检查：\n"
                    "1. 模型文件是否存在\n"
                    "2. 模型路径是否正确\n"
                    "3. 显存是否足够\n"
                    "4. 依赖是否已安装"
                )
            
            try:
                # ⚡ 关键修复：在调用 pipeline 之前，验证 pipeline 组件完整性
                # 检查 pipeline 是否有 unet 属性（更可靠的方法）
                try:
                    # 首先尝试直接访问 unet 属性
                    if not hasattr(pipeline_to_use, 'unet'):
                        raise AttributeError("Pipeline 缺少 'unet' 属性")
                    # 尝试访问 unet（如果使用 CPU offload，可能需要触发加载）
                    unet = pipeline_to_use.unet
                    if unet is None:
                        raise AttributeError("Pipeline.unet 为 None")
                except (AttributeError, KeyError) as unet_check_error:
                    # 如果直接访问失败，尝试通过 components 检查
                    print(f"  ⚠ Pipeline.unet 访问失败: {unet_check_error}")
                    if hasattr(pipeline_to_use, 'components'):
                        try:
                            components = pipeline_to_use.components
                            if isinstance(components, dict):
                                if 'unet' not in components:
                                    raise KeyError(f"Pipeline.components 字典中缺少 'unet' 键，现有键: {list(components.keys())[:10]}")
                            else:
                                raise TypeError(f"Pipeline.components 不是字典，类型: {type(components)}")
                        except Exception as comp_error:
                            print(f"  ⚠ Pipeline.components 检查失败: {comp_error}")
                            # 如果使用 CPU offload，可能需要强制加载组件
                            # 尝试重新加载 pipeline
                            raise RuntimeError(f"Pipeline 组件检查失败: {unet_check_error}, components错误: {comp_error}")
                    else:
                        raise RuntimeError(f"Pipeline 缺少 'unet' 属性且没有 'components' 属性: {unet_check_error}")
                
                # 如果是img2img pipeline，添加image和strength参数
                if pipeline_to_use is self.img2img_pipeline and init_image is not None:
                    kwargs['image'] = init_image
                    kwargs['strength'] = img2img_strength
                    print(f"  ℹ img2img参数: strength={img2img_strength:.2f}")

                # ⚡ 关键修复：在调用 pipeline 前，再次验证并输出调试信息
                print(f"  🔍 [DEBUG] 准备调用 pipeline，类型: {type(pipeline_to_use).__name__}")
                print(f"  🔍 [DEBUG] pipeline_to_use.unet: {pipeline_to_use.unet is not None if hasattr(pipeline_to_use, 'unet') else 'N/A'}")
                if hasattr(pipeline_to_use, 'components'):
                    try:
                        components = pipeline_to_use.components
                        if isinstance(components, dict):
                            print(f"  🔍 [DEBUG] components 键: {list(components.keys())[:10]}")
                    except Exception as e:
                        print(f"  ⚠ 无法访问 components: {e}")

                result = pipeline_to_use(
                    width=self.width,
                    height=self.height,
                    **kwargs,
                )
            except (KeyError, AttributeError, RuntimeError) as e:
                error_str = str(e).lower()
                error_type = type(e).__name__
                import traceback
                if isinstance(e, KeyError) and 'unet' in str(e):
                    print(f"  ✗ Pipeline 调用失败（KeyError 'unet'）: {e}")
                    print(f"  📋 完整错误堆栈:\n{traceback.format_exc()}")
                    # 尝试诊断问题
                    print(f"  🔍 [DEBUG] pipeline_to_use 类型: {type(pipeline_to_use).__name__}")
                    print(f"  🔍 [DEBUG] hasattr(pipeline_to_use, 'unet'): {hasattr(pipeline_to_use, 'unet')}")
                    if hasattr(pipeline_to_use, 'unet'):
                        try:
                            unet = pipeline_to_use.unet
                            print(f"  🔍 [DEBUG] pipeline_to_use.unet: {type(unet).__name__ if unet is not None else 'None'}")
                        except Exception as unet_err:
                            print(f"  🔍 [DEBUG] 访问 unet 时出错: {unet_err}")
                    if hasattr(pipeline_to_use, 'components'):
                        try:
                            components = pipeline_to_use.components
                            if isinstance(components, dict):
                                print(f"  🔍 [DEBUG] components 键: {list(components.keys())[:10]}")
                                print(f"  🔍 [DEBUG] 'unet' in components: {'unet' in components}")
                            else:
                                print(f"  🔍 [DEBUG] components 类型: {type(components)}")
                        except Exception as comp_err:
                            print(f"  🔍 [DEBUG] 访问 components 时出错: {comp_err}")
                elif 'unet' in error_str:
                    print(f"  ✗ Pipeline 组件错误（unet）: {error_type}: {e}")
                    print(f"  📋 完整错误堆栈:\n{traceback.format_exc()}")
                    print(f"  ⚠ 尝试重新加载 SDXL pipeline（不使用 CPU offload）...")
                    try:
                        # 强制重新加载 SDXL pipeline，但不使用 CPU offload（避免 components 字典不完整）
                        instantid_pipeline = self.pipeline if is_instantid_pipeline else None
                        original_use_cpu_offload = self.image_config.get("enable_cpu_offload", True)
                        # 临时禁用 CPU offload，确保 components 字典完整
                        self.image_config["enable_cpu_offload"] = False
                        try:
                            self._load_sdxl_pipeline(load_lora=False)
                        finally:
                            # 恢复原始配置
                            self.image_config["enable_cpu_offload"] = original_use_cpu_offload
                        # 验证重新加载的 pipeline
                        if self.pipeline is not None:
                            # 验证 unet 属性（更可靠的方法）
                            if hasattr(self.pipeline, 'unet') and self.pipeline.unet is not None:
                                # 将 pipeline 赋值给 sdxl_pipeline（如果之前是这样的话）
                                if pipeline_to_use is self.sdxl_pipeline or (is_instantid_pipeline and use_ip_adapter):
                                    self.sdxl_pipeline = self.pipeline
                                pipeline_to_use = self.pipeline
                                print(f"  ✓ SDXL pipeline 重新加载成功（通过 unet 属性验证）")
                                # 重新尝试调用
                                if pipeline_to_use is self.img2img_pipeline and init_image is not None:
                                    kwargs['image'] = init_image
                                    kwargs['strength'] = img2img_strength
                                result = pipeline_to_use(
                                    width=self.width,
                                    height=self.height,
                                    **kwargs,
                                )
                            else:
                                raise RuntimeError("重新加载的 pipeline 仍然缺少有效的 'unet' 属性")
                        else:
                            raise RuntimeError("无法重新加载 SDXL pipeline")
                        # 恢复 InstantID pipeline（如果之前在使用）
                        if instantid_pipeline:
                            self.pipeline = instantid_pipeline
                    except Exception as reload_err:
                        import traceback
                        print(f"  ✗ 重新加载 pipeline 失败: {reload_err}")
                        print(f"  📋 详细错误:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Pipeline 组件错误且无法修复: {e}") from e
                else:
                    raise
            except TypeError as e:
                if "image must be passed" in str(
                        e) or "image" in str(e).lower():
                    print(f"  ✗ Pipeline 调用失败，错误信息: {e}")
                    print(f"  ℹ kwargs 内容: {list(kwargs.keys())}")
                    print(f"  ℹ pipeline_to_use 类型: {type(pipeline_to_use)}")
                    print(
                        f"  ℹ pipeline_to_use 是否为 img2img_pipeline: {
                            pipeline_to_use is self.img2img_pipeline}")
                    # 最后一次尝试：创建一个全新的、干净的 kwargs
                    clean_kwargs = {
                        "prompt": kwargs.get("prompt"),
                        "negative_prompt": kwargs.get("negative_prompt"),
                        "guidance_scale": kwargs.get("guidance_scale"),
                        "num_inference_steps": kwargs.get("num_inference_steps"),
                        "generator": kwargs.get("generator"),
                    }
                    # 移除所有 None 值（除了 negative_prompt）
                    clean_kwargs = {k: v for k, v in clean_kwargs.items(
                    ) if v is not None or k == 'negative_prompt'}
                    print(f"  ℹ 使用干净的 kwargs 重试: {list(clean_kwargs.keys())}")
                    result = pipeline_to_use(
                        width=self.width,
                        height=self.height,
                        **clean_kwargs,
                    )
                else:
                    raise

        image = result.images[0]
        # 确保 output_path 是 Path 对象
        if isinstance(output_path, str):
            output_path = Path(output_path)
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

        output_dir = Path(
            output_dir or self.paths_config.get(
                "image_output",
                "outputs/images"))
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

        # ============================================================
        # Execution Planner v2 集成：检测是否是 v2 格式
        # ============================================================
        is_v2_format = False
        planner = None
        if scenes:
            first_scene = scenes[0]
            # 检查是否是 v2 格式（通过 version 字段或 v2 特有字段）
            if first_scene.get("version") == "v2" or (
                "intent" in first_scene and 
                "visual_constraints" in first_scene and 
                "generation_policy" in first_scene
            ):
                is_v2_format = True
                print("\n" + "="*60)
                print("🎯 检测到 Scene JSON v2 格式，启用 Execution Planner v2")
                print("="*60)
                try:
                    from model_selector import ModelSelector
                    planner = ModelSelector(self.config)
                    print("✓ Execution Planner v2 已初始化")
                except Exception as e:
                    print(f"⚠ Execution Planner v2 初始化失败: {e}，将使用默认逻辑")
                    planner = None
                    is_v2_format = False

        # 自动生成 face_style_auto（如果不存在，且不是 v2 格式）
        if not is_v2_format:
            try:
                from face_style_auto_generator import generate_face_styles_for_episode
                print("\n[自动生成 face_style_auto]")
                generate_face_styles_for_episode(
                    scenes, smooth=True, overwrite_existing=False)
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
                print(
                    f"  ✓ 匹配到场景模板: {
                        matched_profile.get(
                            'scene_name',
                            '未知')}")

        saved_paths: List[Path] = []
        previous_scene = None  # 跟踪前一个场景，用于连贯性控制
        previous_image_path = None  # 前一个场景的图像路径，用于img2img
        
        # 预先创建 ModelManager 实例（如果可能用到，避免重复创建）
        manager = None
        manager_models_root = None
        manager_config_path = None

        for idx, scene in enumerate(scenes, start=1):
            # v2 兼容：scene_id 优先，其次 id，最后回退 idx
            scene_id = scene.get("scene_id", scene.get('id', idx - 1))
            print(f"\n{'='*60}")
            print(f"处理场景 {idx}/{len(scenes)} (场景ID={scene_id})")
            print(f"{'='*60}")
            
            # 在每个场景生成前清理GPU缓存（除了第一个场景）
            if idx > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            # 判断是否需要主角（使用 LoRA）
            needs_character = self._needs_character(scene)
            
            # 识别场景中的角色，自动选择对应的参考图像（提前识别，用于决定是否使用 ModelManager）
            identified_characters = self._identify_characters_in_scene(scene)
            primary_character = identified_characters[0] if identified_characters else None
            
            # 根据角色ID选择对应的参考图像
            face_reference = self._select_face_reference_image(
                idx, character_id=primary_character)
            
            # MVP 流程：如果检测到科普主持人且有参考图，直接使用 ModelManager（与 mvp_main 一致）
            # 提前检测，避免加载普通 pipeline
            # 检查 face_reference 是否为有效路径
            face_reference_valid = False
            if face_reference:
                if isinstance(face_reference, (str, Path)):
                    face_reference_path = Path(face_reference)
                    if face_reference_path.exists() and face_reference_path.is_file():
                        face_reference_valid = True
                else:
                    # 可能是其他类型，尝试转换
                    try:
                        face_reference_path = Path(str(face_reference))
                        if face_reference_path.exists() and face_reference_path.is_file():
                            face_reference_valid = True
                    except:
                        pass
            
            # 如果 primary_character 是科普主持人，但 face_reference 无效，尝试从配置读取
            if primary_character in ["kepu_gege", "weilai_jiejie"] and not face_reference_valid:
                # 尝试从配置读取参考图路径
                model_selection = self.image_config.get('model_selection', {})
                character_config = model_selection.get('character', {})
                config_face_path = character_config.get('face_image_path')
                if config_face_path and Path(config_face_path).exists():
                    face_reference = Path(config_face_path)
                    face_reference_valid = True
                    print(f"  ℹ 从配置读取参考图: {face_reference.name}")
            
            use_model_manager = primary_character in ["kepu_gege", "weilai_jiejie"] and face_reference_valid
            
            # 调试信息
            if primary_character in ["kepu_gege", "weilai_jiejie"]:
                print(f"  🔍 调试: primary_character={primary_character}, face_reference={face_reference}, face_reference_valid={face_reference_valid}, use_model_manager={use_model_manager}")
            
            if use_model_manager:
                # 使用 ModelManager，不需要加载普通 pipeline
                if manager is None:
                    from model_manager import ModelManager
                    from PIL import Image
                    
                    print(f"  🎯 检测到科普主持人场景，将使用 ModelManager（避免重复加载模型）")
                    
                    # 创建 ModelManager 实例（只创建一次，复用）
                    models_root = self.models_root
                    config_path = str(self.config_path) if hasattr(self, 'config_path') else None
                    manager = ModelManager(models_root=str(models_root), lazy_load=True, config_path=config_path)
                    manager_models_root = models_root
                    manager_config_path = config_path
                
                # 传递前一个场景信息，用于连贯性控制
                prompt = self.build_prompt(
                    scene,
                    include_character=needs_character,
                    script_data=script_data,
                    previous_scene=previous_scene)
                filename = f"scene_{idx:03d}.png"
                output_path = output_dir / filename
                
                if output_path.exists() and not overwrite:
                    print(f"跳过 {output_path}（文件已存在）")
                    scene["image_path"] = str(output_path)
                    saved_paths.append(output_path)
                    previous_scene = scene
                    previous_image_path = str(output_path)
                    continue
                
                print(f"生成场景图像 {idx}/{len(scenes)}: {prompt[:80]}...")
                print(f"  ✓ 使用 ModelManager（科普主持人场景）")
                print(f"  使用面部参考图: {face_reference.name if hasattr(face_reference, 'name') else face_reference}")
                
                try:
                    from PIL import Image
                    # 加载参考图（确保路径有效）
                    face_reference_path = Path(face_reference) if not isinstance(face_reference, Path) else face_reference
                    if not face_reference_path.exists():
                        raise FileNotFoundError(f"参考图不存在: {face_reference_path}")
                    
                    face_image = Image.open(face_reference_path)
                    print(f"  ✅ 已加载参考图: {face_reference_path.name}")
                    
                    # 从 scene 获取尺寸
                    width = scene.get("width", self.width)
                    height = scene.get("height", self.height)
                    
                    # 精简 prompt（避免超过 77 tokens）
                    # ModelManager._build_character_prompt 会添加角色描述，所以这里只保留核心场景描述
                    prompt_short = prompt
                    if len(prompt) > 40:  # 保留前 40 字符，给角色描述留出空间
                        # 尝试保留最重要的部分（场景描述）
                        scene_prompt = scene.get("prompt", "")
                        if scene_prompt and len(scene_prompt) <= 40:
                            prompt_short = scene_prompt
                        else:
                            prompt_short = prompt[:40]
                        print(f"  ℹ Prompt 已精简: {len(prompt)} -> {len(prompt_short)} 字符")
                    
                    # 使用 ModelManager 生成（自动使用 Flux + InstantID）
                    # ModelManager 会自动从配置读取 num_inference_steps 和 guidance_scale
                    image = manager.generate(
                        task="host_face",  # 会自动切换到 host_face_instantid
                        prompt=prompt_short,  # 使用精简后的 prompt
                        negative_prompt=None,
                        width=width,
                        height=height,
                        num_inference_steps=None,  # None 表示从配置读取
                        guidance_scale=None,  # None 表示从配置读取
                        seed=None,
                        face_image=face_image,
                        face_strength=0.8
                    )
                    
                    # 保存图像
                    image.save(output_path)
                    print(f"  ✅ ModelManager 生成成功: {output_path}")
                    scene["image_path"] = str(output_path)
                    saved_paths.append(output_path)
                    
                    # 更新前一个场景信息
                    previous_scene = scene
                    previous_image_path = str(output_path)
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    continue  # 跳过后续的普通流程
                    
                except Exception as e:
                    print(f"  ⚠ ModelManager 生成失败: {e}，回退到普通流程")
                    import traceback
                    traceback.print_exc()
                    # 继续使用普通流程
            
            # ============================================================
            # Execution Planner v2: 如果是 v2 格式，使用 Planner 决策
            # ============================================================
            planner_decision = None
            if is_v2_format and planner:
                try:
                    planner_decision = planner.select_engine_for_scene_v2(scene)
                    print(f"  🎯 Execution Planner 决策:")
                    print(f"     引擎: {planner_decision['engine']}")
                    print(f"     模式: {planner_decision['mode']}")
                    print(f"     锁脸: {planner_decision['lock_face']}")
                    print(f"     任务类型: {planner_decision['task_type']}")
                    
                    # 根据 Planner 决策调整参数
                    if planner_decision['lock_face'] and not face_reference:
                        # 如果需要锁脸但没有参考图，尝试从角色ID获取
                        character_id = scene.get("character", {}).get("id")
                        if character_id:
                            face_reference = self._select_face_reference_image(
                                idx, character_id=character_id)
                            if face_reference:
                                print(f"  ✓ 根据 Planner 决策，已加载角色参考图: {face_reference.name}")
                except Exception as e:
                    print(f"  ⚠ Execution Planner 决策失败: {e}，使用默认逻辑")
                    planner_decision = None

            # ⚡ 关键修复：检查 Execution Planner 是否要求使用语义化 prompt（FLUX 专用）
            use_semantic_prompt = False
            if planner_decision and planner_decision.get('use_semantic_prompt', False):
                use_semantic_prompt = True
                print(f"  ✓ Execution Planner 要求使用语义化 prompt（FLUX 专用）")
            
            # 普通流程：传递前一个场景信息，用于连贯性控制
            prompt = self.build_prompt(
                scene,
                include_character=needs_character,
                script_data=script_data,
                previous_scene=previous_scene,
                use_semantic_prompt=use_semantic_prompt)  # ⚡ 新增：传递语义化 prompt 标志
            filename = f"scene_{idx:03d}.png"
            # 重要：始终使用output_dir生成的文件路径，忽略JSON中可能错误的image_path
            output_path = output_dir / filename

            # ⚡ 关键修复：检查 Execution Planner 是否要求使用人设锚点图
            # ⚡ 核心规则：直接使用参考图（hanli_mid.jpg），不拷贝 anchor
            use_character_anchor = False
            character_anchor_path = None
            if planner_decision and planner_decision.get('use_character_anchor', False):
                use_character_anchor = True
                # ⚡ 核心规则：直接使用参考图，不查找 anchor 目录
                character_id = scene.get("character", {}).get("id", primary_character if primary_character else "hanli")
                
                # ⚡ 关键修复：使用绝对路径查找，避免相对路径问题
                # 优先级：配置中的 face_image_path > hanli_mid.jpg > hanli_mid.png
                if character_id == "hanli":
                    # 优先级 1：配置中的 face_image_path（通常是绝对路径）
                    face_image_path = self.image_config.get("face_image_path")
                    if face_image_path:
                        face_image_path_obj = Path(face_image_path)
                        # 如果是相对路径，转换为绝对路径
                        if not face_image_path_obj.is_absolute():
                            # 尝试从当前工作目录或项目根目录查找
                            base_paths = [
                                Path.cwd(),
                                Path(__file__).parent.parent,  # gen_video 目录
                                Path(__file__).parent.parent.parent,  # 项目根目录
                            ]
                            for base in base_paths:
                                test_path = base / face_image_path
                                if test_path.exists():
                                    face_image_path_obj = test_path
                                    break
                        
                        if face_image_path_obj.exists():
                            character_anchor_path = face_image_path_obj
                            print(f"  ✓ 使用人设锚点图（参考图）: {character_anchor_path.name}")
                        else:
                            # 优先级 2：hanli_mid.jpg（尝试多个路径）
                            base_paths = [
                                Path(__file__).parent / "reference_image",  # gen_video/reference_image
                                Path(__file__).parent.parent / "reference_image",  # 项目根目录/reference_image
                                Path.cwd() / "gen_video" / "reference_image",  # 当前目录/gen_video/reference_image
                            ]
                            found = False
                            for base in base_paths:
                                default_path_jpg = base / "hanli_mid.jpg"
                                if default_path_jpg.exists():
                                    character_anchor_path = default_path_jpg
                                    print(f"  ✓ 使用人设锚点图（参考图）: {character_anchor_path.name}")
                                    found = True
                                    break
                            
                            if not found:
                                print(f"  ⚠ 警告：Execution Planner 要求使用人设锚点图，但未找到")
                                print(f"  ℹ 已尝试以下路径：")
                                for base in base_paths:
                                    print(f"     - {base / 'hanli_mid.jpg'}")
                                # 回退到使用普通参考图
                                use_character_anchor = False
                    else:
                        # 配置中没有 face_image_path，直接查找 hanli_mid.jpg
                        base_paths = [
                            Path(__file__).parent / "reference_image",
                            Path(__file__).parent.parent / "reference_image",
                            Path.cwd() / "gen_video" / "reference_image",
                        ]
                        found = False
                        for base in base_paths:
                            default_path_jpg = base / "hanli_mid.jpg"
                            if default_path_jpg.exists():
                                character_anchor_path = default_path_jpg
                                print(f"  ✓ 使用人设锚点图（参考图）: {character_anchor_path.name}")
                                found = True
                                break
                        
                        if not found:
                            print(f"  ⚠ 警告：未找到参考图 hanli_mid.jpg")
                            use_character_anchor = False
                else:
                    # 其他角色：尝试查找对应的参考图
                    base_paths = [
                        Path(__file__).parent / "reference_image",
                        Path(__file__).parent.parent / "reference_image",
                        Path.cwd() / "gen_video" / "reference_image",
                    ]
                    found = False
                    for base in base_paths:
                        char_ref_path = base / f"{character_id}_mid.jpg"
                        if char_ref_path.exists():
                            character_anchor_path = char_ref_path
                            print(f"  ✓ 使用人设锚点图（参考图）: {character_anchor_path.name}")
                            found = True
                            break
                    
                    if not found:
                        print(f"  ⚠ 警告：未找到角色参考图: {character_id}_mid.jpg")
                        use_character_anchor = False
            
            # 如果要求使用人设锚点图，优先使用它作为 reference_image
            if use_character_anchor and character_anchor_path:
                reference_image = character_anchor_path
                print(f"  🎯 使用人设锚点图作为 reference_image: {reference_image.name}")
            else:
                reference_image = self._select_reference_image(scene, idx)

            if primary_character and face_reference:
                print(
                    f"  ✓ 识别到角色: {primary_character}，使用对应参考图像: {
                        face_reference.name}")

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
            
            # 注意：ModelManager 检测已在循环开始前完成，这里不再重复检测
            # 如果 use_model_manager 为 True，应该已经在前面处理并 continue 了
            
            try:
                # 如果启用场景连贯性，且不是第一个场景，使用前一个场景作为img2img参考
                init_image = None
                if idx > 1 and previous_image_path and Path(
                        previous_image_path).exists():
                    # 检查是否应该使用img2img（同一环境或连续场景）
                    current_env = scene.get("scene_name") or script_data.get(
                        "title", "") if script_data else ""
                    prev_env = previous_scene.get(
                        "scene_name") or "" if previous_scene else ""

                    # 如果环境相同或相似，使用img2img保持连贯性
                    use_continuity = False
                    if current_env and prev_env:
                        # 检查环境关键词匹配
                        env_keywords = [
                            "desert",
                            "chamber",
                            "corridor",
                            "遗迹",
                            "沙漠",
                            "地下",
                            "underground"]
                        if any(kw in current_env.lower() and kw in prev_env.lower()
                               for kw in env_keywords):
                            use_continuity = True
                    elif idx <= 5:  # 前5个场景通常在同一环境（更宽松的条件）
                        use_continuity = True

                    if use_continuity and self.use_img2img:
                        init_image = Path(previous_image_path)
                        print(
                            f"  ✓ 使用前一个场景作为参考（场景连贯性），strength={
                                self.img2img_strength * 0.4:.2f}")
                    elif idx <= 5 and self.use_img2img and previous_image_path:
                        # 即使环境关键词不匹配，前5个场景也尝试使用img2img保持连贯性
                        init_image = Path(previous_image_path)
                        print(
                            f"  ✓ 使用前一个场景作为参考（相邻场景连贯性），strength={
                                self.img2img_strength * 0.4:.2f}")

                # 根据 Execution Planner 决策选择引擎和参数
                model_engine = None
                task_type = None
                style_lora_from_planner = None
                if planner_decision:
                    model_engine = planner_decision['engine']
                    task_type = planner_decision['task_type']
                    # ⚡ 关键修复：应用 Execution Planner 返回的风格锚点配置
                    style_anchor = planner_decision.get('style_anchor')
                    # ⚡ 关键修复：检查是否需要禁用 LoRA（wide + top_down + lying 场景）
                    disable_character_lora = planner_decision.get('disable_character_lora', False)
                    disable_style_lora = planner_decision.get('disable_style_lora', False)
                    
                    if disable_style_lora:
                        # 禁用风格 LoRA
                        style_lora_from_planner = ""  # 空字符串表示禁用
                        print(f"  ✓ Execution Planner 禁用风格 LoRA（wide + top_down + lying 场景，避免姿态冲突）")
                    elif style_anchor and style_anchor.get('enabled', False):
                        style_lora_name = style_anchor.get('name', 'anime_style')
                        # 映射 fanren_style 到实际的 LoRA 名称
                        if style_lora_name == "fanren_style":
                            style_lora_name = "anime_style"  # 使用配置中的 anime_style LoRA
                        style_lora_from_planner = style_lora_name
                        style_lora_weight = style_anchor.get('weight', 0.35)
                        print(f"  ✓ Execution Planner 指定风格锚点: {style_lora_name}，权重: {style_lora_weight}")
                    
                    # 如果禁用角色 LoRA，确保 character_lora 为 None
                    if disable_character_lora:
                        character_lora = None  # 这里需要传递到 generate_image，但当前代码结构不支持，需要在 generate_image 中处理
                        print(f"  ✓ Execution Planner 禁用角色 LoRA（wide + top_down + lying 场景，避免姿态冲突）")
                    # 如果 Planner 决定不锁脸，但当前有 face_reference，根据决策决定是否使用
                    if not planner_decision['lock_face'] and face_reference:
                        # Planner 决定不锁脸，但保留参考图用于其他用途（如风格参考）
                        # 这里可以根据需要决定是否传递 face_reference
                        pass  # 暂时保留，后续可以根据需要调整
                
                # ⚡ 关键修复：如果要求使用人设锚点图，确保传递正确的 reference_image
                final_reference_image = reference_image
                if use_character_anchor and character_anchor_path:
                    final_reference_image = character_anchor_path
                    print(f"  🎯 传递人设锚点图到 generate_image: {final_reference_image.name}")
                
                # ⚡ 修复：人物场景必须始终传递 face_reference 给 generate_image
                # 原因：增强模式（PuLID）依赖 face_reference 注入身份；如果只在 lock_face/flux1 传递，
                # 会导致 B 类/SDXL 过渡镜头（lock_face=False）完全丢失身份一致性。
                face_ref_for_flux = None
                if needs_character and face_reference:
                    face_ref_for_flux = face_reference
                    if planner_decision:
                        print(
                            f"  ✓ 人物场景：传递 face_reference（用于增强模式身份注入）: {face_ref_for_flux.name if hasattr(face_ref_for_flux, 'name') else face_ref_for_flux}"
                        )
                else:
                    # 纯场景：不传 reference，避免误伤
                    face_ref_for_flux = None
                
                path = self.generate_image(
                    prompt,
                    output_path,
                    reference_image_path=final_reference_image,  # ⚡ 修复：使用人设锚点图
                    face_reference_image_path=face_ref_for_flux,  # ⚡ 修复：FLUX 引擎时传递 face_reference
                    use_lora=needs_character if not planner_decision else False,  # v2 格式不使用 LoRA，改用 InstantID
                    style_lora=style_lora_from_planner,  # ⚡ 新增：使用 Execution Planner 指定的风格 LoRA
                    scene=scene,
                    init_image=init_image,  # 传递前一个场景图像用于连贯性
                    model_engine=model_engine,  # 使用 Planner 决策的引擎
                    task_type=task_type,  # 使用 Planner 决策的任务类型
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
                import traceback
                error_type = type(exc).__name__
                error_msg = str(exc)
                print(f"✗ 场景 {idx} 生成失败: {error_type}: {error_msg}")
                # 如果是 KeyError 'unet'，输出完整堆栈
                if isinstance(exc, KeyError) and 'unet' in error_msg:
                    print(f"  📋 完整错误堆栈（KeyError 'unet'）:\n{traceback.format_exc()}")
                elif 'unet' in error_msg.lower():
                    print(f"  📋 完整错误堆栈（包含 'unet'）:\n{traceback.format_exc()}")
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

    def _get_character_profile(
            self, character_id: str = "hanli") -> Dict[str, Any]:
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
                if profile_scene_name and (
                        profile_scene_name in scene_name_lower or scene_name_lower in profile_scene_name):
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
            if ("城市" in scene_name or "市集" in scene_name) and (
                    "沙漠" not in scene_name and "沙地" not in scene_name):
                for key, profile in self.scene_profiles.items():
                    profile_scene_name = profile.get("scene_name", "").lower()
                    if "城市" in profile_scene_name or "市集" in profile_scene_name or "city" in key.lower(
                    ) or "market" in key.lower():
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

