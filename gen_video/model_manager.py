#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型协调调用系统
统一接口，自动路由，按任务选择最优模型
"""

from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import torch
import yaml

from pipelines.flux_pipeline import FluxPipeline
from pipelines.kolors_pipeline import KolorsPipeline
from pipelines.sd3_pipeline import SD3TurboPipeline
from pipelines.hunyuan_pipeline import HunyuanPipeline
from pipelines.flux_instantid_pipeline import FluxInstantIDPipeline


class ModelManager:
    """
    多模型管理器
    统一接口，自动路由，按任务选择最优模型
    """
    
    def __init__(self, models_root: Optional[str] = None, lazy_load: bool = True, config_path: Optional[str] = None):
        """
        初始化 ModelManager
        
        Args:
            models_root: 模型根目录，默认使用当前项目的 models 目录
            lazy_load: 是否延迟加载（只在需要时加载模型）
            config_path: 配置文件路径（用于读取 LoRA alpha 等配置）
        """
        if models_root is None:
            models_root = Path(__file__).parent / "models"
        else:
            models_root = Path(models_root)
        
        self.models_root = models_root
        self.lazy_load = lazy_load
        
        # 加载配置文件（用于读取 LoRA alpha 等配置）
        self.config = {}
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        
        # 模型路径配置
        self.model_paths = {
            "flux1": str(models_root / "flux1-dev"),
            "flux2": str(models_root / "flux2-dev"),
            "flux1-instantid": str(models_root / "flux1-dev"),  # Flux.1 + InstantID
            "kolors": str(models_root / "kolors-base"),
            "hunyuan": str(models_root / "hunyuan-dit" / "t2i"),
            "sd3": str(models_root / "sd3-turbo"),
        }
        
        # InstantID 路径配置
        self.instantid_paths = {
            "instantid": str(models_root / "instantid"),
            "controlnet": str(models_root / "instantid" / "ControlNet"),
            "ip_adapter": str(models_root / "instantid" / "ip-adapter"),  # InstantID 原版（SDXL 用）
            "ip_adapter_flux": str(models_root / "instantid" / "ip-adapter-flux"),  # Flux 专用版本
        }
        
        # 人脸参考图片目录
        self.face_references_dir = models_root / "face_references"
        self.face_references_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline 缓存（延迟加载）
        self.pipelines: Dict[str, Any] = {}
        
        # LoRA 配置（可选）
        # 优先使用改进版 LoRA（v2），如果不存在则使用原版
        lora_root_v2 = models_root / "lora" / "host_person_v2"
        lora_root_v1 = models_root / "lora" / "host_person"
        
        # 检查哪个版本存在（最终模型已更新为 checkpoint-500）
        final_model = lora_root_v2 / "pytorch_lora_weights.safetensors"
        v1_model = lora_root_v1 / "pytorch_lora_weights.safetensors"
        
        if final_model.exists():
            lora_path = str(final_model)
            lora_version = "v2（最终版，基于 checkpoint-500）"
        elif v1_model.exists():
            lora_path = str(v1_model)
            lora_version = "v1（原版）"
        else:
            lora_path = None
            lora_version = "未找到"
        
        # 从配置文件读取 LoRA alpha（优先使用 model_selection.character.lora.alpha）
        default_alpha = 0.6  # 默认值
        if self.config:
            image_config = self.config.get('image', {})
            model_selection = image_config.get('model_selection', {})
            character_config = model_selection.get('character', {})
            lora_config = character_config.get('lora', {})
            if 'alpha' in lora_config:
                default_alpha = float(lora_config['alpha'])
                print(f"  ℹ 从配置文件读取 LoRA alpha: {default_alpha}")
        
        self.lora_configs: Dict[str, Dict[str, Any]] = {
            "host_face": {
                "lora_path": lora_path,  # 自动选择可用的 LoRA 版本
                "lora_alpha": default_alpha  # 从配置文件读取（默认 0.6）
            },
            "character_face": {
                "lora_path": lora_path,  # 复用主持人 LoRA
                "lora_alpha": default_alpha  # 从配置文件读取（默认 0.6）
            }
        }
        
        if lora_path:
            print(f"  ℹ 使用 LoRA: {lora_version}")
        
        # 加载角色描述文件（用于配合 LoRA 固定形象）
        self.character_profiles = self._load_character_profiles()
        
        # 任务路由表
        self.routing_table = {
            # 人脸相关（支持 InstantID）
            "host_face": "flux1",  # 科普主持人脸（默认使用 Flux.1，如果提供 face_image 则使用 InstantID）
            "host_face_instantid": "flux1-instantid",  # 科普主持人脸 + InstantID（固定人脸）
            "character_face": "flux1",  # 角色人脸（默认使用 Flux.1）
            "character_face_instantid": "flux1-instantid",  # 角色人脸 + InstantID
            "realistic_face": "flux1",  # 真实感人脸（默认使用 Flux.1）
            "realistic_face_instantid": "flux1-instantid",  # 真实感人脸 + InstantID
            
            # 科学背景
            "science_background": "flux2",  # 科学背景图（冲击力强）
            "quantum_particle": "flux2",  # 量子/粒子
            "space_cosmos": "flux2",  # 太空/宇宙
            
            # 实验室/医学
            "lab_scene": "flux1",  # 实验室场景（更干净自然）
            "medical_scene": "flux1",  # 医学场景
            
            # 官方风格
            "official_style": "hunyuan",  # 官方感科教宣传图
            "chinese_scene": "hunyuan",  # 中文场景
            "education_style": "hunyuan",  # 教育风格
            
            # 快速生成
            "fast_background": "sd3",  # 快速背景
            "batch_generation": "sd3",  # 批量生成
            "variations": "sd3",  # 备选图
        }
        
        # 如果不需要延迟加载，预加载所有模型
        if not lazy_load:
            self._load_all_pipelines()
    
    def _get_pipeline(self, model_name: str):
        """获取 Pipeline（延迟加载）"""
        if model_name in self.pipelines:
            return self.pipelines[model_name]
        
        model_path = self.model_paths.get(model_name)
        if not model_path:
            raise ValueError(f"未知的模型: {model_name}")
        
        if not Path(model_path).exists():
            raise RuntimeError(f"模型路径不存在: {model_path}")
        
        # 根据模型名称创建对应的 Pipeline
        if model_name == "flux1":
            pipeline = FluxPipeline(model_path, model_type="flux1")
        elif model_name == "flux2":
            pipeline = FluxPipeline(model_path, model_type="flux2")
        elif model_name == "flux1-instantid":
            # 使用 Flux + InstantID Pipeline
            # 优先使用 Flux 专用的 IP-Adapter
            instantid_path = self.instantid_paths.get("ip_adapter_flux")
            if not Path(instantid_path).exists():
                # 如果 Flux 版本不存在，尝试使用原版（虽然不兼容，但至少可以提示）
                instantid_path = self.instantid_paths.get("ip_adapter")
            controlnet_path = self.instantid_paths.get("controlnet")
            
            # 默认禁用 IP-Adapter，使用纯 Flux + LoRA 模式（效果更好）
            # 如果用户需要 IP-Adapter，可以通过环境变量或配置启用
            use_ip_adapter = False  # 默认禁用，因为 Flux IP-Adapter 效果不如 LoRA
            
            pipeline = FluxInstantIDPipeline(
                model_path=model_path,
                instantid_path=instantid_path,
                controlnet_path=controlnet_path,
                model_type="flux1",
                use_ip_adapter=use_ip_adapter
            )
        elif model_name == "kolors":
            # Kolors tokenizer 有严重 bug，暂时禁用，自动使用 Flux.1 替代
            print("  ⚠️  警告: Kolors tokenizer 存在严重 bug（即使很短提示词也会溢出）")
            print("  ℹ️  自动切换到 Flux.1（效果类似，更稳定）")
            # 使用 Flux.1 替代
            flux1_path = self.model_paths.get("flux1")
            if flux1_path and Path(flux1_path).exists():
                pipeline = FluxPipeline(flux1_path, model_type="flux1")
                # 更新缓存键，使用 flux1 而不是 kolors
                model_name = "flux1"
            else:
                raise RuntimeError(
                    "Kolors tokenizer 有 bug 且 Flux.1 不可用。"
                    "建议：使用其他模型或等待 Kolors 修复"
                )
        elif model_name == "hunyuan":
            pipeline = HunyuanPipeline(model_path)
        elif model_name == "sd3":
            pipeline = SD3TurboPipeline(model_path)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 延迟加载：只在第一次使用时加载
        if not self.lazy_load:
            pipeline.load()
        
        # 如果 model_name 被修改（如 kolors -> flux1），使用新的名称缓存
        # 但也要保留原始请求的映射，以便后续查找
        self.pipelines[model_name] = pipeline
        return pipeline
    
    def route(self, task: str) -> str:
        """
        根据任务路由到对应的模型
        
        Args:
            task: 任务类型
            
        Returns:
            模型名称
        """
        model_name = self.routing_table.get(task)
        if model_name is None:
            # 默认使用 Flux.2
            print(f"⚠️  未知任务 '{task}'，使用默认模型 flux2")
            return "flux2"
        return model_name
    
    def _load_face_image(self, face_image_name: Optional[str] = None, task: Optional[str] = None) -> Optional[Image.Image]:
        """
        从 face_references 目录加载人脸图片
        
        Args:
            face_image_name: 人脸图片文件名（如 "host_face.png"）
            task: 任务类型（用于自动查找）
            
        Returns:
            PIL Image 或 None
        """
        if face_image_name:
            # 直接加载指定文件
            face_path = self.face_references_dir / face_image_name
            if face_path.exists():
                try:
                    return Image.open(face_path)
                except Exception as e:
                    print(f"  ⚠️  无法加载人脸图片 {face_image_name}: {e}")
                    return None
            else:
                print(f"  ⚠️  人脸图片不存在: {face_path}")
                return None
        
        # 根据任务类型自动查找
        if task:
            # 任务类型到文件名的映射
            task_to_filename = {
                "host_face": "host_face.png",
                "character_face": "character_face.png",
                "realistic_face": "realistic_face.png",
            }
            
            filename = task_to_filename.get(task)
            if filename:
                face_path = self.face_references_dir / filename
                if face_path.exists():
                    try:
                        print(f"  ✅ 自动加载人脸图片: {filename}")
                        return Image.open(face_path)
                    except Exception as e:
                        print(f"  ⚠️  无法加载人脸图片 {filename}: {e}")
        
        return None
    
    def generate(
        self,
        task: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        face_image: Optional[Image.Image] = None,
        face_image_name: Optional[str] = None,
        face_strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """
        生成图像（统一接口）
        
        Args:
            task: 任务类型（自动路由到对应模型）
            prompt: 提示词
            negative_prompt: 负面提示词
            width: 图像宽度
            height: 图像高度
            num_inference_steps: 推理步数（None 时使用模型默认值）
            guidance_scale: 引导强度（None 时使用模型默认值）
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            PIL Image
        """
        # 加载人脸图片（如果提供了文件名或需要自动查找）
        if face_image is None:
            loaded_face_image = self._load_face_image(face_image_name, task)
            if loaded_face_image:
                face_image = loaded_face_image
        
        # 如果提供了人脸图像，自动切换到 InstantID 模式
        if face_image is not None and task in ["host_face", "character_face", "realistic_face"]:
            task = f"{task}_instantid"
            print(f"  🎯 检测到人脸图像，切换到 InstantID 模式: {task}")
        
        # 路由到对应模型
        model_name = self.route(task)
        
        # 检查 pipeline 是否已缓存（避免重复创建）
        if model_name in self.pipelines:
            pipeline = self.pipelines[model_name]
            print(f"  🔍 调试: 使用缓存的 pipeline {model_name} (loaded={pipeline.loaded})")
        else:
            pipeline = self._get_pipeline(model_name)
            print(f"  🔍 调试: 创建新的 pipeline {model_name} (loaded={pipeline.loaded})")
        
        # 延迟加载
        if self.lazy_load:
            if not pipeline.loaded:
                print(f"  🔍 调试: 准备加载 pipeline {model_name} (loaded={pipeline.loaded})")
                pipeline.load()
                # 验证 loaded 状态是否正确设置
                if hasattr(pipeline, 'loaded'):
                    print(f"  🔍 调试: pipeline {model_name} 加载完成 (loaded={pipeline.loaded})")
                else:
                    print(f"  ⚠️  警告: pipeline {model_name} 没有 loaded 属性")
            else:
                print(f"  ⏭️  Pipeline {model_name} 已加载，跳过重复加载 (loaded={pipeline.loaded})")
        elif not pipeline.loaded:
            # 非延迟加载模式，但pipeline还未加载，需要加载
            print(f"  ⚠️  警告: 非延迟加载模式，但pipeline未加载，强制加载...")
            pipeline.load()
        
        # 优化提示词（针对特定任务）
        optimized_prompt, optimized_negative = self._optimize_prompt(task, prompt, negative_prompt)
        
        # 如果是 host_face 任务，添加科学主持人的角色描述（配合 LoRA 使用）
        base_task = task.replace("_instantid", "")
        if base_task == "host_face" and "host_person" in self.character_profiles:
            # 先构建角色描述
            optimized_prompt = self._build_character_prompt("host_person", optimized_prompt)
            print(f"  ✅ 已添加科学主持人角色描述（配合 LoRA 使用）")
            
            # 在角色描述后，强制在最前面添加真实感关键词（确保真实风格，避免动漫）
            # 这些关键词必须在最前面，权重最高
            real_style_keywords = "photorealistic, realistic, high quality, detailed, professional photography"
            if real_style_keywords.lower() not in optimized_prompt.lower():
                optimized_prompt = f"{real_style_keywords}, {optimized_prompt}"
                print(f"  ℹ 添加真实感关键词（最前面）：{real_style_keywords}（避免动漫风格）")
            
            # 确保有中国/亚洲人特征（在真实感关键词后）
            if "chinese" not in optimized_prompt.lower() and "asian" not in optimized_prompt.lower() and "中国" not in optimized_prompt and "亚洲" not in optimized_prompt:
                # 在真实感关键词后插入
                optimized_prompt = optimized_prompt.replace(real_style_keywords, f"{real_style_keywords}, Chinese, Asian", 1)
                print(f"  ℹ 添加中国/亚洲人特征：Chinese, Asian（确保中国人形象）")
            
            # 仅使用LoRA时，添加更精确的人脸特征描述以增强效果
            if face_image is not None and "instantid" in task.lower():
                # 如果FaceAnalyzer不可用，仅使用LoRA，需要更强的提示词
                # 在真实感关键词后插入
                face_keywords = "detailed facial features, accurate face"
                if face_keywords.lower() not in optimized_prompt.lower():
                    optimized_prompt = optimized_prompt.replace(real_style_keywords, f"{real_style_keywords}, {face_keywords}", 1)
                    print(f"  ℹ 仅使用LoRA模式，添加精确人脸特征描述以增强效果")
        
        # 使用模型默认参数（如果未指定）
        # 优先从配置文件读取（image.model_selection.character）
        if num_inference_steps is None:
            # 尝试从配置文件读取
            if self.config:
                image_config = self.config.get('image', {})
                model_selection = image_config.get('model_selection', {})
                character_config = model_selection.get('character', {})
                if 'num_inference_steps' in character_config:
                    num_inference_steps = int(character_config['num_inference_steps'])
                    print(f"  ℹ 从配置文件读取推理步数: {num_inference_steps}")
            
            # 如果配置文件没有，使用模型默认值
            if num_inference_steps is None:
                if model_name == "sd3":
                    num_inference_steps = 8  # SD3 Turbo 默认步数少
                elif model_name in ["flux1", "flux2", "flux1-instantid"]:
                    num_inference_steps = 28  # Flux 优化：28步已足够，速度提升约30%
                elif model_name == "kolors":
                    num_inference_steps = 22  # Kolors 默认步数
                else:
                    num_inference_steps = 50  # 其他模型默认
        
        if guidance_scale is None:
            # 尝试从配置文件读取
            if self.config:
                image_config = self.config.get('image', {})
                model_selection = image_config.get('model_selection', {})
                character_config = model_selection.get('character', {})
                if 'guidance_scale' in character_config:
                    guidance_scale = float(character_config['guidance_scale'])
                    print(f"  ℹ 从配置文件读取引导强度: {guidance_scale}")
            
            # 如果配置文件没有，使用模型默认值
            if guidance_scale is None:
                if model_name == "sd3":
                    guidance_scale = 1.0  # SD3 Turbo 低引导
                elif model_name in ["flux1", "flux2", "flux1-instantid"]:
                    guidance_scale = 3.5  # Flux 默认引导
                else:
                    guidance_scale = 7.5  # 其他模型默认
        
        print(f"🎨 使用模型: {model_name} (任务: {task})")
        if optimized_prompt != prompt:
            print(f"  ℹ 优化后的提示词: {optimized_prompt[:100]}...")
        if optimized_negative != (negative_prompt or ""):
            print(f"  ℹ 优化后的负面提示词: {optimized_negative[:100]}...")
        
        # 检查是否需要加载 LoRA
        # 注意：host_face_instantid 和 character_face_instantid 也应该使用对应的 LoRA
        lora_kwargs = {}
        # 获取基础任务名（去掉 _instantid 后缀）
        base_task = task.replace("_instantid", "")
        if base_task in self.lora_configs:
            lora_cfg = self.lora_configs[base_task]
            if lora_cfg.get("lora_path"):
                lora_kwargs["lora_path"] = lora_cfg["lora_path"]
                lora_kwargs["lora_alpha"] = lora_cfg.get("lora_alpha", 1.0)
                print(f"  ✅ 已配置 LoRA: {Path(lora_cfg['lora_path']).name} (alpha={lora_cfg.get('lora_alpha', 1.0)})")
        elif task in self.lora_configs:
            # 兼容旧的任务名
            lora_cfg = self.lora_configs[task]
            if lora_cfg.get("lora_path"):
                lora_kwargs["lora_path"] = lora_cfg["lora_path"]
                lora_kwargs["lora_alpha"] = lora_cfg.get("lora_alpha", 1.0)
                print(f"  ✅ 已配置 LoRA: {Path(lora_cfg['lora_path']).name} (alpha={lora_cfg.get('lora_alpha', 1.0)})")
        
        # 生成图像
        generate_kwargs = {
            "prompt": optimized_prompt,
            "negative_prompt": optimized_negative,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            **lora_kwargs,
        }
        
        # 如果使用 InstantID，添加人脸相关参数
        if "instantid" in model_name and face_image is not None:
            generate_kwargs["face_image"] = face_image
            generate_kwargs["face_strength"] = face_strength
        
        generate_kwargs.update(kwargs)
        
        image = pipeline.generate(**generate_kwargs)
        
        return image
    
    def _optimize_prompt(self, task: str, prompt: str, negative_prompt: Optional[str] = None) -> tuple[str, str]:
        """
        优化提示词（根据任务类型智能添加约束，不写死）
        
        Args:
            task: 任务类型
            prompt: 原始提示词
            negative_prompt: 原始负面提示词
            
        Returns:
            (优化后的提示词, 优化后的负面提示词)
        """
        optimized_prompt = prompt
        optimized_negative = negative_prompt or ""
        
        # 检查是否是 InstantID 任务
        is_instantid_task = task.endswith("_instantid")
        base_task = task.replace("_instantid", "") if is_instantid_task else task
        
        # 针对 host_face 任务（科普主持人脸）的智能优化
        if base_task in ["host_face", "character_face", "realistic_face"] or task in ["host_face", "character_face", "realistic_face"]:
            prompt_lower = prompt.lower()
            
            # 检查是否明确提到人物相关关键词
            has_person_keywords = any(kw in prompt_lower for kw in [
                "人", "人物", "角色", "person", "character", "people", "man", "woman",
                "主持人", "host", "presenter", "科普主持人", "face", "portrait"
            ])
            
            # 如果没有明确提到人物，添加"人物"关键词
            if not has_person_keywords:
                optimized_prompt = f"人物，{prompt}"
                print(f"  💡 提示词优化: 添加'人物'关键词，确保生成人物图像")
            
            # 检查是否明确指定了性别
            has_male = any(kw in prompt_lower for kw in ["男", "male", "man", "gentleman", "先生", "男士"])
            has_female = any(kw in prompt_lower for kw in ["女", "female", "woman", "lady", "女士", "小姐", "女孩"])
            
            # 检查是否是"科普主持人"场景（只有明确提到时才添加约束）
            is_science_host = any(kw in prompt for kw in ["科普主持人", "科普", "science host", "science presenter"])
            is_host = any(kw in prompt_lower for kw in ["主持人", "host", "presenter"])
            
            # 只在明确是"科普主持人"且未指定性别时，才添加"男性"约束
            if is_science_host and not has_male and not has_female:
                # 检查 prompt 中是否已经有 male/man/男 等关键词，避免重复
                prompt_lower_check = prompt.lower()
                if not any(kw in prompt_lower_check for kw in ["male", "man", "男", "男士", "先生"]):
                    # 只添加一个简洁的性别约束，避免重复关键词
                    # 同时添加中国/亚洲人特征，确保生成中国人形象（放在最前面，权重最高）
                    optimized_prompt = f"Chinese male, Asian male, {prompt}"
                    print(f"  ℹ 检测到科普主持人场景，自动添加性别和种族约束：Chinese male, Asian male（简洁）")
                else:
                    # 即使已有性别信息，也添加中国/亚洲人特征（确保中国人形象）
                    if "chinese" not in prompt_lower_check and "asian" not in prompt_lower_check and "中国" not in prompt and "亚洲" not in prompt:
                        optimized_prompt = f"Chinese, Asian, {prompt}"
                        print(f"  ℹ 添加中国/亚洲人特征：Chinese, Asian（确保中国人形象）")
                    else:
                        print(f"  ℹ 提示词中已包含性别和种族信息，不重复添加")
            # 如果用户明确指定了性别，完全尊重用户意图，不做任何修改
            elif has_female or has_male:
                # 用户已明确指定性别，不添加任何约束
                pass
            # 如果只是普通"主持人"但未指定性别，也不强制添加（让用户自由选择）
            elif is_host and not has_male and not has_female:
                # 不强制添加性别，保持用户原始意图
                pass
            
            # 只在明确是"科普"场景时，才添加专业风格约束
            if is_science_host:
                # 检查是否已有专业相关词汇
                has_professional = any(kw in prompt for kw in ["专业", "professional", "正式", "formal", "商务"])
                if not has_professional:
                    optimized_prompt = f"{optimized_prompt}, 专业形象"
                    print(f"  ℹ 检测到科普场景，添加专业形象约束")
            
            # 负面提示词：添加通用风格约束和性别约束（强化真实感）
            # 强化负面提示词，确保排除所有动漫、卡通风格
            style_negative = "cartoon, anime, animation, animated, fantasy, 卡通, 动漫, 动画, 幻想, 不专业, 不正式, low quality, blurry, distorted, illustration, drawing, sketch, 插画, 绘画, 手绘, 2d, stylized, artistic style, comic style, manga style, 3d render, cgi, computer graphics, digital art, concept art, game character, video game, animated character, cartoon character, anime character, manga character, chibi, kawaii, moe, cel shading, toon shading"
            
            # 如果是科普主持人且未指定性别，添加女性排除（强化）
            if is_science_host and not has_male and not has_female:
                gender_negative = "female, woman, girl, 女性, 女人, 女孩, 女士, 小姐, 女性特征, 女性形象, feminine, female features, female appearance, feminine appearance, woman features"
                style_negative = f"{gender_negative}, {style_negative}"
            
            # 只在负面提示词中不包含这些词时才添加
            if optimized_negative:
                if "cartoon" not in optimized_negative.lower() and "卡通" not in optimized_negative:
                    optimized_negative = f"{optimized_negative}, {style_negative}".strip(", ")
            else:
                optimized_negative = style_negative.strip(", ")
        
        return optimized_prompt, optimized_negative
    
    def _load_character_profiles(self) -> Dict[str, Any]:
        """加载角色描述配置文件"""
        profile_path = Path(__file__).parent / "character_profiles.yaml"
        if not profile_path.exists():
            print(f"  ⚠️  角色描述文件不存在: {profile_path}，将不使用角色描述")
            return {}
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                profiles = data.get("characters", {})
                if profiles:
                    print(f"  ✅ 已加载角色描述文件: {len(profiles)} 个角色")
                return profiles
        except Exception as e:
            print(f"  ⚠️  加载角色描述文件失败: {e}，将不使用角色描述")
            return {}
    
    def _build_character_prompt(self, character_id: str, prompt: str) -> str:
        """
        根据角色描述构建完整的提示词（优化版：限制长度，避免超过 77 tokens）
        
        Args:
            character_id: 角色ID（如 "host_person"）
            prompt: 原始提示词
            
        Returns:
            包含角色描述的完整提示词（精简版，确保不超过 77 tokens）
        """
        if character_id not in self.character_profiles:
            return prompt
        
        profile = self.character_profiles[character_id]
        parts = []
        
        # 1. 身份描述（精简，只保留核心）
        identity = profile.get("identity", "")
        if identity:
            # 只取第一个核心描述，移除权重标记
            identity_short = identity.split(",")[0].split(":")[0].strip()
            if identity_short:
                parts.append(identity_short)
        
        # 2. 面部特征（精简，只保留前 2 个关键词）
        face_keywords = profile.get("face_keywords", "")
        if face_keywords:
            # 只保留前 2 个关键词
            face_list = [f.strip() for f in face_keywords.split(",")[:2]]
            if face_list:
                parts.append(", ".join(face_list))
        
        # 3. 发型（精简，只保留第一个核心描述）
        hair = profile.get("hair", {})
        hair_keywords = hair.get("prompt_keywords", "")
        if hair_keywords:
            # 只保留第一个核心描述（移除权重）
            hair_first = hair_keywords.split(",")[0].split(":")[0].strip()
            if hair_first:
                parts.append(hair_first)
        elif not hair_keywords:
            # 使用备用字段（只保留样式）
            hair_style = hair.get("style", "")
            if hair_style:
                parts.append(hair_style.split()[0] if hair_style else "")
        
        # 4. 服饰（精简，只保留第一个核心描述）
        clothes = profile.get("clothes", {})
        clothes_keywords = clothes.get("prompt_keywords", "")
        if clothes_keywords:
            # 只保留第一个核心描述（移除权重）
            clothes_first = clothes_keywords.split(",")[0].split(":")[0].strip()
            if clothes_first:
                parts.append(clothes_first)
        elif not clothes_keywords:
            # 使用备用字段（只保留样式）
            clothes_style = clothes.get("style", "")
            if clothes_style:
                parts.append(clothes_style.split()[0] if clothes_style else "")
        
        # 5. 组合：角色描述 + 原始 prompt（确保不超过 77 tokens）
        # 估算 token 数量（粗略：1 token ≈ 0.75 个单词，中文 1 字 ≈ 1 token）
        character_desc = ", ".join(parts)
        
        # 如果角色描述太长，进一步精简
        # 目标：角色描述 < 20 tokens，prompt < 57 tokens，总计 < 77 tokens
        if len(character_desc) > 30:  # 粗略估算：30 字符 ≈ 20 tokens
            # 只保留身份和第一个特征
            parts = parts[:2] if len(parts) >= 2 else parts
            character_desc = ", ".join(parts)
        
        # 如果 prompt 太长，截断（保留前 40 字符）
        prompt_short = prompt
        if len(prompt) > 40:
            prompt_short = prompt[:40] + "..."
        
        enhanced_prompt = f"{character_desc}, {prompt_short}"
        
        # 最终检查：如果还是太长，只保留核心部分
        if len(enhanced_prompt) > 60:  # 60 字符 ≈ 40 tokens（安全范围）
            # 只保留：身份 + prompt 前 30 字符
            if parts:
                enhanced_prompt = f"{parts[0]}, {prompt[:30]}"
            else:
                enhanced_prompt = prompt[:50]
        
        return enhanced_prompt
    
    def unload(self, model_name: Optional[str] = None):
        """
        卸载模型，释放显存
        
        Args:
            model_name: 模型名称，None 时卸载所有模型
        """
        if model_name is None:
            # 卸载所有模型
            for pipeline in self.pipelines.values():
                pipeline.unload()
            self.pipelines.clear()
            torch.cuda.empty_cache()
            print("✅ 所有模型已卸载")
        else:
            # 卸载指定模型
            if model_name in self.pipelines:
                self.pipelines[model_name].unload()
                del self.pipelines[model_name]
                torch.cuda.empty_cache()
                print(f"✅ {model_name} 已卸载")
    
    def list_models(self) -> Dict[str, bool]:
        """列出所有模型及其状态"""
        status = {}
        for model_name, model_path in self.model_paths.items():
            exists = Path(model_path).exists()
            loaded = model_name in self.pipelines and self.pipelines[model_name].loaded
            status[model_name] = {
                "exists": exists,
                "loaded": loaded,
                "path": model_path
            }
        return status


# 使用示例
if __name__ == "__main__":
    # 创建 ModelManager
    manager = ModelManager(lazy_load=True)
    
    # 示例 1: 生成科普主持人脸
    print("\n" + "="*80)
    print("示例 1: 生成科普主持人脸")
    print("="*80)
    # img1 = manager.generate(
    #     task="host_face",
    #     prompt="一位温暖亲和的中国科普主持人，正面对镜头，专业形象",
    #     width=1024,
    #     height=1024
    # )
    # img1.save("host_face.png")
    # print("✅ 已保存: host_face.png")
    
    # 示例 2: 生成科学背景
    print("\n" + "="*80)
    print("示例 2: 生成科学背景")
    print("="*80)
    # img2 = manager.generate(
    #     task="science_background",
    #     prompt="量子计算机核心光学元件，蓝色光晕，高科技，未来感",
    #     width=1024,
    #     height=1024
    # )
    # img2.save("science_background.png")
    # print("✅ 已保存: science_background.png")
    
    # 列出所有模型状态
    print("\n" + "="*80)
    print("模型状态")
    print("="*80)
    status = manager.list_models()
    for model_name, info in status.items():
        exists = "✅" if info["exists"] else "❌"
        loaded = "已加载" if info["loaded"] else "未加载"
        print(f"{exists} {model_name}: {loaded} ({info['path']})")

