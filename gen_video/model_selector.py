#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型选择器 - 根据任务类型自动选择最适合的模型
"""

from typing import Dict, Any, Optional, Literal
from enum import Enum


class TaskType(Enum):
    """任务类型"""
    CHARACTER = "character"  # 人物生成（主持人）
    SCENE = "scene"         # 场景生成（科普背景）
    BATCH = "batch"          # 批量生成（备选图）


class ModelSelector:
    """模型选择器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_config = config.get("image", {})
        self.model_selection = self.image_config.get("model_selection", {})
    
    def select_engine(
        self,
        task_type: Optional[TaskType] = None,
        prompt: Optional[str] = None,
        scene_context: Optional[Dict[str, Any]] = None,
        manual_engine: Optional[str] = None
    ) -> str:
        """
        根据任务类型和提示词选择最适合的模型引擎
        
        Args:
            task_type: 任务类型（人物/场景/批量）
            prompt: 生成提示词
            scene_context: 场景上下文信息
            manual_engine: 手动指定的引擎（优先级最高）
        
        Returns:
            模型引擎名称
        """
        # 如果手动指定了引擎，直接使用
        if manual_engine and manual_engine != "auto":
            return manual_engine
        
        # 如果没有指定任务类型，尝试自动检测
        if task_type is None:
            task_type = self._detect_task_type(prompt, scene_context)
        
        if task_type == TaskType.CHARACTER:
            # 人物生成：统一使用 InstantID (SDXL + InstantID)
            # 原因：
            # 1. 韩立需要InstantID保证人脸一致性
            # 2. 其他角色也需要使用InstantID，因为SDXL支持风格LoRA，可以保持风格统一
            # 3. 如果使用Flux，风格LoRA无法应用，会导致风格不统一
            return "instantid"
        
        elif task_type == TaskType.SCENE:
            # 场景生成：根据提示词内容选择
            return self._select_scene_engine(prompt, scene_context)
        
        elif task_type == TaskType.BATCH:
            # 批量生成：使用 SD3 Turbo
            return "sd3-turbo"
        
        else:
            # 默认使用当前方案（InstantID）
            return self.image_config.get("engine", "instantid")
    
    def _detect_task_type(
        self,
        prompt: Optional[str],
        scene_context: Optional[Dict[str, Any]]
    ) -> TaskType:
        """自动检测任务类型"""
        # 如果场景上下文中有 face_reference_image_path，判断为人物生成
        if scene_context:
            if scene_context.get("face_reference_image_path"):
                return TaskType.CHARACTER
            if scene_context.get("character_lora"):
                return TaskType.CHARACTER
        
        # 如果提示词中包含人物相关关键词，判断为人物生成
        if prompt:
            character_keywords = [
                "主持人", "讲解员", "人物", "角色", "人像", "肖像",
                "han li", "hanli", "韩立", "主角", "main character", "cultivator",
                "presenter", "host", "character", "portrait", "person"
            ]
            prompt_lower = prompt.lower()
            if any(keyword in prompt_lower for keyword in character_keywords):
                return TaskType.CHARACTER
        
        # 默认判断为场景生成
        return TaskType.SCENE
    
    def _select_scene_engine(
        self,
        prompt: Optional[str],
        scene_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """根据提示词选择场景生成引擎"""
        if not prompt:
            # 默认使用 Flux.1（更稳定，flux2 加载可能失败）
            return "flux1"
        
        prompt_lower = prompt.lower()
        
        # Flux.2 场景关键词（科学背景图、太空/粒子/量子类，冲击力强）
        flux2_keywords = [
            "太空", "宇宙", "粒子", "量子", "科学背景", "科技背景",
            "冲击", "震撼", "爆炸", "能量", "光束", "粒子效果",
            "space", "particle", "quantum", "scientific background",
            "impact", "explosive", "energy", "beam"
        ]
        
        # Flux.1 场景关键词（实验室/医学，更干净自然）
        flux1_keywords = [
            "实验室", "医学", "医疗", "医院", "手术", "实验设备",
            "干净", "自然", "清晰", "专业",
            "laboratory", "medical", "hospital", "surgery", "clean", "natural"
        ]
        
        # 中文场景关键词（优先使用 Hunyuan-DiT）
        chinese_scene_keywords = [
            "科技馆", "博物馆", "天文台",
            "中国", "中文", "科教", "科普", "教育",
            "研究", "学术", "教学"
        ]
        
        # 真实感场景关键词（优先使用 Kolors）
        realism_keywords = [
            "真实", "照片", "摄影", "手部", "光影", "细节",
            "realistic", "photorealistic", "photo", "photography"
        ]
        
        # 批量生成关键词（使用 SD3 Turbo）
        batch_keywords = [
            "批量", "备选", "多个", "variations", "batch", "multiple"
        ]
        
        # 检查是否包含批量生成关键词
        if any(keyword in prompt_lower for keyword in batch_keywords):
            return "sd3-turbo"
        
        # 检查是否包含 Flux.2 关键词（优先）
        if any(keyword in prompt_lower for keyword in flux2_keywords):
            return "flux2"  # 科学背景图、太空/粒子/量子类，冲击力强
        
        # 检查是否包含 Flux.1 关键词
        if any(keyword in prompt_lower for keyword in flux1_keywords):
            return "flux1"  # 实验室/医学，更干净自然
        
        # 检查是否包含中文场景关键词
        if any(keyword in prompt for keyword in chinese_scene_keywords):
            return "hunyuan-dit"
        
        # 检查是否包含真实感关键词
        if any(keyword in prompt_lower for keyword in realism_keywords):
            return "kolors"  # 使用 Kolors（真人质感强，中文 prompt 理解优秀）
        
        # 默认使用 Flux.1（更稳定，flux2 加载可能失败）
        return "flux1"
    
    def select_engine_for_scene_v2(
        self,
        scene: Dict[str, Any],
        manual_engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        基于 Scene JSON v2 自动选择图像生成引擎（Execution Planner v1）
        
        核心策略：
        - 默认用 SDXL（人物稳定）
        - 只有"它做不好"的时候才切 Flux（世界/氛围）
        
        Args:
            scene: Scene JSON v2 格式的场景数据
            manual_engine: 手动指定的引擎（优先级最高）
        
        Returns:
            {
                "engine": "instantid" | "sdxl" | "flux1" | "flux2",
                "mode": "instantid" | "normal" | "cinematic",
                "lock_face": bool,
                "task_type": "character" | "scene"
            }
        """
        # 如果手动指定了引擎，直接使用
        if manual_engine and manual_engine != "auto":
            return {
                "engine": manual_engine,
                "mode": "normal",
                "lock_face": False,
                "task_type": "scene"
            }
        
        # 提取关键字段（安全读取，避免 KeyError）
        character = scene.get("character", {}) or {}
        camera = scene.get("camera", {}) or {}
        intent = scene.get("intent", {}) or {}
        scene_role = scene.get("scene_role", "")
        
        # 判断是否有角色
        character_present = character.get("present", False)
        face_visible = character.get("face_visible", False)
        visibility = character.get("visibility", "low")  # high/mid/low
        camera_shot = camera.get("shot", "medium")
        intent_type = intent.get("type", "")
        
        # ============================================================
        # Rule 1: 有人物 + 近景/特写 → SDXL + InstantID（最高优先级）
        # ============================================================
        if character_present:
            # 判断是否应该锁脸
            should_lock_face = False
            
            # 条件1: face_visible 明确为 True
            if face_visible:
                should_lock_face = True
            
            # 条件2: visibility 为 high 或 mid
            if visibility in ["high", "mid"]:
                should_lock_face = True
            
            # 条件3: 镜头是 close_up 或 medium
            if camera_shot in ["close_up", "extreme_close", "medium"]:
                should_lock_face = True
            
            if should_lock_face:
                return {
                    "engine": "instantid",  # SDXL + InstantID
                    "mode": "instantid",
                    "lock_face": True,
                    "task_type": "character"
                }
            
            # ============================================================
            # Rule 2: 三类镜头分流（人设锚点方案）
            # 
            # 🟢 A类：叙事/氛围镜头（FLUX，禁用 InstantID/LoRA，reference=人设锚点图）
            #   用在：躺沙漠、远景、背影、剪影
            # 
            # 🟡 B类：过渡人物镜头（SDXL，reference=人设锚点图，不用 InstantID）
            #   用在：站立、走路、回头
            # 
            # 🔴 C类：情绪/表情镜头（InstantID，reference=人设锚点图，中近景）
            #   用在：回忆、痛苦、施法特写
            # ============================================================
            camera_angle = camera.get("angle", "eye_level")
            character_pose = character.get("pose", "")
            
            # 检测镜头类型
            is_wide_topdown_lying = (
                camera_shot == "wide" and 
                camera_angle == "top_down" and 
                character_pose in ["lying_motionless", "lying"]
            )
            
            is_narrative_shot = (
                camera_shot == "wide" or 
                visibility == "low" or
                is_wide_topdown_lying or
                character_pose in ["lying_motionless", "lying", "back_view"]
            )
            
            is_transition_shot = (
                camera_shot == "medium" and
                character_pose in ["standing", "walking", "turning"]
            )
            
            is_emotion_shot = (
                camera_shot in ["close", "medium"] and
                character_pose in ["thinking", "pain", "casting", "expression"]
            )
            
            # 🟢 A类：叙事/氛围镜头 → FLUX
            if is_narrative_shot:
                result = {
                    "engine": "flux1",  # 使用 FLUX.1（更稳定）
                    "mode": "cinematic",
                    "lock_face": False,
                    "task_type": "character",
                    "shot_category": "narrative",  # ⚡ 新增：镜头类别
                    "use_character_anchor": True,  # ⚡ 新增：必须使用人设锚点图
                    "style_anchor": {
                        "enabled": False
                    },
                    "disable_character_lora": True,  # 禁用角色 LoRA
                    "disable_style_lora": True,  # 禁用风格 LoRA
                    "disable_ip_adapter": False,  # ⚡ 关键修复：不禁用 IP-Adapter，需要使用参考图
                    "treat_as_silhouette": True,  # 标记为"剪影+氛围"镜头
                    "use_semantic_prompt": True,  # 使用语义化 prompt（FLUX 优势）
                }
                print(f"  🟢 A类镜头（叙事/氛围）：使用 FLUX 引擎（世界观一致性 > 人脸一致性）")
                print(f"  ✓ 必须引用人设锚点图（确保形象一致性）")
                print(f"  ✓ 禁用 LoRA（避免姿态冲突），但使用 IP-Adapter 引用参考图")
                return result
            
            # 🟡 B类：过渡人物镜头 → SDXL
            if is_transition_shot:
                result = {
                    "engine": "sdxl",
                    "mode": "normal",
                    "lock_face": False,
                    "task_type": "character",
                    "shot_category": "transition",  # ⚡ 新增：镜头类别
                    "use_character_anchor": True,  # ⚡ 新增：必须使用人设锚点图
                    "style_anchor": {
                        "type": "lora",
                        "name": "fanren_style",
                        "weight": 0.35,
                        "enabled": True
                    }
                }
                print(f"  🟡 B类镜头（过渡人物）：使用 SDXL 引擎")
                print(f"  ✓ 必须引用人设锚点图（确保形象一致性）")
                return result
            
            # 🔴 C类：情绪/表情镜头 → InstantID
            if is_emotion_shot:
                result = {
                    "engine": "instantid",
                    "mode": "face_lock",
                    "lock_face": True,
                    "task_type": "character",
                    "shot_category": "emotion",  # ⚡ 新增：镜头类别
                    "use_character_anchor": True,  # ⚡ 新增：必须使用人设锚点图
                    "style_anchor": {
                        "type": "lora",
                        "name": "fanren_style",
                        "weight": 0.35,
                        "enabled": True
                    }
                }
                print(f"  🔴 C类镜头（情绪/表情）：使用 InstantID 引擎（锁脸）")
                print(f"  ✓ 必须引用人设锚点图（确保形象一致性）")
                return result
            
            # 默认：根据镜头类型选择
            if camera_shot == "wide" or visibility == "low":
                # 远景场景，使用 SDXL + 风格锚点
                result = {
                    "engine": "sdxl",
                    "mode": "normal",
                    "lock_face": False,
                    "task_type": "character",
                    "shot_category": "default",
                    "use_character_anchor": True,  # ⚡ 新增：必须使用人设锚点图
                    "style_anchor": {
                        "type": "lora",
                        "name": "fanren_style",
                        "weight": 0.35,
                        "enabled": True
                    }
                }
                return result
            
            # ============================================================
            # Rule 3: 人物存在但中景 → SDXL（不锁脸，但用 SDXL 保证一致性）+ 风格锚点
            # ⚡ 关键修复：禁用 InstantID 时，必须绑定风格锚点
            # ============================================================
            if camera_shot == "medium":
                return {
                    "engine": "sdxl",
                    "mode": "normal",
                    "lock_face": False,
                    "task_type": "character",
                    "style_anchor": {  # ⚡ 新增：风格锚点配置
                        "type": "lora",
                        "name": "fanren_style",  # 凡人修仙传风格 LoRA
                        "weight": 0.35,  # 低权重，只绑定风格，不抢戏
                        "enabled": True
                    }
                }
            
            # 其他情况（人物存在但镜头类型不明确）→ 默认 SDXL + 风格锚点
            return {
                "engine": "sdxl",
                "mode": "normal",
                "lock_face": False,
                "task_type": "character",
                "style_anchor": {  # ⚡ 新增：风格锚点配置
                    "type": "lora",
                    "name": "fanren_style",  # 凡人修仙传风格 LoRA
                    "weight": 0.35,  # 低权重，只绑定风格，不抢戏
                    "enabled": True
                }
            }
        
        # ============================================================
        # Rule 4: 没有人物，是世界观镜头 → Flux
        # ============================================================
        if not character_present:
            # 判断是否是世界观/环境镜头
            world_intent_types = [
                "title_reveal",
                "introduce_world",
                "establish_world",
                "opening",
                "transition"
            ]
            
            if intent_type in world_intent_types or scene_role in ["opening", "establishing", "transition"]:
                # 根据场景类型选择 Flux 版本
                # 如果是科学/太空类，用 flux2；否则用 flux1
                visual_constraints = scene.get("visual_constraints", {}) or {}
                environment = str(visual_constraints.get("environment", "")).lower()
                
                flux2_keywords = [
                    "space", "particle", "quantum", "scientific",
                    "太空", "宇宙", "粒子", "量子", "科学"
                ]
                
                if any(kw in environment for kw in flux2_keywords):
                    return {
                        "engine": "flux2",
                        "mode": "cinematic",
                        "lock_face": False,
                        "task_type": "scene"
                    }
                else:
                    return {
                        "engine": "flux1",
                        "mode": "cinematic",
                        "lock_face": False,
                        "task_type": "scene"
                    }
        
        # ============================================================
        # Fallback Rule: 默认用 SDXL（人物驱动小说推文）
        # ============================================================
        return {
            "engine": "sdxl",
            "mode": "normal",
            "lock_face": False,
            "task_type": "scene"
        }
    
    def get_engine_config(self, engine: str) -> Dict[str, Any]:
        """获取指定引擎的配置"""
        model_selection = self.image_config.get("model_selection", {})
        
        if engine == "flux-instantid":
            return model_selection.get("character", {})
        elif engine == "hunyuan-dit":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("hunyuan_dit", {})
        elif engine == "kolors":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("kolors", {})
        elif engine == "realistic-vision":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("realistic_vision", {})
        elif engine == "sd3-turbo":
            scene_config = model_selection.get("scene", {})
            return scene_config.get("sd3_turbo", {})
        else:
            # 默认配置（InstantID 或 SDXL）
            if engine == "instantid":
                return self.image_config.get("instantid", {})
            else:
                return self.image_config.get("sdxl", {})

