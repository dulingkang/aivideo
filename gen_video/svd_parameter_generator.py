#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVD参数自动生成模块
根据场景类型自动生成最优的SVD参数（motion_bucket_id、noise_aug_strength等）
"""

from typing import Dict, Any, Optional


class SVDParameterGenerator:
    """SVD参数自动生成器"""
    
    def __init__(self):
        """初始化参数生成器"""
        # 场景类型到参数的映射
        self.scene_type_params = {
            # 基于运动强度的通用参数映射（更健壮，不依赖关键词匹配）
            # 轻微运动（gentle）：表情变化、微动、环境微动
            "gentle": {
                "motion_bucket_id": 1.4,  # 较低，减少过快运动
                "noise_aug_strength": 0.00025,  # 保持稳定
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            # 中等运动（moderate）：物体运动、镜头运动、人物中等动作
            "moderate": {
                "motion_bucket_id": 1.6,  # 中等，平衡运动明显度和稳定性
                "noise_aug_strength": 0.0003,  # 保持流畅
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            # 明显运动（dynamic）：快速动作、明显物体运动
            "dynamic": {
                "motion_bucket_id": 1.8,  # 较高，确保动作明显
                "noise_aug_strength": 0.0003,  # 保持流畅
                "num_inference_steps": 45,  # 提高稳定性
                "decode_chunk_size": 7,  # 降低，提高连贯性
            },
            
            # 人物场景（基于镜头类型）
            "character_close_up": {
                "motion_bucket_id": 1.4,  # 特写：较低，避免人物变形
                "noise_aug_strength": 0.00025,
                "num_inference_steps": 45,  # 较高，提高稳定性
                "decode_chunk_size": 8,
            },
            "character_medium": {
                "motion_bucket_id": 1.5,  # 中景：中等
                "noise_aug_strength": 0.00025,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            "character_wide": {
                "motion_bucket_id": 1.6,  # 远景：稍高，可以更多运动
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            
            # 环境场景
            "environment_static": {
                "motion_bucket_id": 1.4,  # 环境微动
                "noise_aug_strength": 0.00025,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            "environment_dynamic": {
                "motion_bucket_id": 1.6,  # 环境有明显运动
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            
            # 物体运动场景（基于运动类型）
            "object_unfurling": {
                "motion_bucket_id": 1.7,  # 展开运动：中等偏高
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 45,
                "decode_chunk_size": 7,
            },
            "object_rotating": {
                "motion_bucket_id": 1.6,  # 旋转：中等
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            "object_floating": {
                "motion_bucket_id": 1.4,  # 飘动：轻微运动
                "noise_aug_strength": 0.00025,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            
            # 镜头运动场景
            "camera_pan": {
                "motion_bucket_id": 1.6,  # 镜头移动：中等
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            "camera_zoom": {
                "motion_bucket_id": 1.5,  # 缩放：中等偏低
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            "camera_dolly": {
                "motion_bucket_id": 1.6,  # 跟拍：中等
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            
            # 动作场景（基于运动强度）
            "action_dynamic": {
                "motion_bucket_id": 1.8,  # 明显动作
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 45,
                "decode_chunk_size": 7,
            },
            "action_moderate": {
                "motion_bucket_id": 1.6,  # 中等动作
                "noise_aug_strength": 0.0003,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
            "action_gentle": {
                "motion_bucket_id": 1.4,  # 轻微动作
                "noise_aug_strength": 0.00025,
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            },
        }
    
    def generate_params(
        self,
        scene: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        根据场景和分析结果生成SVD参数
        
        Args:
            scene: 场景JSON数据
            analysis: 场景分析结果（来自SceneMotionAnalyzer）
            base_config: 基础配置（可选）
        
        Returns:
            SVD参数
        """
        if base_config is None:
            base_config = {
                "num_inference_steps": 40,
                "decode_chunk_size": 8,
            }
        
        # 优先使用运动强度（更健壮，不依赖关键词匹配）
        motion_intensity = None
        if analysis:
            motion_intensity = analysis.get("motion_intensity")
        
        # 如果analysis中有明确的运动强度，优先使用
        if motion_intensity and motion_intensity in ["gentle", "moderate", "dynamic"]:
            type_params = self.scene_type_params.get(motion_intensity)
            if type_params:
                params = {
                    "motion_bucket_id": type_params["motion_bucket_id"],
                    "noise_aug_strength": type_params["noise_aug_strength"],
                    "num_inference_steps": type_params.get("num_inference_steps", base_config.get("num_inference_steps", 40)),
                    "decode_chunk_size": type_params.get("decode_chunk_size", base_config.get("decode_chunk_size", 8)),
                }
                # 如果analysis中有覆盖值，使用覆盖值（但运动强度参数优先级更高）
                if analysis.get("motion_bucket_id_override") is not None:
                    params["motion_bucket_id"] = analysis["motion_bucket_id_override"]
                if analysis.get("noise_aug_strength_override") is not None:
                    params["noise_aug_strength"] = analysis["noise_aug_strength_override"]
                return params
        
        # 否则，使用场景类型分类
        scene_type = self._classify_scene_type(scene, analysis)
        
        # 获取该类型的参数
        type_params = self.scene_type_params.get(scene_type, {
            "motion_bucket_id": 1.5,
            "noise_aug_strength": 0.00025,
            "num_inference_steps": 40,
            "decode_chunk_size": 8,
        })
        
        # 合并参数
        params = {
            "motion_bucket_id": type_params["motion_bucket_id"],
            "noise_aug_strength": type_params["noise_aug_strength"],
            "num_inference_steps": type_params.get("num_inference_steps", base_config.get("num_inference_steps", 40)),
            "decode_chunk_size": type_params.get("decode_chunk_size", base_config.get("decode_chunk_size", 8)),
        }
        
        # 如果analysis中有覆盖值，使用覆盖值
        if analysis:
            if analysis.get("motion_bucket_id_override") is not None:
                params["motion_bucket_id"] = analysis["motion_bucket_id_override"]
            if analysis.get("noise_aug_strength_override") is not None:
                params["noise_aug_strength"] = analysis["noise_aug_strength_override"]
        
        return params
    
    def _classify_scene_type(
        self,
        scene: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        分类场景类型
        
        Returns:
            场景类型字符串
        """
        if analysis:
            # 优先使用analysis的结果
            if analysis.get("has_object_motion"):
                object_type = analysis.get("object_motion_type", "")
                if object_type == "unfurling":
                    return "object_unfurling"
                elif object_type == "rotating":
                    return "object_rotating"
                elif object_type == "floating":
                    return "object_floating"
            
            camera_motion = analysis.get("camera_motion_type", "static")
            if camera_motion != "static":
                if camera_motion == "pan":
                    return "camera_pan"
                elif camera_motion == "zoom":
                    return "camera_zoom"
                elif camera_motion == "dolly":
                    return "camera_dolly"
            
            motion_intensity = analysis.get("motion_intensity", "static")
            if motion_intensity == "dynamic":
                return "action_dynamic"
            elif motion_intensity == "moderate":
                return "action_moderate"
            elif motion_intensity == "gentle":
                return "action_gentle"
        
        # 从scene推断
        visual = scene.get("visual", {})
        composition = (visual.get("composition", "") or "").lower()
        camera = (scene.get("camera", "") or "").lower()
        
        # 检查是否有角色
        has_character = bool(scene.get("characters")) or "han li" in composition.lower() or "韩立" in composition.lower()
        
        if has_character:
            # 人物场景
            if "close" in composition or "close-up" in composition or "特写" in composition:
                return "character_close_up"
            elif "medium" in composition or "mid" in composition or "中景" in composition:
                return "character_medium"
            elif "wide" in composition or "distant" in composition or "远景" in composition:
                return "character_wide"
            else:
                return "character_medium"  # 默认中景
        else:
            # 环境场景
            description = (scene.get("description", "") or "").lower()
            if any(kw in description for kw in ["雾气", "粒子", "飘动", "流动", "mist", "particle", "floating"]):
                return "environment_dynamic"
            else:
                return "environment_static"
        
        # 默认
        return "environment_static"

