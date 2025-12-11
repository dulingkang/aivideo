#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景类型自动识别模块
增强的场景分类器，用于精确识别场景类型并选择最佳生成策略
"""

from typing import Dict, Any, Optional, List, Tuple


class SceneTypeClassifier:
    """场景类型分类器"""
    
    def __init__(self):
        """初始化分类器"""
        # 场景类型关键词
        self.scene_type_keywords = {
            "character_scene": [
                "han li", "韩立", "character", "person", "人物", "角色", "主角",
                "face", "面容", "表情", "眼神", "手", "手臂", "身影",
            ],
            "environment_scene": [
                "desert", "沙漠", "chamber", "洞府", "corridor", "走廊", "遗迹",
                "sky", "天空", "cloud", "云", "mountain", "山", "landscape", "风景",
            ],
            "object_focus": [
                "scroll", "卷轴", "token", "令牌", "artifact", "法器", "pill", "丹药",
            ],
            "action_scene": [
                "attack", "攻击", "fight", "战斗", "cast", "施法", "strike", "出击",
                "walk", "走", "run", "跑", "jump", "跳", "fly", "飞",
            ],
            "dialogue_scene": [
                "say", "说", "speak", "讲", "talk", "对话", "交谈", "讨论",
            ],
            "static_scene": [
                "still", "静止", "motionless", "不动", "一动不动", "lying still",
            ],
        }
        
        # 镜头类型关键词
        self.camera_type_keywords = {
            "close_up": ["close-up", "closeup", "close up", "特写", "近景", "面部特写"],
            "medium_shot": ["medium shot", "mid shot", "中景", "半身", "上半身"],
            "wide_shot": ["wide shot", "full body", "full figure", "全身", "远景", "wide view"],
        }
    
    def classify(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        分类场景类型
        
        Returns:
            {
                "primary_type": str,  # 主要类型（character_scene, environment_scene等）
                "camera_type": str,  # 镜头类型（close_up, medium_shot, wide_shot）
                "has_character": bool,  # 是否有角色
                "has_object_motion": bool,  # 是否有物体运动
                "has_camera_motion": bool,  # 是否有镜头运动
                "motion_type": str,  # 运动类型
                "recommended_strategy": str,  # 推荐策略（svd, deforum, static）
            }
        """
        result = {
            "primary_type": "environment_scene",
            "camera_type": "medium_shot",
            "has_character": False,
            "has_object_motion": False,
            "has_camera_motion": False,
            "motion_type": "static",
            "recommended_strategy": "svd",
        }
        
        if not scene:
            return result
        
        # 收集所有文本
        description = (scene.get("description", "") or "").lower()
        action = (scene.get("action", "") or "").lower()
        visual = scene.get("visual", {})
        composition = (visual.get("composition", "") or "").lower() if isinstance(visual, dict) else ""
        fx = (visual.get("fx", "") or "").lower() if isinstance(visual, dict) else ""
        camera = (scene.get("camera", "") or "").lower()
        motion = visual.get("motion", {}) if isinstance(visual, dict) else {}
        
        all_text = f"{description} {action} {composition} {fx}".lower()
        
        # 1. 检测是否有角色
        has_character = (
            bool(scene.get("characters")) or
            any(kw in all_text for kw in self.scene_type_keywords["character_scene"])
        )
        result["has_character"] = has_character
        
        # 2. 检测镜头类型
        for cam_type, keywords in self.camera_type_keywords.items():
            if any(kw in composition or kw in camera for kw in keywords):
                result["camera_type"] = cam_type
                break
        
        # 3. 检测物体运动
        object_motion_keywords = [
            "unfurling", "unfold", "展开", "open", "打开",
            "rotate", "旋转", "float", "飘动", "shimmer", "闪烁",
        ]
        has_object_motion = any(kw in all_text for kw in object_motion_keywords)
        result["has_object_motion"] = has_object_motion
        
        # 4. 检测镜头运动
        if isinstance(motion, dict) and motion.get("type") and motion.get("type") != "static":
            result["has_camera_motion"] = True
            result["motion_type"] = motion.get("type", "static")
        elif any(kw in camera for kw in ["pan", "zoom", "dolly", "tilt", "横移", "推进", "拉远"]):
            result["has_camera_motion"] = True
            if "pan" in camera or "横移" in camera:
                result["motion_type"] = "pan"
            elif "zoom" in camera or "推" in camera or "拉" in camera:
                result["motion_type"] = "zoom"
        
        # 5. 确定主要类型
        if has_character:
            if any(kw in all_text for kw in self.scene_type_keywords["action_scene"]):
                result["primary_type"] = "action_scene"
            elif any(kw in all_text for kw in self.scene_type_keywords["dialogue_scene"]):
                result["primary_type"] = "dialogue_scene"
            else:
                result["primary_type"] = "character_scene"
        elif any(kw in all_text for kw in self.scene_type_keywords["object_focus"]):
            result["primary_type"] = "object_focus"
        elif any(kw in all_text for kw in self.scene_type_keywords["static_scene"]):
            result["primary_type"] = "static_scene"
        else:
            result["primary_type"] = "environment_scene"
        
        # 6. 推荐策略
        if result["primary_type"] == "static_scene" and not result["has_object_motion"]:
            result["recommended_strategy"] = "deforum"  # 使用Deforum静态动画
        elif result["has_object_motion"] or result["has_camera_motion"]:
            result["recommended_strategy"] = "svd"  # 使用SVD
        elif result["primary_type"] == "character_scene":
            result["recommended_strategy"] = "svd"  # 人物场景使用SVD
        else:
            result["recommended_strategy"] = "deforum"  # 环境场景可以使用Deforum
        
        return result

