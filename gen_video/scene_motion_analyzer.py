#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景动作和镜头智能分析模块
根据JSON描述自动推断动作类型、镜头运动和视频生成参数
"""

from typing import Dict, Any, Optional


class SceneMotionAnalyzer:
    """场景动作和镜头智能分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 物体运动模式（中英文关键词）
        self.object_motion_patterns = {
            "unfurling": ["unfurling", "unfold", "unfolding", "展开", "舒展开", "展开来", "缓缓展开", "卷轴展开"],
            "opening": ["open", "opening", "打开", "开启", "张开"],
            "spreading": ["spread", "spreading", "扩散", "蔓延"],
            "rotating": ["rotate", "rotating", "spin", "spinning", "turn", "turning", "旋转", "转动", "翻转"],
            "floating": ["float", "floating", "drift", "drifting", "飘动", "漂浮", "流动"],
            "rising": ["rise", "rising", "ascend", "ascending", "上升", "升起", "升腾"],
            "falling": ["fall", "falling", "drop", "dropping", "descend", "descending", "落下", "掉落", "坠落"],
            "shimmering": ["shimmer", "shimmering", "glow", "glowing", "闪烁", "发光", "闪耀"],
            "particles": ["particle", "particles", "粒子", "光点", "星点", "spirit particles"],
        }
        
        # 镜头运动关键词
        self.camera_motion_keywords = {
            "pan": ["pan", "横移", "平移", "左右", "left", "right"],
            "tilt": ["tilt", "上下", "up", "down", "仰", "俯"],
            "zoom": ["zoom", "推", "拉", "推进", "拉远", "in", "out", "淡入", "淡出"],
            "dolly": ["dolly", "跟", "跟随", "tracking"],
            "orbit": ["orbit", "环绕", "旋转", "rotate"],
            "static": ["static", "静止", "固定", "定格"],
        }
        
        # 动态动作关键词（明显动作，需要较高运动参数）
        self.dynamic_keywords = [
            "walk", "run", "attack", "fight", "jump", "fly", "移动", "奔跑", "攻击", "战斗", "跳跃", "飞行",
            "cast", "casting", "施法", "使用法术", "释放", "strike", "dodge", "evade", "出击", "闪避"
        ]
        
        # 中等动作关键词（轻微动作，需要中等运动参数）
        self.moderate_action_keywords = [
            "tilt", "turn", "move", "shift", "adjust", "侧", "转", "移动", "调整", "变化",
            "gather", "collect", "inhale", "exhale", "collecting", "gathering", "吸", "呼", "收集",
            "look", "glance", "gaze", "stare", "看", "注视", "凝视", "瞥",
            "raise", "lower", "lift", "drop", "举", "放", "抬起", "放下"
        ]
        
        # 静态动作关键词
        self.static_keywords = [
            "still", "motionless", "静止", "不动", "一动不动", "躺着不动", "站着不动"
        ]
    
    def analyze(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        智能分析场景动作和镜头需求
        
        Args:
            scene: 场景JSON数据
            
        Returns:
            {
                "has_object_motion": bool,  # 是否有物体运动（如卷轴展开、门打开等）
                "object_motion_type": str,  # 物体运动类型（unfurling, opening, rotating等）
                "camera_motion_type": str,  # 镜头运动类型（pan, zoom, static等）
                "camera_motion_direction": str,  # 镜头运动方向
                "camera_motion_speed": str,  # 镜头运动速度
                "motion_intensity": str,  # 运动强度（static, gentle, moderate, dynamic）
                "use_svd": bool,  # 是否使用SVD生成视频
                "motion_bucket_id_override": Optional[int],  # 运动参数覆盖
                "noise_aug_strength_override": Optional[float],  # 噪声参数覆盖
                "recommended_motion": Dict,  # 推荐的motion字段值
            }
        """
        result = {
            "has_object_motion": False,
            "object_motion_type": None,
            "camera_motion_type": "static",
            "camera_motion_direction": None,
            "camera_motion_speed": "slow",
            "motion_intensity": "static",
            "use_svd": False,
            "motion_bucket_id_override": None,
            "noise_aug_strength_override": None,
            "recommended_motion": None,
        }
        
        if not scene:
            return result
        
        # 收集所有文本信息
        description = (scene.get("description") or "").lower()
        action = (scene.get("action") or "").lower()
        prompt = (scene.get("prompt") or "").lower()
        camera = (scene.get("camera") or "").lower()
        
        visual = scene.get("visual") or {}
        if isinstance(visual, dict):
            composition = (visual.get("composition") or "").lower()
            fx = (visual.get("fx") or "").lower()
            motion_value = visual.get("motion")
        else:
            composition = ""
            fx = ""
            motion_value = None
        
        # 合并所有文本用于分析
        all_text = f"{description} {action} {prompt} {composition} {fx}".lower()
        
        # ========== 1. 检测物体运动 ==========
        # 优先检查composition和fx字段，这些字段通常更明确地描述物体运动
        composition_lower = composition.lower()
        fx_lower = fx.lower()
        composition_fx_text = f"{composition_lower} {fx_lower}".lower()
        
        # 先检查composition和fx（更明确）
        for motion_type, keywords in self.object_motion_patterns.items():
            if any(keyword in composition_fx_text for keyword in keywords):
                result["has_object_motion"] = True
                result["object_motion_type"] = motion_type
                result["use_svd"] = True  # 有物体运动，必须使用SVD
                result["motion_intensity"] = "gentle" if motion_type in ["shimmering", "particles", "floating"] else "moderate"
                print(f"  ℹ 从composition/fx检测到物体运动: {motion_type}")
                break
        
        # 如果composition/fx没有检测到，再检查all_text（包括description等）
        if not result["has_object_motion"]:
            for motion_type, keywords in self.object_motion_patterns.items():
                if any(keyword in all_text for keyword in keywords):
                    result["has_object_motion"] = True
                    result["object_motion_type"] = motion_type
                    result["use_svd"] = True  # 有物体运动，必须使用SVD
                    result["motion_intensity"] = "gentle" if motion_type in ["shimmering", "particles", "floating"] else "moderate"
                    print(f"  ℹ 从description等检测到物体运动: {motion_type}")
                    break
        
        # ========== 2. 检测镜头运动 ==========
        # 优先从motion字段读取（如果存在，说明明确指定了镜头运动）
        if motion_value:
            if isinstance(motion_value, dict):
                motion_type = motion_value.get("type", "")
                if motion_type and motion_type != "static":
                    result["camera_motion_type"] = motion_type
                    result["camera_motion_direction"] = motion_value.get("direction")
                    result["camera_motion_speed"] = motion_value.get("speed", "slow")
                    result["use_svd"] = True  # 有镜头运动，必须使用SVD
                    # 根据镜头运动类型调整运动参数，确保镜头移动明显
                    if motion_type in ["pan", "tilt", "dolly", "orbit"]:
                        result["motion_intensity"] = "moderate"
                        result["motion_bucket_id_override"] = 1.8  # 降低到1.8，减少闪动，保持移动明显
                        result["noise_aug_strength_override"] = 0.0003  # 保持0.0003，减少闪动
                        print(f"  ℹ 检测到镜头运动（{motion_type}），使用SVD并设置运动参数（motion_bucket_id=1.8，减少闪动）")
                    elif motion_type == "zoom":
                        result["motion_intensity"] = "moderate"
                        result["motion_bucket_id_override"] = 1.7  # 缩放运动可以更低，减少闪动
                        result["noise_aug_strength_override"] = 0.0003
                        print(f"  ℹ 检测到镜头缩放（{motion_type}），使用SVD并设置运动参数（motion_bucket_id=1.7，减少闪动）")
        
        # 如果motion字段没有指定，从camera字段推断
        if result["camera_motion_type"] == "static":
            for motion_type, keywords in self.camera_motion_keywords.items():
                if any(keyword in camera for keyword in keywords):
                    result["camera_motion_type"] = motion_type
                    if motion_type != "static":
                        result["use_svd"] = True  # 有镜头运动，必须使用SVD
                        result["motion_intensity"] = "moderate"
                        result["motion_bucket_id_override"] = 2.0
                        result["noise_aug_strength_override"] = 0.0003
                        print(f"  ℹ 从camera字段推断镜头运动（{motion_type}），使用SVD并设置运动参数")
                    break
        
        # 从description推断镜头运动（如果还没有检测到）
        if result["camera_motion_type"] == "static":
            if "淡入" in description or "fade in" in all_text:
                result["camera_motion_type"] = "zoom"
                result["camera_motion_direction"] = "in"
                result["use_svd"] = True
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 1.8
                result["noise_aug_strength_override"] = 0.0003
            elif "淡出" in description or "fade out" in all_text:
                result["camera_motion_type"] = "zoom"
                result["camera_motion_direction"] = "out"
                result["use_svd"] = True
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 1.8
                result["noise_aug_strength_override"] = 0.0003
            elif "推进" in description or "push" in all_text or "推近" in description:
                result["camera_motion_type"] = "zoom"
                result["camera_motion_direction"] = "in"
                result["use_svd"] = True
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 1.8
                result["noise_aug_strength_override"] = 0.0003
            elif "拉远" in description or "pull" in all_text or "拉镜" in description:
                result["camera_motion_type"] = "zoom"
                result["camera_motion_direction"] = "out"
                result["use_svd"] = True
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 1.8
                result["noise_aug_strength_override"] = 0.0003
            elif "横移" in description or "pan" in all_text or "平移" in description:
                result["camera_motion_type"] = "pan"
                result["camera_motion_direction"] = "left_to_right"
                result["use_svd"] = True
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 2.0
                result["noise_aug_strength_override"] = 0.0003
        
        # ========== 3. 检测运动强度 ==========
        if result["has_object_motion"]:
            # 根据物体运动类型确定强度
            if result["object_motion_type"] in ["unfurling", "opening", "spreading"]:
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 1.8  # 降低到1.8，减少闪动，保持运动明显
                result["noise_aug_strength_override"] = 0.0003  # 降低到0.0003，减少闪动，保持流畅
                print(f"  ℹ 物体展开运动，设置运动参数（motion_bucket_id=1.8, noise_aug_strength=0.0003，减少闪动）")
            elif result["object_motion_type"] in ["shimmering", "particles", "floating"]:
                result["motion_intensity"] = "gentle"
                result["motion_bucket_id_override"] = 1.5  # 轻微运动
                result["noise_aug_strength_override"] = 0.00025  # SVD-XT
            elif result["object_motion_type"] in ["rotating", "rising", "falling"]:
                result["motion_intensity"] = "moderate"
                result["motion_bucket_id_override"] = 1.8  # 降低到1.8，减少闪动
                result["noise_aug_strength_override"] = 0.0003  # 降低到0.0003，减少闪动，保持流畅
        else:
            # 检查人物动作（根据动作类型调整参数，使动作更自然）
            if any(keyword in all_text for keyword in self.dynamic_keywords):
                # 明显动作（如攻击、奔跑、跳跃）：使用较高运动参数，确保动作明显
                result["motion_intensity"] = "dynamic"
                result["use_svd"] = True
                result["motion_bucket_id_override"] = 2.5  # 较高运动（SVD-XT最大2，但可以接近）
                result["noise_aug_strength_override"] = 0.0004  # 稍高噪声，增加动作自然度
                print(f"  ℹ 检测到明显人物动作，使用较高运动参数（motion_bucket_id=2.5）")
            elif any(keyword in all_text for keyword in self.moderate_action_keywords):
                # 中等动作（如转头、抬手、吸气）：使用中等运动参数，确保动作自然流畅，减少闪动
                result["motion_intensity"] = "moderate"
                result["use_svd"] = True
                result["motion_bucket_id_override"] = 1.8  # 降低到1.8，减少闪动，保持动作明显
                result["noise_aug_strength_override"] = 0.0003  # 降低到0.0003，减少闪动，保持流畅
                print(f"  ℹ 检测到中等人物动作，使用中等运动参数（motion_bucket_id=1.8, noise_aug_strength=0.0003，减少闪动）")
            elif any(keyword in all_text for keyword in self.static_keywords):
                # 完全静态：使用静态图像动画
                result["motion_intensity"] = "static"
                result["use_svd"] = False
                print(f"  ℹ 检测到完全静态场景，使用静态图像动画")
            else:
                # 默认情况（可能有轻微动作，如表情变化、微动）：使用轻微运动参数，但确保有自然微动
                result["motion_intensity"] = "gentle"
                result["use_svd"] = True
                result["motion_bucket_id_override"] = 1.8  # 轻微运动，但比完全静态稍高，确保有自然微动
                result["noise_aug_strength_override"] = 0.0003  # 稍高噪声，增加自然度和流畅度
                print(f"  ℹ 检测到轻微动作或默认场景，使用轻微运动参数（motion_bucket_id=1.8, noise_aug_strength=0.0003）")
        
        # ========== 4. 生成推荐的motion字段 ==========
        if result["camera_motion_type"] != "static":
            result["recommended_motion"] = {
                "type": result["camera_motion_type"],
                "direction": result["camera_motion_direction"] or self._infer_direction(result["camera_motion_type"], description, all_text),
                "speed": result["camera_motion_speed"]
            }
        elif result["has_object_motion"]:
            # 有物体运动但镜头静止，建议添加轻微镜头运动
            result["recommended_motion"] = {
                "type": "pan",
                "direction": "left_to_right",
                "speed": "slow"
            }
        
        return result
    
    def _infer_direction(self, motion_type: str, description: str, all_text: str) -> Optional[str]:
        """推断镜头运动方向"""
        if motion_type == "pan":
            if "左" in description or "left" in all_text:
                return "right_to_left"
            elif "右" in description or "right" in all_text:
                return "left_to_right"
            else:
                return "left_to_right"  # 默认
        elif motion_type == "zoom":
            if "in" in all_text or "推" in description or "近" in description:
                return "in"
            elif "out" in all_text or "拉" in description or "远" in description:
                return "out"
        elif motion_type == "tilt":
            if "上" in description or "up" in all_text or "仰" in description:
                return "up"
            elif "下" in description or "down" in all_text or "俯" in description:
                return "down"
        return None

