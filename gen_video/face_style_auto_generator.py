#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_style_auto 自动生成器
根据场景的 mood / lighting / action 自动生成 face_style_auto 字段
"""

from typing import Dict, Any, List, Optional
from collections import Counter


# 1) 基础映射表（根据 doc.md 规范）

EXPRESSION_MAP = {
    "neutral": "neutral",
    "focused": "focused",
    "serious": "serious",
    "calm": "calm",
    "alert": "alert",
    "determined": "determined",
    "happy_soft": "happy_soft",
    "smirk": "smirk",
    "angry_low": "angry_low",
    "tired_soft": "tired_soft"
}

LIGHTING_MAP = {
    "bright_normal": "bright_normal",
    "soft": "soft",
    "dramatic": "dramatic",
    "rim_light": "rim_light",
    "dark_lowkey": "dark_lowkey",
    "warm_indoor": "warm_indoor",
    "cool_moonlight": "cool_moonlight",
    "magic_glow": "magic_glow",
    "subterranean_dark": "subterranean_dark"
}

DETAIL_MAP = {
    "natural": "natural",
    "soft_concentrated": "soft_concentrated",
    "detailed": "detailed",
    "subtle": "subtle",
    "cinematic": "cinematic",
    "sharp_dynamic": "sharp_dynamic"
}

# 2) 动作关键词到细节偏好（action -> detail）

ACTION_DETAIL_OVERRIDES = {
    "walking": "natural",
    "walking_forward": "natural",
    "detect_spiritual_fluctuation": "soft_concentrated",
    "fight": "sharp_dynamic",
    "fight_ready": "sharp_dynamic",
    "slash": "sharp_dynamic",
    "idle": "natural",
    "think": "soft_concentrated",
    "injured": "tired_soft",
    "meditate": "soft_concentrated",
    "cast_spell": "sharp_dynamic",
    "defend": "determined",
}

# 3) mood 到 expression 的映射

MOOD_EXPRESSION_MAP = {
    "serious": "serious",
    "alert": "alert",
    "calm": "calm",
    "tense": "focused",
    "focused": "focused",
    "determined": "determined",
    "happy": "happy_soft",
    "angry": "angry_low",
    "tired": "tired_soft",
    "confident": "smirk",
}

# 4) 默认值

DEFAULT_STYLE = {
    "expression": "neutral",
    "lighting": "bright_normal",
    "detail": "natural",
    "strength": 0.85  # 用于 downstream 强度参数（0..1）
}


def pick_expression(mood: Optional[str], action: Optional[str]) -> str:
    """根据 mood 和 action 选择表情"""
    # 优先用 mood
    if mood:
        m = mood.lower()
        # 直接匹配
        if m in MOOD_EXPRESSION_MAP:
            return MOOD_EXPRESSION_MAP[m]
        # 近似匹配
        if "focus" in m:
            return "focused"
        if "angry" in m or "rage" in m:
            return "angry_low"
        if "serious" in m:
            return "serious"
        if "calm" in m:
            return "calm"
        if "alert" in m:
            return "alert"
        if "determined" in m:
            return "determined"
    
    # 根据 action 推断
    if action:
        a = action.lower()
        if "fight" in a or "slash" in a or "attack" in a or "battle" in a:
            return "determined"
        if "detect" in a or "search" in a or "observe" in a:
            return "focused"
        if "think" in a or "meditate" in a:
            return "focused"
        if "injured" in a or "hurt" in a:
            return "tired_soft"
        if "smile" in a or "laugh" in a:
            return "happy_soft"
    
    return DEFAULT_STYLE["expression"]


def pick_lighting(lighting: Optional[str], visual_lighting_hint: Optional[str] = None) -> str:
    """根据 lighting 字段和 visual 提示选择光照"""
    # lighting 字段优先
    if lighting:
        l = lighting.lower()
        # 直接匹配
        if l in LIGHTING_MAP:
            return LIGHTING_MAP[l]
        # 近似判断
        if "day" in l or "sun" in l or "bright" in l:
            return "bright_normal"
        if "night" in l or "moon" in l:
            return "cool_moonlight"
        if "magic" in l or "glow" in l or "spiritual" in l:
            return "magic_glow"
        if "dark" in l or "shadow" in l:
            return "dark_lowkey"
        if "dramatic" in l or "tense" in l:
            return "dramatic"
        if "warm" in l or "indoor" in l:
            return "warm_indoor"
        if "soft" in l:
            return "soft"
    
    # 从 visual 字段获取提示
    if visual_lighting_hint:
        vh = visual_lighting_hint.lower()
        if "rim" in vh or "edge" in vh:
            return "rim_light"
        if "dramatic" in vh:
            return "dramatic"
        if "soft" in vh:
            return "soft"
    
    return DEFAULT_STYLE["lighting"]


def pick_detail(action: Optional[str], desired_detail: Optional[str] = None) -> str:
    """根据 action 和 desired_detail 选择细节级别"""
    # 优先 desired_detail（如果有）
    if desired_detail and desired_detail in DETAIL_MAP:
        return desired_detail
    
    # 根据 action 决定
    if action:
        a = action.lower()
        # 精确匹配
        for k, v in ACTION_DETAIL_OVERRIDES.items():
            if k in a:
                return v
        # 基于关键词的兜底
        if "fight" in a or "battle" in a or "attack" in a or "slash" in a:
            return "sharp_dynamic"
        if "talk" in a or "sit" in a or "stand" in a:
            return "soft_concentrated"
        if "detect" in a or "think" in a or "meditate" in a:
            return "soft_concentrated"
        if "walk" in a or "move" in a:
            return "natural"
    
    return DEFAULT_STYLE["detail"]


def auto_face_style_for_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    为单个场景自动生成 face_style_auto
    
    Args:
        scene: 场景字典，包含 mood, lighting, action, visual, camera 等字段
    
    Returns:
        dict: {expression, lighting, detail, strength}
    """
    mood = scene.get("mood")
    lighting = scene.get("lighting")
    action = scene.get("action")
    visual = scene.get("visual", {})
    
    # 从 visual 字段获取光照提示
    visual_hint = None
    if isinstance(visual, dict):
        visual_hint = visual.get("environment") or visual.get("composition")
    
    desired_detail = scene.get("desired_detail")  # 可选手动覆盖
    
    expr = pick_expression(mood, action)
    light = pick_lighting(lighting, visual_hint)
    detail = pick_detail(action, desired_detail)
    
    # 根据镜头类型微调强度（特写提高强度，远景降低）
    strength = DEFAULT_STYLE["strength"]
    camera = scene.get("camera", "")
    if isinstance(camera, str):
        camera_lower = camera.lower()
        if "close" in camera_lower or "closeup" in camera_lower or "close_shot" in camera_lower or "portrait" in camera_lower:
            strength = min(1.0, strength + 0.1)
        elif "wide" in camera_lower or "establish" in camera_lower or "long_shot" in camera_lower:
            strength = max(0.6, strength - 0.05)
    
    return {
        "expression": expr,
        "lighting": light,
        "detail": detail,
        "strength": round(strength, 2)
    }


def smooth_face_styles(styles: List[Dict[str, Any]], window: int = 2) -> List[Dict[str, Any]]:
    """
    序列平滑（避免两镜头间风格突变）
    
    简单滑动窗口平滑：对 strength 做平均，对 categorical 字段取窗口内最频繁值
    """
    out = []
    n = len(styles)
    
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        window_styles = styles[lo:hi]
        
        # average strength
        avg_strength = sum(s.get("strength", DEFAULT_STYLE["strength"]) for s in window_styles) / len(window_styles)
        
        # most common categorical
        expr = Counter(s["expression"] for s in window_styles).most_common(1)[0][0]
        light = Counter(s["lighting"] for s in window_styles).most_common(1)[0][0]
        detail = Counter(s["detail"] for s in window_styles).most_common(1)[0][0]
        
        out.append({
            "expression": expr,
            "lighting": light,
            "detail": detail,
            "strength": round(avg_strength, 2)
        })
    
    return out


def generate_face_styles_for_episode(
    scenes: List[Dict[str, Any]], 
    smooth: bool = True,
    overwrite_existing: bool = False
) -> List[Dict[str, Any]]:
    """
    为整集场景批量生成 face_style_auto
    
    Args:
        scenes: 场景列表
        smooth: 是否进行序列平滑
        overwrite_existing: 是否覆盖已存在的 face_style_auto
    
    Returns:
        生成的 face_style_auto 列表
    """
    styles = []
    
    for scene in scenes:
        # 如果已存在且不覆盖，保留原有
        if not overwrite_existing and scene.get("face_style_auto"):
            styles.append(scene["face_style_auto"])
        else:
            style = auto_face_style_for_scene(scene)
            styles.append(style)
            # 自动更新到场景中
            scene["face_style_auto"] = style
    
    if smooth and len(styles) > 1:
        styles = smooth_face_styles(styles, window=1)
        # 更新回场景
        for scene, style in zip(scenes, styles):
            scene["face_style_auto"] = style
    
    return styles


def to_instantid_params(face_style: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 face_style_auto 转换为 InstantID 参数
    
    Returns:
        包含 InstantID 相关参数的字典
    """
    expression = face_style.get("expression", "neutral")
    lighting = face_style.get("lighting", "bright_normal")
    detail = face_style.get("detail", "natural")
    strength = face_style.get("strength", 0.85)
    
    # 根据 detail 调整面部权重
    detail_scale_map = {
        "detailed": 1.15,
        "sharp_dynamic": 1.1,
        "cinematic": 1.05,
        "natural": 1.0,
        "soft_concentrated": 0.95,
        "subtle": 0.9,
    }
    detail_scale = detail_scale_map.get(detail, 1.0)
    
    # 根据 expression 调整权重（需要清晰表情时提高）
    expression_scale_map = {
        "focused": 1.1,
        "alert": 1.1,
        "determined": 1.05,
        "serious": 1.0,
        "neutral": 1.0,
        "calm": 0.95,
        "happy_soft": 0.95,
        "smirk": 0.95,
        "tired_soft": 0.9,
        "angry_low": 1.0,
    }
    expression_scale = expression_scale_map.get(expression, 1.0)
    
    # 综合强度
    final_strength = strength * detail_scale * expression_scale
    
    # 提高最小 multiplier，避免过度降低人脸相似度
    # 最小 0.7 而不是 0.3，确保人脸相似度不会太低
    return {
        "ip_adapter_scale_multiplier": min(1.0, max(0.7, final_strength)),  # 从 0.3 提高到 0.7
        "face_kps_scale_multiplier": min(1.3, max(0.7, final_strength)),
        "expression": expression,
        "lighting": lighting,
        "detail": detail,
        "strength": strength,
    }


# -----------------------
# 示例运行

if __name__ == "__main__":
    example_scenes = [
        {"id": 1, "mood": "serious", "lighting": "day", "action": "walking_forward", "camera": "wide_shot_low_angle"},
        {"id": 2, "mood": "alert", "lighting": "day", "action": "detect_spiritual_fluctuation", "camera": "medium_shot_front"}
    ]
    
    styles = generate_face_styles_for_episode(example_scenes, smooth=True)
    
    for idx, s in enumerate(styles):
        print(f"scene {example_scenes[idx]['id']} -> face_style_auto: {s}")
        print(f"  -> instantid_params: {to_instantid_params(s)}")

