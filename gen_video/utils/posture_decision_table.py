"""
动作 → 姿态 → 镜头决策表（工程级映射）

将导演语义、动作类型、姿态类型映射到镜头推荐和参数配置。
"""

from typing import Dict, Optional, List, Tuple
from enum import Enum


class PostureType(Enum):
    """姿态类型"""
    LYING = "lying"
    SITTING = "sitting"
    KNEELING = "kneeling"
    CROUCHING = "crouching"
    STANDING = "standing"


class ShotType(Enum):
    """镜头类型"""
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    FULL = "full"
    AMERICAN = "american"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE = "close"
    EXTREME_CLOSE = "extreme_close"


# 动作 → 姿态 → 镜头决策表
POSTURE_SHOT_DECISION_TABLE = {
    PostureType.LYING: {
        "recommended_shots": [ShotType.FULL, ShotType.WIDE, ShotType.MEDIUM],
        "needs_ground": True,
        "needs_full_body": True,  # 但可以覆盖（中近景躺着也常见）
        "reference_strength_reduction": 20,  # 降低参考强度，让 prompt 控制姿态
        "priority": "high"  # 高优先级，确保姿态正确
    },
    PostureType.SITTING: {
        "recommended_shots": [ShotType.FULL, ShotType.MEDIUM, ShotType.MEDIUM_CLOSE],
        "needs_ground": True,
        "needs_full_body": False,  # 坐姿可以中近景
        "reference_strength_reduction": 10,
        "priority": "medium"
    },
    PostureType.KNEELING: {
        "recommended_shots": [ShotType.FULL, ShotType.MEDIUM],
        "needs_ground": True,
        "needs_full_body": True,
        "reference_strength_reduction": 15,
        "priority": "high"
    },
    PostureType.CROUCHING: {
        "recommended_shots": [ShotType.MEDIUM, ShotType.MEDIUM_CLOSE],
        "needs_ground": True,
        "needs_full_body": False,
        "reference_strength_reduction": 10,
        "priority": "medium"
    },
    PostureType.STANDING: {
        "recommended_shots": [ShotType.FULL, ShotType.MEDIUM, ShotType.MEDIUM_CLOSE],
        "needs_ground": False,
        "needs_full_body": False,
        "reference_strength_reduction": 0,
        "priority": "low"  # 默认姿态，不需要特殊处理
    }
}


def get_posture_decision(
    posture_type: Optional[str],
    current_shot: Optional[str] = None
) -> Dict[str, any]:
    """
    根据姿态类型获取决策配置
    
    Args:
        posture_type: 姿态类型（"lying", "sitting", etc.）
        current_shot: 当前镜头类型（可选）
    
    Returns:
        决策配置字典
    """
    if not posture_type:
        return {
            "recommended_shots": [ShotType.MEDIUM],
            "needs_ground": False,
            "needs_full_body": False,
            "reference_strength_reduction": 0,
            "priority": "low"
        }
    
    try:
        posture_enum = PostureType(posture_type.lower())
        decision = POSTURE_SHOT_DECISION_TABLE.get(posture_enum)
        if decision:
            # 转换为字符串列表（便于使用）
            decision_copy = decision.copy()
            decision_copy["recommended_shots"] = [s.value for s in decision["recommended_shots"]]
            return decision_copy
    except ValueError:
        pass
    
    # 默认返回
    return {
        "recommended_shots": [ShotType.MEDIUM.value],
        "needs_ground": False,
        "needs_full_body": False,
        "reference_strength_reduction": 0,
        "priority": "low"
    }


def recommend_shot_for_posture(
    posture_type: Optional[str],
    current_shot: Optional[str] = None,
    needs_ground: bool = False,
    needs_environment: bool = False
) -> str:
    """
    根据姿态推荐镜头类型
    
    Args:
        posture_type: 姿态类型
        current_shot: 当前镜头类型
        needs_ground: 是否需要显示地面
        needs_environment: 是否需要显示环境
    
    Returns:
        推荐的镜头类型（字符串）
    """
    decision = get_posture_decision(posture_type, current_shot)
    recommended_shots = decision["recommended_shots"]
    
    # 如果需要环境和地面，优先选择 wide/full
    if needs_environment and needs_ground:
        for shot in [ShotType.WIDE.value, ShotType.FULL.value]:
            if shot in recommended_shots:
                return shot
    
    # 如果需要地面，优先选择 full
    if needs_ground:
        if ShotType.FULL.value in recommended_shots:
            return ShotType.FULL.value
        if ShotType.WIDE.value in recommended_shots:
            return ShotType.WIDE.value
    
    # 如果当前镜头在推荐列表中，保持当前镜头
    if current_shot and current_shot in recommended_shots:
        return current_shot
    
    # 返回第一个推荐镜头
    return recommended_shots[0] if recommended_shots else ShotType.MEDIUM.value

