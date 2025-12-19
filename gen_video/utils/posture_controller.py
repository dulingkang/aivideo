"""
Posture Controller - 姿态显式控制模块

解决扩散模型默认"站立"先验问题，通过显式姿态指令确保生成正确的姿态。
"""

from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


# 姿态模板库（英文，必须，因为 Flux 对英文理解更好）
POSTURE_TEMPLATES = {
    "lying": {
        "positive": "lying on the ground, body fully reclined, back touching the ground, head resting on the surface, arms relaxed by the side, legs extended flat, horizontal position, prone position",
        "negative": "standing pose, upright posture, vertical position, person standing, person upright, walking, running, sitting pose"
    },
    "sitting": {
        "positive": "sitting on the ground, legs bent, body in seated position, resting on the surface",
        "negative": "standing pose, upright posture, vertical position, person standing, lying down, prone position"
    },
    "kneeling": {
        "positive": "kneeling on the ground, on knees, body lowered, kneeling position",
        "negative": "standing pose, upright posture, vertical position, person standing"
    },
    "crouching": {
        "positive": "crouching position, body lowered, squatting, crouched down",
        "negative": "standing pose, upright posture, vertical position, person standing"
    },
    "standing": {
        "positive": "standing, upright posture, vertical position",
        "negative": "lying down, sitting, prone position, horizontal position"
    }
}

# 中文动作到姿态的映射
ACTION_TO_POSTURE = {
    "躺": "lying",
    "卧": "lying",
    "伏": "lying",
    "倒地": "lying",
    "lying": "lying",
    "lie": "lying",
    "prone": "lying",
    "坐": "sitting",
    "sitting": "sitting",
    "sit": "sitting",
    "跪": "kneeling",
    "kneeling": "kneeling",
    "kneel": "kneeling",
    "蹲": "crouching",
    "crouching": "crouching",
    "crouch": "crouching",
    "站": "standing",
    "standing": "standing",
    "stand": "standing"
}

# 动作上下文模式（组合判断）
ACTION_CONTEXT_PATTERNS = {
    "lying": [
        # 模式：(保持不动/静止) + (脚下/地面) = 躺
        (["保持不动", "静止", "不动", "motionless", "still"], ["脚下", "地面", "floor", "ground", "沙地", "sand"]),
        # 模式：(体会/感受) + (脚下/地面) = 躺
        (["体会", "感受", "feel"], ["脚下", "地面", "floor", "ground"]),
        # 模式：(躺在) + (沙漠/地面)
        (["躺在", "lying in"], ["沙漠", "desert", "地面", "ground"]),
    ],
    "sitting": [
        # 模式：(坐在) + (石头/地面)
        (["坐在", "sitting on"], ["石头", "stone", "地面", "ground"]),
    ]
}


class PostureController:
    """
    姿态控制器 - 显式注入姿态指令，避免默认站立先验
    
    架构设计：
    - 规则引擎层：导演语义判断（"保持不动"+"脚下" → lying）
    - 模板库：姿态描述模板（英文，用于 Flux）
    - 决策表：动作 → 姿态 → 镜头的映射关系
    """
    
    def __init__(self):
        self.templates = POSTURE_TEMPLATES
        self.action_to_posture = ACTION_TO_POSTURE
        self.context_patterns = ACTION_CONTEXT_PATTERNS
    
    def analyze_director_semantics(self, prompt: str) -> Dict[str, Any]:
        """
        导演语义分析（规则引擎层）
        
        判断导演意图，而不是语言学必然。
        例如："保持不动"+"脚下" → 可能是 lying，但不是必然。
        
        Args:
            prompt: 原始提示词
        
        Returns:
            {
                "posture_hint": "lying_candidate" | "sitting_candidate" | None,
                "needs_ground": bool,
                "needs_full_body": bool,
                "confidence": float
            }
        """
        prompt_lower = prompt.lower()
        result = {
            "posture_hint": None,
            "needs_ground": False,
            "needs_full_body": False,
            "confidence": 0.0
        }
        
        # 检测地面相关关键词
        ground_keywords = ["脚下", "地面", "floor", "ground", "沙地", "sand", "地上"]
        has_ground = any(kw in prompt_lower for kw in ground_keywords)
        result["needs_ground"] = has_ground
        
        # 导演语义规则：组合判断
        # 规则1: "保持不动" + "脚下" → lying 候选（但不是必然）
        motionless_keywords = ["保持不动", "静止", "不动", "motionless", "still"]
        has_motionless = any(kw in prompt_lower for kw in motionless_keywords)
        
        if has_motionless and has_ground:
            # 排除明确不是 lying 的情况
            standing_keywords = ["站", "站立", "standing", "直立", "upright"]
            if not any(kw in prompt_lower for kw in standing_keywords):
                result["posture_hint"] = "lying_candidate"
                result["confidence"] = 0.7  # 中等置信度
                result["needs_full_body"] = True
        
        # 规则2: "体会" + "脚下" → lying 候选
        feel_keywords = ["体会", "感受", "feel"]
        if any(kw in prompt_lower for kw in feel_keywords) and has_ground:
            if result["posture_hint"] is None:
                result["posture_hint"] = "lying_candidate"
                result["confidence"] = 0.6
                result["needs_full_body"] = True
        
        # 规则3: 明确动作词（优先级最高）
        if "躺" in prompt_lower or "lying" in prompt_lower:
            result["posture_hint"] = "lying"
            result["confidence"] = 0.95
            result["needs_full_body"] = True
        elif "坐" in prompt_lower or "sitting" in prompt_lower:
            result["posture_hint"] = "sitting"
            result["confidence"] = 0.95
            result["needs_full_body"] = True
        
        return result
    
    def detect_posture(self, prompt: str, action_type: Optional[str] = None) -> Optional[str]:
        """
        检测姿态类型
        
        Args:
            prompt: 原始提示词
            action_type: 场景分析器检测到的动作类型（可选）
        
        Returns:
            姿态类型: "lying", "sitting", "kneeling", "crouching", "standing", None
        """
        prompt_lower = prompt.lower()
        
        # 1. 如果提供了 action_type，直接映射
        if action_type:
            posture = self.action_to_posture.get(action_type.lower())
            if posture:
                logger.info(f"  检测到姿态（从 action_type）: {posture}")
                return posture
        
        # 2. 直接关键词匹配
        for action_key, posture in self.action_to_posture.items():
            if action_key in prompt_lower:
                logger.info(f"  检测到姿态（关键词匹配）: {posture} (关键词: {action_key})")
                return posture
        
        # 3. 上下文模式匹配（组合判断）
        for posture, patterns in self.context_patterns.items():
            for condition1_keywords, condition2_keywords in patterns:
                has_condition1 = any(kw in prompt_lower for kw in condition1_keywords)
                has_condition2 = any(kw in prompt_lower for kw in condition2_keywords)
                if has_condition1 and has_condition2:
                    logger.info(f"  检测到姿态（上下文模式）: {posture} (条件1: {condition1_keywords}, 条件2: {condition2_keywords})")
                    return posture
        
        return None
    
    def get_posture_prompt(self, posture: str, use_chinese: bool = False) -> Dict[str, str]:
        """
        获取姿态相关的 prompt 片段
        
        Args:
            posture: 姿态类型 ("lying", "sitting", etc.)
            use_chinese: 是否使用中文（Flux 支持中文，但英文更精确）
        
        Returns:
            {"positive": "...", "negative": "..."}
        """
        if posture not in self.templates:
            return {"positive": "", "negative": ""}
        
        template = self.templates[posture]
        
        if use_chinese:
            # 中文版本（备用）
            chinese_templates = {
                "lying": {
                    "positive": "躺在地上，身体完全贴地，背部接触地面，头部贴地，双臂平放，双腿伸直，水平位置，俯卧姿势",
                    "negative": "站立姿势，直立姿态，垂直位置，人物站立，人物直立，行走，跑步，坐姿"
                },
                "sitting": {
                    "positive": "坐在地上，双腿弯曲，身体处于坐姿，贴地休息",
                    "negative": "站立姿势，直立姿态，垂直位置，人物站立，躺下，俯卧姿势"
                }
            }
            if posture in chinese_templates:
                return chinese_templates[posture]
        
        return template
    
    def inject_posture(self, prompt: str, action_type: Optional[str] = None, use_chinese: bool = False) -> Dict[str, str]:
        """
        注入姿态指令到 prompt
        
        Args:
            prompt: 原始提示词
            action_type: 场景分析器检测到的动作类型（可选）
            use_chinese: 是否使用中文
        
        Returns:
            {
                "enhanced_prompt": "原始prompt + 姿态positive",
                "negative_prompt": "姿态negative",
                "posture": "检测到的姿态类型"
            }
        """
        # 检测姿态
        posture = self.detect_posture(prompt, action_type)
        
        if not posture:
            return {
                "enhanced_prompt": prompt,
                "negative_prompt": "",
                "posture": None
            }
        
        # 获取姿态模板
        posture_prompt = self.get_posture_prompt(posture, use_chinese)
        
        # 构建增强 prompt：姿态指令放在最前面（最高优先级）
        enhanced_prompt = f"{posture_prompt['positive']}, {prompt}"
        
        # 负面 prompt
        negative_prompt = posture_prompt['negative']
        
        logger.info(f"  ✓ 姿态注入: {posture}")
        logger.info(f"  ✓ 姿态指令: {posture_prompt['positive'][:50]}...")
        
        return {
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "posture": posture
        }

