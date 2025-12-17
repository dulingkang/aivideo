#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Planner V3 - 智能场景路由与参数控制

核心升级:
1. 参考强度控制 (0-100, 参考可灵/即梦)
2. 多引擎路由 (PuLID/InstantID/Flux)
3. 解耦生成决策 (是否使用分离流水线)
4. 多角度参考图选择

参考架构:
- 豆包 Seedream 的参考强度控制
- 可灵 Element Library 的多参考图系统
- 即梦的 Flow Matching 技术
"""

from typing import Dict, Any, Optional, List, Tuple, Literal
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class GenerationMode(Enum):
    """生成模式"""
    STANDARD = "standard"       # 标准模式 (InstantID/PuLID 一次性生成)
    DECOUPLED = "decoupled"     # 解耦模式 (场景+身份分离)
    SCENE_ONLY = "scene_only"   # 纯场景模式 (无人物)


class IdentityEngine(Enum):
    """身份引擎"""
    PULID = "pulid"             # PuLID-FLUX (环境融合更好)
    INSTANTID = "instantid"     # InstantID (锁脸更强)
    IPADAPTER = "ipadapter"     # IP-Adapter (风格迁移)
    NONE = "none"               # 无身份约束


class SceneEngine(Enum):
    """场景引擎"""
    FLUX1 = "flux1"             # Flux.1-dev (高质量)
    FLUX2 = "flux2"             # Flux.2/schnell (速度快)
    SDXL = "sdxl"               # SDXL (稳定)
    HUNYUAN_DIT = "hunyuan_dit" # HunyuanDiT (中文)
    KOLORS = "kolors"           # Kolors (真实感)


@dataclass
class GenerationStrategy:
    """生成策略"""
    # 生成模式
    mode: GenerationMode
    
    # 引擎选择
    scene_engine: SceneEngine
    identity_engine: IdentityEngine
    
    # 参考强度 (0-100)
    reference_strength: int
    
    # 参考图选择
    primary_reference: Optional[str] = None    # 主参考图 (角度匹配)
    expression_reference: Optional[str] = None # 表情参考图
    
    # 参考模式
    reference_mode: Literal["face_only", "full_body"] = "face_only"
    
    # Prompt 权重
    environment_weight: float = 1.0  # 环境描述权重乘数
    character_weight: float = 1.0    # 人物描述权重乘数
    
    # 是否使用解耦
    use_decoupled_pipeline: bool = False
    
    # 质量验证
    verify_face_similarity: bool = True
    similarity_threshold: float = 0.7


class ExecutionPlannerV3:
    """
    Execution Planner V3 - 智能场景路由
    
    核心功能:
    1. 分析场景特征 (镜头类型、相机角度、人物状态)
    2. 计算最优参考强度 (0-100)
    3. 选择最适合的引擎组合
    4. 决定是否使用解耦生成
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 Execution Planner V3
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 参考强度映射表 (基于镜头类型)
        # 注意：中景和全身场景提高参考强度，以增强服饰一致性
        # 基础参考强度映射（根据实测结果调整）
        # 说明：
        # - 远景：统一收敛到 50%，避免过低导致“完全不像”且融合感差
        # - 中景：统一收敛到 60%，作为当前的最佳平衡点
        # - 近景/特写：保持较高强度，依赖参考图锁脸
        self.shot_strength_map = {
            "extreme_wide": 50,   # 超远景: 环境优先，但保持 50% 参考强度
            "wide": 50,           # 远景: 根据测试结果，从 30 调整为 50%
            "full": 50,           # 全身: 与远景保持一致，50%
            "american": 55,       # 7/8身: 介于远景与中景之间
            "medium": 60,         # 中景: 根据测试结果，推荐 60%（效果最好）
            "medium_close": 60,   # 中近景: 与中景保持一致，60%
            "close": 65,          # 特写: 从 60% 提高到 65%，提升人脸相似度（当前 0.09 太低）
            "extreme_close": 65,  # 超特写: 从 60% 提高到 65%，与近景保持一致
        }
        
        # 相机角度调整
        self.angle_adjustments = {
            "top_down": -20,      # 俯拍: 降低参考强度
            "bird_eye": -25,      # 鸟瞰
            "low": +10,           # 仰拍: 增加参考强度
            "dutch": 0,           # 荷兰角: 不调整
            "eye_level": 0,       # 平视
        }
        
        # 表情需求调整
        self.emotion_adjustments = {
            "neutral": 0,
            "happy": +5,
            "sad": +5,
            "angry": +10,
            "surprised": +10,
            "pain": +15,           # 痛苦表情需要更强的参考
            "thinking": +5,
        }
        
        logger.info("Execution Planner V3 初始化完成")
    
    def analyze_scene(
        self,
        scene: Dict[str, Any],
        character_profiles: Dict[str, Any] = None
    ) -> GenerationStrategy:
        """
        分析场景并生成策略
        
        Args:
            scene: 场景 JSON (v2 格式)
            character_profiles: 角色档案 (可选)
            
        Returns:
            生成策略
        """
        # 提取场景信息
        camera = scene.get("camera", {})
        character = scene.get("character", {})
        environment = scene.get("environment", {})
        
        shot_type = camera.get("shot", "medium")
        camera_angle = camera.get("angle", "eye_level")
        character_present = character.get("present", True)
        character_pose = character.get("pose", "standing")
        character_emotion = character.get("emotion", "neutral")
        
        logger.info(f"分析场景: shot={shot_type}, angle={camera_angle}, emotion={character_emotion}")
        
        # 1. 计算参考强度
        reference_strength = self._calculate_reference_strength(
            shot_type=shot_type,
            camera_angle=camera_angle,
            emotion=character_emotion,
            character_present=character_present
        )
        
        # 2. 选择生成模式
        mode = self._select_generation_mode(
            shot_type=shot_type,
            character_present=character_present,
            reference_strength=reference_strength
        )
        
        # 3. 选择引擎
        scene_engine, identity_engine = self._select_engines(
            shot_type=shot_type,
            mode=mode,
            scene=scene
        )
        
        # 4. 选择参考图
        primary_ref, expression_ref = self._select_references(
            camera_angle=camera_angle,
            emotion=character_emotion,
            character_id=character.get("id"),
            character_profiles=character_profiles
        )
        
        # 5. 计算 Prompt 权重
        env_weight, char_weight = self._calculate_prompt_weights(
            shot_type=shot_type,
            reference_strength=reference_strength
        )
        
        # 6. 是否使用解耦
        use_decoupled = self._should_use_decoupled(
            shot_type=shot_type,
            reference_strength=reference_strength,
            mode=mode
        )
        
        # 构建策略
        strategy = GenerationStrategy(
            mode=mode,
            scene_engine=scene_engine,
            identity_engine=identity_engine,
            reference_strength=reference_strength,
            primary_reference=str(primary_ref) if primary_ref else None,
            expression_reference=str(expression_ref) if expression_ref else None,
            reference_mode="face_only" if shot_type in ["close", "extreme_close"] else "full_body",
            environment_weight=env_weight,
            character_weight=char_weight,
            use_decoupled_pipeline=use_decoupled,
            verify_face_similarity=True,
            similarity_threshold=0.7
        )
        
        self._log_strategy(strategy)
        
        return strategy
    
    def _calculate_reference_strength(
        self,
        shot_type: str,
        camera_angle: str,
        emotion: str,
        character_present: bool
    ) -> int:
        """
        计算参考强度 (0-100)
        
        参考可灵/即梦的策略:
        - 远景: 环境优先，弱参考 (20-40)
        - 中景: 平衡 (50-70)
        - 特写: 人脸优先，强参考 (70-90)
        """
        if not character_present:
            return 0  # 无人物场景
        
        # 基础强度 (基于镜头类型)
        base_strength = self.shot_strength_map.get(shot_type, 60)
        
        # 角度调整
        angle_adj = self.angle_adjustments.get(camera_angle, 0)
        
        # 表情调整
        emotion_adj = self.emotion_adjustments.get(emotion, 0)
        
        # 计算最终强度
        final_strength = base_strength + angle_adj + emotion_adj
        
        # 限制范围
        final_strength = max(0, min(100, final_strength))
        
        logger.debug(f"参考强度计算: base={base_strength}, angle_adj={angle_adj}, "
                    f"emotion_adj={emotion_adj}, final={final_strength}")
        
        return final_strength
    
    def _select_generation_mode(
        self,
        shot_type: str,
        character_present: bool,
        reference_strength: int
    ) -> GenerationMode:
        """选择生成模式"""
        if not character_present:
            return GenerationMode.SCENE_ONLY
        
        # 远景且参考强度低，考虑使用解耦模式
        if shot_type in ["extreme_wide", "wide"] and reference_strength < 40:
            return GenerationMode.DECOUPLED
        
        return GenerationMode.STANDARD
    
    def _select_engines(
        self,
        shot_type: str,
        mode: GenerationMode,
        scene: Dict[str, Any]
    ) -> Tuple[SceneEngine, IdentityEngine]:
        """选择引擎组合"""
        
        # 场景引擎选择
        scene_engine = SceneEngine.FLUX1  # 默认 Flux1
        
        # 身份引擎选择
        # 统一使用 PuLID，保持所有镜头类型的一致性
        if mode == GenerationMode.SCENE_ONLY:
            identity_engine = IdentityEngine.NONE
        else:
            # 所有场景统一使用 PuLID，确保处理方式一致
            identity_engine = IdentityEngine.PULID
        
        return scene_engine, identity_engine
    
    def _select_references(
        self,
        camera_angle: str,
        emotion: str,
        character_id: Optional[str],
        character_profiles: Dict[str, Any] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """选择参考图"""
        primary_ref = None
        expression_ref = None
        
        if character_profiles and character_id:
            profile = character_profiles.get(character_id)
            if profile:
                # 根据角度选择主参考图
                if hasattr(profile, 'get_reference_for_scene'):
                    primary_ref, expression_ref = profile.get_reference_for_scene(
                        camera_angle=camera_angle,
                        emotion=emotion
                    )
        
        return primary_ref, expression_ref
    
    def _calculate_prompt_weights(
        self,
        shot_type: str,
        reference_strength: int
    ) -> Tuple[float, float]:
        """
        计算 Prompt 权重
        
        当参考强度低时，环境权重应该高
        """
        # 环境权重: 参考强度越低，环境权重越高
        env_weight = 1.0 + (100 - reference_strength) / 100 * 0.5  # 1.0 - 1.5
        
        # 人物权重: 参考强度越高，人物权重越高
        char_weight = 0.8 + reference_strength / 100 * 0.4  # 0.8 - 1.2
        
        return round(env_weight, 2), round(char_weight, 2)
    
    def _should_use_decoupled(
        self,
        shot_type: str,
        reference_strength: int,
        mode: GenerationMode
    ) -> bool:
        """决定是否使用解耦生成"""
        # 近景/特写（参考强度 >= 70%）强制使用 STANDARD 模式，不使用解耦
        if reference_strength >= 70:
            return False
        
        # 解耦模式
        if mode == GenerationMode.DECOUPLED:
            return True
        
        # 远景且低参考强度
        if shot_type in ["extreme_wide", "wide", "full"] and reference_strength < 50:
            return True
        
        return False
    
    def _log_strategy(self, strategy: GenerationStrategy):
        """记录策略"""
        logger.info("=" * 50)
        logger.info("生成策略:")
        logger.info(f"  模式: {strategy.mode.value}")
        logger.info(f"  场景引擎: {strategy.scene_engine.value}")
        logger.info(f"  身份引擎: {strategy.identity_engine.value}")
        logger.info(f"  参考强度: {strategy.reference_strength}%")
        logger.info(f"  参考模式: {strategy.reference_mode}")
        logger.info(f"  环境权重: {strategy.environment_weight}x")
        logger.info(f"  人物权重: {strategy.character_weight}x")
        logger.info(f"  解耦生成: {strategy.use_decoupled_pipeline}")
        logger.info("=" * 50)
    
    # ==========================================
    # 便捷方法
    # ==========================================
    
    def get_ip_adapter_scale(
        self,
        shot_type: str,
        base_scale: float = 0.8
    ) -> float:
        """
        根据镜头类型获取 IP-Adapter 权重
        
        这是为兼容现有代码提供的便捷方法
        """
        strength = self._calculate_reference_strength(
            shot_type=shot_type,
            camera_angle="eye_level",
            emotion="neutral",
            character_present=True
        )
        
        # 将 0-100 映射到 0.3-0.95
        scale = 0.3 + (strength / 100) * 0.65
        
        return round(scale, 2)
    
    def get_reference_strength_for_scene(
        self,
        scene: Dict[str, Any]
    ) -> int:
        """
        获取场景的参考强度
        
        便捷方法，直接返回参考强度值
        """
        camera = scene.get("camera", {})
        character = scene.get("character", {})
        
        return self._calculate_reference_strength(
            shot_type=camera.get("shot", "medium"),
            camera_angle=camera.get("angle", "eye_level"),
            emotion=character.get("emotion", "neutral"),
            character_present=character.get("present", True)
        )
    
    def build_weighted_prompt(
        self,
        scene: Dict[str, Any],
        strategy: GenerationStrategy = None
    ) -> str:
        """
        构建带权重的 Prompt
        
        参考豆包/可灵的 Prompt 策略:
        - 环境描述放在前面
        - 使用加权语法 (xxx:1.3)
        - 自动添加镜头类型描述以控制构图
        """
        if strategy is None:
            strategy = self.analyze_scene(scene)
        
        env = scene.get("environment", {})
        char = scene.get("character", {})
        visual = scene.get("visual", {})
        camera = scene.get("camera", {})
        
        parts = []
        
        # 0. 镜头类型描述（最重要，放在最前面以控制构图）
        shot_type = camera.get("shot", "medium")
        shot_descriptions = {
            "extreme_wide": "extreme wide shot, person very small, distant view, environment dominates, landscape composition, vast scenery",
            "wide": "wide shot, person very small in frame, distant view, showing vast environment, landscape composition, person is tiny",
            "full": "full body shot, person medium-small in frame",
            "american": "american shot, 7/8 body, person medium size",
            "medium": "medium shot, upper body, person medium size",
            "medium_close": "medium close-up, chest and head, person large",
            "close": "close-up shot, head and shoulders, face prominent",
            "extreme_close": "extreme close-up, face only, person fills frame"
        }
        shot_desc = shot_descriptions.get(shot_type, "medium shot")
        # 远景使用更高权重（3.0）确保构图正确，其他镜头使用 2.5
        shot_weight = 3.0 if shot_type in ["extreme_wide", "wide"] else 2.5
        parts.append(f"({shot_desc}:{shot_weight})")
        
        # 1. 环境描述 (加权)
        env_desc = env.get("description", "")
        if env_desc:
            # 所有镜头类型使用相同的处理方式，保持一致性
            if strategy.environment_weight > 1.0:
                parts.append(f"({env_desc}:{strategy.environment_weight})")
            else:
                parts.append(env_desc)
        
        # 2. 光照和氛围
        lighting = env.get("lighting", "")
        atmosphere = env.get("atmosphere", "")
        # 所有镜头类型保留完整的光照和氛围描述，保持一致性
        if lighting:
            parts.append(f"({lighting}:1.2)")
        if atmosphere:
            parts.append(f"({atmosphere}:1.2)")
        
        # 3. 人物描述 (根据参考强度决定详细程度)
        if char.get("present", True):
            char_desc = char.get("description", char.get("basic_appearance", ""))
            if char_desc:
                # 所有镜头类型使用相同的处理方式，保持完整描述以确保人脸相似度
                # 参考图主要控制人脸特征，但服饰细节需要 prompt 指导
                if strategy.character_weight != 1.0:
                    parts.append(f"({char_desc}:{strategy.character_weight})")
                else:
                    parts.append(char_desc)
            else:
                # 如果既没有 description 也没有 basic_appearance，使用默认值
                parts.append("a person")
            
            # 3.1 添加表情/情绪描述（如果指定）
            emotion = char.get("emotion", "")
            if emotion and emotion != "neutral":
                emotion_map = {
                    "determined": "determined expression, focused eyes, resolute face",
                    "angry": "angry expression, fierce eyes, stern face",
                    "sad": "sad expression, melancholic eyes, sorrowful face",
                    "happy": "happy expression, smiling, joyful face",
                    "calm": "calm expression, peaceful eyes, serene face",
                    "serious": "serious expression, intense eyes, stern face",
                    "surprised": "surprised expression, wide eyes, shocked face",
                    "fearful": "fearful expression, worried eyes, anxious face"
                }
                emotion_desc = emotion_map.get(emotion.lower(), f"{emotion} expression")
                parts.append(f"({emotion_desc}:1.3)")
            
            # 3.2 添加动作/姿态描述（如果指定）
            pose = char.get("pose", "")
            action = char.get("action", "")
            if action:
                # 动作描述优先级更高
                parts.append(f"({action}:1.4)")
            elif pose and pose != "standing":
                # 如果没有动作，使用姿态
                parts.append(f"({pose}:1.3)")
        
        # 4. 视觉构图（如果用户明确指定，则使用用户的描述）
        composition = visual.get("composition", "")
        if composition:
            parts.append(composition)
        
        return ", ".join(filter(None, parts))


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    planner = ExecutionPlannerV3()
    
    # 测试场景
    test_scenes = [
        {
            "camera": {"shot": "wide", "angle": "top_down"},
            "character": {"present": True, "emotion": "neutral"},
            "environment": {"description": "misty mountain valley at dawn"}
        },
        {
            "camera": {"shot": "medium", "angle": "eye_level"},
            "character": {"present": True, "emotion": "sad"},
            "environment": {"description": "ancient temple interior"}
        },
        {
            "camera": {"shot": "close", "angle": "eye_level"},
            "character": {"present": True, "emotion": "angry"},
            "environment": {"description": "battlefield"}
        },
    ]
    
    print("\n" + "=" * 60)
    print("Execution Planner V3 测试")
    print("=" * 60)
    
    for i, scene in enumerate(test_scenes):
        print(f"\n场景 {i+1}:")
        print(f"  相机: {scene['camera']}")
        print(f"  人物: {scene['character']}")
        
        strategy = planner.analyze_scene(scene)
        
        print(f"\n  策略:")
        print(f"    参考强度: {strategy.reference_strength}%")
        print(f"    身份引擎: {strategy.identity_engine.value}")
        print(f"    解耦生成: {strategy.use_decoupled_pipeline}")
        print(f"    环境权重: {strategy.environment_weight}x")
    
    print("\n" + "=" * 60)
    print("测试完成!")
