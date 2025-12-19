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

# 导入智能场景分析器
try:
    from utils.scene_analyzer import analyze_scene, HybridSceneAnalyzer, LocalSceneAnalyzer
    SCENE_ANALYZER_AVAILABLE = True
except ImportError:
    SCENE_ANALYZER_AVAILABLE = False
import logging
logger = logging.getLogger(__name__)
if not SCENE_ANALYZER_AVAILABLE:
    logger.warning("场景分析器未找到，将使用硬编码规则")

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
        
        # ⚡ 初始化 LLM 客户端（如果配置了 LLM 模式）
        self.llm_client = None
        prompt_engine_config = self.config.get("prompt_engine", {})
        scene_analyzer_mode = prompt_engine_config.get("scene_analyzer_mode", "local")
        
        # ⚡ 调试日志：检查配置读取
        logger.info(f"  [DEBUG] prompt_engine_config keys: {list(prompt_engine_config.keys())}")
        logger.info(f"  [DEBUG] scene_analyzer_mode: {scene_analyzer_mode}")
        
        if scene_analyzer_mode in ["llm", "hybrid"]:
            try:
                llm_api_config = prompt_engine_config.get("llm_api", {})
                logger.info(f"  [DEBUG] llm_api_config keys: {list(llm_api_config.keys()) if llm_api_config else 'None'}")
                if llm_api_config.get("api_key"):
                    from utils.scene_analyzer import OpenAILLMClient
                    self.llm_client = OpenAILLMClient(
                        api_key=llm_api_config.get("api_key"),
                        model=llm_api_config.get("model", "gpt-4o-mini"),
                        base_url=llm_api_config.get("base_url")
                    )
                    logger.info(f"  ✓ LLM 场景分析器已初始化 (model: {llm_api_config.get('model', 'gpt-4o-mini')})")
                else:
                    logger.warning("  ⚠ 配置了 LLM 模式但未提供 API Key，将使用本地模式")
            except Exception as e:
                logger.warning(f"  ⚠ LLM 客户端初始化失败: {e}，将使用本地模式")
                import traceback
                logger.debug(f"  [DEBUG] LLM 初始化异常详情: {traceback.format_exc()}")
        
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
        
        # v2 shot 字段归一化：close_up/medium_close_up 等映射到 planner 可识别的 shot_type
        shot_type_raw = camera.get("shot", "medium")
        shot_type = self._normalize_shot_type(shot_type_raw)
        camera_angle = camera.get("angle", "eye_level")
        character_present = character.get("present", True)
        character_pose = character.get("pose", "standing")
        character_emotion = character.get("emotion", "neutral")
        
        logger.info(f"分析场景: shot={shot_type}, angle={camera_angle}, emotion={character_emotion}")
        
        # ⚡ 关键修复：检测"lying"动作，降低参考强度，让 prompt 有更大的控制权
        # 因为参考图可能是站立的，会强烈影响姿态，所以需要降低参考强度
        is_lying_action = False
        if character_present:
            # 检查场景描述中是否包含"lying"相关关键词
            scene_text = str(scene.get("prompt", "")).lower() + " " + str(scene.get("description", "")).lower()
            lying_keywords = ["lying", "lie", "躺", "lying on", "lie on", "prone", "保持不动", "静止", "脚下", "地面", "floor", "ground", "沙地"]
            if any(kw in scene_text for kw in lying_keywords):
                # 进一步检查是否是"lying"动作（需要组合判断）
                motionless_keywords = ["保持不动", "静止", "不动", "motionless", "still"]
                ground_keywords = ["脚下", "地面", "floor", "ground", "沙地"]
                if any(kw in scene_text for kw in motionless_keywords) and any(kw in scene_text for kw in ground_keywords):
                    is_lying_action = True
                    logger.info("  ⚡ 检测到'lying'动作，将降低参考强度以让 prompt 控制姿态")
        
        # 1. 计算参考强度
        reference_strength = self._calculate_reference_strength(
            shot_type=shot_type,
            camera_angle=camera_angle,
            emotion=character_emotion,
            character_present=character_present
        )
        
        # ⚡ 关键修复：如果是"lying"动作，降低参考强度（从 60% 降到 40%），让 prompt 有更大控制权
        if is_lying_action:
            reference_strength = max(40, reference_strength - 20)  # 至少降低 20%，但不低于 40%
            logger.info(f"  ⚡ 'lying'动作：参考强度从 {reference_strength + 20}% 降低到 {reference_strength}%")

        
        # ⚡ 脸可见/特写场景：强制提高参考强度，避免"完全不像"的人像
        # 说明：
        # - v2 常见 shot=close_up，但如果 reference_strength 偏低，会导致身份注入不够
        # - 但对于"lying"动作，不强制提高（因为需要让 prompt 控制姿态）
        if character_present and not is_lying_action:
            face_visible = bool(character.get("face_visible", False))
            visibility = str(character.get("visibility", "") or "").lower()
            body_coverage = str(character.get("body_coverage", "") or "").lower()
            # 近景/特写兜底
            if face_visible or visibility in ("high", "mid") or body_coverage in ("head_only", "head", "face"):
                if shot_type in ("close", "extreme_close"):
                    reference_strength = max(reference_strength, 85)
                elif shot_type in ("medium_close",):
                    reference_strength = max(reference_strength, 75)
                elif shot_type in ("medium",):
                    reference_strength = max(reference_strength, 65)
        
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

    @staticmethod
    def _normalize_shot_type(value: Any) -> str:
        """将不同写法的 shot 归一化为 planner 内部使用的枚举字符串。"""
        if not value:
            return "medium"
        s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        # 常见同义词
        if s in ("close_up", "closeup"):
            return "close"
        if s in ("medium_close_up", "medium_closeup"):
            return "medium_close"
        if s in ("extreme_close_up", "extreme_closeup"):
            return "extreme_close"
        # 兜底：保持原样（让 shot_strength_map 或上层逻辑处理）
        return s
    
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
        strategy: GenerationStrategy = None,
        original_prompt: Optional[str] = None
    ) -> str:
        """
        构建带权重的 Prompt
        
        参考豆包/可灵的 Prompt 策略:
        - 环境描述放在前面
        - 使用加权语法 (xxx:1.3)
        - 自动添加镜头类型描述以控制构图
        
        Args:
            scene: 场景字典
            strategy: 生成策略（如果为 None，会自动分析）
            original_prompt: 原始 prompt（如果提供，会优先使用它，而不是从 scene 构建）
        """
        if strategy is None:
            strategy = self.analyze_scene(scene)
        
        # ⚡ 如果提供了原始 prompt，优先使用它（保留完整信息）
        if original_prompt and original_prompt.strip():
            # 从 scene 中提取镜头类型描述，添加到原始 prompt 前面
            camera = scene.get("camera", {})
            shot_type = camera.get("shot", "medium")
            
            # ⚡ 关键修复：使用智能场景分析器（支持本地规则和 LLM）
            if SCENE_ANALYZER_AVAILABLE:
                try:
                    # 使用场景分析器分析 prompt
                    # 读取配置：是否使用 LLM 分析（默认使用本地规则）
                    prompt_engine_config = self.config.get("prompt_engine", {})
                    use_llm = prompt_engine_config.get("scene_analyzer_mode", "local") in ["llm", "hybrid"]
                    
                    # ⚡ 关键修复：传递 LLM 客户端，确保 LLM 模式正常工作
                    analysis_result = analyze_scene(
                        prompt=original_prompt,
                        current_shot_type=shot_type,
                        use_llm=use_llm,
                        llm_client=self.llm_client if use_llm else None
                    )
                    
                    if use_llm and self.llm_client:
                        logger.info("  ✓ 使用 LLM 场景分析器（更智能的分析）")
                    else:
                        logger.info("  ✓ 使用本地场景分析器（快速规则引擎）")
                    
                    # ⚡ 工程级优化：使用决策表整合规则层和 LLM 层
                    logger.info(f"  ✓ LLM 场景分析完成，开始处理结果...")
                    # 1. 获取 LLM 的语义理解结果
                    llm_posture_type = analysis_result.posture_type if analysis_result else None
                    llm_recommended_shot = analysis_result.recommended_shot_type.value if analysis_result else shot_type
                    logger.debug(f"  [DEBUG] LLM 返回: posture_type={llm_posture_type}, recommended_shot={llm_recommended_shot}")
                    
                    # 2. 使用规则引擎（PostureController）做导演语义判断
                    logger.debug(f"  [DEBUG] 开始规则引擎分析...")
                    posture_hint = None
                    try:
                        from utils.posture_controller import PostureController
                        logger.debug(f"  [DEBUG] PostureController 导入成功")
                        posture_controller = PostureController()
                        logger.debug(f"  [DEBUG] PostureController 实例化成功，开始分析...")
                        director_semantics = posture_controller.analyze_director_semantics(original_prompt)
                        logger.debug(f"  [DEBUG] 规则引擎分析完成: {director_semantics}")
                        posture_hint = director_semantics.get("posture_hint")
                        if posture_hint:
                            logger.info(f"  ✓ 规则引擎检测到姿态提示: {posture_hint} (置信度: {director_semantics.get('confidence', 0):.2f})")
                        else:
                            logger.debug(f"  [DEBUG] 规则引擎未检测到姿态提示")
                    except ImportError as e:
                        logger.debug(f"  [DEBUG] PostureController 导入失败: {e}")
                        pass
                    except Exception as e:
                        logger.warning(f"  ⚠ 规则引擎分析出错: {e}")
                        import traceback
                        logger.debug(f"  [DEBUG] 规则引擎异常详情: {traceback.format_exc()}")
                        pass
                    
                    # 3. 使用决策表决定最终姿态和镜头
                    final_posture_type = llm_posture_type
                    if posture_hint and posture_hint.endswith("_candidate"):
                        # 规则引擎的候选提示，如果 LLM 没有检测到，使用规则引擎的结果
                        if not final_posture_type:
                            final_posture_type = posture_hint.replace("_candidate", "")
                            logger.info(f"  ✓ 使用规则引擎的姿态判断: {final_posture_type}")
                    
                    # 4. 根据姿态决策表推荐镜头
                    if final_posture_type:
                        try:
                            from utils.posture_decision_table import recommend_shot_for_posture
                            recommended_shot = recommend_shot_for_posture(
                                posture_type=final_posture_type,
                                current_shot=shot_type,
                                needs_ground=analysis_result.needs_ground_visible if analysis_result else False,
                                needs_environment=analysis_result.needs_environment_visible if analysis_result else False
                            )
                            if recommended_shot != shot_type:
                                shot_type = recommended_shot
                                logger.info(f"  ✓ 决策表推荐镜头类型: {recommended_shot} (原: {camera.get('shot', 'medium')}, 姿态: {final_posture_type})")
                        except ImportError:
                            # 回退到 LLM 推荐
                            if llm_recommended_shot != shot_type:
                                shot_type = llm_recommended_shot
                                logger.info(f"  ✓ LLM 推荐镜头类型: {llm_recommended_shot} (原: {camera.get('shot', 'medium')})")
                    else:
                        # 没有检测到姿态，使用 LLM 推荐
                        if llm_recommended_shot != shot_type:
                            shot_type = llm_recommended_shot
                            logger.info(f"  ✓ LLM 推荐镜头类型: {llm_recommended_shot} (原: {camera.get('shot', 'medium')})")
                    
                    logger.info(f"  ✓ 镜头类型决策完成: {shot_type}")
                    # 5. 生成增强描述（由 Execution Planner 根据规则生成，不在 LLM 中处理）
                    logger.info(f"  ✓ 开始生成增强描述...")
                    enhancement_descriptions = []
                    if analysis_result:
                        # 根据姿态决策表决定是否需要全身
                        if final_posture_type:
                            try:
                                from utils.posture_decision_table import get_posture_decision
                                decision = get_posture_decision(final_posture_type)
                                if decision.get("needs_full_body"):
                                    enhancement_descriptions.append("全身可见，完整身体")
                            except ImportError:
                                pass
                        
                        # 添加地面/环境描述
                        if analysis_result.needs_ground_visible:
                            enhancement_descriptions.append("地面可见，脚可见")
                        if analysis_result.needs_environment_visible:
                            enhancement_descriptions.append("环境可见，风景构图")
                    logger.info(f"  ✓ 增强描述生成完成: {len(enhancement_descriptions)} 条")
                except Exception as e:
                    logger.warning(f"场景分析器失败，回退到硬编码规则: {e}")
                    import traceback
                    logger.debug(f"  [DEBUG] 场景分析器异常详情: {traceback.format_exc()}")
                    analysis_result = None
                    enhancement_descriptions = []
            else:
                # 回退到硬编码规则（原有逻辑）
                logger.debug(f"  [DEBUG] 使用硬编码规则（场景分析器不可用）")
                analysis_result = None
                enhancement_descriptions = []
            
            logger.info(f"  ✓ 场景分析完成，开始构建 prompt...")
            # ⚡ 关键修复：Flux 使用 T5 编码器，不支持权重语法 (xxx:1.5)
            # ⚡ 关键修复：Flux 支持中文，直接使用中文描述，不需要翻译
            shot_descriptions = {
                "extreme_wide": "极远景，人物很小，远景，环境占主导，风景构图，广阔风景",
                "wide": "远景，人物在画面中可见，远景，显示广阔环境，风景构图，全身可见",
                "full": "全身镜头，人物中等大小，地面可见，脚可见，全身清晰显示",
                "american": "美式镜头，7/8身体，人物中等大小",
                "medium": "中景，上半身，人物中等大小",
                "medium_close": "中近景，胸部和头部，人物较大",
                "close": "近景，头部和肩膀，脸部突出",
                "extreme_close": "特写，只有脸部，人物填满画面"
            }
            shot_desc = shot_descriptions.get(shot_type, "medium shot")
            # Flux 不支持权重语法，不使用 shot_weight
            
            # 检查原始 prompt 是否已经包含镜头类型描述
            prompt_lower = original_prompt.lower()
            has_shot_desc = any(
                keyword in prompt_lower 
                for keyword in ["shot", "wide", "medium", "close", "full body", "upper body"]
            )
            
            if not has_shot_desc:
                # 如果原始 prompt 没有镜头类型描述，添加它
                enhanced_prompt = original_prompt.strip()
                
                # ⚡ 关键修复：如果场景分析器提供了增强描述，优先使用它们
                # 特别是"lying"等姿态描述，需要放在最前面（镜头类型之后），确保不被覆盖
                if enhancement_descriptions:
                    # ⚡ 关键修复：使用统一的去重工具，避免重复
                    logger.debug(f"  [DEBUG] 开始去重处理，enhancement_descriptions: {len(enhancement_descriptions)} 条")
                    try:
                        from utils.prompt_deduplicator import filter_duplicates
                        logger.debug(f"  [DEBUG] filter_duplicates 导入成功")
                        
                        # 合并已有文本（镜头类型描述 + 原始 prompt）
                        existing_texts = [shot_desc, enhanced_prompt]
                        logger.debug(f"  [DEBUG] existing_texts: {existing_texts}")
                        
                        # 使用去重工具过滤
                        # ⚡ 关键修复：提高阈值，更严格地检测重复（从0.6提高到0.5）
                        logger.debug(f"  [DEBUG] 调用 filter_duplicates...")
                        filtered_enhancements = filter_duplicates(
                            new_descriptions=enhancement_descriptions,
                            existing_texts=existing_texts,
                            threshold=0.5  # 50% 重叠认为是重复（更严格）
                        )
                        logger.debug(f"  [DEBUG] filter_duplicates 完成，结果: {len(filtered_enhancements)} 条")
                    except ImportError:
                        # 如果去重工具不可用，使用简单检查
                        logger.warning("去重工具不可用，使用简单检查")
                        enhanced_prompt_lower = enhanced_prompt.lower()
                        shot_desc_lower = shot_desc.lower()
                        combined_lower = f"{shot_desc_lower}, {enhanced_prompt_lower}"
                        
                        filtered_enhancements = []
                        for desc in enhancement_descriptions:
                            desc_lower = desc.lower()
                            desc_keywords = desc_lower.split("，")  # 中文逗号分隔
                            if len(desc_keywords) == 1:
                                desc_keywords = desc_lower.split(", ")  # 英文逗号分隔
                            
                            existing_count = sum(1 for kw in desc_keywords[:3] if kw.strip() in combined_lower)
                            if existing_count < 2:
                                filtered_enhancements.append(desc)
                            else:
                                logger.debug(f"  跳过重复的增强描述: {desc[:50]}...")
                    
                    # ⚡ 工程级优化：使用 final_posture_type（已由规则层和 LLM 层整合）
                    # 优先使用 LLM 返回的姿态描述，如果没有则使用 PostureController 模板
                    logger.info(f"  ✓ 开始处理姿态指令，final_posture_type: {final_posture_type}")
                    final_parts = []
                    other_descriptions = filtered_enhancements if filtered_enhancements else []
                    logger.debug(f"  [DEBUG] other_descriptions: {len(other_descriptions)} 条")
                    
                    # 获取姿态指令（优先级：LLM 返回 > PostureController 模板）
                    posture_instruction = None
                    if analysis_result and analysis_result.posture_positive:
                        # LLM 已经返回了精确的姿态描述，直接使用
                        posture_instruction = analysis_result.posture_positive
                        logger.info(f"  ✓ LLM 已返回姿态描述: {analysis_result.posture_type}")
                        logger.info(f"  ✓ 姿态指令: {posture_instruction[:80]}...")
                    elif final_posture_type:
                        # LLM 没有返回描述，但检测到了姿态类型，使用 PostureController 模板
                        try:
                            from utils.posture_controller import PostureController
                            posture_controller = PostureController()
                            posture_prompt = posture_controller.get_posture_prompt(final_posture_type, use_chinese=False)
                            posture_instruction = posture_prompt.get("positive", "")
                            if posture_instruction:
                                logger.info(f"  ✓ 使用 PostureController 模板生成姿态描述: {final_posture_type}")
                                logger.info(f"  ✓ 姿态指令: {posture_instruction[:80]}...")
                        except ImportError:
                            pass
                    
                    # 如果有姿态指令，添加到 final_parts 最前面（最高优先级）
                    if posture_instruction:
                        final_parts = [posture_instruction]
                        logger.debug(f"  姿态指令已添加到 final_parts，当前长度: {len(final_parts)}")
                    else:
                        # 没有检测到姿态，检查是否有增强描述
                        if not filtered_enhancements:
                            # 如果没有增强描述，直接返回
                            return f"{shot_desc}, {enhanced_prompt}"
                        
                        # 如果有增强描述但没有检测到姿态，检查是否有姿态关键词
                        pose_keywords = ["lying", "躺", "sitting", "坐", "prone", "水平位置", "俯卧", "水平"]
                        pose_descriptions = []
                        
                        for desc in filtered_enhancements:
                            desc_lower = desc.lower()
                            if any(kw in desc_lower for kw in pose_keywords):
                                pose_descriptions.append(desc)
                            else:
                                other_descriptions.append(desc)
                        
                        # ⚡ 关键修复：姿态描述放在最最前面（最高优先级），确保不被任何其他描述覆盖
                        if pose_descriptions:
                            final_parts.extend(pose_descriptions)
                            logger.info(f"  ✓ 已添加姿态描述（最高优先级，放在最最前面）: {len(pose_descriptions)} 条")
                    
                    # ⚡ 关键修复：统一处理 other_descriptions
                    # 如果 PostureController 检测到姿态，final_parts 已经包含姿态指令，只需要处理其他描述
                    if "other_descriptions" not in locals():
                        other_descriptions = filtered_enhancements if filtered_enhancements else []
                    
                    # 镜头类型描述
                    if "final_parts" not in locals():
                        final_parts = []
                    final_parts.append(shot_desc)
                    
                    # 其他增强描述（场景、地面等）
                    if other_descriptions:
                        final_parts.extend(other_descriptions)
                        logger.info(f"  ✓ 已添加场景增强描述: {len(other_descriptions)} 条")
                    
                    # ⚡ 关键修复：分离角色描述和原始 prompt，确保角色描述在姿态和场景之后，但在原始 prompt 之前
                    logger.info(f"  ✓ 开始分离角色描述和场景描述...")
                    # 检查 enhanced_prompt 是否包含角色描述（通常在开头）
                    enhanced_prompt_parts = enhanced_prompt.split(", ")
                    logger.debug(f"  [DEBUG] enhanced_prompt_parts: {len(enhanced_prompt_parts)} 部分")
                    # ⚡ 关键修复：更精确的角色关键词识别，避免误判场景描述为角色描述
                    character_keywords = [
                        "male", "female", "character", "person", 
                        "robe", "clothing", "hair", "韩立",
                        "cultivator robe", "xianxia", "immortal cultivator",
                        "tied long black hair", "forehead bangs", "neat",
                        "deep cyan", "flowing fabric", "not armor"
                    ]
                    character_parts = []
                    scene_parts = []
                    
                    for part in enhanced_prompt_parts:
                        part_lower = part.lower()
                        # 检查是否是角色描述（更精确的匹配）
                        is_character = any(kw in part_lower for kw in character_keywords)
                        # ⚡ 关键修复：排除明显是场景描述的内容（如 "cinematic", "dramatic", "landscape" 等）
                        scene_keywords = ["cinematic", "dramatic", "landscape", "nature", "desert", "floor", "ground", "沙地"]
                        is_scene = any(kw in part_lower for kw in scene_keywords)
                        
                        if is_character and not is_scene:
                            character_parts.append(part)
                        else:
                            scene_parts.append(part)
                    
                    # ⚡ 关键修复：构建最终 prompt 的顺序：
                    # 姿态描述 -> 镜头类型 -> 场景描述 -> 角色描述（服饰+形象）-> 原始 prompt
                    # 这样确保角色描述在原始 prompt 之前，有足够的权重
                    
                    # ⚡ 关键修复：使用 PromptDeduplicator 去除重复描述
                    logger.info(f"  ✓ 开始最终去重处理...")
                    try:
                        from utils.prompt_deduplicator import PromptDeduplicator
                        logger.debug(f"  [DEBUG] PromptDeduplicator 导入成功")
                        deduplicator = PromptDeduplicator()
                        logger.debug(f"  [DEBUG] PromptDeduplicator 实例化成功")
                        
                        # 去重场景部分
                        if scene_parts:
                            logger.debug(f"  [DEBUG] 开始去重场景部分，共 {len(scene_parts)} 条")
                            scene_text = ", ".join(scene_parts)
                            # 检查场景部分内部是否有重复
                            scene_parts_clean = deduplicator.merge_prompt_parts(scene_parts)
                            logger.debug(f"  [DEBUG] 场景部分去重完成，结果: {len(scene_parts_clean.split(', ')) if scene_parts_clean else 0} 条")
                            if scene_parts_clean:
                                final_parts.append(scene_parts_clean)
                        
                        # 去重角色部分
                        if character_parts:
                            character_text = ", ".join(character_parts)
                            # 检查角色部分内部是否有重复
                            character_parts_clean = deduplicator.merge_prompt_parts(character_parts)
                            if character_parts_clean:
                                final_parts.append(character_parts_clean)
                                logger.info(f"  ✓ 已添加角色描述（服饰+形象，已去重）")
                        
                        # ⚡ 关键修复：添加原始 prompt（如果还有剩余内容）
                        # 注意：enhanced_prompt 可能已经被 PostureController 处理过，需要确保原始内容也被添加
                        if enhanced_prompt.strip():
                            # 检查 enhanced_prompt 是否已经在 final_parts 中（通过场景/角色部分）
                            # 如果没有，添加它
                            enhanced_prompt_lower = enhanced_prompt.lower()
                            already_included = False
                            for part in final_parts:
                                if enhanced_prompt_lower in part.lower() or part.lower() in enhanced_prompt_lower:
                                    already_included = True
                                    break
                            
                            if not already_included:
                                final_parts.append(enhanced_prompt)
                                logger.info(f"  ✓ 已添加原始 prompt 到 final_parts")
                        
                        # 最终合并并去重
                        final_prompt = deduplicator.merge_prompt_parts(final_parts)
                        logger.info(f"  ✓ 已对最终 prompt 进行去重处理")
                        return final_prompt
                    except ImportError:
                        # 如果去重工具不可用，使用原始逻辑
                        if scene_parts:
                            final_parts.append(", ".join(scene_parts))
                        if character_parts:
                            final_parts.append(", ".join(character_parts))
                            logger.info(f"  ✓ 已添加角色描述（服饰+形象，{len(character_parts)} 条）")
                        if enhanced_prompt.strip():
                            final_parts.append(enhanced_prompt)
                        return ", ".join(final_parts)
                
                # Flux 不支持权重语法，直接使用自然语言描述
                return f"{shot_desc}, {enhanced_prompt}"
            else:
                # 如果已经有镜头类型描述，但需要增强场景描述
                if enhancement_descriptions:
                    enhanced_prompt = f"{', '.join(enhancement_descriptions)}, {original_prompt.strip()}"
                    logger.info(f"  ✓ 已应用场景分析器增强描述: {len(enhancement_descriptions)} 条")
                    return enhanced_prompt
                
                # 如果已经有镜头类型描述，直接返回原始 prompt
                return original_prompt.strip()
        
        # 如果没有提供原始 prompt，从 scene 字典构建（原有逻辑）
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
