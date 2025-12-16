#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Engine V2 - 工程化实现
基于架构设计文档的完整实现，支持可扩展、可观测、可调优的Prompt工程系统
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
import hashlib
import logging
import time
from pathlib import Path
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Shot:
    """单个镜头定义"""
    shot_id: int
    description: str
    action: Optional[str] = None
    mood: Optional[str] = None
    duration_secs: Optional[float] = None
    key_frames: Optional[List[int]] = None


@dataclass
class Character:
    """角色定义"""
    id: str
    desc: str
    appearance: Optional[str] = None


@dataclass
class SceneStruct:
    """场景结构"""
    shots: List[Shot] = field(default_factory=list)
    characters: List[Character] = field(default_factory=list)


@dataclass
class StyleTemplate:
    """风格模板"""
    template_id: str
    pre_prompt: str = ""
    post_prompt: str = ""
    negative_list: List[str] = field(default_factory=list)
    preferred_camera: Optional[Dict[str, Any]] = None


@dataclass
class CameraDSL:
    """相机DSL定义"""
    shot: str = "medium_shot"
    motion: str = "static"
    lens: str = "35mm"
    dof: str = "shallow"
    speed: str = "normal"


@dataclass
class PromptPackage:
    """Prompt处理包 - 模块间传递的核心对象"""
    raw_text: str
    rewritten_text: str = ""
    scene_struct: SceneStruct = field(default_factory=SceneStruct)
    style: Optional[StyleTemplate] = None
    camera: List[Dict[str, Any]] = field(default_factory=list)
    negative: str = ""
    final_prompt: str = ""
    model_target: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        result = asdict(self)
        # 处理嵌套对象
        result['scene_struct'] = {
            'shots': [asdict(s) for s in self.scene_struct.shots],
            'characters': [asdict(c) for c in self.scene_struct.characters]
        }
        if self.style:
            result['style'] = asdict(self.style)
        return result


@dataclass
class UserRequest:
    """用户请求结构"""
    text: str
    scene_type: str = "general"
    style: Optional[str] = None
    character_id: Optional[str] = None
    target_model: str = "auto"
    user_tier: str = "basic"
    params: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 缓存接口（支持Redis和内存缓存）
# ============================================================================

class CacheInterface:
    """缓存接口"""
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    def get(self, key: str) -> Any:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        raise NotImplementedError


class MemoryCache(CacheInterface):
    """内存缓存实现"""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def exists(self, key: str) -> bool:
        if key not in self._cache:
            return False
        value, expire_time = self._cache[key]
        if expire_time > 0 and time.time() > expire_time:
            del self._cache[key]
            return False
        return True
    
    def get(self, key: str) -> Any:
        if not self.exists(key):
            return None
        return self._cache[key][0]
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        expire_time = time.time() + ttl if ttl > 0 else 0
        self._cache[key] = (value, expire_time)


# ============================================================================
# LLM客户端接口
# ============================================================================

class LLMClient:
    """LLM客户端接口"""
    
    def rewrite_prompt(self, text: str, scene: str = "general") -> str:
        """重写prompt"""
        raise NotImplementedError
    
    def decompose_shots(self, text: str) -> SceneStruct:
        """分解为镜头结构"""
        raise NotImplementedError


class SimpleLLMClient(LLMClient):
    """本地规则LLM客户端（完全本地运行，无需外部API）"""
    
    def __init__(self):
        """初始化本地规则引擎"""
        # 场景类型关键词映射
        self.scene_keywords = {
            "novel": ["cinematic", "dramatic", "epic", "storytelling", "narrative"],
            "drama": ["dramatic", "emotional", "intense", "theatrical"],
            "scientific": ["scientific", "technical", "precise", "realistic", "accurate"],
            "government": ["professional", "authoritative", "formal", "official"],
            "enterprise": ["corporate", "professional", "modern", "sophisticated"],
            "general": ["high quality", "professional", "detailed"]
        }
        
        # 质量增强词库
        self.quality_enhancers = [
            "high-resolution", "cinematic", "detailed", "professional",
            "photorealistic", "excellent composition", "visual appeal",
            "sharp focus", "masterpiece", "best quality"
        ]
        
        # 构图关键词
        self.composition_keywords = [
            "rule of thirds", "symmetrical composition", "leading lines",
            "balanced framing", "dynamic composition", "cinematic framing"
        ]
        
        # 光线关键词
        self.lighting_keywords = [
            "rim light", "soft light", "dramatic lighting", "natural lighting",
            "golden hour", "cinematic lighting", "volumetric lighting"
        ]
        
        # 动作关键词
        self.action_keywords = {
            "walking": "walking slowly",
            "running": "running gracefully",
            "sitting": "sitting calmly",
            "standing": "standing firmly",
            "looking": "gazing intently",
            "moving": "moving smoothly",
            "flowing": "flowing elegantly",
            "rotating": "rotating slowly",
            "expanding": "expanding gradually"
        }
        
        # 环境关键词
        self.environment_keywords = {
            "山谷": "mountain valley",
            "瀑布": "waterfall",
            "彩虹": "rainbow",
            "雪山": "snow-capped mountains",
            "绿树": "green trees",
            "阳光": "sunlight",
            "云层": "clouds",
            "月光": "moonlight",
            "剑": "sword",
            "桥": "bridge",
            "实验室": "laboratory",
            "太空": "space",
            "黑洞": "black hole"
        }
    
    def rewrite_prompt(self, text: str, scene: str = "general") -> str:
        """
        使用本地规则重写prompt（完全本地运行，无需LLM API）
        
        Args:
            text: 原始文本
            scene: 场景类型
            
        Returns:
            重写后的prompt
        """
        if not text or not text.strip():
            return ""
        
        parts = []
        
        # 1. 保留原始文本（作为核心描述）
        parts.append(text.strip())
        
        # 2. 根据场景类型添加风格关键词
        scene_keywords = self.scene_keywords.get(scene, self.scene_keywords["general"])
        if scene_keywords:
            parts.append(", ".join(scene_keywords[:2]))  # 使用前2个关键词
        
        # 3. 添加构图关键词
        parts.append(self.composition_keywords[0])
        
        # 4. 添加光线关键词
        parts.append(self.lighting_keywords[0])
        
        # 5. 添加质量增强词
        parts.append(", ".join(self.quality_enhancers[:3]))  # 使用前3个
        
        # 6. 检测并增强动作描述
        text_lower = text.lower()
        for action_key, action_enhanced in self.action_keywords.items():
            if action_key in text_lower:
                parts.append(action_enhanced)
                break
        
        # 7. 检测并增强环境描述
        for env_cn, env_en in self.environment_keywords.items():
            if env_cn in text:
                parts.append(f"in {env_en}")
                break
        
        return ", ".join(parts)
    
    def decompose_shots(self, text: str) -> SceneStruct:
        """
        使用本地规则分解镜头（完全本地运行，无需LLM API）
        
        Args:
            text: 重写后的文本
            
        Returns:
            SceneStruct对象
        """
        shots = []
        
        # 策略1: 按句号分割（中文）
        sentences_cn = [s.strip() for s in text.split('。') if s.strip()]
        
        # 策略2: 按句号分割（英文）
        sentences_en = [s.strip() for s in text.split('.') if s.strip()]
        
        # 选择更合适的分割方式
        if len(sentences_cn) > len(sentences_en):
            sentences = sentences_cn
        else:
            sentences = sentences_en
        
        # 如果没有明确的分割，尝试按逗号分割长句
        if len(sentences) <= 1:
            # 按逗号分割，但只处理较长的文本
            if len(text) > 50:
                parts = [p.strip() for p in text.split(',') if p.strip()]
                if len(parts) > 1:
                    sentences = parts[:3]  # 最多3个部分
        
        # 如果还是没有分割，将整个文本作为一个镜头
        if len(sentences) == 0:
            sentences = [text]
        
        # 为每个句子创建镜头（过滤掉太短的片段）
        for i, sentence in enumerate(sentences):
            # 过滤掉太短的片段（少于5个字符）
            if len(sentence.strip()) < 5:
                continue
            
            # 估算时长（根据文本长度）
            duration = max(1.5, min(3.0, len(sentence) / 20))
            
            # 检测动作
            action = self._detect_action(sentence)
            
            # 检测情绪
            mood = self._detect_mood(sentence)
            
            shots.append(Shot(
                shot_id=len(shots) + 1,
                description=sentence,
                action=action,
                mood=mood,
                duration_secs=duration,
                key_frames=[0, int(duration * 12)] if duration >= 2.0 else [0]  # 假设24fps
            ))
        
        # 如果没有镜头，创建一个默认镜头
        if len(shots) == 0:
            shots.append(Shot(
                shot_id=1,
                description=text[:100],  # 使用前100个字符
                duration_secs=2.0
            ))
        
        return SceneStruct(shots=shots)
    
    def _detect_shot_type(self, text: str) -> str:
        """检测镜头类型"""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["close", "close-up", "face", "portrait"]):
            return "close_up"
        elif any(kw in text_lower for kw in ["wide", "establishing", "landscape", "view"]):
            return "wide_shot"
        elif any(kw in text_lower for kw in ["medium", "mid"]):
            return "medium_shot"
        else:
            return "medium_shot"  # 默认
    
    def _detect_action(self, text: str) -> Optional[str]:
        """检测动作"""
        text_lower = text.lower()
        for action_key, action_enhanced in self.action_keywords.items():
            if action_key in text_lower:
                return action_enhanced
        return None
    
    def _detect_mood(self, text: str) -> Optional[str]:
        """检测情绪/氛围"""
        mood_keywords = {
            "dramatic": ["dramatic", "intense", "powerful"],
            "calm": ["calm", "peaceful", "serene", "tranquil"],
            "mysterious": ["mysterious", "mystical", "enigmatic"],
            "epic": ["epic", "grand", "magnificent"],
            "romantic": ["romantic", "tender", "gentle"]
        }
        
        text_lower = text.lower()
        for mood, keywords in mood_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return mood
        
        return "calm"  # 默认


# ============================================================================
# 风格模板存储
# ============================================================================

class StyleStore:
    """风格模板存储"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.styles: Dict[str, StyleTemplate] = {}
        self._load_default_styles()
        if config_path:
            self._load_from_file(config_path)
    
    def _load_default_styles(self):
        """加载默认风格模板"""
        default_styles = {
            "xianxia_v2": StyleTemplate(
                template_id="xianxia_v2",
                pre_prompt="ethereal, mist, soft rim light, flowing cloth, cinematic composition",
                post_prompt="film grain low, 35mm lens, shallow depth of field",
                negative_list=["oversaturated", "watermark", "disfigured hands"],
                preferred_camera={"shot": "medium", "motion": "slow", "lens": "35mm"}
            ),
            "novel": StyleTemplate(
                template_id="novel",
                pre_prompt="cinematic scene, dramatic backlight, film texture",
                post_prompt="35mm lens, shallow depth of field, cinematic color grading",
                negative_list=["modern elements", "anachronistic", "low quality"],
                preferred_camera={"shot": "wide", "motion": "dolly_in", "lens": "35mm"}
            ),
            "scientific": StyleTemplate(
                template_id="scientific",
                pre_prompt="high-tech scientific visualization, clean lighting",
                post_prompt="professional documentary look, realistic details",
                negative_list=["unrealistic physics", "impossible phenomena"],
                preferred_camera={"shot": "medium", "motion": "static", "lens": "50mm"}
            ),
            "general": StyleTemplate(
                template_id="general",
                pre_prompt="high quality, cinematic, detailed",
                post_prompt="professional, photorealistic",
                negative_list=["low quality", "blurry", "distorted"],
                preferred_camera={"shot": "medium", "motion": "static", "lens": "35mm"}
            )
        }
        self.styles.update(default_styles)
    
    def _load_from_file(self, config_path: str):
        """从文件加载风格模板"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
                
                for template_id, template_data in data.items():
                    # 避免重复传递template_id
                    template_data_copy = template_data.copy()
                    if 'template_id' in template_data_copy:
                        del template_data_copy['template_id']
                    self.styles[template_id] = StyleTemplate(
                        template_id=template_id,
                        **template_data_copy
                    )
        except Exception as e:
            logger.warning(f"加载风格配置失败: {e}")
    
    def get(self, template_id: str) -> StyleTemplate:
        """获取风格模板"""
        return self.styles.get(template_id, self.styles["general"])
    
    def get_negative(self, template_id: str) -> str:
        """获取负面提示词"""
        template = self.get(template_id)
        return ", ".join(template.negative_list)


# ============================================================================
# 核心模块实现
# ============================================================================

class PromptRewriter:
    """Prompt重写器 - 语义增强 + 语法"""
    
    def __init__(self, llm_client: LLMClient, cache: Optional[CacheInterface] = None):
        self.llm = llm_client
        self.cache = cache or MemoryCache()
    
    def rewrite(self, text: str, req: UserRequest) -> str:
        """重写用户输入"""
        # 检查缓存
        cache_key = f"rewrite:{hashlib.md5(text.encode()).hexdigest()}"
        if self.cache.exists(cache_key):
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info(f"缓存命中: {cache_key}")
                # 注意：metrics 在 PromptEngine 层面统一管理，这里不直接访问
                return cached_result
        
        # 调用LLM重写
        rewritten = self.llm.rewrite_prompt(text, req.scene_type)
        
        # 缓存结果（24小时）
        self.cache.set(cache_key, rewritten, ttl=86400)
        
        return rewritten


class SceneDecomposer:
    """场景分解器 - 拆成结构"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def decompose(self, rewritten: str) -> SceneStruct:
        """分解为镜头结构"""
        return self.llm.decompose_shots(rewritten)


class CameraEngine:
    """相机引擎 - 镜头语言"""
    
    def __init__(self):
        self.dsl_templates = {
            "medium_shot": "medium shot",
            "wide_shot": "wide establishing shot",
            "close_up": "close-up shot",
            "dolly_in": "slow camera dolly forward",
            "dolly_out": "slow camera dolly backward",
            "pan": "smooth camera pan",
            "35mm": "35mm lens",
            "50mm": "50mm lens",
            "85mm": "85mm lens",
            "shallow": "shallow depth of field",
            "deep": "deep depth of field"
        }
    
    def make_camera(self, scene_struct: SceneStruct, style: Optional[StyleTemplate]) -> List[Dict[str, Any]]:
        """为每个镜头生成相机DSL"""
        cameras = []
        preferred = style.preferred_camera if style else {}
        
        for shot in scene_struct.shots:
            # 使用风格模板的偏好，或默认值
            dsl = CameraDSL(
                shot=preferred.get("shot", "medium_shot"),
                motion=preferred.get("motion", "static"),
                lens=preferred.get("lens", "35mm"),
                dof=preferred.get("dof", "shallow"),
                speed=preferred.get("speed", "normal")
            )
            
            # 转换为自然语言
            camera_prompt = self._dsl_to_prompt(dsl)
            
            cameras.append({
                "shot_id": shot.shot_id,
                "dsl": camera_prompt
            })
        
        return cameras
    
    def _dsl_to_prompt(self, dsl: CameraDSL) -> str:
        """将DSL转换为自然语言prompt"""
        parts = []
        parts.append(self.dsl_templates.get(dsl.shot, dsl.shot))
        parts.append(self.dsl_templates.get(dsl.motion, dsl.motion))
        parts.append(self.dsl_templates.get(dsl.lens, dsl.lens))
        parts.append(self.dsl_templates.get(dsl.dof, dsl.dof))
        return ", ".join(parts)


class NegativePromptGenerator:
    """负面提示词生成器"""
    
    def __init__(self, style_store: StyleStore):
        self.style_store = style_store
        self.base_negative = [
            "bad anatomy", "distorted face", "extra limbs", "low resolution",
            "motion blur", "flickering", "inconsistent lighting", "deformed hands",
            "oversaturated colors", "unrealistic shadows", "blurry", "distorted",
            "deformed", "bad hands", "text", "watermark", "jittery", "unstable",
            "sudden movement", "abrupt changes", "low quality", "worst quality"
        ]
        
        self.model_specific = {
            "hunyuanvideo": [
                "green vertical bars", "color shift", "frame skipping",
                "unnatural motion", "jittery frames"
            ],
            "cogvideox": [
                "inconsistent face", "face distortion", "unnatural movement",
                "frame flickering", "motion artifacts"
            ],
            "flux": [
                "bad anatomy", "distorted proportions", "unrealistic details"
            ]
        }
    
    def generate(
        self,
        model_type: str,
        style_template_id: str,
        custom_negatives: Optional[List[str]] = None
    ) -> str:
        """生成负面提示词"""
        negatives = self.base_negative.copy()
        
        # 添加模型特定负面词
        if model_type in self.model_specific:
            negatives.extend(self.model_specific[model_type])
        
        # 添加风格模板负面词
        style_negatives = self.style_store.get_negative(style_template_id)
        if style_negatives:
            negatives.append(style_negatives)
        
        # 添加自定义负面词
        if custom_negatives:
            negatives.extend(custom_negatives)
        
        # 去重
        unique_negatives = list(dict.fromkeys(negatives))
        return ", ".join(unique_negatives)


class PromptQA:
    """Prompt质量检查与修复"""
    
    def __init__(self):
        self.required_fields = ["subject", "action", "environment", "composition", "lighting", "style"]
        self.sensitive_words = ["violence", "explicit", "illegal"]  # 示例敏感词列表
        self.max_tokens = 200  # 最大token数（示例）
    
    def check(self, pkg: PromptPackage) -> Dict[str, Any]:
        """检查PromptPackage的完整性"""
        result = {
            "valid": True,
            "score": 0,
            "max_score": len(self.required_fields) + 2,  # +2 for safety and length
            "missing_fields": [],
            "safety_issues": [],
            "suggestions": []
        }
        
        # 检查必需字段
        final_prompt = pkg.final_prompt or pkg.rewritten_text
        prompt_lower = final_prompt.lower()
        
        field_keywords = {
            "subject": ["subject", "character", "person", "object"],
            "action": ["action", "motion", "movement", "doing"],
            "environment": ["environment", "location", "setting", "scene"],
            "composition": ["composition", "framing", "shot", "angle"],
            "lighting": ["lighting", "light", "illumination"],
            "style": ["style", "aesthetic", "look"]
        }
        
        for field, keywords in field_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                result["score"] += 1
            else:
                result["missing_fields"].append(field)
        
        # 安全检查
        for word in self.sensitive_words:
            if word in prompt_lower:
                result["safety_issues"].append(word)
                result["valid"] = False
        
        # 长度检查
        word_count = len(final_prompt.split())
        if word_count < 10:
            result["suggestions"].append("Prompt太短，建议至少10个词")
        elif word_count > self.max_tokens:
            result["suggestions"].append(f"Prompt过长，建议精简到{self.max_tokens}个词以内")
        else:
            result["score"] += 1
        
        # 安全性评分
        if not result["safety_issues"]:
            result["score"] += 1
        
        return result
    
    def fix(self, pkg: PromptPackage, rewriter: Optional[PromptRewriter] = None) -> PromptPackage:
        """自动修复PromptPackage"""
        check_result = self.check(pkg)
        
        if check_result["valid"] and check_result["score"] == check_result["max_score"]:
            return pkg  # 无需修复
        
        # 如果有安全问题，需要人工审核
        if check_result["safety_issues"]:
            logger.warning(f"检测到安全问题: {check_result['safety_issues']}")
            # 这里可以触发人工审核流程
            pkg.metadata["requires_review"] = True
            return pkg
        
        # 补充缺失字段
        if check_result["missing_fields"]:
            # 可以调用rewriter重新生成
            if rewriter:
                pkg.rewritten_text = rewriter.rewrite(
                    pkg.raw_text,
                    UserRequest(text=pkg.raw_text)
                )
        
        return pkg


# ============================================================================
# Model Adapter层
# ============================================================================

class ModelAdapter:
    """模型适配器基类"""
    
    def build_prompt(self, pkg: PromptPackage) -> str:
        """构建模型特定的prompt"""
        raise NotImplementedError
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证和调整参数"""
        return params


class CogVideoXAdapter(ModelAdapter):
    """CogVideoX模型适配器"""
    
    def build_prompt(self, pkg: PromptPackage) -> str:
        """构建CogVideoX格式的prompt"""
        parts = []
        
        # 添加风格前缀
        if pkg.style:
            parts.append(pkg.style.pre_prompt)
        
        # 添加镜头描述
        for shot in pkg.scene_struct.shots:
            camera = next(
                (c for c in pkg.camera if c["shot_id"] == shot.shot_id),
                {"dsl": ""}
            )
            shot_text = f"{shot.description}. CAMERA: {camera['dsl']}."
            parts.append(shot_text)
        
        # 添加风格后缀
        if pkg.style:
            parts.append(pkg.style.post_prompt)
        
        return " ".join(parts)
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """CogVideoX参数验证"""
        # CogVideoX固定分辨率720x480
        params["width"] = 720
        params["height"] = 480
        params["fps"] = 8
        return params


class HunyuanVideoAdapter(ModelAdapter):
    """HunyuanVideo模型适配器"""
    
    def build_prompt(self, pkg: PromptPackage) -> str:
        """构建HunyuanVideo格式的prompt"""
        parts = []
        
        # HunyuanVideo需要详细的描述
        parts.append(pkg.rewritten_text)
        
        # 添加风格
        if pkg.style:
            parts.append(pkg.style.pre_prompt)
            parts.append(pkg.style.post_prompt)
        
        # 添加相机语言
        if pkg.camera:
            camera_dsls = [c["dsl"] for c in pkg.camera]
            parts.append(", ".join(camera_dsls))
        
        return ", ".join(parts)
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """HunyuanVideo参数验证"""
        # HunyuanVideo支持多种分辨率
        width = params.get("width", 1280)
        height = params.get("height", 768)
        # 确保是8的倍数
        params["width"] = (width // 8) * 8
        params["height"] = (height // 8) * 8
        return params


class FluxAdapter(ModelAdapter):
    """Flux模型适配器"""
    
    def build_prompt(self, pkg: PromptPackage) -> str:
        """构建Flux格式的prompt"""
        # Flux需要简洁清晰的prompt
        parts = [pkg.rewritten_text]
        
        if pkg.style:
            parts.append(pkg.style.pre_prompt)
        
        return ", ".join(parts)
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Flux参数验证"""
        # Flux支持多种分辨率
        return params


# ============================================================================
# PromptEngine Orchestrator（核心流程）
# ============================================================================

class PromptEngine:
    """Prompt Engine主类 - 整合所有模块"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        style_store: Optional[StyleStore] = None,
        cache: Optional[CacheInterface] = None,
        enable_metrics: bool = True
    ):
        """
        初始化Prompt Engine
        
        Args:
            llm_client: LLM客户端（可选，默认使用SimpleLLMClient）
            style_store: 风格存储（可选，默认使用内置风格）
            cache: 缓存接口（可选，默认使用内存缓存）
            enable_metrics: 是否启用指标收集
        """
        self.llm = llm_client or SimpleLLMClient()
        self.style_store = style_store or StyleStore()
        self.cache = cache or MemoryCache()
        self.enable_metrics = enable_metrics
        
        # 初始化模块
        self.rewriter = PromptRewriter(self.llm, self.cache)
        self.decomposer = SceneDecomposer(self.llm)
        self.camera_engine = CameraEngine()
        self.negative_generator = NegativePromptGenerator(self.style_store)
        self.qa = PromptQA()
        
        # 初始化适配器
        self.adapters = {
            "cogvideox": CogVideoXAdapter(),
            "hunyuanvideo": HunyuanVideoAdapter(),
            "flux": FluxAdapter()
        }
        
        # 指标收集
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "qa_fixes": 0,
            "errors": 0,
            "avg_processing_time": 0.0  # 平均处理时间（秒）
        }
    
    def run(self, user_request: UserRequest) -> PromptPackage:
        """
        执行完整的Prompt处理流程
        
        Args:
            user_request: 用户请求
            
        Returns:
            PromptPackage对象
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # 初始化PromptPackage
            pkg = PromptPackage(
                raw_text=user_request.text,
                metadata={
                    "scene_type": user_request.scene_type,
                    "user_tier": user_request.user_tier,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # 1. Prompt Rewriter
            logger.info("步骤1: Prompt Rewriter")
            # 检查缓存是否命中（在调用前检查）
            cache_key = f"rewrite:{hashlib.md5(pkg.raw_text.encode()).hexdigest()}"
            cache_hit_before = self.rewriter.cache.exists(cache_key)
            pkg.rewritten_text = self.rewriter.rewrite(pkg.raw_text, user_request)
            # 如果缓存命中，更新指标
            if cache_hit_before:
                self.metrics["cache_hits"] = self.metrics.get("cache_hits", 0) + 1
            
            # 2. Scene Decomposer
            logger.info("步骤2: Scene Decomposer")
            pkg.scene_struct = self.decomposer.decompose(pkg.rewritten_text)
            
            # 3. Style Controller
            logger.info("步骤3: Style Controller")
            style_id = user_request.style or user_request.scene_type
            pkg.style = self.style_store.get(style_id)
            
            # 4. Camera Engine
            logger.info("步骤4: Camera Engine")
            pkg.camera = self.camera_engine.make_camera(pkg.scene_struct, pkg.style)
            
            # 5. Negative Prompt Generator
            logger.info("步骤5: Negative Prompt Generator")
            model_type = self._select_model(user_request)
            pkg.negative = self.negative_generator.generate(
                model_type=model_type,
                style_template_id=pkg.style.template_id if pkg.style else "general",
                custom_negatives=user_request.params.get("custom_negatives")
            )
            
            # 6. Prompt QA
            logger.info("步骤6: Prompt QA")
            pkg = self.qa.fix(pkg, self.rewriter)
            qa_result = self.qa.check(pkg)
            if not qa_result["valid"]:
                self.metrics["qa_fixes"] += 1
                logger.warning(f"QA检查发现问题: {qa_result}")
            
            # 7. Model Adapter
            logger.info("步骤7: Model Adapter")
            adapter = self.adapters.get(model_type, self.adapters["cogvideox"])
            pkg.final_prompt = adapter.build_prompt(pkg)
            pkg.model_target = model_type
            
            # 8. Token限制检查（CLIP限制77 tokens）
            if model_type == "flux":
                try:
                    from transformers import CLIPTokenizer
                    tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14"
                    )
                    tokens = tokenizer(pkg.final_prompt, truncation=False, return_tensors="pt")
                    token_count = tokens.input_ids.shape[1]
                    
                    if token_count > 77:
                        logger.warning(f"Prompt超过77 tokens限制 ({token_count} tokens)，开始截断")
                        # 截断到77 tokens
                        tokens_truncated = tokenizer(
                            pkg.final_prompt,
                            truncation=True,
                            max_length=77,
                            return_tensors="pt"
                        )
                        # 解码回文本（可能略有差异，但确保不超过77 tokens）
                        pkg.final_prompt = tokenizer.decode(tokens_truncated.input_ids[0], skip_special_tokens=True)
                        final_token_count = tokenizer(pkg.final_prompt, truncation=False, return_tensors="pt").input_ids.shape[1]
                        logger.info(f"Prompt已截断到 {final_token_count} tokens")
                    else:
                        logger.info(f"Prompt token数: {token_count} (在限制内)")
                except Exception as e:
                    logger.warning(f"无法检查token数: {e}，跳过token限制检查")
            
            # 验证参数
            if user_request.params:
                user_request.params = adapter.validate_params(user_request.params)
            
            # 记录处理时间
            processing_time = time.time() - start_time
            pkg.metadata["processing_time"] = processing_time
            pkg.metadata["qa_score"] = qa_result["score"]
            pkg.metadata["qa_max_score"] = qa_result["max_score"]
            
            # 更新平均处理时间
            total = self.metrics["total_requests"]
            current_avg = self.metrics.get("avg_processing_time", 0.0)
            if total > 0:
                self.metrics["avg_processing_time"] = (current_avg * (total - 1) + processing_time) / total
            else:
                self.metrics["avg_processing_time"] = processing_time
            
            logger.info(f"Prompt处理完成，耗时: {processing_time:.2f}秒")
            
            return pkg
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Prompt处理失败: {e}", exc_info=True)
            raise
    
    def _select_model(self, req: UserRequest) -> str:
        """选择目标模型"""
        if req.target_model != "auto":
            return req.target_model
        
        # 根据场景类型自动选择
        model_mapping = {
            "novel": "cogvideox",
            "drama": "cogvideox",
            "scientific": "hunyuanvideo",
            "general": "cogvideox"
        }
        return model_mapping.get(req.scene_type, "cogvideox")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        return self.metrics.copy()
    
    def get_adapter(self, model_type: str) -> ModelAdapter:
        """获取模型适配器"""
        return self.adapters.get(model_type, self.adapters["cogvideox"])


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """测试Prompt Engine V2"""
    print("=" * 60)
    print("Prompt Engine V2 测试")
    print("=" * 60)
    
    # 创建引擎
    engine = PromptEngine()
    
    # 测试用例1：小说推文
    print("\n【测试1】小说推文")
    print("-" * 60)
    req = UserRequest(
        text="那夜他手握长剑，踏入断桥，月光如水洒在剑身上",
        scene_type="novel",
        style="xianxia_v2",
        user_tier="professional"
    )
    
    pkg = engine.run(req)
    print(f"原始文本: {req.text}")
    print(f"重写文本: {pkg.rewritten_text}")
    print(f"最终Prompt: {pkg.final_prompt}")
    print(f"Negative Prompt: {pkg.negative[:100]}...")
    print(f"目标模型: {pkg.model_target}")
    print(f"QA评分: {pkg.metadata.get('qa_score', 0)}/{pkg.metadata.get('qa_max_score', 0)}")
    
    # 测试用例2：科学场景
    print("\n【测试2】科学场景")
    print("-" * 60)
    req2 = UserRequest(
        text="黑洞在太空中旋转，周围有吸积盘",
        scene_type="scientific",
        target_model="hunyuanvideo"
    )
    
    pkg2 = engine.run(req2)
    print(f"最终Prompt: {pkg2.final_prompt}")
    print(f"目标模型: {pkg2.model_target}")
    
    # 显示指标
    print("\n【指标统计】")
    print("-" * 60)
    metrics = engine.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

