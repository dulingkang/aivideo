#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Engine - 专业级AIGC工厂的Prompt工程系统
包含6个核心模块：
1. Prompt Rewriter（Prompt重写器）
2. Scene Decomposer（场景语义解析器）
3. Style Controller（风格控制器）
4. Camera Engine（相机语言引擎）
5. Negative Prompt Generator（反向提示词生成器）
6. Prompt QA（质量评分器）
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class PromptComponents:
    """Prompt组件结构"""
    shot: str = ""  # 镜头类型
    subject: str = ""  # 主体
    action: str = ""  # 动作
    environment: str = ""  # 环境
    emotion: str = ""  # 情绪/氛围
    fx: str = ""  # 特效
    style: str = ""  # 风格
    camera: str = ""  # 相机语言
    lighting: str = ""  # 光线
    composition: str = ""  # 构图


class PromptRewriter:
    """Prompt重写器 - 利用LLM统一所有用户输入的结构"""
    
    def __init__(self, use_llm: bool = False, llm_api: Optional[Any] = None):
        """
        初始化Prompt重写器
        
        Args:
            use_llm: 是否使用LLM进行重写（需要API）
            llm_api: LLM API接口（可选）
        """
        self.use_llm = use_llm
        self.llm_api = llm_api
        
        # 基础关键词库
        self.quality_enhancers = [
            "high-resolution", "cinematic", "detailed", "professional",
            "photorealistic", "excellent composition", "visual appeal",
            "sharp focus", "masterpiece", "best quality"
        ]
        
        self.composition_keywords = [
            "rule of thirds", "symmetrical composition", "leading lines",
            "balanced framing", "dynamic composition"
        ]
        
        self.lighting_keywords = [
            "rim light", "soft light", "dramatic lighting", "natural lighting",
            "golden hour", "cinematic lighting", "volumetric lighting"
        ]
    
    def rewrite(
        self,
        user_input: str,
        scene_type: str = "general",
        enhance_quality: bool = True
    ) -> str:
        """
        重写用户输入的prompt
        
        Args:
            user_input: 用户原始输入
            scene_type: 场景类型
            enhance_quality: 是否增强质量关键词
        
        Returns:
            重写后的prompt
        """
        if not user_input or not user_input.strip():
            return ""
        
        # 如果使用LLM，调用LLM API
        if self.use_llm and self.llm_api:
            return self._rewrite_with_llm(user_input, scene_type)
        
        # 否则使用规则基础的重写
        return self._rewrite_with_rules(user_input, scene_type, enhance_quality)
    
    def _rewrite_with_llm(self, user_input: str, scene_type: str) -> str:
        """使用LLM重写prompt"""
        # TODO: 实现LLM API调用
        # 这里提供一个模板prompt
        llm_prompt = f"""请将以下用户输入重写为专业的视频生成prompt，要求：
1. 语法正确，描述清晰
2. 补充镜头细节（如：wide shot, close-up, medium shot）
3. 添加构图词汇（如：rule of thirds, symmetrical composition）
4. 添加光线描述（如：rim light, soft light, dramatic lighting）
5. 添加风格描述（根据场景类型：{scene_type}）
6. 使用英文，专业术语

用户输入：{user_input}

请直接输出重写后的prompt，不要添加其他说明。"""
        
        # 这里应该调用LLM API
        # rewritten = self.llm_api.generate(llm_prompt)
        # return rewritten
        
        # 临时返回规则重写的结果
        return self._rewrite_with_rules(user_input, scene_type, True)
    
    def _rewrite_with_rules(
        self,
        user_input: str,
        scene_type: str,
        enhance_quality: bool
    ) -> str:
        """使用规则重写prompt"""
        parts = []
        
        # 1. 基础描述（用户输入）
        parts.append(user_input.strip())
        
        # 2. 添加质量关键词
        if enhance_quality:
            parts.append(", ".join(self.quality_enhancers[:3]))
        
        # 3. 添加构图关键词
        parts.append(self.composition_keywords[0])
        
        # 4. 添加光线关键词
        parts.append(self.lighting_keywords[0])
        
        # 5. 根据场景类型添加风格
        if scene_type == "scientific":
            parts.append("scientific visualization, clean lighting, realistic details")
        elif scene_type == "novel" or scene_type == "drama":
            parts.append("cinematic scene, dramatic backlight, film texture")
        elif scene_type == "government" or scene_type == "enterprise":
            parts.append("professional documentary look, authoritative tone")
        
        return ", ".join(parts)


class SceneDecomposer:
    """场景语义解析器 - 将用户输入拆解为结构化组件"""
    
    def __init__(self):
        """初始化场景解析器"""
        self.shot_types = {
            "wide": ["wide shot", "establishing shot", "full shot"],
            "medium": ["medium shot", "mid shot"],
            "close": ["close-up", "closeup", "tight shot"],
            "extreme_close": ["extreme close-up", "macro shot"],
            "aerial": ["aerial view", "bird's eye view", "overhead shot"]
        }
        
        self.action_keywords = [
            "walking", "running", "sitting", "standing", "looking", "gazing",
            "moving", "flowing", "rotating", "expanding", "contracting"
        ]
    
    def decompose(
        self,
        user_input: str,
        scene: Optional[Dict[str, Any]] = None
    ) -> PromptComponents:
        """
        解析用户输入为结构化组件
        
        Args:
            user_input: 用户输入文本
            scene: 场景配置（可选）
        
        Returns:
            PromptComponents对象
        """
        components = PromptComponents()
        
        # 从scene中提取信息（如果提供）
        if scene:
            components.shot = scene.get('camera', {}).get('shot_type', '') or scene.get('visual', {}).get('composition', '')
            components.subject = scene.get('description', '') or scene.get('prompt', '')
            components.action = scene.get('motion', {}).get('type', '') or scene.get('action', '')
            components.environment = scene.get('environment', '') or scene.get('location', '')
            components.emotion = scene.get('mood', '') or scene.get('visual', {}).get('mood', '')
            components.lighting = scene.get('visual', {}).get('lighting', '')
            components.style = scene.get('style', '') or scene.get('visual', {}).get('style', '')
        
        # 如果从scene中提取的信息不足，尝试从user_input解析
        if not components.shot:
            components.shot = self._extract_shot_type(user_input)
        
        if not components.subject:
            components.subject = self._extract_subject(user_input)
        
        if not components.action:
            components.action = self._extract_action(user_input)
        
        if not components.environment:
            components.environment = self._extract_environment(user_input)
        
        # 设置默认值
        if not components.shot:
            components.shot = "wide shot, establishing"
        if not components.emotion:
            components.emotion = "calm but determined"
        
        return components
    
    def _extract_shot_type(self, text: str) -> str:
        """从文本中提取镜头类型"""
        text_lower = text.lower()
        for shot_type, keywords in self.shot_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return keywords[0]  # 返回第一个关键词
        return ""
    
    def _extract_subject(self, text: str) -> str:
        """从文本中提取主体"""
        # 简单的关键词匹配
        subjects = ["man", "woman", "person", "character", "scientist", "cat", "dog"]
        text_lower = text.lower()
        for subject in subjects:
            if subject in text_lower:
                return subject
        return ""
    
    def _extract_action(self, text: str) -> str:
        """从文本中提取动作"""
        text_lower = text.lower()
        for action in self.action_keywords:
            if action in text_lower:
                return f"{action} slowly" if "slow" in text_lower else action
        return ""
    
    def _extract_environment(self, text: str) -> str:
        """从文本中提取环境"""
        environments = ["field", "lab", "room", "street", "beach", "forest", "space"]
        text_lower = text.lower()
        for env in environments:
            if env in text_lower:
                return env
        return ""
    
    def to_prompt(self, components: PromptComponents) -> str:
        """
        将组件组合为完整的prompt
        
        Args:
            components: PromptComponents对象
        
        Returns:
            组合后的prompt字符串
        """
        parts = []
        
        if components.shot:
            parts.append(components.shot)
        if components.subject:
            parts.append(components.subject)
        if components.action:
            parts.append(components.action)
        if components.environment:
            parts.append(f"in {components.environment}")
        if components.emotion:
            parts.append(f"with {components.emotion} emotion")
        if components.fx:
            parts.append(components.fx)
        if components.style:
            parts.append(components.style)
        if components.camera:
            parts.append(components.camera)
        if components.lighting:
            parts.append(components.lighting)
        if components.composition:
            parts.append(components.composition)
        
        return ", ".join(parts)


class StyleController:
    """风格控制器 - 针对不同业务场景建立固定的提示词规范"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化风格控制器
        
        Args:
            config_path: 风格配置文件路径（可选）
        """
        self.styles = self._load_default_styles()
        
        if config_path and os.path.exists(config_path):
            self._load_styles_from_config(config_path)
    
    def _load_default_styles(self) -> Dict[str, Dict[str, Any]]:
        """加载默认风格模板"""
        return {
            "novel": {
                "description": "Cinematic scene, Chinese fantasy style",
                "keywords": [
                    "35mm lens", "dramatic backlight",
                    "hair and clothes fluttering in wind", "film texture",
                    "shallow depth of field", "cinematic color grading"
                ],
                "lighting": "dramatic rim light, soft ambient",
                "composition": "rule of thirds, dynamic framing"
            },
            "drama": {
                "description": "Cinematic scene, dramatic storytelling",
                "keywords": [
                    "35mm lens", "dramatic backlight",
                    "cinematic motion", "film texture",
                    "shallow depth of field"
                ],
                "lighting": "dramatic lighting, high contrast",
                "composition": "cinematic framing, leading lines"
            },
            "scientific": {
                "description": "High-tech scientific visualization",
                "keywords": [
                    "clean lighting", "realistic details",
                    "professional documentary look", "soft camera motion",
                    "authoritative tone", "precise visualization"
                ],
                "lighting": "clean, even lighting, soft shadows",
                "composition": "balanced, symmetrical composition"
            },
            "government": {
                "description": "Professional government presentation style",
                "keywords": [
                    "clean lighting", "realistic details",
                    "professional documentary look", "authoritative tone",
                    "high-quality visualization"
                ],
                "lighting": "professional studio lighting",
                "composition": "formal, balanced composition"
            },
            "enterprise": {
                "description": "Professional corporate style",
                "keywords": [
                    "clean lighting", "modern aesthetic",
                    "professional look", "sophisticated tone"
                ],
                "lighting": "professional lighting, soft shadows",
                "composition": "modern, clean composition"
            },
            "chinese_modern": {
                "description": "Chinese modern aesthetic",
                "keywords": [
                    "calm tone", "minimalistic elegance",
                    "cool color palette with warm highlights",
                    "symmetrical composition", "refined style"
                ],
                "lighting": "soft, natural lighting",
                "composition": "symmetrical, balanced composition"
            },
            "general": {
                "description": "High-quality video generation",
                "keywords": [
                    "cinematic", "high quality", "detailed",
                    "professional", "photorealistic"
                ],
                "lighting": "natural lighting",
                "composition": "balanced composition"
            }
        }
    
    def _load_styles_from_config(self, config_path: str):
        """从配置文件加载风格"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_styles = json.load(f)
                self.styles.update(custom_styles)
        except Exception as e:
            print(f"⚠ 加载风格配置失败: {e}")
    
    def get_style(self, scene_type: str) -> Dict[str, Any]:
        """
        获取指定场景类型的风格模板
        
        Args:
            scene_type: 场景类型
        
        Returns:
            风格配置字典
        """
        return self.styles.get(scene_type, self.styles["general"])
    
    def apply_style(
        self,
        base_prompt: str,
        scene_type: str,
        include_keywords: bool = True
    ) -> str:
        """
        将风格应用到prompt
        
        Args:
            base_prompt: 基础prompt
            scene_type: 场景类型
            include_keywords: 是否包含关键词
        
        Returns:
            应用风格后的prompt
        """
        style = self.get_style(scene_type)
        parts = [base_prompt]
        
        # 添加风格描述
        if style.get("description"):
            parts.append(style["description"])
        
        # 添加关键词
        if include_keywords and style.get("keywords"):
            parts.extend(style["keywords"][:3])  # 使用前3个关键词
        
        return ", ".join(parts)
    
    def get_lighting(self, scene_type: str) -> str:
        """获取场景类型对应的光线描述"""
        style = self.get_style(scene_type)
        return style.get("lighting", "natural lighting")
    
    def get_composition(self, scene_type: str) -> str:
        """获取场景类型对应的构图描述"""
        style = self.get_style(scene_type)
        return style.get("composition", "balanced composition")


class CameraEngine:
    """相机语言引擎 - 自动补充镜头描述词"""
    
    def __init__(self):
        """初始化相机引擎"""
        self.viewpoints = {
            "pov": "first-person POV",
            "third_person": "third-person view",
            "aerial": "aerial view",
            "eye_level": "eye-level shot"
        }
        
        self.shot_types = {
            "wide": "wide establishing shot",
            "medium": "medium shot",
            "close": "close-up shot",
            "extreme_close": "extreme close-up",
            "full": "full body shot"
        }
        
        self.movements = {
            "static": "static camera",
            "pan": "smooth camera pan",
            "tilt": "smooth camera tilt",
            "push_in": "slow camera dolly forward",
            "dolly_out": "slow camera dolly backward",
            "zoom_in": "gentle camera zoom in",
            "zoom_out": "gentle camera zoom out",
            "track": "smooth camera tracking movement",
            "orbit": "slow camera orbit around subject"
        }
        
        self.dof_types = {
            "shallow": "shallow depth of field",
            "deep": "deep depth of field",
            "bokeh": "beautiful bokeh background"
        }
        
        self.focal_lengths = {
            "wide": "24mm lens",
            "normal": "35mm lens",
            "portrait": "85mm lens",
            "telephoto": "135mm lens"
        }
    
    def generate_camera_prompt(
        self,
        shot_type: str = "wide",
        movement: str = "static",
        viewpoint: str = "third_person",
        dof: str = "shallow",
        focal_length: str = "normal"
    ) -> str:
        """
        生成相机语言prompt
        
        Args:
            shot_type: 镜头类型
            movement: 镜头运动
            viewpoint: 视角
            dof: 景深类型
            focal_length: 焦段
        
        Returns:
            相机语言prompt字符串
        """
        parts = []
        
        # 镜头类型
        if shot_type in self.shot_types:
            parts.append(self.shot_types[shot_type])
        
        # 镜头运动
        if movement in self.movements:
            parts.append(self.movements[movement])
        
        # 视角
        if viewpoint in self.viewpoints:
            parts.append(self.viewpoints[viewpoint])
        
        # 景深
        if dof in self.dof_types:
            parts.append(self.dof_types[dof])
        
        # 焦段
        if focal_length in self.focal_lengths:
            parts.append(self.focal_lengths[focal_length])
        
        return ", ".join(parts)
    
    def enhance_prompt_with_camera(
        self,
        base_prompt: str,
        camera_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        用相机语言增强prompt
        
        Args:
            base_prompt: 基础prompt
            camera_config: 相机配置字典
        
        Returns:
            增强后的prompt
        """
        if not camera_config:
            # 使用默认相机配置
            camera_prompt = self.generate_camera_prompt()
        else:
            camera_prompt = self.generate_camera_prompt(
                shot_type=camera_config.get("shot_type", "wide"),
                movement=camera_config.get("movement", "static"),
                viewpoint=camera_config.get("viewpoint", "third_person"),
                dof=camera_config.get("dof", "shallow"),
                focal_length=camera_config.get("focal_length", "normal")
            )
        
        return f"{base_prompt}, {camera_prompt}"


class NegativePromptGenerator:
    """反向提示词生成器 - 避免生成坏图/假脸/乱动"""
    
    def __init__(self):
        """初始化反向提示词生成器"""
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
            "svd": [
                "motion blur", "temporal inconsistency", "frame artifacts"
            ],
            "flux": [
                "bad anatomy", "distorted proportions", "unrealistic details"
            ]
        }
    
    def generate(
        self,
        model_type: str = "general",
        scene_type: Optional[str] = None,
        custom_negatives: Optional[List[str]] = None
    ) -> str:
        """
        生成反向提示词
        
        Args:
            model_type: 模型类型
            scene_type: 场景类型（可选）
            custom_negatives: 自定义负面词（可选）
        
        Returns:
            反向提示词字符串
        """
        negatives = self.base_negative.copy()
        
        # 添加模型特定的负面词
        if model_type in self.model_specific:
            negatives.extend(self.model_specific[model_type])
        
        # 添加场景特定的负面词
        if scene_type:
            scene_negatives = self._get_scene_negatives(scene_type)
            negatives.extend(scene_negatives)
        
        # 添加自定义负面词
        if custom_negatives:
            negatives.extend(custom_negatives)
        
        # 去重并组合
        unique_negatives = list(dict.fromkeys(negatives))  # 保持顺序的去重
        return ", ".join(unique_negatives)
    
    def _get_scene_negatives(self, scene_type: str) -> List[str]:
        """获取场景特定的负面词"""
        scene_negatives_map = {
            "scientific": ["unrealistic physics", "impossible phenomena"],
            "government": ["unprofessional", "casual", "informal"],
            "novel": ["modern elements", "anachronistic"],
            "drama": ["static", "boring", "lifeless"]
        }
        return scene_negatives_map.get(scene_type, [])


class PromptQA:
    """Prompt质量评分器 - 检查prompt是否缺少关键字段"""
    
    def __init__(self):
        """初始化Prompt QA"""
        self.required_fields = {
            "subject": ["subject", "character", "person", "object", "entity"],
            "action": ["action", "motion", "movement", "doing", "performing"],
            "environment": ["environment", "location", "setting", "scene", "place"],
            "composition": ["composition", "framing", "shot", "angle", "view"],
            "lighting": ["lighting", "light", "illumination", "brightness"],
            "style": ["style", "aesthetic", "look", "visual style"]
        }
        
        self.quality_keywords = [
            "high quality", "cinematic", "detailed", "professional",
            "photorealistic", "excellent", "masterpiece"
        ]
    
    def check(self, prompt: str) -> Dict[str, Any]:
        """
        检查prompt质量
        
        Args:
            prompt: 待检查的prompt
        
        Returns:
            检查结果字典
        """
        prompt_lower = prompt.lower()
        results = {
            "score": 0,
            "max_score": len(self.required_fields) + 1,  # +1 for quality keywords
            "missing_fields": [],
            "has_quality_keywords": False,
            "word_count": len(prompt.split()),
            "suggestions": []
        }
        
        # 检查必需字段
        for field, keywords in self.required_fields.items():
            found = any(keyword in prompt_lower for keyword in keywords)
            if found:
                results["score"] += 1
            else:
                results["missing_fields"].append(field)
        
        # 检查质量关键词
        has_quality = any(keyword in prompt_lower for keyword in self.quality_keywords)
        if has_quality:
            results["score"] += 1
            results["has_quality_keywords"] = True
        else:
            results["suggestions"].append("建议添加质量关键词（如：high quality, cinematic）")
        
        # 生成改进建议
        if results["missing_fields"]:
            for field in results["missing_fields"]:
                suggestions_map = {
                    "subject": "建议添加主体描述（如：a person, a character）",
                    "action": "建议添加动作描述（如：walking, moving）",
                    "environment": "建议添加环境描述（如：in a room, on a field）",
                    "composition": "建议添加构图描述（如：wide shot, close-up）",
                    "lighting": "建议添加光线描述（如：soft light, dramatic lighting）",
                    "style": "建议添加风格描述（如：cinematic, realistic）"
                }
                if field in suggestions_map:
                    results["suggestions"].append(suggestions_map[field])
        
        # 检查词数
        if results["word_count"] < 15:
            results["suggestions"].append("Prompt太短，建议至少15-20个词")
        elif results["word_count"] > 100:
            results["suggestions"].append("Prompt可能过长，建议精简到50-80个词")
        
        return results
    
    def auto_fix(
        self,
        prompt: str,
        scene: Optional[Dict[str, Any]] = None,
        style_controller: Optional[StyleController] = None,
        camera_engine: Optional[CameraEngine] = None
    ) -> str:
        """
        自动修复prompt，补充缺失字段
        
        Args:
            prompt: 原始prompt
            scene: 场景配置（可选）
            style_controller: 风格控制器（可选）
            camera_engine: 相机引擎（可选）
        
        Returns:
            修复后的prompt
        """
        check_result = self.check(prompt)
        
        if check_result["score"] == check_result["max_score"]:
            return prompt  # 已经完整，无需修复
        
        parts = [prompt]
        
        # 补充缺失字段
        missing = check_result["missing_fields"]
        
        # 补充构图
        if "composition" in missing:
            if camera_engine:
                camera_prompt = camera_engine.generate_camera_prompt()
                parts.append(camera_prompt)
            else:
                parts.append("wide shot, balanced composition")
        
        # 补充光线
        if "lighting" in missing:
            if scene and style_controller:
                scene_type = scene.get("type", "general")
                lighting = style_controller.get_lighting(scene_type)
                parts.append(lighting)
            else:
                parts.append("natural lighting")
        
        # 补充风格
        if "style" in missing:
            if scene and style_controller:
                scene_type = scene.get("type", "general")
                style = style_controller.get_style(scene_type)
                if style.get("description"):
                    parts.append(style["description"])
            else:
                parts.append("cinematic, high quality")
        
        # 补充质量关键词
        if not check_result["has_quality_keywords"]:
            parts.append("high quality, cinematic, detailed")
        
        return ", ".join(parts)


class PromptEngine:
    """Prompt Engine主类 - 整合所有模块"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_llm_rewriter: bool = False,
        llm_api: Optional[Any] = None
    ):
        """
        初始化Prompt Engine
        
        Args:
            config_path: 配置文件路径（可选）
            use_llm_rewriter: 是否使用LLM重写器
            llm_api: LLM API接口（可选）
        """
        self.rewriter = PromptRewriter(use_llm=use_llm_rewriter, llm_api=llm_api)
        self.decomposer = SceneDecomposer()
        self.style_controller = StyleController(config_path)
        self.camera_engine = CameraEngine()
        self.negative_generator = NegativePromptGenerator()
        self.qa = PromptQA()
    
    def process(
        self,
        user_input: str,
        scene: Optional[Dict[str, Any]] = None,
        model_type: str = "general",
        scene_type: str = "general",
        camera_config: Optional[Dict[str, Any]] = None,
        return_components: bool = False
    ) -> Dict[str, Any]:
        """
        完整的Prompt处理流程
        
        Args:
            user_input: 用户输入
            scene: 场景配置（可选）
            model_type: 模型类型（用于生成negative prompt）
            scene_type: 场景类型
            camera_config: 相机配置（可选）
            return_components: 是否返回组件结构
        
        Returns:
            处理结果字典，包含：
            - prompt: 最终prompt
            - negative_prompt: 反向提示词
            - components: PromptComponents（如果return_components=True）
            - qa_result: QA检查结果
        """
        # 1. Prompt Rewriter（语义增强 + 语法）
        rewritten = self.rewriter.rewrite(user_input, scene_type)
        
        # 2. Scene Decomposer（拆成结构）
        components = self.decomposer.decompose(rewritten, scene)
        
        # 3. Style Controller（按场景补风格词）
        base_prompt = self.decomposer.to_prompt(components)
        styled_prompt = self.style_controller.apply_style(
            base_prompt, scene_type, include_keywords=True
        )
        
        # 4. Camera Engine（加入镜头语言）
        if camera_config or not components.camera:
            final_prompt = self.camera_engine.enhance_prompt_with_camera(
                styled_prompt, camera_config
            )
        else:
            final_prompt = styled_prompt
        
        # 5. Negative Prompt Generator
        negative_prompt = self.negative_generator.generate(
            model_type=model_type,
            scene_type=scene_type
        )
        
        # 6. Prompt QA（检查与修复）
        qa_result = self.qa.check(final_prompt)
        if qa_result["score"] < qa_result["max_score"]:
            # 自动修复
            final_prompt = self.qa.auto_fix(
                final_prompt,
                scene=scene,
                style_controller=self.style_controller,
                camera_engine=self.camera_engine
            )
            # 重新检查
            qa_result = self.qa.check(final_prompt)
        
        result = {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "qa_result": qa_result
        }
        
        if return_components:
            result["components"] = components
        
        return result
    
    def quick_process(
        self,
        user_input: str,
        scene_type: str = "general",
        model_type: str = "general"
    ) -> Tuple[str, str]:
        """
        快速处理（只返回prompt和negative_prompt）
        
        Args:
            user_input: 用户输入
            scene_type: 场景类型
            model_type: 模型类型
        
        Returns:
            (prompt, negative_prompt) 元组
        """
        result = self.process(
            user_input=user_input,
            scene_type=scene_type,
            model_type=model_type
        )
        return result["prompt"], result["negative_prompt"]


if __name__ == "__main__":
    """测试Prompt Engine"""
    print("=" * 60)
    print("Prompt Engine 测试")
    print("=" * 60)
    
    # 创建引擎
    engine = PromptEngine()
    
    # 测试用例1：简单输入
    print("\n【测试1】简单用户输入")
    print("-" * 60)
    user_input = "一个男人在雪地里走路"
    result = engine.process(
        user_input=user_input,
        scene_type="novel",
        model_type="cogvideox"
    )
    print(f"输入: {user_input}")
    print(f"输出Prompt: {result['prompt']}")
    print(f"Negative Prompt: {result['negative_prompt'][:100]}...")
    print(f"QA评分: {result['qa_result']['score']}/{result['qa_result']['max_score']}")
    if result['qa_result']['suggestions']:
        print(f"建议: {', '.join(result['qa_result']['suggestions'])}")
    
    # 测试用例2：带场景配置
    print("\n【测试2】带场景配置")
    print("-" * 60)
    scene = {
        "type": "scientific",
        "description": "a black hole in space",
        "motion": {"type": "rotating"},
        "visual": {
            "lighting": "dramatic",
            "style": "scientific"
        }
    }
    result = engine.process(
        user_input="黑洞旋转",
        scene=scene,
        scene_type="scientific",
        model_type="hunyuanvideo"
    )
    print(f"输入: 黑洞旋转")
    print(f"输出Prompt: {result['prompt']}")
    print(f"QA评分: {result['qa_result']['score']}/{result['qa_result']['max_score']}")
    
    # 测试用例3：快速处理
    print("\n【测试3】快速处理")
    print("-" * 60)
    prompt, negative = engine.quick_process(
        "科学家在实验室工作",
        scene_type="scientific",
        model_type="hunyuanvideo"
    )
    print(f"Prompt: {prompt}")
    print(f"Negative: {negative[:80]}...")

