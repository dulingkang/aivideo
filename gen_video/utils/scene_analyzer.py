#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能场景分析器
分析 prompt，自动识别场景需求（镜头类型、场景元素、动作等）
支持本地规则引擎和 LLM 两种模式
"""

import json
import re
import traceback
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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


@dataclass
class SceneAnalysisResult:
    """场景分析结果"""
    # 推荐的镜头类型
    recommended_shot_type: ShotType
    # 需要显示的元素
    needs_ground_visible: bool = False
    needs_environment_visible: bool = False
    needs_full_body: bool = False
    # 动作类型
    action_type: Optional[str] = None  # "lying", "sitting", "standing", "walking", etc.
    # 姿态类型（与 action_type 对应，但更精确）
    posture_type: Optional[str] = None  # "lying", "sitting", "kneeling", "crouching", "standing"
    # 姿态描述（正面，英文，用于 Flux）
    posture_positive: Optional[str] = None  # "lying on the ground, body fully reclined, ..."
    # 姿态负面提示词（英文）
    posture_negative: Optional[str] = None  # "standing pose, upright posture, ..."
    # 场景元素
    scene_elements: List[str] = None  # ["desert", "sand", "floor", etc.]
    # 需要增强的描述
    enhancement_descriptions: List[str] = None
    # 置信度（0-1）
    confidence: float = 0.8


class LocalSceneAnalyzer:
    """本地规则场景分析器（快速、免费）"""
    
    def __init__(self):
        """初始化本地规则引擎"""
        # 地面相关关键词
        self.ground_keywords = [
            "脚下", "地面", "floor", "ground", "desert floor", "沙地", "土地",
            "floor visible", "ground visible", "feet on", "foot on"
        ]
        
        # 环境相关关键词
        self.environment_keywords = [
            "desert", "沙漠", "landscape", "environment", "scenery", "background",
            "vast", "wide", "distant", "far", "horizon", "sky", "天空",
            "mountain", "山", "valley", "山谷", "forest", "森林"
        ]
        
        # 动作关键词映射
        self.action_keywords = {
            "lying": ["lying", "lie", "躺", "lying on", "lie on", "prone", "supine", "horizontal"],
            "sitting": ["sitting", "sit", "坐", "sitting on", "seated"],
            "kneeling": ["kneeling", "kneel", "跪", "on knees"],
            "crouching": ["crouching", "crouch", "蹲", "squatting"],
            "standing": ["standing", "stand", "站", "upright"],
            "walking": ["walking", "walk", "走", "striding"],
            "running": ["running", "run", "跑", "sprinting"]
        }
        
        # ⚡ 关键修复：动作上下文模式（需要组合判断）
        # 例如："保持不动" + "脚下" = "lying" 或 "sitting"
        self.action_context_patterns = {
            "lying": [
                # 模式：(保持不动/静止) + (脚下/地面/floor) = 躺
                (["保持不动", "静止", "不动", "motionless", "still"], ["脚下", "地面", "floor", "ground", "沙地"]),
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
        
        # 全身相关关键词
        self.full_body_keywords = [
            "full body", "全身", "whole body", "entire body",
            "feet visible", "legs visible", "完整身体"
        ]
    
    def analyze(self, prompt: str, current_shot_type: Optional[str] = None) -> SceneAnalysisResult:
        """
        分析 prompt，返回场景需求
        
        Args:
            prompt: 原始 prompt
            current_shot_type: 当前镜头类型（如果已指定）
        
        Returns:
            SceneAnalysisResult: 分析结果
        """
        prompt_lower = prompt.lower()
        
        # 1. 检测需要显示地面
        needs_ground = any(keyword in prompt_lower for keyword in self.ground_keywords)
        
        # 2. 检测需要显示环境
        needs_environment = any(keyword in prompt_lower for keyword in self.environment_keywords)
        
        # 3. 检测动作类型
        action_type = None
        
        # 3.1 先检查直接关键词匹配
        for action, keywords in self.action_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                action_type = action
                break
        
        # 3.2 如果没有直接匹配，检查上下文模式（组合判断）
        if action_type is None:
            for action, patterns in self.action_context_patterns.items():
                for pattern_group in patterns:
                    # pattern_group 是一个元组：(条件1列表, 条件2列表)
                    condition1_keywords, condition2_keywords = pattern_group
                    # 检查是否同时满足两个条件
                    has_condition1 = any(keyword in prompt_lower for keyword in condition1_keywords)
                    has_condition2 = any(keyword in prompt_lower for keyword in condition2_keywords)
                    
                    if has_condition1 and has_condition2:
                        action_type = action
                        logger.info(f"  检测到动作上下文模式: {action} (条件1: {condition1_keywords}, 条件2: {condition2_keywords})")
                        break
                
                if action_type is not None:
                    break
        
        # 4. 检测需要全身
        needs_full_body = (
            any(keyword in prompt_lower for keyword in self.full_body_keywords) or
            needs_ground or
            action_type in ["lying", "sitting", "kneeling", "crouching"]
        )
        
        # 5. 提取场景元素
        scene_elements = []
        if any(keyword in prompt_lower for keyword in ["desert", "沙漠", "sand", "沙"]):
            scene_elements.append("desert")
        if any(keyword in prompt_lower for keyword in ["mountain", "山", "peak", "山顶"]):
            scene_elements.append("mountain")
        
        # 6. 推荐镜头类型
        recommended_shot = self._recommend_shot_type(
            needs_ground=needs_ground,
            needs_environment=needs_environment,
            needs_full_body=needs_full_body,
            action_type=action_type,
            current_shot_type=current_shot_type
        )
        
        # 7. 生成增强描述
        # ⚡ 关键修复：Flux 不支持权重语法，使用自然语言描述
        # ⚡ 关键修复：传入推荐的镜头类型，避免与镜头类型描述重复
        enhancement_descriptions = self._generate_enhancements(
            needs_ground=needs_ground,
            needs_environment=needs_environment,
            action_type=action_type,
            scene_elements=scene_elements,
            recommended_shot_type=recommended_shot.value,  # 传入推荐的镜头类型
            use_flux=True  # 当前使用 Flux，不使用权重语法
        )
        
        # ⚡ 关键优化：如果检测到动作类型，尝试获取姿态描述
        posture_type = action_type  # 本地规则引擎中，posture_type 与 action_type 相同
        posture_positive = None
        posture_negative = None
        
        if action_type:
            try:
                from utils.posture_controller import PostureController
                posture_controller = PostureController()
                posture_prompt = posture_controller.get_posture_prompt(action_type, use_chinese=False)
                posture_positive = posture_prompt.get("positive")
                posture_negative = posture_prompt.get("negative")
            except ImportError:
                pass  # PostureController 不可用，保持为 None
        
        return SceneAnalysisResult(
            recommended_shot_type=recommended_shot,
            needs_ground_visible=needs_ground,
            needs_environment_visible=needs_environment,
            needs_full_body=needs_full_body,
            action_type=action_type,
            posture_type=posture_type,
            posture_positive=posture_positive,
            posture_negative=posture_negative,
            scene_elements=scene_elements,
            enhancement_descriptions=enhancement_descriptions,
            confidence=0.8  # 本地规则引擎的置信度
        )
    
    def _recommend_shot_type(
        self,
        needs_ground: bool,
        needs_environment: bool,
        needs_full_body: bool,
        action_type: Optional[str],
        current_shot_type: Optional[str]
    ) -> ShotType:
        """推荐镜头类型"""
        # 如果当前是近景/特写，但需要显示地面/环境，改为全身或远景
        if current_shot_type in ["close", "medium_close", "extreme_close"]:
            if needs_ground or needs_full_body:
                return ShotType.FULL
            elif needs_environment:
                return ShotType.WIDE
        
        # 如果需要显示环境，推荐远景
        if needs_environment and not needs_ground:
            return ShotType.WIDE
        
        # 如果需要显示地面或全身，推荐全身镜头
        if needs_ground or needs_full_body:
            return ShotType.FULL
        
        # 默认中景
        return ShotType.MEDIUM
    
    def _generate_enhancements(
        self,
        needs_ground: bool,
        needs_environment: bool,
        action_type: Optional[str],
        scene_elements: List[str],
        recommended_shot_type: Optional[str] = None,  # ⚡ 新增：推荐的镜头类型
        use_flux: bool = True  # ⚡ 新增：是否使用 Flux（Flux 不支持权重语法）
    ) -> List[str]:
        """
        生成增强描述
        
        Args:
            needs_ground: 是否需要显示地面
            needs_environment: 是否需要显示环境
            action_type: 动作类型
            scene_elements: 场景元素列表
            recommended_shot_type: 推荐的镜头类型（用于避免与镜头类型描述重复）
            use_flux: 是否使用 Flux（Flux 不支持权重语法，使用自然语言）
        """
        enhancements = []
        
        # ⚡ 关键修复：Flux 使用 T5 编码器，不支持权重语法 (xxx:1.5)
        # 使用自然语言描述，通过重复和位置来强调重要性
        
        # ⚡ 关键修复：Flux 支持中文，直接使用中文描述，不需要翻译成英文
        
        # ⚡ 关键修复：检查镜头类型是否已经包含地面描述，避免重复
        # 镜头类型描述映射（用于检查是否已包含地面描述）
        shot_type_ground_keywords = {
            "full": ["地面可见", "脚可见", "地面", "脚"],
            "wide": ["地面", "地面可见"],
            "extreme_wide": ["地面", "地面可见"]
        }
        
        # 检查推荐的镜头类型是否已经包含地面描述
        shot_has_ground = False
        if recommended_shot_type and recommended_shot_type in shot_type_ground_keywords:
            shot_has_ground = True  # full/wide/extreme_wide 镜头类型通常已经包含地面描述
        
        # 环境增强
        if needs_environment:
            if "desert" in scene_elements or "沙漠" in scene_elements:
                if use_flux:
                    # Flux：使用中文，更简洁，避免重复
                    # ⚡ 关键修复：避免与"地面"描述重复
                    enhancements.append("广阔的沙漠景观，沙丘，沙漠风景")
                else:
                    # SDXL/CLIP：支持权重语法
                    enhancements.append("(vast desert landscape, sand dunes, desert floor visible:2.0)")
            elif "mountain" in scene_elements or "山" in scene_elements:
                if use_flux:
                    enhancements.append("山景，广阔的风景，环境清晰可见，广阔的背景")
                else:
                    enhancements.append("(mountain landscape, vast scenery, environment visible:2.0)")
            else:
                if use_flux:
                    enhancements.append("景观可见，环境可见，背景清晰显示，广阔的风景")
                else:
                    enhancements.append("(landscape visible, environment visible, background visible:2.0)")
        
        # 地面增强
        # ⚡ 关键修复：如果镜头类型已经包含地面描述，就不需要再添加地面增强
        if needs_ground and not shot_has_ground:
            if use_flux:
                # Flux：使用中文，更简洁，避免重复
                # ⚡ 关键修复：检查是否与环境描述重复（环境描述可能已包含"沙漠地面"）
                ground_desc = "地面可见，脚可见"
                if enhancements:  # 如果已有环境描述
                    try:
                        from .prompt_deduplicator import is_duplicate
                        if not is_duplicate(ground_desc, enhancements, threshold=0.4):
                            enhancements.append(ground_desc)
                    except ImportError:
                        # 如果去重工具不可用，检查关键词
                        combined = " ".join(enhancements).lower()
                        if "地面" not in combined and "ground" not in combined:
                            enhancements.append(ground_desc)
                else:
                    enhancements.append(ground_desc)
            else:
                enhancements.append("(ground visible, floor visible, feet on ground, full body visible:2.0)")
        
        # 动作增强
        # ⚡ 关键修复：使用 PostureController 的模板（更精确的姿态描述）
        try:
            from utils.posture_controller import PostureController
            posture_controller = PostureController()
            
            if action_type:
                posture_prompt = posture_controller.get_posture_prompt(action_type, use_chinese=use_flux)
                if posture_prompt["positive"]:
                    # 姿态指令放在最前面（最高优先级）
                    enhancements.insert(0, posture_prompt["positive"])
                    logger.info(f"  ✓ 使用 PostureController 模板: {action_type}")
        except ImportError:
            # 回退到原有逻辑
            if action_type == "lying":
                if use_flux:
                    # Flux：使用中文，自然描述躺姿
                    enhancements.insert(0, "躺在地上，身体贴地，水平位置")
                else:
                    enhancements.insert(0, "(lying on ground, horizontal position, prone position, body fully on ground, back touching ground, legs extended flat, arms flat, not standing, not upright, not sitting:2.5)")
            elif action_type == "sitting":
                if use_flux:
                    enhancements.append("坐在地上，坐姿，全身可见，双腿弯曲")
                else:
                    enhancements.append("(sitting on ground, seated position, full body visible:2.0)")
            elif action_type == "kneeling":
                if use_flux:
                    enhancements.append("跪在地上，跪姿，全身可见")
                else:
                    enhancements.append("(kneeling on ground, on knees, full body visible:2.0)")
        
        return enhancements


class OpenAILLMClient:
    """OpenAI LLM 客户端（用于场景分析）"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = None):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: OpenAI API Key
            model: 模型名称（默认 gpt-4o-mini）
            base_url: API 基础 URL（可选，用于自定义端点）
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url if base_url else None
            )
            self.model = model
        except ImportError:
            raise ImportError("需要安装 openai 库: pip install openai")
    
    def analyze_scene(self, prompt: str) -> str:
        """
        调用 LLM 分析场景
        
        Args:
            prompt: 分析 prompt
        
        Returns:
            JSON 格式的分析结果
        """
        try:
            # 添加超时设置（30秒）
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的图像生成场景分析专家。请仔细分析提示词，返回准确的JSON格式结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度，提高准确性
                response_format={"type": "json_object"},  # 强制返回 JSON
                timeout=30.0  # 30秒超时
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM 返回空内容")
            return content
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            logger.debug(f"API 调用异常详情: {traceback.format_exc()}")
            raise


class LLMSceneAnalyzer:
    """LLM 场景分析器（更智能、灵活）"""
    
    def __init__(self, llm_client=None):
        """
        初始化 LLM 分析器
        
        Args:
            llm_client: LLM 客户端（需要实现 analyze_scene 方法）
        """
        self.llm_client = llm_client
    
    def analyze(self, prompt: str, current_shot_type: Optional[str] = None) -> SceneAnalysisResult:
        """
        使用 LLM 分析 prompt
        
        Args:
            prompt: 原始 prompt
            current_shot_type: 当前镜头类型
        
        Returns:
            SceneAnalysisResult: 分析结果
        """
        if not self.llm_client:
            raise ValueError("LLM client 未配置")
        
        # 构建分析 prompt
        # ⚡ 工程级优化：精简 prompt，移除硬规则，LLM 只做语义理解
        # 规则判断下沉到 PostureController 和 Execution Planner
        analysis_prompt = f"""你是图像生成场景分析专家。
请分析给定提示词的「动作、姿态、镜头需求」，并结构化输出。

提示词：{prompt}
当前镜头类型：{current_shot_type or "未指定"}

请只返回 JSON（不要其他内容）：

{{
    "recommended_shot_type": "extreme_wide|wide|full|american|medium|medium_close|close|extreme_close",
    "needs_ground_visible": true/false,
    "needs_environment_visible": true/false,
    "action_type": "lying|sitting|standing|walking|running|null",
    "posture_type": "lying|sitting|kneeling|crouching|standing|null",
    "posture_positive": "英文姿态描述，用于生成模型",
    "posture_negative": "英文负面姿态描述，用于排除不需要的姿态",
    "scene_elements": ["desert", "mountain", ...],
    "confidence": 0.0-1.0
}}

分析原则：
1. 根据语义推断真实姿态，而不是默认站立
2. 若人物与地面产生直接接触（躺、坐、跪等），应优先考虑相应的姿态类型
3. posture_positive / negative 使用英文，便于生成模型理解
4. 推荐镜头应支持姿态与环境同时可见（如果需要）
5. 如果无法确定姿态，posture_type 设为 null，posture_positive/negative 设为空字符串"""
        
        try:
            # 调用 LLM
            logger.debug(f"调用 LLM 分析场景，prompt 长度: {len(analysis_prompt)}")
            response = self.llm_client.analyze_scene(analysis_prompt)
            logger.debug(f"LLM 返回响应，长度: {len(response) if response else 0}")
            
            if not response:
                raise ValueError("LLM 返回空响应")
            
            # ⚡ 关键修复：解析 JSON 响应，处理可能的 markdown 代码块包裹
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            logger.debug(f"清理后的响应前200字符: {response_clean[:200]}")
            
            try:
                result_dict = json.loads(response_clean)
                logger.info(f"  ✓ LLM JSON 解析成功，字段: {list(result_dict.keys())}")
                logger.debug(f"  [DEBUG] LLM 返回的完整结果: {json.dumps(result_dict, ensure_ascii=False, indent=2)[:500]}")
            except json.JSONDecodeError as je:
                logger.error(f"JSON 解析失败: {je}")
                logger.error(f"响应内容: {response_clean[:500]}")
                raise
            
            # ⚡ 工程级优化：LLM 只返回语义理解，规则判断下沉到代码层
            # 如果 LLM 没有返回姿态描述，使用 PostureController 生成
            posture_type = result_dict.get("posture_type")
            posture_positive = result_dict.get("posture_positive", "")
            posture_negative = result_dict.get("posture_negative", "")
            
            # 如果 LLM 返回了 posture_type 但没有描述，使用模板生成
            if posture_type and not posture_positive:
                try:
                    from utils.posture_controller import PostureController
                    posture_controller = PostureController()
                    posture_prompt = posture_controller.get_posture_prompt(posture_type, use_chinese=False)
                    posture_positive = posture_prompt.get("positive", "")
                    posture_negative = posture_prompt.get("negative", "")
                    logger.debug(f"  使用 PostureController 模板补充姿态描述: {posture_type}")
                except ImportError:
                    pass
            
            # 转换为 SceneAnalysisResult
            # ⚡ 注意：needs_full_body 和 enhancement_descriptions 由 Execution Planner 根据规则决定
            result = SceneAnalysisResult(
                recommended_shot_type=ShotType(result_dict.get("recommended_shot_type", "medium")),
                needs_ground_visible=result_dict.get("needs_ground_visible", False),
                needs_environment_visible=result_dict.get("needs_environment_visible", False),
                needs_full_body=False,  # 由 Execution Planner 根据姿态决策表决定
                action_type=result_dict.get("action_type"),
                posture_type=posture_type,
                posture_positive=posture_positive,
                posture_negative=posture_negative,
                scene_elements=result_dict.get("scene_elements", []),
                enhancement_descriptions=[],  # 由 Execution Planner 生成，不在 LLM 中处理
                confidence=result_dict.get("confidence", 0.8)
            )
            
            logger.info(f"  ✓ LLM 场景分析完成: posture_type={posture_type}, shot_type={result.recommended_shot_type.value}")
            return result
        except Exception as e:
            logger.error(f"LLM 分析失败: {e}，回退到本地规则引擎")
            logger.debug(f"LLM 分析异常详情: {traceback.format_exc()}")
            # 回退到本地规则引擎
            local_analyzer = LocalSceneAnalyzer()
            return local_analyzer.analyze(prompt, current_shot_type)


class HybridSceneAnalyzer:
    """混合场景分析器（本地规则 + LLM）"""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        """
        初始化混合分析器
        
        Args:
            use_llm: 是否使用 LLM（默认 False，使用本地规则）
            llm_client: LLM 客户端（如果 use_llm=True）
        """
        self.use_llm = use_llm
        self.local_analyzer = LocalSceneAnalyzer()
        self.llm_analyzer = LLMSceneAnalyzer(llm_client) if use_llm and llm_client else None
    
    def analyze(self, prompt: str, current_shot_type: Optional[str] = None) -> SceneAnalysisResult:
        """
        分析 prompt（优先使用 LLM，失败时回退到本地规则）
        
        Args:
            prompt: 原始 prompt
            current_shot_type: 当前镜头类型
        
        Returns:
            SceneAnalysisResult: 分析结果
        """
        if self.use_llm and self.llm_analyzer:
            try:
                return self.llm_analyzer.analyze(prompt, current_shot_type)
            except Exception as e:
                logger.warning(f"LLM 分析失败，回退到本地规则: {e}")
        
        # 使用本地规则引擎
        return self.local_analyzer.analyze(prompt, current_shot_type)


# 便捷函数
def analyze_scene(
    prompt: str,
    current_shot_type: Optional[str] = None,
    use_llm: bool = False,
    llm_client=None
) -> SceneAnalysisResult:
    """
    分析场景需求（便捷函数）
    
    Args:
        prompt: 原始 prompt
        current_shot_type: 当前镜头类型
        use_llm: 是否使用 LLM
        llm_client: LLM 客户端
    
    Returns:
        SceneAnalysisResult: 分析结果
    """
    analyzer = HybridSceneAnalyzer(use_llm=use_llm, llm_client=llm_client)
    return analyzer.analyze(prompt, current_shot_type)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试本地规则引擎
    analyzer = LocalSceneAnalyzer()
    
    test_prompts = [
        "韩立, Gray-green desert floor, 韩立保持不动，静静体会脚下炽热的沙地。",
        "韩立站在山顶，远眺群山",
        "韩立躺在沙漠中，仰望星空",
        "韩立坐在石头上，沉思",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        result = analyzer.analyze(prompt)
        print(f"推荐镜头类型: {result.recommended_shot_type.value}")
        print(f"需要显示地面: {result.needs_ground_visible}")
        print(f"需要显示环境: {result.needs_environment_visible}")
        print(f"需要全身: {result.needs_full_body}")
        print(f"动作类型: {result.action_type}")
        print(f"场景元素: {result.scene_elements}")
        print(f"增强描述: {result.enhancement_descriptions}")

