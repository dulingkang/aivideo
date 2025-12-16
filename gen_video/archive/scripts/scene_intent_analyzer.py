#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景意图智能分析器
通用分析场景描述，提取用户意图，不依赖特殊规则
"""

from typing import Dict, Any, Optional, List, Tuple
import re


class SceneIntentAnalyzer:
    """场景意图智能分析器"""
    
    def __init__(self):
        """初始化分析器"""
        # 通用实体类型（不针对特定角色或物体）
        self.entity_patterns = {
            "character": {
                "keywords": ["character", "person", "figure", "man", "woman", "角色", "人物", "人"],
                "exclude_keywords": ["no character", "no person", "无人物", "无角色"]
            },
            "object": {
                "keywords": ["object", "item", "artifact", "scroll", "weapon", "物体", "物品", "道具"],
                "exclude_keywords": []
            },
            "environment": {
                "keywords": ["sky", "ground", "desert", "forest", "mountain", "gravel", "sand", "sandstone", "天空", "地面", "沙漠", "森林", "山", "砂砾", "沙砾", "沙地"],
                "exclude_keywords": []
            }
        }
        
        # 通用动作类型
        self.action_patterns = {
            "static": ["still", "motionless", "静止", "不动", "一动不动"],
            "dynamic": ["move", "walk", "run", "attack", "cast", "移动", "奔跑", "攻击", "施法"],
            "object_motion": ["unfurl", "open", "rotate", "float", "rise", "展开", "打开", "旋转", "飘动", "上升"]
        }
        
        # 通用视角类型
        self.viewpoint_patterns = {
            "front": ["facing camera", "front view", "face forward", "正面", "面向镜头"],
            "back": ["back view", "from behind", "背影", "背后", "rear view"],
            "side": ["side view", "profile", "侧面", "侧身"],
            "top": ["top-down", "aerial", "俯视", "鸟瞰"],
            "close": ["close-up", "extreme close-up", "特写", "近景"],
            "wide": ["wide shot", "distant", "远景", "远距离"]
        }
    
    def analyze(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析场景意图
        
        Args:
            scene: 场景JSON数据
            
        Returns:
            {
                "entities": List[Dict],  # 检测到的实体（角色、物体、环境）
                "primary_entity": Optional[Dict],  # 主要实体
                "action_type": str,  # 动作类型（static/dynamic/object_motion）
                "viewpoint": Dict,  # 视角信息
                "emphasis": List[str],  # 需要强调的关键词
                "exclusions": List[str],  # 需要排除的关键词
                "prompt_priority": List[Tuple[str, float]],  # Prompt优先级列表 (text, weight)
            }
        """
        if not scene:
            return self._empty_result()
        
        # 收集所有文本
        all_text = self._collect_all_text(scene)
        
        # 分析实体
        entities = self._analyze_entities(scene, all_text)
        primary_entity = self._identify_primary_entity(entities, all_text)
        
        # 分析动作
        action_type = self._analyze_action(scene, all_text)
        
        # 分析视角
        viewpoint = self._analyze_viewpoint(scene, all_text)
        
        # 提取强调项
        emphasis = self._extract_emphasis(scene, all_text, primary_entity)
        
        # 提取排除项
        exclusions = self._extract_exclusions(scene, all_text)
        
        # 构建Prompt优先级
        prompt_priority = self._build_prompt_priority(
            scene, entities, primary_entity, action_type, viewpoint, emphasis, exclusions
        )
        
        # 综合权重计算（智能分析）
        weight_adjustments = self._calculate_weight_adjustments(
            entities, primary_entity, action_type, viewpoint, emphasis, all_text
        )
        
        return {
            "entities": entities,
            "primary_entity": primary_entity,
            "action_type": action_type,
            "viewpoint": viewpoint,
            "emphasis": emphasis,
            "exclusions": exclusions,
            "prompt_priority": prompt_priority,
            "weight_adjustments": weight_adjustments,  # 新增：综合权重调整建议
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            "entities": [],
            "primary_entity": None,
            "action_type": "static",
            "viewpoint": {"type": "front", "weight": 1.0},
            "emphasis": [],
            "exclusions": [],
            "prompt_priority": [],
        }
    
    def _collect_all_text(self, scene: Dict[str, Any]) -> str:
        """收集场景中的所有文本"""
        texts = []
        
        # 收集各个字段
        fields = ["description", "action", "prompt", "camera", "narration"]
        for field in fields:
            value = scene.get(field)
            if value:
                texts.append(str(value).lower())
        
        # 收集visual字段
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            for field in ["composition", "environment", "character_pose", "fx"]:
                value = visual.get(field)
                if value:
                    texts.append(str(value).lower())
        
        return " ".join(texts)
    
    def _analyze_entities(self, scene: Dict[str, Any], all_text: str) -> List[Dict[str, Any]]:
        """分析场景中的实体"""
        entities = []
        
        # 检查是否有角色
        has_character = False
        for keyword in self.entity_patterns["character"]["keywords"]:
            if keyword in all_text:
                # 检查排除关键词
                excluded = any(ekw in all_text for ekw in self.entity_patterns["character"]["exclude_keywords"])
                if not excluded:
                    has_character = True
                    break
        
        if has_character:
            entities.append({
                "type": "character",
                "weight": 1.5,
                "keywords": self._extract_character_keywords(scene, all_text)
            })
        
        # 检查是否有物体
        for keyword in self.entity_patterns["object"]["keywords"]:
            if keyword in all_text:
                entities.append({
                    "type": "object",
                    "name": keyword,
                    "weight": 1.8,  # 物体通常需要更高权重
                    "keywords": [keyword]
                })
                break
        
        # 检查环境
        for keyword in self.entity_patterns["environment"]["keywords"]:
            if keyword in all_text:
                entities.append({
                    "type": "environment",
                    "name": keyword,
                    "weight": 1.2,
                    "keywords": [keyword]
                })
                break
        
        return entities
    
    def _extract_character_keywords(self, scene: Dict[str, Any], all_text: str) -> List[str]:
        """提取角色关键词"""
        keywords = []
        
        # 从character_pose提取
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            character_pose = visual.get("character_pose", "")
            if character_pose:
                keywords.append(character_pose)
        
        # 从description/prompt中提取角色描述
        description = scene.get("description", "")
        prompt = scene.get("prompt", "")
        
        # 提取包含角色的句子
        for text in [description, prompt]:
            if text:
                # 简单提取：包含"角色"、"人物"等关键词的短语
                sentences = re.split(r'[，,。.]', text)
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in ["character", "person", "角色", "人物"]):
                        keywords.append(sentence.strip())
        
        return keywords[:3]  # 最多返回3个关键词
    
    def _identify_primary_entity(self, entities: List[Dict], all_text: str) -> Optional[Dict]:
        """识别主要实体"""
        if not entities:
            return None
        
        # 特殊处理：检测"以环境/物体为主，人物为辅"的场景
        # 关键词：sees, revealing, showing, 只见, 映入眼帘, 展现
        observation_keywords = [
            "sees", "see", "revealing", "reveals", "showing", "shows", "revealed",
            "只见", "映入眼帘", "展现", "露出", "显示", "透出"
        ]
        has_observation_keyword = any(kw in all_text for kw in observation_keywords)
        
        # 如果检测到观察关键词，且后面跟着环境/物体描述，优先选择环境/物体
        if has_observation_keyword:
            # 查找环境实体
            env_entity = next((e for e in entities if e.get("type") == "environment"), None)
            # 查找物体实体
            obj_entity = next((e for e in entities if e.get("type") == "object"), None)
            
            # 如果同时有角色、环境/物体，且检测到观察关键词，优先选择环境/物体
            has_character = any(e.get("type") == "character" for e in entities)
            if has_character and (env_entity or obj_entity):
                # 优先选择物体（如果存在），否则选择环境
                if obj_entity:
                    print(f"  ℹ 检测到观察场景（人物观察物体），主要实体：物体")
                    return obj_entity
                elif env_entity:
                    print(f"  ℹ 检测到观察场景（人物观察环境），主要实体：环境")
                    return env_entity
        
        # 优先选择权重最高的实体
        primary = max(entities, key=lambda e: e.get("weight", 1.0))
        
        # 如果有多实体，检查文本中的提及频率
        if len(entities) > 1:
            mention_counts = {}
            for entity in entities:
                count = sum(all_text.count(kw) for kw in entity.get("keywords", []))
                mention_counts[entity["type"]] = count
            
            # 选择提及最多的
            if mention_counts:
                most_mentioned = max(mention_counts.items(), key=lambda x: x[1])
                for entity in entities:
                    if entity["type"] == most_mentioned[0]:
                        return entity
        
        return primary
    
    def _analyze_action(self, scene: Dict[str, Any], all_text: str) -> str:
        """分析动作类型"""
        # 检查静态动作
        if any(kw in all_text for kw in self.action_patterns["static"]):
            return "static"
        
        # 检查物体运动
        if any(kw in all_text for kw in self.action_patterns["object_motion"]):
            return "object_motion"
        
        # 检查动态动作
        if any(kw in all_text for kw in self.action_patterns["dynamic"]):
            return "dynamic"
        
        return "static"  # 默认静态
    
    def _analyze_viewpoint(self, scene: Dict[str, Any], all_text: str) -> Dict[str, Any]:
        """分析视角"""
        viewpoint = {"type": "front", "weight": 1.0, "explicit": False}
        
        # 检查character_pose中是否有明确的"facing camera"
        visual = scene.get("visual", {}) or {}
        if isinstance(visual, dict):
            character_pose = visual.get("character_pose", "")
            if character_pose and "facing camera" in character_pose.lower():
                viewpoint["type"] = "front"
                viewpoint["weight"] = 2.0  # 明确要求时提高到2.0
                viewpoint["explicit"] = True
                return viewpoint
        
        # 检查各个视角类型
        for vp_type, keywords in self.viewpoint_patterns.items():
            if any(kw in all_text for kw in keywords):
                viewpoint["type"] = vp_type
                # 根据视角类型设置权重
                if vp_type == "front":
                    # 如果明确包含"facing camera"，进一步提高权重
                    if "facing camera" in all_text.lower():
                        viewpoint["weight"] = 2.0
                        viewpoint["explicit"] = True
                    else:
                        viewpoint["weight"] = 1.8
                elif vp_type == "back":
                    viewpoint["weight"] = 1.0  # 如果明确要求背影，不增加权重
                elif vp_type in ["close", "wide"]:
                    viewpoint["weight"] = 1.5
                else:
                    viewpoint["weight"] = 1.2
                break
        
        return viewpoint
    
    def _extract_emphasis(self, scene: Dict[str, Any], all_text: str, primary_entity: Optional[Dict]) -> List[str]:
        """提取需要强调的关键词"""
        emphasis = []
        
        # 如果主要实体是物体，强调物体
        if primary_entity and primary_entity.get("type") == "object":
            emphasis.extend(primary_entity.get("keywords", []))
        
        # 从composition中提取关键物体
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            composition = visual.get("composition", "")
            if composition:
                # 提取名词短语（简单实现）
                words = composition.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        emphasis.append(word)
        
        return emphasis[:5]  # 最多5个强调项
    
    def _extract_exclusions(self, scene: Dict[str, Any], all_text: str) -> List[str]:
        """提取需要排除的关键词"""
        exclusions = []
        
        # 检查是否有"no person"、"no character"等排除描述
        exclude_patterns = [
            "no person", "no character", "no human", "无人物", "无角色", "无人类"
        ]
        
        for pattern in exclude_patterns:
            if pattern in all_text:
                exclusions.extend(["person", "character", "human", "人物", "角色", "人"])
                break
        
        # 如果主要实体是物体，排除角色
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            composition = visual.get("composition", "")
            if composition and "no person" in composition.lower():
                exclusions.extend(["person", "character", "human"])
        
        return list(set(exclusions))  # 去重
    
    def _build_prompt_priority(
        self,
        scene: Dict[str, Any],
        entities: List[Dict],
        primary_entity: Optional[Dict],
        action_type: str,
        viewpoint: Dict,
        emphasis: List[str],
        exclusions: List[str]
    ) -> List[Tuple[str, float]]:
        """构建Prompt优先级列表"""
        priority = []
        
        # 1. 风格（固定）
        priority.append(("xianxia fantasy", 1.0))
        
        # 2. 主要实体（如果有）
        if primary_entity:
            entity_text = " ".join(primary_entity.get("keywords", []))
            if entity_text:
                priority.append((entity_text, primary_entity.get("weight", 1.5)))
        
        # 3. 强调项
        if emphasis:
            emphasis_text = ", ".join(emphasis)
            priority.append((emphasis_text, 1.8))
        
        # 4. 视角
        if viewpoint["type"] != "front" or viewpoint["weight"] > 1.0:
            vp_keywords = self.viewpoint_patterns.get(viewpoint["type"], [])
            if vp_keywords:
                priority.append((vp_keywords[0], viewpoint["weight"]))
        
        # 5. 动作描述
        description = scene.get("description", "")
        if description:
            priority.append((description, 1.3))
        
        # 6. 构图描述
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            composition = visual.get("composition", "")
            if composition:
                priority.append((composition, 1.4))
        
        return priority
    
    def _calculate_weight_adjustments(
        self,
        entities: List[Dict],
        primary_entity: Optional[Dict],
        action_type: str,
        viewpoint: Dict,
        emphasis: List[str],
        all_text: str
    ) -> Dict[str, float]:
        """
        综合计算权重调整建议（智能分析）
        
        考虑因素：
        1. 主要实体的重要性
        2. 视角类型和明确程度
        3. 动作类型（静态/动态）
        4. 强调项的数量和重要性
        5. 场景复杂度
        
        Returns:
            {
                "character_weight": float,  # 角色描述权重
                "viewpoint_weight": float,  # 视角权重
                "camera_weight": float,    # 镜头类型权重
                "action_weight": float,    # 动作描述权重
                "composition_weight": float,  # 构图描述权重
                "entity_weight": float,   # 主要实体权重
            }
        """
        adjustments = {
            "character_weight": 1.5,  # 默认角色权重
            "viewpoint_weight": 1.0,   # 默认视角权重
            "camera_weight": 1.3,     # 默认镜头权重
            "action_weight": 1.2,      # 默认动作权重
            "composition_weight": 1.4, # 默认构图权重
            "entity_weight": 1.5,      # 默认实体权重
        }
        
        # 1. 根据主要实体类型调整权重
        if primary_entity:
            entity_type = primary_entity.get('type')
            base_weight = primary_entity.get('weight', 1.5)
            
            if entity_type == 'object':
                # 物体场景：提高物体权重，降低角色权重
                adjustments["entity_weight"] = max(base_weight, 1.8)
                adjustments["character_weight"] = 0.0  # 无角色
            elif entity_type == 'character':
                # 角色场景：提高角色权重，确保角色可见
                adjustments["character_weight"] = max(base_weight, 1.8)  # 从1.7提高到1.8
                adjustments["entity_weight"] = max(base_weight, 1.7)
                # 如果视角明确，进一步提高角色和视角权重
                if viewpoint.get('explicit') and viewpoint['type'] == 'front':
                    adjustments["character_weight"] = 2.0
                    adjustments["viewpoint_weight"] = max(viewpoint.get('weight', 1.0), 2.0)
                elif viewpoint.get('type') == 'front' and viewpoint.get('weight', 1.0) >= 1.5:
                    # 明确要求正面：提高角色和视角权重
                    adjustments["viewpoint_weight"] = max(viewpoint.get('weight', 1.0), 1.8)
                elif viewpoint.get('type') == 'back':
                    # 明确要求背面：降低角色权重
                    adjustments["character_weight"] = max(base_weight, 1.3)
        
        # 2. 根据视角类型调整镜头权重
        viewpoint_type = viewpoint.get('type', 'front')
        if viewpoint_type in ['close', 'wide']:
            # 特写或远景：提高镜头权重
            adjustments["camera_weight"] = max(viewpoint.get('weight', 1.0), 1.8)
        elif viewpoint_type == 'front':
            # 正面视角：中等镜头权重
            adjustments["camera_weight"] = 1.5
        
        # 3. 根据动作类型调整动作权重
        if action_type == 'dynamic':
            # 动态动作：提高动作权重
            adjustments["action_weight"] = 1.5
        elif action_type == 'static':
            # 静态动作：降低动作权重，提高构图权重
            adjustments["action_weight"] = 1.0
            adjustments["composition_weight"] = 1.6
        elif action_type == 'object_motion':
            # 物体运动：提高构图权重（物体是构图的一部分）
            adjustments["composition_weight"] = 1.8
        
        # 4. 根据强调项数量调整权重
        if len(emphasis) >= 3:
            # 多个强调项：提高实体和构图权重
            adjustments["entity_weight"] = min(adjustments["entity_weight"] + 0.2, 2.0)
            adjustments["composition_weight"] = min(adjustments["composition_weight"] + 0.2, 2.0)
        elif len(emphasis) == 0 and primary_entity:
            # 无强调项但有主要实体：适度提高实体权重
            adjustments["entity_weight"] = min(adjustments["entity_weight"] + 0.1, 1.8)
        
        # 5. 根据场景复杂度调整（简单启发式：文本长度）
        text_length = len(all_text.split())
        if text_length > 50:
            # 复杂场景：适度降低各权重，避免冲突
            for key in adjustments:
                if adjustments[key] > 1.0:
                    adjustments[key] = max(adjustments[key] - 0.1, 1.0)
        elif text_length < 20:
            # 简单场景：适度提高关键权重
            if adjustments["entity_weight"] < 1.8:
                adjustments["entity_weight"] = min(adjustments["entity_weight"] + 0.2, 1.8)
        
        return adjustments

