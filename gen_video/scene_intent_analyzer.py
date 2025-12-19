#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景意图智能分析器（生产可用入口）

说明：
- `image_generator.py` / Prompt 系统依赖 `SceneIntentAnalyzer`
- 该实现历史上曾放在 `gen_video/archive/scripts/scene_intent_analyzer.py`
- 为避免运行期 `ModuleNotFoundError`，这里提供一个稳定的模块入口
"""

from __future__ import annotations

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
                "exclude_keywords": ["no character", "no person", "无人物", "无角色"],
            },
            "object": {
                "keywords": ["object", "item", "artifact", "scroll", "weapon", "物体", "物品", "道具"],
                "exclude_keywords": [],
            },
            "environment": {
                "keywords": [
                    "sky",
                    "ground",
                    "desert",
                    "forest",
                    "mountain",
                    "gravel",
                    "sand",
                    "sandstone",
                    "天空",
                    "地面",
                    "沙漠",
                    "森林",
                    "山",
                    "砂砾",
                    "沙砾",
                    "沙地",
                ],
                "exclude_keywords": [],
            },
        }

        # 通用动作类型
        self.action_patterns = {
            "static": ["still", "motionless", "静止", "不动", "一动不动"],
            "dynamic": ["move", "walk", "run", "attack", "cast", "移动", "奔跑", "攻击", "施法"],
            "object_motion": ["unfurl", "open", "rotate", "float", "rise", "展开", "打开", "旋转", "飘动", "上升"],
        }

        # 通用视角类型
        self.viewpoint_patterns = {
            "front": ["facing camera", "front view", "face forward", "正面", "面向镜头"],
            "back": ["back view", "from behind", "背影", "背后", "rear view"],
            "side": ["side view", "profile", "侧面", "侧身"],
            "top": ["top-down", "aerial", "俯视", "鸟瞰"],
            "close": ["close-up", "extreme close-up", "特写", "近景"],
            "wide": ["wide shot", "distant", "远景", "远距离"],
        }

    def analyze(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """分析场景意图（返回结构见旧版本注释）。"""
        if not scene:
            return self._empty_result()

        all_text = self._collect_all_text(scene)
        entities = self._analyze_entities(scene, all_text)
        primary_entity = self._identify_primary_entity(entities, all_text)
        action_type = self._analyze_action(scene, all_text)
        viewpoint = self._analyze_viewpoint(scene, all_text)
        emphasis = self._extract_emphasis(scene, all_text, primary_entity)
        exclusions = self._extract_exclusions(scene, all_text)
        prompt_priority = self._build_prompt_priority(scene, entities, primary_entity, action_type, viewpoint, emphasis, exclusions)
        weight_adjustments = self._calculate_weight_adjustments(entities, primary_entity, action_type, viewpoint, emphasis, all_text)

        return {
            "entities": entities,
            "primary_entity": primary_entity,
            "action_type": action_type,
            "viewpoint": viewpoint,
            "emphasis": emphasis,
            "exclusions": exclusions,
            "prompt_priority": prompt_priority,
            "weight_adjustments": weight_adjustments,
        }

    def _empty_result(self) -> Dict[str, Any]:
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
        texts: List[str] = []
        fields = ["description", "action", "prompt", "camera", "narration"]
        for field in fields:
            value = scene.get(field)
            if value:
                texts.append(str(value).lower())

        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            for field in ["composition", "environment", "character_pose", "fx"]:
                value = visual.get(field)
                if value:
                    texts.append(str(value).lower())
        return " ".join(texts)

    def _analyze_entities(self, scene: Dict[str, Any], all_text: str) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []

        # character
        has_character = False
        for keyword in self.entity_patterns["character"]["keywords"]:
            if keyword in all_text:
                excluded = any(ekw in all_text for ekw in self.entity_patterns["character"]["exclude_keywords"])
                if not excluded:
                    has_character = True
                    break
        if has_character:
            entities.append({"type": "character", "weight": 1.5, "keywords": self._extract_character_keywords(scene, all_text)})

        # object
        for keyword in self.entity_patterns["object"]["keywords"]:
            if keyword in all_text:
                entities.append({"type": "object", "name": keyword, "weight": 1.8, "keywords": [keyword]})
                break

        # environment
        for keyword in self.entity_patterns["environment"]["keywords"]:
            if keyword in all_text:
                entities.append({"type": "environment", "name": keyword, "weight": 1.2, "keywords": [keyword]})
                break

        return entities

    def _extract_character_keywords(self, scene: Dict[str, Any], all_text: str) -> List[str]:
        keywords: List[str] = []
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            character_pose = visual.get("character_pose", "")
            if character_pose:
                keywords.append(character_pose)

        description = scene.get("description", "")
        prompt = scene.get("prompt", "")
        for text in [description, prompt]:
            if text:
                sentences = re.split(r"[，,。.]", text)
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in ["character", "person", "角色", "人物"]):
                        keywords.append(sentence.strip())
        return keywords[:3]

    def _identify_primary_entity(self, entities: List[Dict[str, Any]], all_text: str) -> Optional[Dict[str, Any]]:
        if not entities:
            return None

        observation_keywords = [
            "sees",
            "see",
            "revealing",
            "reveals",
            "showing",
            "shows",
            "revealed",
            "只见",
            "映入眼帘",
            "展现",
            "露出",
            "显示",
            "透出",
        ]
        has_observation_keyword = any(kw in all_text for kw in observation_keywords)
        if has_observation_keyword:
            env_entity = next((e for e in entities if e.get("type") == "environment"), None)
            obj_entity = next((e for e in entities if e.get("type") == "object"), None)
            has_character = any(e.get("type") == "character" for e in entities)
            if has_character and (env_entity or obj_entity):
                if obj_entity:
                    return obj_entity
                if env_entity:
                    return env_entity

        primary = max(entities, key=lambda e: e.get("weight", 1.0))
        if len(entities) > 1:
            mention_counts: Dict[str, int] = {}
            for entity in entities:
                count = sum(all_text.count(kw) for kw in entity.get("keywords", []))
                mention_counts[str(entity.get("type", ""))] = count
            if mention_counts:
                most_mentioned = max(mention_counts.items(), key=lambda x: x[1])
                for entity in entities:
                    if entity.get("type") == most_mentioned[0]:
                        return entity
        return primary

    def _analyze_action(self, scene: Dict[str, Any], all_text: str) -> str:
        if any(kw in all_text for kw in self.action_patterns["static"]):
            return "static"
        if any(kw in all_text for kw in self.action_patterns["object_motion"]):
            return "object_motion"
        if any(kw in all_text for kw in self.action_patterns["dynamic"]):
            return "dynamic"
        return "static"

    def _analyze_viewpoint(self, scene: Dict[str, Any], all_text: str) -> Dict[str, Any]:
        viewpoint: Dict[str, Any] = {"type": "front", "weight": 1.0, "explicit": False}
        visual = scene.get("visual", {}) or {}
        if isinstance(visual, dict):
            character_pose = visual.get("character_pose", "")
            if character_pose and "facing camera" in character_pose.lower():
                viewpoint["type"] = "front"
                viewpoint["weight"] = 2.0
                viewpoint["explicit"] = True
                return viewpoint

        for vp_type, keywords in self.viewpoint_patterns.items():
            if any(kw in all_text for kw in keywords):
                viewpoint["type"] = vp_type
                if vp_type == "front":
                    if "facing camera" in all_text.lower():
                        viewpoint["weight"] = 2.0
                        viewpoint["explicit"] = True
                    else:
                        viewpoint["weight"] = 1.8
                elif vp_type == "back":
                    viewpoint["weight"] = 1.0
                elif vp_type in ["close", "wide"]:
                    viewpoint["weight"] = 1.5
                else:
                    viewpoint["weight"] = 1.2
                break
        return viewpoint

    def _extract_emphasis(self, scene: Dict[str, Any], all_text: str, primary_entity: Optional[Dict[str, Any]]) -> List[str]:
        emphasis: List[str] = []
        if primary_entity and primary_entity.get("type") == "object":
            emphasis.extend(primary_entity.get("keywords", []))

        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            composition = visual.get("composition", "")
            if composition:
                words = composition.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        emphasis.append(word)
        return emphasis[:5]

    def _extract_exclusions(self, scene: Dict[str, Any], all_text: str) -> List[str]:
        exclusions: List[str] = []
        exclude_patterns = ["no person", "no character", "no human", "无人物", "无角色", "无人类"]
        if any(pattern in all_text for pattern in exclude_patterns):
            exclusions.extend(["person", "character", "human", "人物", "角色", "人"])
        return list(set(exclusions))

    def _build_prompt_priority(
        self,
        scene: Dict[str, Any],
        entities: List[Dict[str, Any]],
        primary_entity: Optional[Dict[str, Any]],
        action_type: str,
        viewpoint: Dict[str, Any],
        emphasis: List[str],
        exclusions: List[str],
    ) -> List[Tuple[str, float]]:
        priority: List[Tuple[str, float]] = []
        priority.append(("xianxia fantasy", 1.0))
        if primary_entity:
            entity_text = " ".join(primary_entity.get("keywords", []))
            if entity_text:
                priority.append((entity_text, float(primary_entity.get("weight", 1.5))))
        if emphasis:
            priority.append((", ".join(emphasis), 1.8))
        if viewpoint.get("type") != "front" or float(viewpoint.get("weight", 1.0)) > 1.0:
            vp_keywords = self.viewpoint_patterns.get(str(viewpoint.get("type")), [])
            if vp_keywords:
                priority.append((vp_keywords[0], float(viewpoint.get("weight", 1.0))))
        description = scene.get("description", "")
        if description:
            priority.append((str(description), 1.3))
        visual = scene.get("visual", {})
        if isinstance(visual, dict):
            composition = visual.get("composition", "")
            if composition:
                priority.append((str(composition), 1.4))
        return priority

    def _calculate_weight_adjustments(
        self,
        entities: List[Dict[str, Any]],
        primary_entity: Optional[Dict[str, Any]],
        action_type: str,
        viewpoint: Dict[str, Any],
        emphasis: List[str],
        all_text: str,
    ) -> Dict[str, float]:
        adjustments = {
            "character_weight": 1.5,
            "viewpoint_weight": 1.0,
            "camera_weight": 1.3,
            "action_weight": 1.2,
            "composition_weight": 1.4,
            "entity_weight": 1.5,
        }

        if primary_entity:
            entity_type = primary_entity.get("type")
            base_weight = float(primary_entity.get("weight", 1.5))
            if entity_type == "object":
                adjustments["entity_weight"] = max(base_weight, 1.8)
                adjustments["character_weight"] = 0.0
            elif entity_type == "character":
                adjustments["character_weight"] = max(base_weight, 1.8)
                adjustments["entity_weight"] = max(base_weight, 1.7)
                if viewpoint.get("explicit") and viewpoint.get("type") == "front":
                    adjustments["character_weight"] = 2.0
                    adjustments["viewpoint_weight"] = max(float(viewpoint.get("weight", 1.0)), 2.0)
                elif viewpoint.get("type") == "front" and float(viewpoint.get("weight", 1.0)) >= 1.5:
                    adjustments["viewpoint_weight"] = max(float(viewpoint.get("weight", 1.0)), 1.8)
                elif viewpoint.get("type") == "back":
                    adjustments["character_weight"] = max(base_weight, 1.3)

        viewpoint_type = str(viewpoint.get("type", "front"))
        if viewpoint_type in ["close", "wide"]:
            adjustments["camera_weight"] = max(float(viewpoint.get("weight", 1.0)), 1.8)
        elif viewpoint_type == "front":
            adjustments["camera_weight"] = 1.5

        if action_type == "dynamic":
            adjustments["action_weight"] = 1.5
        elif action_type == "static":
            adjustments["action_weight"] = 1.0
            adjustments["composition_weight"] = 1.6
        elif action_type == "object_motion":
            adjustments["composition_weight"] = 1.8

        if len(emphasis) >= 3:
            adjustments["entity_weight"] = min(adjustments["entity_weight"] + 0.2, 2.0)
            adjustments["composition_weight"] = min(adjustments["composition_weight"] + 0.2, 2.0)
        elif len(emphasis) == 0 and primary_entity:
            adjustments["entity_weight"] = min(adjustments["entity_weight"] + 0.1, 1.8)

        text_length = len(all_text.split())
        if text_length > 50:
            for key in list(adjustments.keys()):
                if adjustments[key] > 1.0:
                    adjustments[key] = max(adjustments[key] - 0.1, 1.0)
        elif text_length < 20:
            if adjustments["entity_weight"] < 1.8:
                adjustments["entity_weight"] = min(adjustments["entity_weight"] + 0.2, 1.8)

        return adjustments


