"""
Prompt 语义增强器（Semantic Enhancer）

基于语义节点进行增强，而不是字符串后处理。
这是语义层的增强组件，供策略层使用。
"""

from typing import List
from .semantic import PromptNode, PromptAST


class SemanticEnhancer:
    """
    语义增强器
    
    对 PromptNode 进行语义级增强，添加标记、调整内容等。
    所有增强都是基于语义理解，而不是字符串匹配。
    """
    
    def enhance_ast(self, ast: PromptAST) -> PromptAST:
        """
        增强整个 AST
        
        Args:
            ast: 输入的 PromptAST
            
        Returns:
            增强后的 PromptAST
        """
        result_ast = ast.copy()
        
        for node in result_ast.nodes:
            # 应用各种增强规则
            self._enhance_pose_disambiguation(node)
            self._enhance_sky_object_visibility(node)
            self._enhance_action_clarity(node)
        
        return result_ast
    
    def _enhance_pose_disambiguation(self, node: PromptNode) -> None:
        """
        增强姿势歧义消除
        
        检测水平姿势（lying, sitting等），添加排除词和标记。
        """
        if node.type not in ["action", "composition"]:
            return
        
        content_lower = node.content.lower()
        
        # 检测lying姿势（需要特别强调，避免被误判为sitting或standing）
        lying_keywords = ["lying", "lie", "躺", "卧", "prone", "supine"]
        has_lying = any(kw in content_lower for kw in lying_keywords)
        
        # 检测sitting姿势（需要排除，避免与lying混淆）
        sitting_keywords = ["sitting", "sit", "坐"]
        has_sitting = any(kw in content_lower for kw in sitting_keywords)
        
        if has_lying:
            # 添加语义标记
            node.tags.add("horizontal_pose")
            node.tags.add("lying_pose")
            
            # 检测是否已有排除词
            has_exclusion = any(kw in content_lower for kw in [
                "not standing", "not stand", "not sitting", "not sit", "排除站立", "非站立", "非坐"
            ])
            
            # ⚡ 修复：如果没有排除词，添加明确的排除词（排除standing和sitting）
            if not has_exclusion:
                node.content += ", NOT standing, NOT sitting, horizontal position, prone, supine"
                # 提高权重，确保lying被正确生成
                node.weight = max(node.weight, 2.8)
                node.priority += 3.0  # 提高优先级
        elif has_sitting:
            # sitting姿势也需要排除standing
            node.tags.add("horizontal_pose")
            has_exclusion = any(kw in content_lower for kw in [
                "not standing", "not stand", "排除站立", "非站立"
            ])
            if not has_exclusion:
                node.content += ", not standing"
    
    def _enhance_sky_object_visibility(self, node: PromptNode) -> None:
        """
        增强天空物体可见性
        
        检测天空中的关键物体（数量词+天体名词），增强可见性描述。
        """
        if node.type not in ["environment", "composition"]:
            return
        
        content_lower = node.content.lower()
        
        # 检测天体名词
        celestial_objects = ["sun", "suns", "moon", "moons", "lunar", "solar", "star", "stars",
                           "太阳", "月亮", "日", "月", "星", "星辰"]
        # 检测数量词
        quantity_words = ["three", "four", "five", "multiple", "several", "many",
                        "三", "四", "五", "多个", "数个", "许多"]
        # 检测可见性描述
        visibility_words = ["visible", "prominent", "clear", "distinct", "large", "bright",
                          "可见", "明显", "清晰", "突出", "大", "明亮"]
        
        has_celestial = any(obj in content_lower for obj in celestial_objects)
        has_quantity = any(qty in content_lower for qty in quantity_words)
        has_visibility = any(vis in content_lower for vis in visibility_words)
        
        # 如果包含天体物体和数量词，但缺少可见性描述，增强可见性
        if has_celestial and has_quantity and not has_visibility:
            # 添加语义标记
            node.tags.add("sky_object")
            node.tags.add("needs_visibility_enhancement")
            
            # 提高权重
            node.weight = max(node.weight, 2.5)
            
            # 根据物体类型添加相应的增强描述（语义级）
            if "sun" in content_lower or "solar" in content_lower or "日" in content_lower:
                node.content += ", large and prominent, clearly visible, bright and distinct, dominating the sky"
            elif "lunar" in content_lower or "moon" in content_lower or "月" in content_lower:
                node.content += ", clearly visible and distinguishable in the sky"
            else:
                node.content += ", large and prominent, clearly visible"
    
    def _enhance_action_clarity(self, node: PromptNode) -> None:
        """
        增强动作清晰度
        
        检测关键动作，确保不被忽略。
        """
        if node.type not in ["action", "composition"]:
            return
        
        content_lower = node.content.lower()
        
        # 检测关键动作（需要明确环境关系的）
        key_actions = ["lying on", "sitting on", "standing on", "lying in", "sitting in"]
        has_key_action = any(kw in content_lower for kw in key_actions)
        
        if has_key_action:
            node.tags.add("pose_sensitive")
            # 如果权重过低，提高权重
            if node.weight < 1.5:
                node.weight = max(node.weight, 1.8)

