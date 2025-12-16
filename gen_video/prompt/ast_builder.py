"""
Prompt AST Builder

将字符串解析为 PromptNode AST，这是从字符串到语义结构的转换层。
"""

from typing import List, Optional
import re
from .semantic import PromptNode, PromptAST
from .token_estimator import TokenEstimator
from .semantic_patterns import get_pattern_registry


class ASTBuilder:
    """
    AST 构建器
    
    将字符串 prompt parts 解析为 PromptNode AST。
    """
    
    def __init__(self, token_estimator: Optional[TokenEstimator] = None, pattern_registry=None):
        """
        初始化 AST 构建器
        
        Args:
            token_estimator: Token 估算器（可选，用于计算优先级）
            pattern_registry: 语义模式注册表（可选，默认使用全局实例）
        """
        self.token_estimator = token_estimator
        self.pattern_registry = pattern_registry or get_pattern_registry()
    
    def parse_part(self, part: str, index: int = 0) -> PromptNode:
        """
        解析单个 prompt part 为 PromptNode
        
        Args:
            part: prompt 部分字符串，如 "(Han Li lying on sand:1.5)"
            index: 原始索引（用于保持顺序）
            
        Returns:
            PromptNode 对象
        """
        # 提取权重（如果有）
        weight_match = re.search(r':(\d+\.?\d*)', part)
        weight = float(weight_match.group(1)) if weight_match else 1.0
        
        # 提取内容（移除权重和括号）
        content = re.sub(r'^\(|\)$', '', part)
        content = re.sub(r':\d+\.?\d*\)?$', '', content).strip()
        
        # 推断类型
        node_type = self._infer_node_type(content)
        
        # 计算基础优先级（基于类型）
        base_priority = self._get_base_priority(node_type)
        
        # 根据权重调整优先级
        priority = base_priority * min(1.0 + (weight - 1.0) * 0.1, 1.5)
        
        # 检测语义标记
        tags = self._detect_semantic_tags(content, node_type)
        
        # 检测是否为硬约束
        hard = self._is_hard_constraint(node_type, content, tags)
        
        return PromptNode(
            type=node_type,
            content=content,
            weight=weight,
            priority=priority,
            hard=hard,
            tags=tags,
            index=index
        )
    
    def parse_parts(self, parts: List[str]) -> PromptAST:
        """
        解析多个 prompt parts 为 PromptAST
        
        Args:
            parts: prompt 部分字符串列表
            
        Returns:
            PromptAST 对象
        """
        nodes = []
        for i, part in enumerate(parts):
            node = self.parse_part(part, index=i)
            nodes.append(node)
        
        return PromptAST(nodes)
    
    def _infer_node_type(self, content: str) -> str:
        """
        推断节点类型（使用语义模式注册表）
        
        ⚡ 不再硬编码词语，而是使用可配置的语义模式
        
        Args:
            content: 纯内容字符串（不含权重）
            
        Returns:
            节点类型
        """
        # 使用语义模式注册表进行类型推断
        return self.pattern_registry.infer_type(content)
    
    def _get_base_priority(self, node_type: str) -> float:
        """
        获取基础优先级（从旧的 _analyze_importance 迁移）
        
        Args:
            node_type: 节点类型
            
        Returns:
            基础优先级数值
        """
        base_priority = {
            "constraint": 20.0,   # 约束条件最高优先级
            "character": 15.0,    # 角色描述
            "composition": 12.0,  # 构图描述
            "style": 12.0,        # 风格描述
            "fx": 11.0,           # 特效
            "environment": 10.0,  # 环境描述
            "action": 7.5,        # 动作描述
            "camera": 6.5,        # 镜头描述
            "background": 4.0,    # 背景一致性
            "other": 3.0
        }
        return base_priority.get(node_type, 3.0)
    
    def _detect_semantic_tags(self, content: str, node_type: str) -> set:
        """
        检测语义标记
        
        Args:
            content: 内容字符串
            node_type: 节点类型
            
        Returns:
            标记集合
        """
        tags = set()
        content_lower = content.lower()
        
        # 姿势标记
        if any(kw in content_lower for kw in ["lying", "lie", "躺", "卧", "sitting", "sit", "坐", "prone"]):
            tags.add("horizontal_pose")
        
        # 天空物体标记
        if any(kw in content_lower for kw in ["sun", "suns", "moon", "moons", "lunar", "solar", "太阳", "月亮"]):
            tags.add("sky_object")
            if any(kw in content_lower for kw in ["three", "four", "five", "multiple", "三", "四", "五"]):
                tags.add("multiple_sky_objects")
        
        # 远景标记
        if any(kw in content_lower for kw in ["wide shot", "distant view", "far away", "远景", "远距离"]):
            tags.add("wide_shot")
        
        # Top-down 标记
        if any(kw in content_lower for kw in ["top-down", "top down", "bird's eye", "俯视"]):
            tags.add("top_down")
        
        # 关键动作标记
        if any(kw in content_lower for kw in ["lying on", "sitting on", "standing on"]):
            tags.add("pose_sensitive")
        
        return tags
    
    def _is_hard_constraint(self, node_type: str, content: str, tags: set) -> bool:
        """
        判断是否为硬约束（不可删除）
        
        Args:
            node_type: 节点类型
            content: 内容字符串
            tags: 语义标记
            
        Returns:
            是否为硬约束
        """
        # 约束条件类型总是硬约束
        if node_type == "constraint":
            return True
        
        # 包含关键角色特征的也是硬约束（InstantID 人设级）
        if node_type == "character":
            content_lower = content.lower()
            has_key_features = any(kw in content_lower for kw in [
                "robe", "cultivator", "道袍", "修仙", "xianxia", "hair", "长发"
            ])
            if has_key_features:
                return True
        
        return False

