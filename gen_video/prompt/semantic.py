"""
Prompt 语义层（Semantic Layer）

定义 PromptNode 数据结构，将 Prompt 从字符串提升为语义 AST。
这是三层架构的第一层：语义层，不碰字符串操作。
"""

from dataclasses import dataclass, field
from typing import Literal, Set, Optional, List


@dataclass
class PromptNode:
    """
    Prompt 语义节点
    
    核心原则：
    - content 永远不含权重标记 `( ) :`
    - weight 是数值，不是字符串
    - hard = True 表示不可删除（如 InstantID 人设级）
    - tags 用于语义标记，供策略层使用
    """
    type: Literal[
        "constraint",      # 约束条件（如 single person）
        "character",       # 角色描述
        "action",          # 动作描述
        "composition",     # 构图描述（包含动作和场景主体）
        "camera",          # 镜头描述
        "environment",     # 环境描述
        "style",           # 风格描述
        "fx",              # 特效描述
        "background",      # 背景一致性
        "other"            # 其他
    ]
    content: str           # 纯语义内容，不含权重标记
    weight: float = 1.0    # 权重（数值）
    priority: float = 1.0  # 语义优先级（用于优化选择）
    hard: bool = False     # 是否不可删除（硬约束）
    tags: Set[str] = field(default_factory=set)  # 语义标记（如 "horizontal_pose", "sky_object"）
    index: int = 0         # 原始顺序（用于保持逻辑顺序）
    
    def __post_init__(self):
        """确保 content 不含权重标记"""
        # 移除可能的权重标记（如果传入时带上了）
        import re
        # 移除 ( ) 和权重
        self.content = re.sub(r'^\(|\)$', '', self.content)
        self.content = re.sub(r':\d+\.?\d*\)?$', '', self.content).strip()
    
    def to_string(self, include_weight: bool = True) -> str:
        """
        渲染为字符串（仅用于最终输出）
        
        Args:
            include_weight: 是否包含权重标记
            
        Returns:
            渲染后的字符串，如 "(content:1.5)" 或 "content"
        """
        if include_weight and self.weight != 1.0:
            return f"({self.content}:{self.weight:.2f})"
        return self.content
    
    def copy(self) -> 'PromptNode':
        """创建副本"""
        return PromptNode(
            type=self.type,
            content=self.content,
            weight=self.weight,
            priority=self.priority,
            hard=self.hard,
            tags=self.tags.copy(),
            index=self.index
        )


class PromptAST:
    """
    Prompt 抽象语法树
    
    管理一组 PromptNode，提供统一的语义操作接口。
    """
    
    def __init__(self, nodes: Optional[List[PromptNode]] = None):
        """
        初始化 AST
        
        Args:
            nodes: 初始节点列表
        """
        self.nodes: List[PromptNode] = nodes or []
    
    def add_node(self, node: PromptNode) -> None:
        """添加节点"""
        self.nodes.append(node)
    
    def get_nodes_by_type(self, node_type: str) -> List[PromptNode]:
        """按类型获取节点"""
        return [n for n in self.nodes if n.type == node_type]
    
    def get_nodes_by_tag(self, tag: str) -> List[PromptNode]:
        """按标记获取节点"""
        return [n for n in self.nodes if tag in n.tags]
    
    def sort_by_priority(self, reverse: bool = True) -> None:
        """按优先级排序（原地排序）"""
        self.nodes.sort(key=lambda n: n.priority, reverse=reverse)
    
    def sort_by_index(self) -> None:
        """按原始顺序排序（恢复逻辑顺序）"""
        self.nodes.sort(key=lambda n: n.index)
    
    def filter_hard(self) -> List[PromptNode]:
        """获取硬约束节点（不可删除）"""
        return [n for n in self.nodes if n.hard]
    
    def copy(self) -> 'PromptAST':
        """创建副本"""
        return PromptAST([n.copy() for n in self.nodes])
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, index: int) -> PromptNode:
        return self.nodes[index]


