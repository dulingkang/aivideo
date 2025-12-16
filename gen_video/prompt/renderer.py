"""
Prompt 渲染层（Render Layer）

将 PromptNode AST 渲染为最终的 prompt 字符串。
这是三层架构的第三层：渲染层，只在这里拼字符串。
"""

from typing import List, Optional
from .semantic import PromptNode, PromptAST
from .token_estimator import TokenEstimator


class PromptRenderer:
    """
    Prompt 渲染器
    
    负责将 PromptAST 渲染为最终的 prompt 字符串。
    禁止任何语义判断出现在 renderer 中。
    """
    
    def __init__(self, token_estimator: Optional[TokenEstimator] = None):
        """
        初始化渲染器
        
        Args:
            token_estimator: Token 估算器（用于 token 限制）
        """
        self.token_estimator = token_estimator
    
    def render(self, ast: PromptAST, max_tokens: int = 70) -> str:
        """
        渲染 AST 为 prompt 字符串
        
        Args:
            ast: PromptAST 对象
            max_tokens: 最大 token 数
            
        Returns:
            渲染后的 prompt 字符串
        """
        # 1. 选择要渲染的节点（基于优先级和 token 限制）
        selected_nodes = self._select_nodes(ast, max_tokens)
        
        # 2. 按原始顺序排序（保持逻辑顺序）
        selected_nodes.sort(key=lambda n: n.index)
        
        # 3. 渲染每个节点
        rendered_parts = [self._render_node(node) for node in selected_nodes]
        
        # 4. 组合为最终字符串
        return ", ".join(rendered_parts)
    
    def _select_nodes(self, ast: PromptAST, max_tokens: int) -> List[PromptNode]:
        """
        选择要渲染的节点（基于优先级和 token 限制）
        
        Args:
            ast: PromptAST 对象
            max_tokens: 最大 token 数
            
        Returns:
            选中的节点列表
        """
        # 复制 AST 以便排序
        ast_copy = ast.copy()
        ast_copy.sort_by_priority(reverse=True)
        
        selected_nodes = []
        current_tokens = 0
        
        # 必须保留的硬约束节点
        hard_nodes = ast_copy.filter_hard()
        for node in hard_nodes:
            test_prompt = self._render_nodes(selected_nodes + [node])
            test_tokens = self.token_estimator.estimate(test_prompt) if self.token_estimator else len(test_prompt.split())
            
            if test_tokens <= max_tokens:
                selected_nodes.append(node)
                current_tokens = test_tokens
            else:
                # 硬约束即使超限也要保留
                selected_nodes.append(node)
                current_tokens = test_tokens
        
        # 按优先级添加其他节点
        for node in ast_copy.nodes:
            if node in selected_nodes:
                continue
            
            test_prompt = self._render_nodes(selected_nodes + [node])
            test_tokens = self.token_estimator.estimate(test_prompt) if self.token_estimator else len(test_prompt.split())
            
            if test_tokens <= max_tokens:
                selected_nodes.append(node)
                current_tokens = test_tokens
            elif node.priority >= 8.0:  # 高优先级节点，即使超限也尝试保留
                # 可以尝试精简，但这里简化处理，直接保留
                selected_nodes.append(node)
                current_tokens = test_tokens
        
        return selected_nodes
    
    def _render_node(self, node: PromptNode) -> str:
        """
        渲染单个节点为字符串
        
        ⚠️ 禁止任何语义判断出现在这里，只做字符串拼接
        
        Args:
            node: PromptNode 对象
            
        Returns:
            渲染后的字符串，如 "(content:1.5)" 或 "content"
        """
        return node.to_string(include_weight=True)
    
    def _render_nodes(self, nodes: List[PromptNode]) -> str:
        """
        渲染多个节点为字符串（用于 token 估算）
        
        Args:
            nodes: 节点列表
            
        Returns:
            渲染后的字符串
        """
        parts = [self._render_node(node) for node in nodes]
        return ", ".join(parts)


