"""
Prompt构建模块

负责Prompt的构建、优化、解析等功能。

三层架构：
- 语义层（Semantic Layer）：PromptNode, PromptAST
- 策略层（Policy Layer）：PromptPolicy, PolicyEngine
- 渲染层（Render Layer）：PromptRenderer
"""

from .builder import PromptBuilder
from .optimizer import PromptOptimizer
from .parser import PromptParser
from .token_estimator import TokenEstimator

# 三层架构新模块
from .semantic import PromptNode, PromptAST
from .ast_builder import ASTBuilder
from .policy import PromptPolicy, InstantIDPolicy, FluxPolicy, HunyuanVideoPolicy, SDXLPolicy, PolicyEngine
from .renderer import PromptRenderer
from .semantic_patterns import SemanticPattern, SemanticPatternRegistry, get_pattern_registry

__all__ = [
    # 原有模块
    "PromptBuilder",
    "PromptOptimizer", 
    "PromptParser",
    "TokenEstimator",
    # 三层架构新模块
    "PromptNode",
    "PromptAST",
    "ASTBuilder",
    "PromptPolicy",
    "InstantIDPolicy",
    "FluxPolicy",
    "HunyuanVideoPolicy",
    "SDXLPolicy",
    "PolicyEngine",
    "PromptRenderer",
    # 语义模式模块
    "SemanticPattern",
    "SemanticPatternRegistry",
    "get_pattern_registry",
]

















