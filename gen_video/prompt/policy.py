"""
Prompt 策略层（Policy Layer）

模型/场景感知的策略引擎，决定什么时候保留/放大/削弱/分离节点。
这是三层架构的第二层：策略层。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .semantic import PromptNode, PromptAST


class PromptPolicy(ABC):
    """
    Prompt 策略基类
    
    所有策略必须实现 apply 方法，对 AST 进行策略调整。
    """
    
    @abstractmethod
    def apply(self, ast: PromptAST) -> PromptAST:
        """
        应用策略到 AST
        
        Args:
            ast: 输入的 PromptAST
            
        Returns:
            调整后的 PromptAST（可以是新实例或修改原实例）
        """
        pass


class InstantIDPolicy(PromptPolicy):
    """
    InstantID 专用策略
    
    核心原则：
    - 角色描述是硬约束，优先级最高
    - 远景 + lying + 强场景会导致失效，需要削弱
    - 人设图优先，场景图次要
    """
    
    def apply(self, ast: PromptAST) -> PromptAST:
        """
        应用 InstantID 策略
        
        策略规则：
        1. 角色描述：提高优先级，标记为硬约束
        2. 远景 + lying：降低优先级，标记为 pose_sensitive
        3. 强场景描述：适当降低权重，避免压过角色
        """
        result_ast = ast.copy()
        
        for node in result_ast.nodes:
            # 规则1：角色描述强化
            if node.type == "character":
                node.hard = True
                node.priority += 5.0
                node.weight = max(node.weight, 1.3)  # 确保角色可见
            
            # 规则2：远景 + lying 场景削弱
            if node.type == "composition" and "lying" in node.content.lower():
                if "wide_shot" in node.tags or "top_down" in node.tags:
                    node.priority -= 2.0
                    node.tags.add("pose_sensitive")
                    # 添加排除词，避免误判为 standing
                    if "not standing" not in node.content.lower():
                        node.content += ", not standing, horizontal position"
            
            # 规则3：强场景描述适当削弱（避免压过角色）
            if node.type == "environment" and node.weight > 1.5:
                # 如果环境权重过高，适当降低，但保持可见性
                node.weight = min(node.weight, 1.8)
        
        return result_ast


class FluxPolicy(PromptPolicy):
    """
    Flux 专用策略
    
    核心原则：
    - Token 多、语义分散是问题
    - 需要精简但保持核心语义
    - 风格描述很重要
    """
    
    def apply(self, ast: PromptAST) -> PromptAST:
        """
        应用 Flux 策略
        
        策略规则：
        1. 风格描述：提高优先级
        2. 环境描述：保持高权重（Flux 擅长场景）
        3. 角色描述：如果场景为主，适当降低权重
        """
        result_ast = ast.copy()
        
        for node in result_ast.nodes:
            # 规则1：风格描述强化
            if node.type == "style":
                node.priority += 2.0
                node.weight = max(node.weight, 1.2)
            
            # 规则2：环境描述保持高权重
            if node.type == "environment":
                node.weight = max(node.weight, 1.5)
            
            # 规则3：如果场景为主（环境节点多），角色权重适当降低
            env_nodes = result_ast.get_nodes_by_type("environment")
            if len(env_nodes) >= 2 and node.type == "character":
                node.weight = min(node.weight, 1.3)
        
        return result_ast


class HunyuanVideoPolicy(PromptPolicy):
    """
    HunyuanVideo 专用策略
    
    核心原则：
    - 镜头 + 动作歧义是问题
    - 需要明确的运动描述
    - 时间连续性重要
    """
    
    def apply(self, ast: PromptAST) -> PromptAST:
        """
        应用 HunyuanVideo 策略
        
        策略规则：
        1. 镜头描述：提高优先级，确保清晰
        2. 动作描述：消除歧义，添加明确标记
        3. 环境描述：保持稳定，支持时间连续性
        """
        result_ast = ast.copy()
        
        for node in result_ast.nodes:
            # 规则1：镜头描述强化
            if node.type == "camera":
                node.priority += 1.5
                node.weight = max(node.weight, 1.3)
            
            # 规则2：动作描述消除歧义
            if node.type == "action" and "pose_sensitive" in node.tags:
                if "not standing" not in node.content.lower():
                    node.content += ", not standing"
            
            # 规则3：环境描述保持稳定
            if node.type == "environment":
                node.tags.add("temporal_stable")
        
        return result_ast


class SDXLPolicy(PromptPolicy):
    """
    SDXL 专用策略（默认策略）
    
    核心原则：
    - 平衡各种元素
    - 保持原有逻辑
    """
    
    def apply(self, ast: PromptAST) -> PromptAST:
        """
        应用 SDXL 策略（默认策略，基本不做调整）
        """
        return ast.copy()


class PolicyEngine:
    """
    策略引擎
    
    根据模型类型选择合适的策略并应用。
    """
    
    def __init__(self):
        """初始化策略引擎"""
        self.policies = {
            "instantid": InstantIDPolicy(),
            "flux": FluxPolicy(),
            "flux1": FluxPolicy(),
            "hunyuanvideo": HunyuanVideoPolicy(),
            "sdxl": SDXLPolicy(),
            "default": SDXLPolicy()
        }
    
    def apply_policy(self, ast: PromptAST, model_type: str = "default") -> PromptAST:
        """
        应用策略
        
        Args:
            ast: 输入的 PromptAST
            model_type: 模型类型（instantid, flux, hunyuanvideo, sdxl）
            
        Returns:
            应用策略后的 PromptAST
        """
        policy = self.policies.get(model_type.lower(), self.policies["default"])
        return policy.apply(ast)


