"""
语义模式定义（Semantic Patterns）

定义用于类型推断的语义模式，而不是硬编码的词语列表。
支持可配置、可扩展的语义匹配。
"""

from typing import Dict, List, Set, Pattern, Optional
import re
from dataclasses import dataclass, field


@dataclass
class SemanticPattern:
    """
    语义模式
    
    定义一组语义特征，用于匹配和分类。
    """
    keywords: List[str] = field(default_factory=list)  # 关键词列表
    patterns: List[Pattern] = field(default_factory=list)  # 正则表达式模式
    min_matches: int = 1  # 最少匹配数
    weight: float = 1.0  # 模式权重（用于多模式匹配时的优先级）
    
    def matches(self, text: str) -> bool:
        """
        检查文本是否匹配此模式
        
        Args:
            text: 待检查的文本
            
        Returns:
            是否匹配
        """
        text_lower = text.lower()
        matches = 0
        
        # 检查关键词
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                matches += 1
        
        # 检查正则表达式
        for pattern in self.patterns:
            if pattern.search(text):
                matches += 1
        
        return matches >= self.min_matches


class SemanticPatternRegistry:
    """
    语义模式注册表
    
    集中管理所有语义模式，支持动态配置和扩展。
    """
    
    def __init__(self):
        """初始化模式注册表"""
        self.patterns: Dict[str, List[SemanticPattern]] = {
            "constraint": [],
            "character": [],
            "action": [],
            "composition": [],
            "camera": [],
            "environment": [],
            "style": [],
            "fx": [],
            "background": [],
            "other": []
        }
        self._init_default_patterns()
    
    def _init_default_patterns(self):
        """初始化默认模式"""
        
        # ========== 约束条件模式 ==========
        self.patterns["constraint"].append(SemanticPattern(
            keywords=[
                "single person", "lone figure", "only one character", "one person only",
                "sole character", "single individual", "单人", "独行", "只有一个角色",
                "仅一人", "唯一角色", "单独个体"
            ],
            min_matches=1,
            weight=10.0  # 最高优先级
        ))
        
        # ========== 角色描述模式 ==========
        # 模式1：角色名称和特征
        self.patterns["character"].append(SemanticPattern(
            keywords=[
                "han li", "韩立",  # 角色名称（可配置）
                "long black hair", "tied long black hair", "forehead bangs",
                "黑色长发", "长发", "刘海"
            ],
            min_matches=1,
            weight=8.0
        ))
        
        # 模式2：服饰和修仙特征
        self.patterns["character"].append(SemanticPattern(
            keywords=[
                "cultivator robe", "dark green", "deep cyan",
                "xianxia cultivator", "immortal cultivator",
                "道袍", "深绿", "修仙", "仙侠"
            ],
            min_matches=1,
            weight=8.0
        ))
        
        # 模式3：多个角色特征组合（更可靠）
        self.patterns["character"].append(SemanticPattern(
            keywords=["hair", "robe", "cultivator", "长发", "道袍", "修仙"],
            min_matches=2,  # 至少匹配2个特征
            weight=9.0
        ))
        
        # 模式4：性别标记
        self.patterns["character"].append(SemanticPattern(
            patterns=[re.compile(r'^\(?(male|female)', re.IGNORECASE)],
            min_matches=1,
            weight=5.0
        ))
        
        # ========== 动作描述模式 ==========
        self.patterns["action"].append(SemanticPattern(
            keywords=[
                "lying", "lying on", "躺", "卧",
                "sitting", "sit", "坐",
                "standing", "stand", "站",
                "walking", "walk", "走",
                "动作", "姿势", "description"
            ],
            min_matches=1,
            weight=6.0
        ))
        
        # ========== 构图描述模式 ==========
        # 模式1：动作动词
        self.patterns["composition"].append(SemanticPattern(
            keywords=[
                "uses", "method", "flowing", "essence", "energy", "performing", "casting",
                "strains", "tilt", "sees", "revealing", "showing",
                "recalls", "tilts", "dive", "hovers", "expands", "changes", "recognizing",
                "躺", "看见", "转头", "回忆", "使用", "施展", "俯冲", "盘旋", "扩张", "变化"
            ],
            min_matches=1,
            weight=7.0
        ))
        
        # 模式2：场景关系词
        self.patterns["composition"].append(SemanticPattern(
            keywords=[
                "on", "above", "below", "in", "at", "with",
                "在", "上", "下", "中", "看到", "展现"
            ],
            min_matches=1,
            weight=5.0
        ))
        
        # 模式3：特殊场景标记
        self.patterns["composition"].append(SemanticPattern(
            keywords=["composition", "nascent soul"],
            min_matches=1,
            weight=8.0
        ))
        
        # ========== 特效模式 ==========
        self.patterns["fx"].append(SemanticPattern(
            keywords=[
                "essence", "energy flow", "spiritual light", "glow", "fx", "effect",
                "flooding", "visible", "flow", "light", "particles",
                "能量", "光效", "特效", "流动", "可见"
            ],
            min_matches=1,
            weight=6.0
        ))
        
        # ========== 环境描述模式 ==========
        self.patterns["environment"].append(SemanticPattern(
            keywords=[
                "environment", "desert", "chamber", "sky", "background", "gravel", "plain",
                "环境", "沙漠", "天空", "遗迹", "背景", "沙地", "地面"
            ],
            min_matches=1,
            weight=6.0
        ))
        
        # ========== 风格描述模式 ==========
        self.patterns["style"].append(SemanticPattern(
            keywords=["xianxia", "chinese fantasy", "仙侠", "修仙", "古风"],
            min_matches=1,
            weight=7.0
        ))
        
        # ========== 镜头描述模式 ==========
        self.patterns["camera"].append(SemanticPattern(
            keywords=[
                "camera", "shot", "镜头", "俯视", "远景", "中景",
                "facing camera", "front view", "top-down", "bird's eye"
            ],
            min_matches=1,
            weight=5.0
        ))
        
        # ========== 背景一致性模式 ==========
        self.patterns["background"].append(SemanticPattern(
            keywords=["consistent", "same", "背景一致"],
            min_matches=1,
            weight=4.0
        ))
    
    def infer_type(self, text: str) -> str:
        """
        推断文本的类型
        
        Args:
            text: 待推断的文本
            
        Returns:
            推断出的类型（按优先级返回）
        """
        # 按优先级顺序检查（constraint 最高）
        type_priority = [
            "constraint",  # 最高优先级
            "character",
            "composition",
            "fx",
            "environment",
            "style",
            "action",
            "camera",
            "background",
            "other"
        ]
        
        # 计算每个类型的匹配分数
        type_scores: Dict[str, float] = {}
        
        for node_type in type_priority:
            if node_type == "other":
                continue  # other 是默认类型，不需要匹配
            
            for pattern in self.patterns[node_type]:
                if pattern.matches(text):
                    score = pattern.weight
                    if node_type not in type_scores:
                        type_scores[node_type] = 0.0
                    type_scores[node_type] += score
        
        # 返回得分最高的类型
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return "other"
    
    def add_pattern(self, node_type: str, pattern: SemanticPattern):
        """
        添加自定义模式
        
        Args:
            node_type: 节点类型
            pattern: 语义模式
        """
        if node_type not in self.patterns:
            self.patterns[node_type] = []
        self.patterns[node_type].append(pattern)
    
    def remove_pattern(self, node_type: str, pattern_index: int):
        """
        移除模式
        
        Args:
            node_type: 节点类型
            pattern_index: 模式索引
        """
        if node_type in self.patterns and 0 <= pattern_index < len(self.patterns[node_type]):
            del self.patterns[node_type][pattern_index]
    
    def update_pattern(self, node_type: str, pattern_index: int, pattern: SemanticPattern):
        """
        更新模式
        
        Args:
            node_type: 节点类型
            pattern_index: 模式索引
            pattern: 新的语义模式
        """
        if node_type in self.patterns and 0 <= pattern_index < len(self.patterns[node_type]):
            self.patterns[node_type][pattern_index] = pattern


# 全局模式注册表实例（单例模式）
_pattern_registry: Optional[SemanticPatternRegistry] = None


def get_pattern_registry() -> SemanticPatternRegistry:
    """
    获取全局模式注册表实例（单例）
    
    Returns:
        SemanticPatternRegistry 实例
    """
    global _pattern_registry
    if _pattern_registry is None:
        _pattern_registry = SemanticPatternRegistry()
    return _pattern_registry


def reset_pattern_registry():
    """
    重置全局模式注册表（用于测试或重新配置）
    """
    global _pattern_registry
    _pattern_registry = None

