"""
Prompt优化器

负责智能优化Prompt，基于语义重要性保留关键信息，确保不超过token限制。

架构升级：
- 支持新的三层架构（语义层 + 策略层 + 渲染层）
- 保留旧的字符串方法作为兼容层
"""

from typing import List, Dict, Any, Optional
from .token_estimator import TokenEstimator
from .parser import PromptParser
from .semantic import PromptNode, PromptAST
from .ast_builder import ASTBuilder
from .policy import PolicyEngine
from .enhancer import SemanticEnhancer
from .renderer import PromptRenderer
from .semantic_patterns import get_pattern_registry


class PromptOptimizer:
    """Prompt优化器"""
    
    def __init__(self, token_estimator: TokenEstimator, parser: PromptParser, ascii_only_prompt: bool = False):
        """
        初始化Prompt优化器
        
        Args:
            token_estimator: Token估算器
            parser: Prompt解析器
            ascii_only_prompt: 是否只使用ASCII字符
        """
        self.token_estimator = token_estimator
        self.parser = parser
        self.ascii_only_prompt = ascii_only_prompt
        
        # 初始化语义模式注册表
        self.pattern_registry = get_pattern_registry()
        
        # 初始化三层架构组件
        self.ast_builder = ASTBuilder(token_estimator=token_estimator, pattern_registry=self.pattern_registry)
        self.policy_engine = PolicyEngine()
        self.semantic_enhancer = SemanticEnhancer()
        self.renderer = PromptRenderer(token_estimator=token_estimator)
    
    def optimize(
        self, 
        parts: List[str], 
        max_tokens: int = 70,
        model_type: str = "default",
        use_ast: bool = True
    ) -> List[str]:
        """
        智能优化 prompt，基于语义重要性保留关键信息
        
        Args:
            parts: prompt 部分列表
            max_tokens: 最大 token 数
            model_type: 模型类型（instantid, flux, hunyuanvideo, sdxl）
            use_ast: 是否使用新的 AST 架构（默认 True）
        
        Returns:
            优化后的 prompt 部分列表
        """
        if not parts:
            return []
        
        # ⚡ 新架构：使用 AST + 策略 + 渲染
        if use_ast:
            return self._optimize_with_ast(parts, max_tokens, model_type)
        
        # 旧架构：保持向后兼容
        return self._optimize_legacy(parts, max_tokens)
    
    def _optimize_with_ast(
        self, 
        parts: List[str], 
        max_tokens: int,
        model_type: str
    ) -> List[str]:
        """
        使用 AST 架构优化（新方法）
        
        流程：
        1. 字符串 → AST（语义层）
        2. 语义增强（语义层）
        3. 策略应用（策略层）
        4. AST → 字符串（渲染层）
        """
        # 1. 解析为 AST
        ast = self.ast_builder.parse_parts(parts)
        
        # 2. 语义增强
        ast = self.semantic_enhancer.enhance_ast(ast)
        
        # 3. 应用策略（模型感知）
        ast = self.policy_engine.apply_policy(ast, model_type)
        
        # 4. 渲染为字符串
        final_prompt = self.renderer.render(ast, max_tokens)
        
        # 5. 分割为 parts（保持接口兼容）
        # 注意：这里简化处理，实际可以返回单个字符串或parts列表
        # 为了兼容，我们分割为parts
        return [p.strip() for p in final_prompt.split(",") if p.strip()]
    
    def _optimize_legacy(self, parts: List[str], max_tokens: int) -> List[str]:
        """
        旧架构优化方法（保持向后兼容）
        """
        # 1. 分析每个部分的重要性
        analyzed_parts = []
        for i, part in enumerate(parts):
            part_type = self._infer_part_type(part)
            analysis = self._analyze_importance(part, part_type)
            analyzed_parts.append({
                "text": part,
                "type": part_type,
                "analysis": analysis,
                "index": i
            })
        
        # 2. 按重要性排序
        analyzed_parts.sort(key=lambda x: x["analysis"]["importance"], reverse=True)
        
        # 3. 智能选择：优先保留高重要性且 token 效率高的部分
        selected_parts = self._select_parts(analyzed_parts, max_tokens)
        
        # 4. 去重：移除重复和语义相近的描述
        selected_parts = self._remove_duplicate_and_similar(selected_parts)
        
        # 4.5. 处理单个 part 内部的重复词汇
        for part in selected_parts:
            part["text"] = self._remove_internal_duplicates(part["text"])
        
        # 5. 按原始顺序重新排序（保持逻辑顺序）
        selected_parts.sort(key=lambda x: x["index"])
        
        # 6. 返回优化后的文本列表
        optimized_parts = [p["text"] for p in selected_parts]
        
        # 打印优化信息
        if len(optimized_parts) < len(parts):
            current_tokens = self.token_estimator.estimate(", ".join(optimized_parts))
            print(f"  🧠 智能优化: 从 {len(parts)} 个部分精简至 {len(optimized_parts)} 个部分")
            print(f"  📊 Token 使用: {current_tokens}/{max_tokens} tokens")
            for p in selected_parts:
                print(f"    - [{p['type']}] {p['text'][:50]}... (重要性: {p['analysis']['importance']:.1f}, {p['analysis']['token_count']} tokens)")
        
        # 检查角色描述是否被保留
        character_parts_in_result = [p for p in selected_parts if p['type'] == 'character']
        if character_parts_in_result:
            print(f"  ✓ 角色描述已保留: {len(character_parts_in_result)} 个部分")
            for cp in character_parts_in_result:
                print(f"    - {cp['text'][:80]}...")
        else:
            # 检查原始parts中是否有角色描述
            original_character_parts = [p for p in analyzed_parts if p['type'] == 'character']
            if original_character_parts:
                print(f"  ⚠ 警告: 原始prompt中有 {len(original_character_parts)} 个角色描述部分，但优化后被移除了！")
                for cp in original_character_parts:
                    print(f"    - 被移除的角色描述: {cp['text'][:80]}...")
        
        return optimized_parts
    
    def _infer_part_type(self, part: str) -> str:
        """
        推断prompt部分的类型（使用语义模式注册表）
        
        ⚡ 不再硬编码词语，而是使用可配置的语义模式
        
        Args:
            part: prompt部分字符串
            
        Returns:
            推断出的类型
        """
        # 提取纯内容（移除权重标记）
        import re
        content = re.sub(r'^\(|\)$', '', part)
        content = re.sub(r':\d+\.?\d*\)?$', '', content).strip()
        
        # 使用语义模式注册表进行类型推断
        return self.pattern_registry.infer_type(content)
    
    def _analyze_importance(self, part: str, part_type: str) -> Dict[str, Any]:
        """
        分析prompt部分的重要性
        
        关键信息优先级（从高到低）：
        1. composition（构图描述，包含动作和场景主体）
        2. fx（特效，包含能量、光效等关键视觉元素）
        3. environment（环境描述，包含背景和氛围）
        4. character（角色描述，包含外观特征）
        5. action（动作描述）
        6. camera（镜头描述，但避免重复）
        7. style（风格描述）
        8. background（背景一致性，次要）
        
        Args:
            part: prompt部分文本
            part_type: 部分类型
            
        Returns:
            包含重要性、token数量、token效率等信息的字典
        """
        token_count = self.token_estimator.estimate(part)
        
        # 根据类型设置基础重要性
        # 用户反馈：场景表现不够好，composition、fx、environment是关键信息，必须保留
        # 约束条件（如单人约束）具有最高优先级，必须保留
        base_importance = {
            "constraint": 20.0,  # 约束条件（如单人约束）最高优先级，必须保留
            "composition": 12.0,  # 构图描述最重要，包含动作和场景主体
            "fx": 11.0,  # 特效次重要，包含能量、光效等关键视觉元素
            "environment": 10.0,  # 环境描述很重要，包含背景和氛围
            "style": 12.0,  # 风格描述（提高优先级，确保xianxia风格不被移除）
            "character": 15.0,  # 角色描述（提高优先级，确保性别、服饰、修仙气质不被移除）
            "action": 7.5,  # 动作描述
            "camera": 6.5,  # 镜头描述（但避免重复）
            "background": 4.0,  # 背景一致性，次要
            "scene": 7.0,  # 场景描述（兼容旧代码）
            "other": 3.0
        }.get(part_type, 3.0)
        
        # 根据权重标记调整重要性
        import re
        weight_match = re.search(r':(\d+\.?\d*)', part)
        if weight_match:
            weight = float(weight_match.group(1))
            # 权重越高，重要性越高（但不超过基础重要性的1.5倍）
            importance = base_importance * min(1.0 + (weight - 1.0) * 0.1, 1.5)
        else:
            importance = base_importance
        
        # 计算token效率（重要性/ token数）
        token_efficiency = importance / max(token_count, 1)
        
        return {
            "importance": importance,
            "token_count": token_count,
            "token_efficiency": token_efficiency
        }
    
    def _select_parts(self, analyzed_parts: List[Dict], max_tokens: int) -> List[Dict]:
        """选择要保留的prompt部分"""
        selected_parts = []
        current_tokens = 0
        
        # 必须保留的核心部分（composition、fx、environment是关键场景信息，必须保留）
        # 用户反馈：场景表现不够好，composition、fx、environment是关键信息
        # 约束条件（如单人约束）必须保留，具有最高优先级
        # 重要：角色描述（character）必须保留，确保性别、服饰、修仙气质不被移除
        # 重要：区分真正的composition描述（包含动作和场景）和简单的标记（如male）
        must_keep_types = ["constraint", "character", "composition", "fx", "environment", "style", "action"]
        must_keep_parts = [p for p in analyzed_parts if p["type"] in must_keep_types]
        
        # 特别处理：确保包含服饰和修仙气质的角色描述被优先保留
        character_parts = [p for p in must_keep_parts if p["type"] == "character"]
        for char_part in character_parts:
            text_lower = char_part["text"].lower()
            # 如果包含服饰或修仙气质关键词，提高重要性
            if any(kw in text_lower for kw in ["robe", "cultivator", "道袍", "修仙", "xianxia", "服饰", "clothes", "hair", "长发"]):
                char_part["analysis"]["importance"] = max(char_part["analysis"]["importance"], 18.0)  # 提高到18.0，确保不被移除
        
        # 分离真正的composition描述和简单的标记
        constraint_parts = [p for p in must_keep_parts if p["type"] == "constraint"]
        composition_parts = [p for p in must_keep_parts if p["type"] == "composition"]
        
        # 检查composition部分，区分真正的描述和简单标记
        real_composition_parts = []
        simple_marker_parts = []
        for p in composition_parts:
            text_lower = p["text"].lower()
            # 真正的composition描述通常较长，包含动作或场景描述
            is_real_composition = (
                len(p["text"].split()) > 3 or  # 长度超过3个词
                any(kw in text_lower for kw in ["lying", "sees", "strains", "recalls", "uses", "on", "above", "revealing", "han li", "韩立"]) or
                "han li" in text_lower or "韩立" in p["text"]
            )
            if is_real_composition:
                real_composition_parts.append(p)
            else:
                # 简单标记（如(male:1.8)）优先级较低
                simple_marker_parts.append(p)
        
        # 其他必须保留的部分
        other_must_keep = [p for p in must_keep_parts if p["type"] not in ["constraint", "composition"]]
        
        # 特别处理：角色描述必须优先保留（即使token超限也要保留）
        character_parts_sorted = sorted([p for p in other_must_keep if p["type"] == "character"], key=lambda x: x["index"])
        other_must_keep_no_char = [p for p in other_must_keep if p["type"] != "character"]
        
        # 排序：约束条件 > 角色描述（最高优先级）> 真正的composition描述 > 其他 > 简单标记
        must_keep_parts = constraint_parts + character_parts_sorted + \
                          sorted(real_composition_parts, key=lambda x: x["index"]) + \
                          sorted(other_must_keep_no_char, key=lambda x: x["index"]) + \
                          sorted(simple_marker_parts, key=lambda x: x["index"])
        
        # 必须保留的部分，即使超过限制也要保留（但可以精简）
        # 重要：composition描述包含关键动作（如lying, sees, strains等），必须保留，不能精简
        for part_info in must_keep_parts:
            test_parts = [p["text"] for p in selected_parts] + [part_info["text"]]
            test_prompt = ", ".join(test_parts)
            actual_tokens = self.token_estimator.estimate(test_prompt)
            
            if actual_tokens <= max_tokens:
                selected_parts.append(part_info)
                current_tokens = actual_tokens
            else:
                # 如果超过限制，尝试精简（但必须保留）
                # 重要：composition类型且包含关键动作的，不能精简，必须完整保留
                part_text_lower = part_info["text"].lower()
                has_key_action = any(kw in part_text_lower for kw in [
                    "lying", "sees", "strains", "tilts", "recalls", "uses", "performing",
                    "躺", "看见", "转头", "回忆", "使用", "施展"
                ])
                
                # 角色描述必须强制保留（包含服饰、发型、修仙气质等关键信息）
                if part_info["type"] == "character":
                    part_text_lower = part_info["text"].lower()
                    has_clothing_keywords = any(kw in part_text_lower for kw in [
                        "robe", "cultivator", "道袍", "修仙", "xianxia", "服饰", "clothes", 
                        "hair", "长发", "deep cyan", "dark green", "immortal"
                    ])
                    if has_clothing_keywords:
                        # 包含服饰和修仙气质的角色描述，必须完整保留，即使超过限制
                        print(f"  ⚠ 检测到角色描述（包含服饰/修仙气质，{part_info['text'][:50]}...），必须完整保留，即使超过token限制")
                        selected_parts.append(part_info)
                        current_tokens = actual_tokens
                    else:
                        # 其他角色描述也可以精简，但必须保留
                        compact_part = self._compact_part(part_info)
                        test_parts = [p["text"] for p in selected_parts] + [compact_part]
                        test_prompt = ", ".join(test_parts)
                        compact_tokens = self.token_estimator.estimate(test_prompt)
                        
                        if compact_tokens <= max_tokens:
                            part_info["text"] = compact_part
                            part_info["analysis"]["token_count"] = compact_tokens
                            selected_parts.append(part_info)
                            current_tokens = compact_tokens
                        else:
                            # 即使精简后还是超限，也要强制保留（角色描述太重要）
                            print(f"  ⚠ 角色描述精简后仍超限，但必须强制保留: {part_info['text'][:50]}...")
                            selected_parts.append(part_info)
                            current_tokens = actual_tokens
                elif part_info["type"] == "composition" and has_key_action:
                    # 包含关键动作的composition描述，必须完整保留，即使超过限制
                    # 优先移除其他低重要性部分，为关键composition描述腾出空间
                    print(f"  ⚠ 检测到关键动作描述（{part_info['text'][:50]}...），必须完整保留，即使超过token限制")
                    selected_parts.append(part_info)
                    current_tokens = actual_tokens
                else:
                    # 其他类型可以精简
                    compact_part = self._compact_part(part_info)
                    test_parts = [p["text"] for p in selected_parts] + [compact_part]
                    test_prompt = ", ".join(test_parts)
                    compact_tokens = self.token_estimator.estimate(test_prompt)
                    
                    if compact_tokens <= max_tokens:
                        part_info["text"] = compact_part
                        part_info["analysis"]["token_count"] = compact_tokens
                        selected_parts.append(part_info)
                        current_tokens = compact_tokens
        
        # 添加其他高重要性部分（场景、镜头等）
        for part_info in analyzed_parts:
            if part_info in selected_parts:
                continue
            
            test_parts = [p["text"] for p in selected_parts] + [part_info["text"]]
            test_prompt = ", ".join(test_parts)
            actual_tokens = self.token_estimator.estimate(test_prompt)
            
            # 如果还有空间，添加高重要性或高 token 效率的部分
            # 用户反馈：需要充分表达意图，降低重要性阈值，保留更多信息
            if actual_tokens <= max_tokens:
                if (part_info["analysis"]["importance"] >= 6.0 or  # 从7.0降低到6.0，保留更多信息
                    part_info["analysis"]["token_efficiency"] >= 0.4):  # 从0.5降低到0.4，保留更多信息
                    selected_parts.append(part_info)
                    current_tokens = actual_tokens
            else:
                # 如果空间不足，尝试精简这个部分
                # 用户反馈：角色和镜头描述很重要，降低精简阈值
                if part_info["analysis"]["importance"] >= 7.0:  # 从8.0降低到7.0，保留更多重要信息
                    compact_part = self.parser.extract_first_keyword(part_info["text"])
                    compact_tokens = self.token_estimator.estimate(compact_part)
                    if current_tokens + compact_tokens <= max_tokens:
                        part_info["text"] = compact_part
                        part_info["analysis"]["token_count"] = compact_tokens
                        selected_parts.append(part_info)
                        current_tokens += compact_tokens
        
        return selected_parts
    
    def _compact_part(self, part_info: Dict) -> str:
        """精简prompt部分"""
        part_type = part_info["type"]
        text = part_info["text"]
        
        if part_type == "style":
            # 风格描述：xianxia风格必须保留
            text_lower = text.lower()
            if "xianxia" in text_lower or "仙侠" in text or "chinese fantasy" in text_lower:
                # xianxia风格是核心，必须保留，不精简
                return text
            # 风格描述：至少保留"仙侠风格"
            if "仙侠风格" in text and "古风" in text:
                if "修仙" in text:
                    text = text.replace("，修仙", "").replace(", 修仙", "").replace("修仙", "")
            else:
                text = "仙侠风格" if not self.ascii_only_prompt else "xianxia fantasy"
        elif part_type == "action":
            # 动作描述：精简但保留核心动作信息
            import re
            if "躺" in text or "lying" in text.lower():
                if "韩立" in text:
                    text = "(韩立躺在沙地上:1.6)" if not text.startswith("(") else text.split(":")[0] + ":1.6)"
                else:
                    text = "(躺在沙地上:1.6)"
            else:
                # 其他动作，提取前20个字符的核心描述
                if len(text) > 30:
                    if text.startswith("(") and ":" in text:
                        content = text.split(":")[0].strip("()")
                        weight = text.split(":")[1].strip("()")
                        compact_content = content[:20] + "..."
                        text = f"({compact_content}:{weight})"
                    else:
                        text = text[:20] + "..."
        else:
            # 其他类型，提取第一个关键词
            text = self.parser.extract_first_keyword(text)
        
        return text
    
    def _remove_duplicate_facing_camera(self, parts: List[Dict]) -> List[Dict]:
        """移除重复的facing camera描述，只保留权重最高的一个"""
        facing_camera_parts = []
        other_parts = []
        
        for part in parts:
            part_lower = part["text"].lower()
            # 检测是否是facing camera相关的描述
            if any(kw in part_lower for kw in [
                "facing camera", "front view", "face forward", "character facing viewer", 
                "frontal view", "面向镜头", "正面视角", "人物面向观众"
            ]):
                facing_camera_parts.append(part)
            else:
                other_parts.append(part)
        
        # 如果有多个facing camera描述，只保留权重最高的一个
        if len(facing_camera_parts) > 1:
            # 提取权重并排序
            import re
            for part in facing_camera_parts:
                weight_match = re.search(r':(\d+\.?\d*)', part["text"])
                if weight_match:
                    part["_weight"] = float(weight_match.group(1))
                else:
                    part["_weight"] = 1.0
            
            # 按权重降序排序，只保留第一个（权重最高的）
            facing_camera_parts.sort(key=lambda x: x.get("_weight", 1.0), reverse=True)
            kept_part = facing_camera_parts[0]
            if len(facing_camera_parts) > 1:
                removed_count = len(facing_camera_parts) - 1
                print(f"  ✓ 移除 {removed_count} 个重复的 'facing camera' 描述，保留权重最高的 ({kept_part.get('_weight', 1.0):.1f})")
            return [kept_part] + other_parts
        
        return parts

    def _remove_duplicate_and_similar(self, parts: List[Dict]) -> List[Dict]:
        """
        通用的去重函数：移除重复和语义相近的描述
        
        检测并合并：
        1. 完全重复的描述
        2. 语义相近的描述（如同义词、近义词）
        3. 包含相同核心概念的描述
        """
        import re
        
        # 定义语义相近的词汇组（同义词组）
        similar_groups = [
            # 单人相关
            {
                "keywords": ["single person", "lone figure", "only one character", "one person only", 
                           "sole character", "single individual", "单人", "独行", "只有一个角色", 
                           "仅一人", "唯一角色", "单独个体"],
                "merged": "single person, only one character",
                "type": "constraint"
            },
            # 正面朝向相关
            {
                "keywords": ["facing camera", "front view", "face forward", "character facing viewer", 
                           "frontal view", "面向镜头", "正面视角", "人物面向观众", "正面", "面向"],
                "merged": "facing camera, front view",
                "type": "camera"
            },
            # 仙侠风格相关
            {
                "keywords": ["xianxia", "chinese fantasy", "仙侠", "修仙", "古风", "immortal", "cultivator"],
                "merged": "xianxia fantasy",
                "type": "style"
            },
            # 远景/全身相关
            {
                "keywords": ["wide shot", "full body", "full figure", "全身", "远景", "wide view", "full view"],
                "merged": "wide shot, full body",
                "type": "camera"
            },
            # 中景/半身相关
            {
                "keywords": ["medium shot", "mid shot", "upper body", "half body", "中景", "半身", "上半身"],
                "merged": "medium shot, upper body",
                "type": "camera"
            },
            # 特写相关
            {
                "keywords": ["close-up", "closeup", "close up", "face close-up", "portrait shot", "headshot", 
                           "特写", "近景", "面部特写"],
                "merged": "close-up, face close-up",
                "type": "camera"
            }
        ]
        
        # 第一步：处理语义相近的组
        processed_parts = []
        used_indices = set()
        
        for group in similar_groups:
            matching_parts = []
            for i, part in enumerate(parts):
                if i in used_indices:
                    continue
                part_lower = part["text"].lower()
                # 检查是否包含该组的任何关键词
                if any(kw in part_lower for kw in group["keywords"]):
                    matching_parts.append((i, part))
            
            # 如果有多个匹配的部分，合并它们
            if len(matching_parts) > 1:
                # 提取最高权重
                max_weight = 1.0
                min_index = float('inf')
                for idx, part in matching_parts:
                    weight_match = re.search(r':(\d+\.?\d*)', part["text"])
                    if weight_match:
                        weight = float(weight_match.group(1))
                        if weight > max_weight:
                            max_weight = weight
                    if part["index"] < min_index:
                        min_index = part["index"]
                
                # 创建合并后的描述
                merged_text = f"({group['merged']}:{max_weight:.1f})"
                
                # 创建新的part
                merged_part = {
                    "text": merged_text,
                    "type": group["type"],
                    "analysis": matching_parts[0][1]["analysis"].copy(),
                    "index": min_index
                }
                merged_part["analysis"]["token_count"] = self.token_estimator.estimate(merged_text)
                
                processed_parts.append(merged_part)
                # 标记这些部分已使用
                for idx, _ in matching_parts:
                    used_indices.add(idx)
                
                removed_count = len(matching_parts) - 1
                print(f"  ✓ 合并 {len(matching_parts)} 个语义相近的描述（{group['merged']}），移除 {removed_count} 个重复项")
            elif len(matching_parts) == 1:
                # 只有一个匹配，直接保留
                idx, part = matching_parts[0]
                processed_parts.append(part)
                used_indices.add(idx)
        
        # 第二步：处理其他部分（未匹配到任何组的）
        for i, part in enumerate(parts):
            if i not in used_indices:
                processed_parts.append(part)
        
        # 第二步半：检测重复的关键词（如"scroll"、"Three dazzling suns"）
        # 先检测并合并包含相同核心关键词的部分
        keyword_groups = {}  # 关键词 -> [parts]
        for part in processed_parts:
            text_lower = part["text"].lower()
            # 提取核心关键词（名词、主要物体等）
            words = re.findall(r'\b[a-z]+(?:\s+[a-z]+)*\b', text_lower)
            if words:
                # 取前2-3个词作为核心关键词（如"golden scroll", "three dazzling suns"）
                core_keyword = ' '.join(words[:3]) if len(words) >= 3 else words[0]
                # 也检查单个关键词（如"scroll"）
                single_keyword = words[0] if words else None
                
                # 检查是否与其他部分共享核心关键词
                found_group = False
                for keyword, group_parts in keyword_groups.items():
                    if core_keyword in keyword or keyword in core_keyword or \
                       (single_keyword and (single_keyword in keyword or keyword in single_keyword)):
                        keyword_groups[keyword].append(part)
                        found_group = True
                        break
                
                if not found_group:
                    # 创建新的关键词组
                    keyword_groups[core_keyword] = [part]
        
        # 合并包含相同关键词的部分
        merged_parts = []
        for keyword, group_parts in keyword_groups.items():
            if len(group_parts) > 1:
                # 多个部分包含相同关键词，合并它们
                max_weight = 1.0
                min_index = float('inf')
                merged_text_parts = []
                
                for part in group_parts:
                    weight_match = re.search(r':(\d+\.?\d*)', part["text"])
                    if weight_match:
                        weight = float(weight_match.group(1))
                        if weight > max_weight:
                            max_weight = weight
                    if part["index"] < min_index:
                        min_index = part["index"]
                    
                    # 提取文本内容（去除权重）
                    text_content = re.sub(r':\d+\.?\d*\)?$', '', part["text"])
                    text_content = re.sub(r'^\(|\)$', '', text_content).strip()
                    merged_text_parts.append(text_content)
                
                # 合并文本，去重
                unique_parts = []
                seen_words = set()
                for text_part in merged_text_parts:
                    words_in_part = set(text_part.lower().split())
                    # 如果这个部分包含新词汇，添加它
                    if not words_in_part.issubset(seen_words):
                        unique_parts.append(text_part)
                        seen_words.update(words_in_part)
                
                # 创建合并后的描述
                merged_content = ', '.join(unique_parts)
                merged_text = f"({merged_content}:{max_weight:.1f})"
                
                merged_part = {
                    "text": merged_text,
                    "type": group_parts[0]["type"],
                    "analysis": group_parts[0]["analysis"].copy(),
                    "index": min_index
                }
                merged_part["analysis"]["token_count"] = self.token_estimator.estimate(merged_text)
                merged_parts.append(merged_part)
                
                removed_count = len(group_parts) - 1
                print(f"  ✓ 合并 {len(group_parts)} 个包含相同关键词的部分（关键词: {keyword[:30]}...），移除 {removed_count} 个重复项")
            else:
                # 只有一个部分，直接保留
                merged_parts.append(group_parts[0])
        
        # 第三步：检测完全重复的描述（相同或几乎相同的文本）
        # 使用merged_parts（如果存在）或processed_parts
        parts_for_final_check = merged_parts if merged_parts else processed_parts
        final_parts = []
        seen_texts = {}  # 存储规范化文本 -> (part, index_in_final_parts)
        
        for part in parts_for_final_check:
            # 规范化文本（去除权重，用于比较）
            text_normalized = re.sub(r':\d+\.?\d*', '', part["text"]).lower().strip()
            text_normalized = re.sub(r'[()]', '', text_normalized)
            
            # 检查是否已经见过类似的文本
            is_duplicate = False
            duplicate_key = None
            
            for seen_text, (seen_part, seen_idx) in seen_texts.items():
                # 计算相似度（简单的词汇重叠度）
                seen_words = set(seen_text.split())
                current_words = set(text_normalized.split())
                
                if len(seen_words) > 0 and len(current_words) > 0:
                    overlap = len(seen_words & current_words)
                    similarity = overlap / max(len(seen_words), len(current_words))
                    
                    # 如果相似度超过80%，认为是重复
                    if similarity > 0.8:
                        # 比较权重，保留权重更高的那个
                        current_weight_match = re.search(r':(\d+\.?\d*)', part["text"])
                        seen_weight_match = re.search(r':(\d+\.?\d*)', seen_part["text"])
                        
                        current_weight = float(current_weight_match.group(1)) if current_weight_match else 1.0
                        seen_weight = float(seen_weight_match.group(1)) if seen_weight_match else 1.0
                        
                        if current_weight > seen_weight:
                            # 当前部分权重更高，替换旧的部分
                            duplicate_key = seen_text
                            # 从final_parts中移除旧的部分
                            final_parts[seen_idx] = part
                            seen_texts[seen_text] = (part, seen_idx)
                        else:
                            # 已存在的部分权重更高，保留它
                            pass
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                # 添加新部分
                final_parts.append(part)
                seen_texts[text_normalized] = (part, len(final_parts) - 1)
        
        # 计算移除数量
        if len(final_parts) < len(parts_for_final_check):
            removed = len(parts_for_final_check) - len(final_parts)
            print(f"  ✓ 移除 {removed} 个完全重复的描述")
        
        return final_parts
    
    def _remove_internal_duplicates(self, text: str) -> str:
        """
        移除单个 part 内部的重复词汇
        
        处理情况：
        1. 包含关系：如 "scroll" 和 "golden scroll" -> 只保留 "golden scroll"
        2. 语义重复：如 "single person" 和 "only one character" -> 只保留一个
        """
        import re
        
        # 提取权重（如果有）
        weight_match = re.search(r':(\d+\.?\d*)\)?$', text)
        weight = weight_match.group(1) if weight_match else None
        has_paren = text.strip().startswith('(')
        
        # 移除权重和括号，提取内容
        content = re.sub(r':\d+\.?\d*\)?$', '', text)
        content = re.sub(r'^\(|\)$', '', content).strip()
        
        # 分割成词汇列表
        words = [w.strip() for w in content.split(',') if w.strip()]
        
        if len(words) <= 1:
            return text  # 只有一个词，不需要去重
        
        # 定义语义重复组（同义词组）
        semantic_groups = [
            # 单人相关：如果同时存在多个，只保留最简洁的一个
            {
                "keywords": ["single person", "lone figure", "only one character", "one person only", 
                           "sole character", "single individual"],
                "keep": "single person, only one character",  # 默认保留组合
                "single_keep": "single person"  # 如果只需要一个，保留这个
            },
            # 正面朝向相关
            {
                "keywords": ["facing camera", "front view", "face forward", "character facing viewer", 
                           "frontal view"],
                "keep": "facing camera, front view",
                "single_keep": "facing camera"
            },
            # 特写相关
            {
                "keywords": ["close-up", "closeup", "close up", "face close-up", "portrait shot", "headshot"],
                "keep": "close-up, face close-up",
                "single_keep": "close-up"
            }
        ]
        
        # 第一步：处理语义重复组
        processed_words = []
        used_indices = set()
        
        for group in semantic_groups:
            matching_indices = []
            matching_words = []
            for i, word in enumerate(words):
                if i in used_indices:
                    continue
                word_lower = word.lower()
                if any(kw in word_lower for kw in group["keywords"]):
                    matching_indices.append(i)
                    matching_words.append(word)
            
            if len(matching_indices) > 1:
                # 如果匹配到多个语义重复的词，只保留一个最简洁的
                # 优先选择最短且最常用的
                best_word = None
                best_length = float('inf')
                for word in matching_words:
                    if not word or not word.strip():  # 跳过空词
                        continue
                    word_lower = word.lower()
                    # 优先选择 "single person" 或 "only one character"（最简洁）
                    if "single person" in word_lower or "only one character" in word_lower:
                        if len(word) < best_length:
                            best_word = word
                            best_length = len(word)
                
                # 如果没有找到最简洁的，选择最短的
                if best_word is None and matching_words:
                    # 过滤掉空词
                    valid_words = [w for w in matching_words if w and w.strip()]
                    if valid_words:
                        best_word = min(valid_words, key=len)
                
                # 只有当 best_word 不为 None 时才添加
                if best_word and best_word.strip():
                    processed_words.append(best_word)
                    for idx in matching_indices:
                        used_indices.add(idx)
            elif len(matching_indices) == 1:
                # 只有一个匹配，直接保留
                idx = matching_indices[0]
                word = words[idx]
                if word and word.strip():  # 确保不是空词
                    processed_words.append(word)
                    used_indices.add(idx)
        
        # 第二步：处理包含关系（如果一个词是另一个词的一部分）
        remaining_words = [words[i] for i in range(len(words)) if i not in used_indices]
        
        # 按长度排序（长的在前），这样短的词如果被包含在长词中，会被检测到
        remaining_words_sorted = sorted(remaining_words, key=len, reverse=True)
        final_words = []
        
        for word in remaining_words_sorted:
            if not word or not word.strip():  # 跳过空词
                continue
            word_lower = word.lower()
            # 检查是否被已添加的词包含
            is_contained = False
            for existing_word in final_words:
                if not existing_word or not existing_word.strip():  # 跳过空词
                    continue
                existing_lower = existing_word.lower()
                # 检查 word 是否被 existing_word 包含（作为完整词，不是子串）
                # 例如："scroll" 在 "golden scroll" 中
                if word_lower in existing_lower:
                    # 进一步检查：确保是完整的词，而不是部分匹配
                    # 使用正则表达式检查是否是完整的词
                    pattern = r'\b' + re.escape(word_lower) + r'\b'
                    if re.search(pattern, existing_lower):
                        is_contained = True
                        break
                # 或者 existing_word 被 word 包含（word 更长）
                elif existing_lower in word_lower:
                    pattern = r'\b' + re.escape(existing_lower) + r'\b'
                    if re.search(pattern, word_lower):
                        # 移除较短的词，保留较长的词
                        final_words.remove(existing_word)
                        is_contained = False
                        break
            
            if not is_contained:
                final_words.append(word)
        
        # 合并处理后的词汇（过滤掉 None 和空词）
        valid_final_words = [w for w in final_words if w and w.strip()]
        processed_words.extend(valid_final_words)
        
        # 过滤掉 None 和空词
        processed_words = [w for w in processed_words if w and w.strip()]
        
        # 如果词汇有变化，重建文本
        if len(processed_words) != len(words) or set(processed_words) != set(words):
            # 去重并保持顺序
            seen = set()
            unique_words = []
            for word in processed_words:
                if word and word.strip() and word not in seen:
                    seen.add(word)
                    unique_words.append(word)
            
            # 重建文本
            new_content = ', '.join(unique_words)
            if weight:
                new_text = f"({new_content}:{weight})" if has_paren else f"{new_content}:{weight}"
            else:
                new_text = f"({new_content})" if has_paren else new_content
            
            return new_text
        
        return text
    
    def enhance_prompt_part(self, part: str, part_type: str) -> str:
        """
        通用的prompt部分增强方法（兼容层）
        
        ⚡ 新架构：使用 AST + 语义增强器，而不是字符串操作
        
        Args:
            part: prompt部分文本
            part_type: 部分类型
            
        Returns:
            增强后的prompt部分文本
        """
        if not part:
            return part
        
        # ⚡ 使用 AST 架构进行增强
        # 1. 解析为 AST
        node = self.ast_builder.parse_part(part, index=0)
        node.type = part_type  # 使用传入的类型
        
        # 2. 创建临时 AST 进行增强
        temp_ast = PromptAST([node])
        temp_ast = self.semantic_enhancer.enhance_ast(temp_ast)
        
        # 3. 返回增强后的节点字符串
        enhanced_node = temp_ast.nodes[0]
        return enhanced_node.to_string(include_weight=True)




