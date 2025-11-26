#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用Prompt构建方法
基于场景意图分析，移除所有特殊处理
"""

def build_prompt_universal(self, scene: Dict[str, Any], include_character: Optional[bool] = None, script_data: Dict[str, Any] = None, previous_scene: Optional[Dict[str, Any]] = None) -> str:
    """
    通用Prompt构建方法
    基于场景意图分析，不依赖特殊规则
    
    Args:
        scene: 场景数据字典
        include_character: 是否包含角色描述。None 时自动判断
        script_data: 脚本数据（用于场景模板匹配）
        previous_scene: 上一个场景（用于连续性）
    """
    # 1. 分析场景意图
    intent = self.intent_analyzer.analyze(scene)
    
    print(f"  ℹ 场景意图分析:")
    print(f"    - 主要实体: {intent['primary_entity']['type'] if intent['primary_entity'] else 'None'}")
    print(f"    - 动作类型: {intent['action_type']}")
    print(f"    - 视角: {intent['viewpoint']['type']} (权重: {intent['viewpoint']['weight']})")
    print(f"    - 强调项: {intent['emphasis']}")
    print(f"    - 排除项: {intent['exclusions']}")
    
    # 2. 构建Prompt优先级列表
    priority_parts: List[str] = []
    
    # 2.1 风格（固定，最高优先级）
    xianxia_style = "xianxia fantasy" if self.ascii_only_prompt else "仙侠风格"
    priority_parts.append(xianxia_style)
    
    # 2.2 主要实体（如果有）
    if intent['primary_entity']:
        entity = intent['primary_entity']
        entity_text = " ".join(entity.get("keywords", []))
        if entity_text:
            weight = entity.get("weight", 1.5)
            priority_parts.append(f"({entity_text}:{weight})")
            print(f"  ✓ 添加主要实体: {entity_text} (权重: {weight})")
    
    # 2.3 强调项（高权重）
    if intent['emphasis']:
        emphasis_text = ", ".join(intent['emphasis'])
        priority_parts.append(f"({emphasis_text}:1.8)")
        print(f"  ✓ 添加强调项: {emphasis_text}")
    
    # 2.4 视角描述
    viewpoint = intent['viewpoint']
    if viewpoint['type'] != 'front' or viewpoint['weight'] > 1.0:
        vp_keywords = self.intent_analyzer.viewpoint_patterns.get(viewpoint['type'], [])
        if vp_keywords:
            vp_text = vp_keywords[0]
            priority_parts.append(f"({vp_text}:{viewpoint['weight']})")
            print(f"  ✓ 添加视角: {vp_text} (权重: {viewpoint['weight']})")
    
    # 2.5 动作/姿势描述
    description = scene.get("description", "")
    if description:
        priority_parts.append(f"({description}:1.3)")
        print(f"  ✓ 添加动作描述: {description[:50]}...")
    
    # 2.6 构图描述
    visual = scene.get("visual", {})
    if isinstance(visual, dict):
        composition = visual.get("composition", "")
        if composition:
            priority_parts.append(f"({composition}:1.4)")
            print(f"  ✓ 添加构图描述: {composition[:50]}...")
    
    # 2.7 角色描述（如果有角色实体）
    if intent['primary_entity'] and intent['primary_entity'].get("type") == "character":
        # 使用通用角色描述逻辑
        character_pose = visual.get("character_pose", "") if isinstance(visual, dict) else ""
        if character_pose:
            # 检查是否包含朝向信息
            pose_lower = character_pose.lower()
            has_facing = any(kw in pose_lower for kw in ["facing", "front", "正面", "面向", "forward", "toward camera", "facing camera"])
            has_back = any(kw in pose_lower for kw in ["back", "背影", "behind", "rear", "turned away", "facing away"])
            
            if has_facing:
                # 有正面朝向，使用高权重
                priority_parts.append(f"({character_pose}:1.8)")
                priority_parts.append("(facing camera, front view, face forward:1.6)")
                print(f"  ✓ 添加角色姿势（正面朝向）: {character_pose}")
            elif not has_back:
                # 没有明确朝向，默认添加正面
                priority_parts.append(f"({character_pose}:1.3)")
                priority_parts.append("(facing camera, front view, face forward:1.8)")
                print(f"  ✓ 添加角色姿势（默认正面）: {character_pose}")
            else:
                # 明确要求背影
                priority_parts.append(f"({character_pose}:1.3)")
                print(f"  ✓ 添加角色姿势（背影）: {character_pose}")
    
    # 2.8 特效描述
    if isinstance(visual, dict):
        fx = visual.get("fx", "")
        if fx:
            priority_parts.append(f"({fx}:1.2)")
            print(f"  ✓ 添加特效描述: {fx[:50]}...")
    
    # 3. 构建最终Prompt
    final_prompt = ", ".join(filter(None, priority_parts))
    
    # 4. 检查Token限制并优化
    estimated_tokens = self._estimate_clip_tokens(final_prompt)
    if estimated_tokens > 70:
        print(f"  ⚠ 警告: Prompt长度 ({estimated_tokens} tokens) 超过限制，开始精简...")
        final_prompt = self._smart_optimize_prompt(priority_parts, max_tokens=70)
        estimated_tokens = self._estimate_clip_tokens(final_prompt)
        print(f"  ✓ 精简后 Prompt 长度: {estimated_tokens} tokens")
    
    return final_prompt

