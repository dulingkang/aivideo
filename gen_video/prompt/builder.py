"""
Prompt构建器

负责根据场景数据构建完整的Prompt，这是Prompt模块的核心组件。
"""

from typing import Dict, Any, List, Optional
from .token_estimator import TokenEstimator
from .parser import PromptParser
from .optimizer import PromptOptimizer


class PromptBuilder:
    """Prompt构建器"""
    
    def __init__(
        self,
        token_estimator: TokenEstimator,
        parser: PromptParser,
        optimizer: PromptOptimizer,
        intent_analyzer: Any,  # SceneIntentAnalyzer
        character_profiles: Dict[str, Any],
        scene_profiles: Dict[str, Any],
        ascii_only_prompt: bool = False,
        identify_characters_fn: Any = None,  # 角色识别函数
        needs_character_fn: Any = None,  # 判断是否需要角色函数
        clip_tokenizer: Any = None,  # CLIP tokenizer（可选）
    ):
        """
        初始化Prompt构建器
        
        Args:
            token_estimator: Token估算器
            parser: Prompt解析器
            optimizer: Prompt优化器
            intent_analyzer: 场景意图分析器
            character_profiles: 角色配置字典
            scene_profiles: 场景配置字典
            ascii_only_prompt: 是否只使用ASCII字符
            identify_characters_fn: 角色识别函数（从ImageGenerator注入）
            needs_character_fn: 判断是否需要角色函数（从ImageGenerator注入）
            clip_tokenizer: CLIP tokenizer（可选，用于准确计算token数）
        """
        self.token_estimator = token_estimator
        self.parser = parser
        self.optimizer = optimizer
        self.intent_analyzer = intent_analyzer
        self.character_profiles = character_profiles
        self.scene_profiles = scene_profiles
        self.ascii_only_prompt = ascii_only_prompt
        self._identify_characters = identify_characters_fn
        self._needs_character = needs_character_fn
        self._clip_tokenizer = clip_tokenizer
    
    def build(
        self,
        scene: Dict[str, Any],
        include_character: Optional[bool] = None,
        script_data: Dict[str, Any] = None,
        previous_scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        根据场景数据构建 prompt
        
        通用版本：基于场景意图分析，智能构建Prompt，不依赖特殊规则。
        
        Args:
            scene: 场景数据字典
            include_character: 是否包含主角描述。None 时自动判断
            script_data: 脚本数据（用于场景模板匹配）
            previous_scene: 前一个场景（用于连贯性）
            
        Returns:
            构建好的prompt字符串
        """
        # 注意：这个方法非常复杂（约1500行），需要从 ImageGenerator.build_prompt 中完整提取
        # 为了保持代码完整性，这里先创建一个占位实现
        # 实际实现需要逐步迁移
        
        # TODO: 完整迁移 build_prompt 方法的所有逻辑
        # 这是一个大工程，需要仔细提取以下部分：
        # 1. 场景意图分析
        # 2. 角色处理逻辑
        # 3. 无人物场景处理
        # 4. Prompt优化和token管理
        # 5. 中英文翻译
        
        # ========== 第一步：场景意图分析（通用分析，不依赖特殊规则）==========
        intent = self.intent_analyzer.analyze(scene)
        
        print(f"  ℹ 场景意图分析:")
        if intent['primary_entity']:
            print(f"    - 主要实体: {intent['primary_entity']['type']} (权重: {intent['primary_entity'].get('weight', 1.5)})")
        else:
            print(f"    - 主要实体: None")
        print(f"    - 动作类型: {intent['action_type']}")
        print(f"    - 视角: {intent['viewpoint']['type']} (权重: {intent['viewpoint']['weight']})")
        if intent['emphasis']:
            print(f"    - 强调项: {', '.join(intent['emphasis'][:3])}")
        if intent['exclusions']:
            print(f"    - 排除项: {', '.join(intent['exclusions'])}")
        
        # 根据意图分析结果判断是否需要角色
        if include_character is None:
            # 如果主要实体是角色，则需要角色
            if intent['primary_entity'] and intent['primary_entity'].get('type') == 'character':
                include_character = True
            else:
                include_character = False
                # 特殊处理：如果是"人物观察环境/物体"的场景，明确排除人物
                # 检测观察关键词（sees, revealing, showing等）
                all_text_lower = " ".join([
                    str(scene.get("description", "")),
                    str(scene.get("prompt", "")),
                    str(scene.get("visual", {}).get("composition", "") if isinstance(scene.get("visual"), dict) else "")
                ]).lower()
                observation_keywords = ["sees", "revealing", "showing", "只见", "映入眼帘", "展现"]
                has_observation = any(kw in all_text_lower for kw in observation_keywords)
                if has_observation and intent['primary_entity'] and intent['primary_entity'].get('type') != 'character':
                    print(f"  ℹ 检测到观察场景（人物观察{intent['primary_entity'].get('type')}），排除人物，以{intent['primary_entity'].get('type')}为主")
        
        # 使用优先级列表，确保关键信息在前
        priority_parts: List[str] = []  # 高优先级（前 77 tokens）
        secondary_parts: List[str] = []  # 次要信息（可能被截断）
        
        raw_prompt = scene.get("prompt") or ""
        used_prompt_as_camera = False
        
        # 先确定镜头类型（用于后续判断）
        camera_desc = scene.get("camera") or ""
        if not camera_desc and raw_prompt and self._looks_like_camera_prompt(raw_prompt):
            camera_desc = raw_prompt
            used_prompt_as_camera = True
        
        shot_type_for_prompt = {
            "is_wide": False,
            "is_medium": False,
            "is_close": False,
            "is_full_body": False,
            "is_eye_closeup": False,  # 眼睛特写标记
            "is_face_closeup": False,  # 面部特写标记
        }
        
        if camera_desc:
            lowered = camera_desc.lower()
            if any(kw in lowered for kw in ["wide", "long", "遠景", "远景", "全景"]):
                shot_type_for_prompt["is_wide"] = True
            if any(kw in lowered for kw in ["medium", "mid", "中景"]):
                shot_type_for_prompt["is_medium"] = True
            if any(kw in lowered for kw in ["close", "closeup", "portrait", "headshot", "特写", "近景"]):
                # 检查是否是眼睛特写或面部特写场景（需要保持特写）
                is_eye_closeup = any(kw in lowered for kw in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔', 'extreme close'])
                is_face_closeup = any(kw in lowered for kw in ['face', 'facial', 'portrait', 'headshot', '面部', '脸部', '头像', 'close-up on face', 'closeup on face'])
                if is_eye_closeup:
                    # 眼睛特写场景：保持特写标记
                    shot_type_for_prompt["is_close"] = True
                    shot_type_for_prompt["is_eye_closeup"] = True  # 添加眼睛特写标记
                elif is_face_closeup:
                    # 面部特写场景：保持特写标记，不转换为中景
                    shot_type_for_prompt["is_close"] = True
                    shot_type_for_prompt["is_face_closeup"] = True  # 添加面部特写标记
                else:
                    # 其他特写场景：标记为特写，但后续会转换为中景
                    shot_type_for_prompt["is_close"] = True
            if any(kw in lowered for kw in ["full", "全身"]):
                shot_type_for_prompt["is_full_body"] = True
        
        # ========== 第一部分：仙侠风格（最高优先级，放在最前面）==========
        use_chinese_prompt = not self.ascii_only_prompt
        
        if use_chinese_prompt:
            xianxia_style = "仙侠风格"
        else:
            xianxia_style = "xianxia fantasy"
        
        priority_parts.append(xianxia_style)
        print(f"  ✓ 仙侠风格（最高优先级）: {xianxia_style}")
        
        # ========== 基于意图分析添加主要实体（智能综合权重调整）==========
        if intent['primary_entity']:
            entity = intent['primary_entity']
            entity_text = " ".join(entity.get("keywords", []))
            if entity_text:
                # 使用综合权重调整后的实体权重
                weight_adjustments = intent.get('weight_adjustments', {})
                entity_weight = weight_adjustments.get('entity_weight', entity.get("weight", 1.5))
                
                # 如果是物体，使用更高权重并强调（去除重复，使用更简洁的描述）
                if entity.get('type') == 'object':
                    # 去除重复，使用更简洁的描述：只出现一次实体名称，用不同的描述词强调
                    # 对于特定物体（如scroll），添加更具体的描述词，避免生成其他物体
                    if "scroll" in entity_text.lower() or "卷轴" in entity_text.lower():
                        # 检查 entity_text 中是否已经有 "golden"，如果有就不重复添加
                        entity_lower = entity_text.lower()
                        if "golden" not in entity_lower:
                            entity_text = f"{entity_text}, golden scroll"
                        # 添加强调词（去重）
                        emphasis_parts = []
                        if "prominent" not in entity_lower:
                            emphasis_parts.append("prominent")
                        if "clearly visible" not in entity_lower:
                            emphasis_parts.append("clearly visible")
                        if "main element" not in entity_lower:
                            emphasis_parts.append("main element")
                        if emphasis_parts:
                            entity_text = f"{entity_text}, {', '.join(emphasis_parts)}"
                        # 添加排除项
                        exclusion_parts = []
                        if "no weapons" not in entity_lower and "weapon" not in entity_lower:
                            exclusion_parts.append("no weapons")
                        if "no tools" not in entity_lower and "tool" not in entity_lower:
                            exclusion_parts.append("no tools")
                        if exclusion_parts:
                            entity_text = f"{entity_text}, {', '.join(exclusion_parts)}"
                        priority_parts.append(f"({entity_text}:{entity_weight:.2f})")
                        print(f"  ✓ 添加主要物体（卷轴，智能综合权重{entity_weight:.2f}）: {entity_text[:60]}...")
                    elif "city" in entity_text.lower() or "城市" in entity_text.lower() or "immortal city" in entity_text.lower():
                        priority_parts.append(f"({entity_text}, city silhouette, prominent, clearly visible, main element, no people, no characters:{entity_weight:.2f})")
                        print(f"  ✓ 添加主要物体（城市，智能综合权重{entity_weight:.2f}）: {entity_text}")
                    else:
                        priority_parts.append(f"({entity_text}, prominent, clearly visible, main element:{entity_weight:.2f})")
                        print(f"  ✓ 添加主要物体（智能综合权重{entity_weight:.2f}）: {entity_text}")
                else:
                    priority_parts.append(f"({entity_text}:{entity_weight:.2f})")
                    print(f"  ✓ 添加主要实体（智能综合权重{entity_weight:.2f}）: {entity_text}")
        
        # ========== 基于意图分析添加强调项（通用处理）==========
        if intent['emphasis']:
            emphasis_text = ", ".join(intent['emphasis'][:3])  # 最多3个强调项
            priority_parts.append(f"({emphasis_text}:1.8)")
            print(f"  ✓ 添加强调项: {emphasis_text}")
        
        # ========== 无人物场景处理：智能排序，优先最重要的细节 ==========
        if not include_character:
            # 注意：无人物场景的完整处理逻辑非常复杂（约700行）
            # 这里先创建一个简化版本，完整逻辑需要从 ImageGenerator.build_prompt() 中迁移
            # TODO: 完整迁移无人物场景处理逻辑（从 line 1963 到 line 2531）
            print(f"  ℹ 检测到无人物场景，智能构建Prompt（优先最重要的细节）")
            
            # 简化版：提取核心信息
            description_text = self._clean_prompt_text(scene.get("description") or "")
            prompt_text = self._clean_prompt_text(scene.get("prompt") or "")
            visual = scene.get("visual", {}) or {}
            
            # 从composition中提取关键信息
            if isinstance(visual, dict):
                composition = self._clean_prompt_text(visual.get("composition") or "")
                if composition:
                    # 检查 composition 中是否已经有排除项（如 "no person", "no character"）
                    composition_lower = composition.lower()
                    has_exclusion_in_composition = any(kw in composition_lower for kw in [
                        'no person', 'no character', 'no human', 'no people',
                        '无人物', '无角色', '无人', '无人物场景'
                    ])
                    
                    # 如果 composition 中已经有排除项，就不添加额外的排除项
                    exclusion_text = ""
                    # 特殊处理：如果是观察场景，强制添加人物排除项
                    observation_keywords = ["sees", "revealing", "showing", "只见", "映入眼帘", "展现"]
                    has_observation = any(kw in composition_lower for kw in observation_keywords)
                    force_exclude_character = has_observation and not has_exclusion_in_composition
                    
                    if not has_exclusion_in_composition and (intent.get('exclusions') or force_exclude_character):
                        # 如果是观察场景，强制添加人物排除项
                        if force_exclude_character:
                            if self.ascii_only_prompt:
                                exclusion_text = ", no person, no character, no human, no people"
                            else:
                                exclusion_text = ", no person, no character, no human, no people, 无人物, 无角色, 无人"
                            print(f"  ✓ 观察场景：强制添加人物排除项，确保以环境/物体为主")
                        else:
                            # 只添加英文排除项（如果 ascii_only_prompt 为 True）
                            exclusions = intent['exclusions']
                            if self.ascii_only_prompt:
                                # 过滤掉中文，只保留英文
                                exclusions = [e for e in exclusions if not any('\u4e00' <= c <= '\u9fff' for c in e)]
                            # 去重：检查 composition 中是否已经有类似的排除项
                            filtered_exclusions = []
                            for exc in exclusions:
                                exc_lower = exc.lower()
                                # 检查是否与 composition 中的内容重复
                                if not any(kw in composition_lower for kw in exc_lower.split()):
                                    filtered_exclusions.append(exc)
                            if filtered_exclusions:
                                exclusion_text = ", " + ", ".join(filtered_exclusions)
                    
                    # 构建最终 prompt，避免重复 "prominent, main element"
                    composition_clean = composition
                    # 如果 composition 中已经有 "prominent" 或 "main element"，就不重复添加
                    if "prominent" not in composition_lower and "main element" not in composition_lower:
                        composition_clean = f"{composition}, prominent, main element"
                    
                    priority_parts.append(f"({composition_clean}{exclusion_text}:2.0)")
                    print(f"  ✓ 添加主要物体（最高优先级，权重2.0）: {composition[:60]}...")
            
            # 构建最终提示词并检查token数
            priority_prompt = ", ".join(filter(None, priority_parts))
            estimated_tokens = self.token_estimator.estimate(priority_prompt)
            
            # 如果使用中文且SDXL模型对中文支持不好，考虑翻译成英文
            if not self.ascii_only_prompt:
                final_prompt = priority_prompt
                print(f"  ℹ 使用中文 prompt（SDXL可能理解不佳，如果生成效果不好，建议设置 ascii_only_prompt: true）")
            else:
                final_prompt = self._translate_chinese_to_english(priority_prompt)
                print(f"  ℹ 已翻译为英文 prompt")
            
            print(f"  📊 无人物场景Prompt长度: {estimated_tokens} tokens (核心部分: {len(priority_parts)} 项)")
            print(f"  📝 最终Prompt文本: {final_prompt}")
            return final_prompt
        
        # ========== 第二部分：角色/人脸特征（紧跟风格之后，仅当需要角色时）==========
        if include_character:
            # 用户反馈：场景5和7生成了多个人物，在所有人物场景都添加单人约束
            # 在角色描述之前添加单人约束，确保最高优先级
            if self.ascii_only_prompt:
                priority_parts.insert(0, "(single person, lone figure, only one character, one person only, sole character, single individual:2.0)")
            else:
                priority_parts.insert(0, "(单人，独行，只有一个角色，仅一人，唯一角色，单独个体:2.0)")
            print(f"  ✓ 人物场景：在prompt最前面添加单人约束（权重2.0，防止多个人物）")
            # 识别场景中的所有角色
            if self._identify_characters:
                identified_characters = self._identify_characters(scene)
            else:
                identified_characters = []
            
            # 如果识别到其他角色（不仅仅是韩立），使用角色描述生成
            if identified_characters:
                # 优先使用第一个识别的角色（通常是主要角色）
                primary_character = identified_characters[0]
                
                # 通用角色处理：不依赖特定角色名称
                # 使用角色模板（如果存在）
                character_profile = self._get_character_profile(primary_character)
                if character_profile:
                    # 构建角色描述 prompt
                    character_desc = self._build_character_description_prompt(character_profile, shot_type_for_prompt)
                    if character_desc:
                        # 前置角色描述到第2位（在风格之后），确保高优先级
                        # 如果已经有风格描述，插入到第2位；否则追加
                        if len(priority_parts) > 0:
                            priority_parts.insert(1, character_desc)
                        else:
                            priority_parts.append(character_desc)
                        print(f"  ✓ 应用角色描述（前置到第2位）: {character_profile.get('character_name', primary_character)}")
                        print(f"  📝 角色描述内容: {character_desc[:100]}...")  # 添加调试日志，显示角色描述内容
                else:
                    # 如果没有角色模板，从场景描述中提取角色信息
                    print(f"  ⚠ 未找到角色模板: {primary_character}，将从场景描述中提取角色信息")
                    # 尝试从 character_pose 或 description 中提取角色描述
                    visual = scene.get("visual", {}) or {}
                    if isinstance(visual, dict):
                        character_pose = visual.get("character_pose", "")
                        if character_pose:
                            priority_parts.append(f"({character_pose}:1.5)")
                            print(f"  ✓ 使用 character_pose 作为角色描述: {character_pose[:50]}...")
                
                # 基于意图分析处理视角（智能综合权重调整）
                weight_adjustments = intent.get('weight_adjustments', {})
                viewpoint = intent.get('viewpoint', {})
                viewpoint_type = viewpoint.get('type', 'front')
                viewpoint_weight = weight_adjustments.get('viewpoint_weight', viewpoint.get('weight', 1.0))
                
                # 使用综合权重调整后的视角权重
                # 对于所有人物场景，默认添加正面朝向提示（除非明确要求背面）
                viewpoint_explicit = viewpoint.get('explicit', False)
                # 如果视角不是背面，都添加正面朝向提示
                if viewpoint_type != 'back':
                    use_chinese = not self.ascii_only_prompt
                    # 如果明确要求正面，使用更高权重；否则使用默认高权重
                    if viewpoint_explicit and viewpoint_type == 'front':
                        final_weight = 2.0
                    elif viewpoint_type == 'front':
                        final_weight = max(viewpoint_weight, 1.8)  # 至少1.8，确保正面朝向明显
                    else:
                        final_weight = 1.8  # 默认高权重，确保正面朝向
                    
                    facing_prompt = f"(正面，面向镜头，人物面向观众，正面视角:{final_weight:.2f})" if use_chinese else f"(facing camera, front view, face forward, character facing viewer, frontal view:{final_weight:.2f})"
                    # 找到角色描述的位置，在其后插入
                    insert_pos = len(priority_parts)
                    for i, part in enumerate(priority_parts):
                        if "han li" in part.lower() or "character" in part.lower() or "角色" in part:
                            insert_pos = i + 1
                            break
                    priority_parts.insert(insert_pos, facing_prompt)
                    if viewpoint_explicit and viewpoint_type == 'front':
                        print(f"  ✓ 基于智能分析添加正面朝向提示（明确要求，权重{final_weight:.2f}，位置{insert_pos}）")
                    else:
                        print(f"  ✓ 基于智能分析添加正面朝向提示（默认正面，权重{final_weight:.2f}，位置{insert_pos}）")
                
                # 如果有多个角色，添加其他角色的描述
                if len(identified_characters) > 1:
                    for char_id in identified_characters[1:]:
                        char_profile = self._get_character_profile(char_id)
                        if char_profile:
                            char_desc = self._build_character_description_prompt(char_profile, shot_type_for_prompt, compact=True)
                            if char_desc:
                                priority_parts.append(char_desc)
                                print(f"  ✓ 添加其他角色描述: {char_profile.get('character_name', char_id)}")
            else:
                # 如果没有识别到角色，但需要角色，使用通用角色描述
                # 从character_pose或description中提取角色信息
                visual = scene.get("visual", {}) or {}
                if isinstance(visual, dict):
                    character_pose = visual.get("character_pose", "")
                    if character_pose:
                        priority_parts.append(f"({character_pose}:1.5)")
                        print(f"  ✓ 使用 character_pose 作为角色描述: {character_pose[:50]}...")
        
        # ========== 立即添加动作/姿势描述（紧跟在角色后面，确保关键动作信息在前）==========
        # 优先使用中文 description/prompt，如果它们是中文，就不使用 visual 字段中的英文内容
        use_chinese_prompt = not self.ascii_only_prompt
        description_text = self._clean_prompt_text(scene.get("description") or "")
        prompt_text = self._clean_prompt_text(scene.get("prompt") or "")
        
        # 检查 description 或 prompt 是否包含中文
        import re
        has_chinese_desc = bool(re.search(r'[\u4e00-\u9fff]', description_text)) if description_text else False
        has_chinese_prompt = bool(re.search(r'[\u4e00-\u9fff]', prompt_text)) if prompt_text else False
        use_chinese = use_chinese_prompt and (has_chinese_desc or has_chinese_prompt)
        
        # 获取 visual 字段
        visual = scene.get("visual") or {}
        if isinstance(visual, dict) and not use_chinese:
            # 只有当不使用中文时，才使用 visual 字段中的英文内容
            character_pose = self._clean_prompt_text(visual.get("character_pose") or "")
            if character_pose:
                # 检查是否包含正面朝向关键词
                pose_lower = character_pose.lower()
                has_facing = any(kw in pose_lower for kw in ["facing", "front", "正面", "面向", "forward", "toward camera", "facing camera"])
                has_back = any(kw in pose_lower for kw in ["back", "背面", "背后", "from behind", "rear"])
                
                # 基于意图分析的动作类型，动态调整权重（通用处理）
                action_type = intent['action_type']
                if action_type == 'static':
                    # 静态动作，使用较高权重确保姿势准确
                    if not has_back:  # 如果不是明确要求背面，添加正面朝向
                        priority_parts.append(f"({character_pose}, facing camera, front view:1.8)")
                        print(f"  ✓ 使用 visual.character_pose（静态动作，增强正面朝向，权重1.8）: {character_pose}")
                    else:
                        priority_parts.append(f"({character_pose}:1.6)")
                        print(f"  ✓ 使用 visual.character_pose（静态动作，增强权重）: {character_pose}")
                else:
                    # 动态动作或其他，根据是否包含正面朝向调整权重
                    if has_facing:
                        priority_parts.append(f"({character_pose}:1.8)")
                        # 额外强调正面朝向，防止被其他描述覆盖
                        priority_parts.append("(facing camera, front view, face forward, frontal view:1.8)")
                        print(f"  ✓ 使用 visual.character_pose（正面朝向，增强权重）: {character_pose}")
                    elif has_back:
                        priority_parts.append(f"({character_pose}:1.3)")
                        print(f"  ✓ 使用 visual.character_pose（背面朝向）: {character_pose}")
                    else:
                        # 如果没有明确指定朝向，默认添加正面朝向
                        priority_parts.append(f"({character_pose}, facing camera, front view:1.8)")
                        print(f"  ✓ 使用 visual.character_pose（默认正面朝向，增强权重1.8）: {character_pose}")
        elif use_chinese and description_text:
            # 如果使用中文，优先使用 description 作为动作/姿势描述
            # 将 description 添加到 prompt（作为主要描述）
            if description_text not in [p.split(':')[0].strip('()') for p in priority_parts]:
                priority_parts.append(f"({description_text}:1.6)")
                print(f"  ✓ 使用中文 description（作为动作/姿势描述）: {description_text[:50]}...")
        
        # ========== 第二部分：镜头和构图（智能综合权重调整）==========
        # 如果使用中文，不使用 visual 字段中的英文内容；否则使用 visual.composition
        if not use_chinese and isinstance(visual, dict):
            # 优先使用 visual.composition（如果存在），这是最准确的构图描述
            # 使用综合权重调整后的构图权重
            weight_adjustments = intent.get('weight_adjustments', {})
            composition_weight = weight_adjustments.get('composition_weight', 1.4)
            composition = self._clean_prompt_text(visual.get("composition") or "")
            if composition:
                priority_parts.append(f"({composition}:{composition_weight:.2f})")
                print(f"  ✓ 使用 visual.composition（智能综合权重{composition_weight:.2f}）: {composition}")
        
        # 基于意图分析处理镜头类型（智能综合权重调整）
        if camera_desc:
            camera_prompt = self._convert_camera_to_prompt(camera_desc)
            if camera_prompt:
                # 使用综合权重调整后的镜头权重
                weight_adjustments = intent.get('weight_adjustments', {})
                camera_weight = weight_adjustments.get('camera_weight', 1.3)
                viewpoint = intent.get('viewpoint', {})
                viewpoint_type = viewpoint.get('type', 'front')
                
                # 根据视角类型和综合权重调整
                if viewpoint_type in ['close', 'wide']:
                    # 特写或远景：使用综合权重，插入到前面
                    priority_parts.insert(1, f"({camera_prompt}:{camera_weight:.2f})")  # 插入到第2位
                    print(f"  ✓ 使用场景 camera 描述（智能综合权重{camera_weight:.2f}）: {camera_desc} -> {camera_prompt}")
                else:
                    # 其他镜头类型：使用综合权重
                    priority_parts.append(f"({camera_prompt}:{camera_weight:.2f})")
                    print(f"  ✓ 使用场景 camera 描述（智能综合权重{camera_weight:.2f}）: {camera_desc} -> {camera_prompt}")
        
        # 镜头构图约束（极简版，只保留一个）
        # 添加宽高比保护，避免人像被横向拉伸或纵向拉伸（瘦长脸）
        use_chinese = not self.ascii_only_prompt
        if shot_type_for_prompt["is_wide"] or shot_type_for_prompt["is_full_body"]:
            # 远景场景：强制添加正面朝向和排除背影，避免人物太小和背影
            if use_chinese:
                priority_parts.append("(单人，正面视角，面向镜头:1.8)")
                priority_parts.append("(正确宽高比，自然面部比例:1.3)")  # 保护宽高比，防止瘦长脸
            else:
                priority_parts.append("(single person, front view, facing camera:1.8)")
                priority_parts.append("(correct aspect ratio, natural face proportions, no stretch:1.3)")  # 保护宽高比，防止瘦长脸
        elif shot_type_for_prompt["is_medium"]:
            # 中景场景：强制添加正面朝向，避免背影
            if use_chinese:
                priority_parts.append("(中景，正面视角，面向镜头，自然身体比例:1.8)")  # 提高权重，强调正面和自然比例
                priority_parts.append("(修长身材，窄肩，自然姿势:1.3)")  # 强调自然姿势
            else:
                priority_parts.append("(medium shot, front view, facing camera, natural body proportions:1.8)")  # 提高权重，强调正面和自然比例
                priority_parts.append("(slim body, narrow shoulders, natural pose:1.3)")  # 强调自然姿势
        elif shot_type_for_prompt["is_close"]:
            # 检查是否是眼睛特写或面部特写场景（需要保持特写，不转换为中景）
            is_eye_closeup = shot_type_for_prompt.get("is_eye_closeup", False)
            camera_desc_check = scene.get("camera") if scene else ""
            camera_desc_lower = (camera_desc_check or "").lower()
            # 如果没有标记，检查camera字段或prompt中是否有眼睛特写或面部特写关键词
            if not is_eye_closeup:
                is_eye_closeup = any(kw in camera_desc_lower for kw in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔', 'extreme close'])
            is_face_closeup = any(kw in camera_desc_lower for kw in ['face', 'facial', 'portrait', 'headshot', '面部', '脸部', '头像', 'close-up on face', 'closeup on face'])
            
            if is_eye_closeup:
                # 眼睛特写场景：保持特写描述，不转换为中景
                if use_chinese:
                    priority_parts.append("(眼睛特写，极近镜头:2.0)")
                    priority_parts.append("(详细的眼睛，瞳孔清晰可见:1.8)")
                else:
                    priority_parts.append("(extreme close-up on eyes:2.0)")
                    priority_parts.append("(detailed eyes, pupils clearly visible:1.8)")
                print(f"  ✓ 检测到眼睛特写场景，保持特写描述（不转换为中景）")
            elif is_face_closeup:
                # 面部特写场景：保持特写描述，不转换为中景
                if use_chinese:
                    priority_parts.append("(面部特写，近景镜头:2.0)")
                    priority_parts.append("(清晰的面部表情:1.8)")
                else:
                    priority_parts.append("(close-up on face:2.0)")
                    priority_parts.append("(portrait shot, headshot, clear facial expression:1.8)")
                print(f"  ✓ 检测到面部特写场景，保持特写描述（不转换为中景）")
            else:
                # 其他特写场景：避免太近的镜头，使用中景描述
                if use_chinese:
                    priority_parts.append("(中景:1.3)")
                    priority_parts.append("(修长身材，窄肩:1.3)")
                else:
                    priority_parts.append("(medium shot:1.3)")
                    priority_parts.append("(slim body, narrow shoulders:1.3)")
                print(f"  ⚠ 检测到特写镜头，已转换为中景以避免身体过宽和模糊")
        
        # ========== 第三部分：场景背景（增强版，保留完整细节）==========
        # 如果已经使用了中文 description，就不再添加 visual.environment（避免重复和混用中英文）
        # 如果还没有添加 description，才考虑使用 visual.environment
        if not use_chinese and isinstance(visual, dict):
            # 只有当不使用中文时，才使用 visual.environment
            environment_visual = self._clean_prompt_text(visual.get("environment") or "")
            if environment_visual:
                # 不再过度精简，保留完整的环境描述以增强场景表现
                # 环境描述包含场景中的物体、地形、天气等重要信息
                priority_parts.append(f"({environment_visual}:1.4)")
                print(f"  ✓ 使用 visual.environment（完整版）: {environment_visual}")
        
        scene_bg_compact = self._build_scene_background_prompt_compact(scene, script_data)
        if scene_bg_compact:
            # 将背景描述添加到priority_parts的开头（在角色之后），确保高优先级
            # 但不要放在最前面，因为角色描述应该在第一位
            insert_pos = 1 if include_character and priority_parts else 0
            priority_parts.insert(insert_pos, scene_bg_compact)
            print(f"  ✓ 应用场景背景模板（精简版）: {scene_bg_compact}")
        
        # ========== 第五部分：动作描述（智能综合权重调整）==========
        # 使用综合权重调整后的动作权重
        weight_adjustments = intent.get('weight_adjustments', {})
        action_weight = weight_adjustments.get('action_weight', 1.2)
        
        # 如果已经有character_pose，检查是否需要补充动作信息
        use_chinese = not self.ascii_only_prompt
        if isinstance(visual, dict) and visual.get("character_pose") and not use_chinese:
            # 如果 character_pose 存在但不够详细，可以补充 action（仅英文模式）
            character_pose_text = visual.get("character_pose", "").lower()
            # 检查是否包含明确的动作动词
            has_action_verb = any(verb in character_pose_text for verb in 
                                 ["lying", "standing", "walking", "sitting", "running", 
                                  "flying", "attacking", "defending", "casting", "using"])
            if not has_action_verb:
                # 如果没有明确的动作，从 action 字段补充
                raw_action = (scene.get("action") or "")
                if raw_action:
                    action_simple = raw_action.replace("_", " ").lower()
                    if "walk" in action_simple:
                        priority_parts.append(f"(walking:{action_weight:.2f})")
                    elif "stand" in action_simple or "detect" in action_simple:
                        priority_parts.append(f"(standing:{action_weight:.2f})")
                    elif "lie" in action_simple or "lying" in action_simple:
                        priority_parts.append(f"(lying:{action_weight:.2f})")
                    elif "use" in action_simple or "cast" in action_simple:
                        priority_parts.append(f"({action_simple}:{action_weight:.2f})")
        elif not use_chinese:
            # 如果没有 character_pose，从 action 字段添加动作描述（仅英文模式）
            raw_action = (scene.get("action") or "")
            if raw_action:
                action_simple = raw_action.replace("_", " ").lower()
                if "walk" in action_simple:
                    priority_parts.append(f"(walking:{action_weight:.2f})")
                elif "stand" in action_simple or "detect" in action_simple:
                    priority_parts.append(f"(standing:{action_weight:.2f})")
                elif "lie" in action_simple or "lying" in action_simple:
                    priority_parts.append(f"(lying:{action_weight:.2f})")
                elif "use" in action_simple or "cast" in action_simple:
                    priority_parts.append(f"({action_simple}:{action_weight:.2f})")
        
        # ========== 第六部分：风格补充（如果前面没有添加，这里补充）==========
        # 检查是否已经有仙侠风格关键词（应该已经在第一部分添加）
        use_chinese = not self.ascii_only_prompt
        has_xianxia_style = any("xianxia" in p.lower() or "chinese fantasy" in p.lower() or "仙侠" in p for p in priority_parts)
        if not has_xianxia_style:
            # 如果前面没有添加，在这里补充（但优先级较低）
            if len(priority_parts) < 6:  # 如果核心部分较少，添加完整风格
                if use_chinese:
                    priority_parts.append("柔和光影，青色灵气")
                else:
                    priority_parts.append("soft lighting, cyan aura")
                style_text = "柔和光影，青色灵气" if use_chinese else "soft lighting, cyan aura"
                print(f"  ✓ 补充风格细节: {style_text}")
        
        # ========== 第七部分：背景一致性（保证场景连贯，精简）==========
        # 强调背景稳定，避免跳帧和风格漂移
        # 检查是否已经有场景背景描述
        has_scene_bg = any("desert" in p.lower() or "chamber" in p.lower() or "background" in p.lower() or "golden" in p.lower() or "沙漠" in p or "遗迹" in p or "背景" in p for p in priority_parts)
        # 如果有场景背景模板，背景一致性提示更重要，优先保留
        if len(priority_parts) < 7 or has_scene_bg:
            # 如果已经有场景背景描述，使用更简洁的一致性提示
            if has_scene_bg:
                if not self.ascii_only_prompt:
                    priority_parts.append("(背景一致:1.2)")
                else:
                    priority_parts.append("(consistent background:1.2)")
            else:
                if not self.ascii_only_prompt:
                    priority_parts.append("(背景一致，环境稳定:1.3)")
                else:
                    priority_parts.append("(consistent background, stable environment:1.3)")
        
        # ========== 第八部分：相邻场景连贯性（新增）==========
        # 如果有前一个场景，强调场景连续性
        if previous_scene:
            # 检查是否在同一环境（通过episode、title或scene_name判断）
            current_env = scene.get("scene_name") or script_data.get("title", "") if script_data else ""
            prev_env = previous_scene.get("scene_name") or ""
            
            # 如果环境相同或相似，强调连续性
            use_chinese = not self.ascii_only_prompt
            if current_env and prev_env and (current_env == prev_env or 
                any(keyword in current_env.lower() and keyword in prev_env.lower() 
                    for keyword in ["desert", "chamber", "corridor", "遗迹", "沙漠"])):
                if use_chinese:
                    priority_parts.append("(相同位置，连续场景:1.2)")
                    priority_parts.append("(相同环境风格:1.1)")
                else:
                    priority_parts.append("(same location, continuous scene:1.2)")
                    priority_parts.append("(same environment style:1.1)")
                print(f"  ✓ 检测到相邻场景在同一环境，添加连贯性提示")
            
            # 检查角色是否相同（如果都有角色）
            if include_character and self._needs_character and self._needs_character(previous_scene):
                if use_chinese:
                    priority_parts.append("(相同角色外观:1.2)")
                else:
                    priority_parts.append("(same character appearance:1.2)")
                print(f"  ✓ 检测到相邻场景都有角色，添加角色一致性提示")
        
        # ========== 第四部分：场景特效和细节（提升优先级）==========
        # 如果使用中文，不使用 visual.fx 中的英文内容；否则使用 visual.fx
        # fx 描述了场景中的特效、物体、动作等关键视觉元素，应该放在优先级部分
        # 增强：将 fx 提升到 priority_parts，确保场景中的物体和动作能被正确表现
        if not use_chinese and isinstance(visual, dict):
            # 只有当不使用中文时，才使用 visual.fx
            fx = self._clean_prompt_text(visual.get("fx") or "")
            if fx:
                # fx 包含场景中的特效、物体、粒子等细节，对场景表现很重要
                # 检查是否是眼睛特写场景，如果是，提高权重
                camera_desc_lower = (camera_desc or "").lower()
                fx_lower = fx.lower()
                is_eye_closeup = any(kw in camera_desc_lower for kw in ['close-up', 'closeup', 'close up', 'extreme close', 'eye', 'eyes', '特写', '近景', '眼睛'])
                has_eye_detail = any(kw in fx_lower for kw in ['eye', 'pupil', 'glint', 'glow', 'blue', 'light'])
                
                # 如果是眼睛特写或包含眼睛细节，使用更高权重
                if is_eye_closeup or has_eye_detail:
                    priority_parts.append(f"({fx}:2.0)")
                    print(f"  ✓ 使用 visual.fx（眼睛特写/细节增强，权重2.0）: {fx}")
                else:
                    priority_parts.append(f"({fx}:1.5)")
                    print(f"  ✓ 使用 visual.fx（提升优先级，权重1.5）: {fx}")
        
        # ========== 次要信息（可能被截断，但保留用于完整性）==========
        # motion 描述镜头运动，作为补充信息
        if isinstance(visual, dict):
            motion_desc = self._convert_motion_to_prompt(visual.get("motion"))
            if motion_desc:
                secondary_parts.append(f"({motion_desc}:1.0)")
        
        # face_style_auto（次要）
        face_style = scene.get("face_style_auto") or {}
        if isinstance(face_style, dict):
            expression = self._clean_prompt_text(face_style.get("expression") or "")
            if expression:
                secondary_parts.append(f"({expression} expression:1.0)")
        
        # 其他风格标签（国风动漫风格）
        if not self.ascii_only_prompt:
            secondary_parts.append("中国动画风格")
            secondary_parts.append("古代中国奇幻")
            secondary_parts.append("电影级光影")
        else:
            secondary_parts.append("Chinese animation style")
            secondary_parts.append("ancient Chinese fantasy")
            secondary_parts.append("cinematic lighting")
        secondary_parts.append("4k")
        
        # 合并：只使用优先部分，确保关键信息在前 77 tokens 内
        # 使用更准确的 token 估算（考虑括号和权重标记）
        priority_prompt = ", ".join(filter(None, priority_parts))
        
        # 尝试使用CLIP tokenizer进行准确计算，如果不可用则使用保守估算
        estimated_tokens = self.token_estimator.estimate(priority_prompt)
        
        # 如果估算超过 70 tokens（留出安全边界，确保不超过77），使用智能优化
        if estimated_tokens > 70:
            # 尝试使用智能优化（基于语义重要性）
            print(f"  🧠 Prompt 过长 ({estimated_tokens} tokens)，尝试智能优化...")
            optimized_parts = self.optimizer.optimize(priority_parts, max_tokens=70)
            if len(optimized_parts) < len(priority_parts):
                priority_parts = optimized_parts
                priority_prompt = ", ".join(filter(None, priority_parts))
                estimated_tokens = self.token_estimator.estimate(priority_prompt)
                print(f"  ✓ 智能优化完成: {len(optimized_parts)} 个部分，{estimated_tokens} tokens")
            else:
                # 如果智能优化没有效果，使用传统精简方法
                print(f"  ⚠ 智能优化未达到预期，使用传统精简方法...")

        # 确保仙侠风格描述不会被优化阶段剔除
        if not any(self._has_xianxia_keyword(part) for part in priority_parts):
            priority_parts.insert(0, xianxia_style)
            priority_prompt = ", ".join(filter(None, priority_parts))
            estimated_tokens = self.token_estimator.estimate(priority_prompt)
            print("  ✓ 智能优化后补回仙侠风格提示，确保风格一致")
        
        # 注意：由于完整迁移所有token优化和精简逻辑需要约600行代码，这里先实现基本逻辑
        # 完整实现需要从 ImageGenerator.build_prompt() 中迁移（line 2960-3362）
        # TODO: 完整迁移token优化和精简逻辑
        
        # 最终验证：确保不超过 77 tokens
        final_estimated = self.token_estimator.estimate(priority_prompt)
        
        # 如果 tokenizer 可用，使用真实计算；否则使用估算
        if self._clip_tokenizer is not None:
            try:
                tokens = self._clip_tokenizer(priority_prompt, truncation=False, return_tensors="pt")
                final_estimated = tokens.input_ids.shape[1]
                print(f"  ✓ 使用真实 tokenizer 计算: {final_estimated} tokens")
            except Exception as e:
                print(f"  ⚠ Tokenizer 最终验证失败，使用估算: {e}")
        
        if final_estimated > 77:
            print(f"  ⚠ 警告: Prompt 最终长度 ({final_estimated} tokens) 超过 77 tokens 限制，将被 CLIP 自动截断")
            print(f"  ⚠ 建议进一步精简 prompt 以避免信息丢失")
        
        full_prompt = priority_prompt
        
        if not full_prompt:
            # 默认fallback prompt，使用中文（因为用户要求使用中文），不限定具体场景
            if not self.ascii_only_prompt:
                full_prompt = "仙侠风格，修仙世界，灵气能量，详细插画"
            else:
                full_prompt = "xianxia fantasy, cultivation world, spiritual energy, detailed illustration"
        
        # 使用准确的tokenizer计算最终token数（必须使用真实计算）
        final_tokens = self.token_estimator.estimate(priority_prompt)
        # 如果 tokenizer 可用，强制使用真实计算
        if self._clip_tokenizer is not None:
            try:
                tokens = self._clip_tokenizer(priority_prompt, truncation=False, return_tensors="pt")
                final_tokens = tokens.input_ids.shape[1]
                print(f"  ✓ 使用真实 tokenizer 计算: {final_tokens} tokens")
            except Exception as e:
                print(f"  ⚠ Tokenizer 最终验证失败，使用估算: {e}")
        
        if final_tokens > 77:
            print(f"  ⚠ 警告: Prompt 最终长度 ({final_tokens} tokens) 超过 77 tokens 限制，将被 CLIP 自动截断")
            print(f"  ⚠ 建议进一步精简 prompt 以避免信息丢失")
        
        # 如果使用中文且SDXL模型对中文支持不好，考虑翻译成英文
        # 但先检查配置，如果配置允许中文，就使用中文
        if not self.ascii_only_prompt:
            # 使用中文prompt
            final_prompt = priority_prompt
            print(f"  ℹ 使用中文 prompt（SDXL可能理解不佳，如果生成效果不好，建议设置 ascii_only_prompt: true）")
        else:
            # 翻译成英文
            final_prompt = self._translate_chinese_to_english(priority_prompt)
            print(f"  ℹ 已翻译为英文 prompt")
        
        # 重新计算最终prompt的token数（使用真实tokenizer）
        if self._clip_tokenizer is not None:
            try:
                tokens = self._clip_tokenizer(final_prompt, truncation=False, return_tensors="pt")
                final_tokens = tokens.input_ids.shape[1]
                print(f"  ✓ 使用真实 tokenizer 计算最终prompt: {final_tokens} tokens")
            except Exception as e:
                print(f"  ⚠ Tokenizer 最终验证失败，使用估算: {e}")
        
        if final_tokens > 77:
            print(f"  ⚠ 警告: Prompt 最终长度 ({final_tokens} tokens) 超过 77 tokens 限制，将被 CLIP 自动截断")
            print(f"  ⚠ 建议进一步精简 prompt 以避免信息丢失")
        
        part_count = len(priority_parts)
        print(f"  📊 Prompt 最终长度: {final_tokens} tokens (关键部分: {part_count} 项)")
        print(f"  📊 Prompt文本长度: {len(final_prompt)} 字符")
        # 修复中文编码问题：使用repr或确保UTF-8编码
        try:
            # 尝试直接打印，如果失败则使用repr
            print(f"  📝 最终Prompt文本: {final_prompt}")
        except UnicodeEncodeError:
            # 如果遇到编码问题，使用安全的打印方式
            print(f"  📝 最终Prompt文本: {final_prompt.encode('utf-8', errors='replace').decode('utf-8')}")
        # 打印每个部分的详细信息
        print(f"  📋 Prompt组成部分 ({len(priority_parts)} 项):")
        for i, part in enumerate(priority_parts, 1):
            part_tokens = self.token_estimator.estimate(part)
            # 使用真实tokenizer计算每个部分的token数
            if self._clip_tokenizer is not None:
                try:
                    tokens_obj = self._clip_tokenizer(part, truncation=False, return_tensors="pt")
                    part_tokens = tokens_obj.input_ids.shape[1]
                except:
                    pass
            print(f"    {i}. [{part_tokens} tokens] {part[:80]}{'...' if len(part) > 80 else ''}")
        return final_prompt
    
    def _clean_prompt_text(self, text: str) -> str:
        """清理 prompt 文本，支持中文"""
        text = (text or "").strip().strip('"')
        if not text:
            return ""
        
        if self.ascii_only_prompt:
            text = "".join(ch if ord(ch) < 128 else " " for ch in text)
            text = " ".join(t for t in text.split() if t)
        else:
            import re
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _has_xianxia_keyword(self, text: str) -> bool:
        """检测文本中是否包含仙侠相关关键词"""
        if not text:
            return False
        lowered = str(text).lower()
        return (
            any(
                kw in lowered
                for kw in [
                    "xianxia",
                    "immortal cultivator",
                    "cultivation world",
                    "celestial aura",
                    "spiritual energy",
                ]
            )
            or ("仙侠" in str(text))
            or ("修仙" in str(text))
        )
    
    def _get_character_profile(self, character_id: str = "hanli") -> Dict[str, Any]:
        """获取角色模板"""
        return self.character_profiles.get(character_id, {})
    
    def _get_scene_profile(
        self,
        scene_name: str = None,
        episode: int = None,
        profile_key: str = None,
    ) -> Dict[str, Any]:
        """根据场景 key、名称或集数获取场景模板"""
        # 1. 若显式指定模板 key，直接精确匹配
        if profile_key:
            profile = self.scene_profiles.get(profile_key)
            if profile:
                return profile
        
        # 2. 使用场景名称模糊匹配
        if scene_name:
            scene_name_lower = scene_name.lower()
            
            for key, profile in self.scene_profiles.items():
                profile_scene_name = profile.get("scene_name", "").lower()
                if profile_scene_name and (profile_scene_name in scene_name_lower or scene_name_lower in profile_scene_name):
                    return profile
                if key.lower() in scene_name_lower or scene_name_lower in key.lower():
                    return profile
            
            # 关键词匹配
            if "沙漠" in scene_name or "沙地" in scene_name:
                for key, profile in self.scene_profiles.items():
                    profile_scene_name = profile.get("scene_name", "").lower()
                    if "沙漠" in profile_scene_name or "desert" in key.lower():
                        return profile
        
        # 3. 使用集数匹配
        if episode:
            for key, profile in self.scene_profiles.items():
                if profile.get("episode") == episode:
                    return profile
        
        # 默认返回第一个场景模板
        if self.scene_profiles:
            return list(self.scene_profiles.values())[0]
        
        return {}
    
    def _translate_chinese_to_english(self, text: str) -> str:
        """将中文 prompt 翻译成英文"""
        if not text:
            return ""
        
        translations = {
            "仙侠风格": "xianxia fantasy style",
            "仙侠": "xianxia",
            "古风": "ancient Chinese style",
            "修仙": "cultivation",
            "修仙世界": "cultivation world",
            "中国古风": "ancient Chinese style",
            "黑色长发": "long black hair",
            "深绿道袍": "dark green robe",
            "平静眼神": "calm eyes",
            "修长身材": "slim body",
            "窄肩": "narrow shoulders",
            "躺": "lying",
            "沙地": "sand ground",
            "青灰色沙地": "grayish sand ground",
            "一动不动": "motionless",
            "感受": "feeling",
            "燥热": "heat",
            "仙域": "immortal realm",
            "天空": "sky",
            "太阳": "sun",
            "月亮": "moon",
            "虚影": "phantom",
            "出现": "appearing",
            "云雾": "mist",
            "缭绕": "swirling",
            "卷轴": "scroll",
            "灵气": "spiritual energy",
            "粒子": "particles",
            "飘浮": "floating",
            "开场": "opening",
            "远景": "wide shot",
            "俯视": "aerial view",
            "鸟瞰": "bird's eye view",
            "全景": "panoramic view",
            "背景一致": "consistent background",
        }
        
        result = text
        for chinese, english in sorted(translations.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(chinese, english)
        
        return result
    
    def _convert_camera_to_prompt(self, camera_desc: str) -> str:
        """将中文镜头描述转换为镜头关键词（根据配置返回中文或英文）"""
        if not camera_desc:
            return ""
        
        use_chinese = not self.ascii_only_prompt
        camera_keywords = []
        
        # 镜头距离/景别
        if "远景" in camera_desc or "全景" in camera_desc or "广角" in camera_desc:
            if "俯拍" in camera_desc or "俯视" in camera_desc:
                if use_chinese:
                    camera_keywords.append("远景俯拍，鸟瞰视角")
                else:
                    camera_keywords.append("extreme wide shot, aerial view")
            elif "仰拍" in camera_desc or "仰视" in camera_desc:
                if use_chinese:
                    camera_keywords.append("远景仰拍，低角度视角")
                else:
                    camera_keywords.append("extreme wide shot, low angle view")
            else:
                if use_chinese:
                    camera_keywords.append("远景，超长镜头")
                else:
                    camera_keywords.append("extreme wide shot, very long shot")
        elif "中景" in camera_desc or "中距" in camera_desc:
            if use_chinese:
                camera_keywords.append("中景镜头")
            else:
                camera_keywords.append("medium shot, mid shot")
        elif "近景" in camera_desc or "特写" in camera_desc or "close-up" in camera_desc.lower() or "closeup" in camera_desc.lower() or "close up" in camera_desc.lower():
            # 检查是否是眼睛特写或面部特写场景（需要保持特写）
            camera_desc_lower = camera_desc.lower()
            is_eye_closeup = any(kw in camera_desc_lower for kw in ['eye', 'eyes', 'pupil', 'pupils', '眼睛', '瞳孔', 'extreme close'])
            is_face_closeup = any(kw in camera_desc_lower for kw in ['face', 'facial', 'portrait', 'headshot', '面部', '脸部', '头像', 'close-up on face', 'closeup on face'])
            
            if is_eye_closeup:
                # 眼睛特写场景：保持特写，不转换为中景
                if use_chinese:
                    camera_keywords.append("眼睛特写，极近镜头")
                else:
                    camera_keywords.append("extreme close-up on eyes, eye close-up, detailed eyes")
                print(f"  ✓ 检测到眼睛特写场景，保持特写镜头（不转换为中景）")
            elif is_face_closeup:
                # 面部特写场景：保持特写，不转换为中景
                if use_chinese:
                    camera_keywords.append("面部特写，近景镜头")
                else:
                    camera_keywords.append("close-up on face, face close-up, portrait shot, headshot")
                print(f"  ✓ 检测到面部特写场景，保持特写镜头（不转换为中景）")
            else:
                # 其他特写场景：避免太近的镜头，转换为中景
                print(f"  ⚠ 检测到特写/近景镜头描述，为避免身体过宽和模糊，转换为中景")
                if use_chinese:
                    camera_keywords.append("中景镜头")
                else:
                    camera_keywords.append("medium shot, mid shot")  # 转换为中景
        elif "全身" in camera_desc or "全貌" in camera_desc:
            if use_chinese:
                camera_keywords.append("全身镜头")
            else:
                camera_keywords.append("full body shot, full figure")
        elif "长镜头" in camera_desc or "长焦" in camera_desc:
            if use_chinese:
                camera_keywords.append("长焦镜头")
            else:
                camera_keywords.append("long shot, telephoto")
        elif "短焦" in camera_desc or "广角" in camera_desc:
            if use_chinese:
                camera_keywords.append("广角镜头")
            else:
                camera_keywords.append("wide angle shot")
        
        # 镜头运动
        if "推近" in camera_desc or "推进" in camera_desc or "推镜" in camera_desc:
            if use_chinese:
                camera_keywords.append("推镜")
            else:
                camera_keywords.append("push in, dolly in")
        elif "拉远" in camera_desc or "拉镜" in camera_desc or "推远" in camera_desc:
            if use_chinese:
                camera_keywords.append("拉镜")
            else:
                camera_keywords.append("pull out, dolly out")
        elif "跟随" in camera_desc or "跟拍" in camera_desc:
            if use_chinese:
                camera_keywords.append("跟拍")
            else:
                camera_keywords.append("follow shot, tracking shot")
        elif "环绕" in camera_desc or "旋转" in camera_desc:
            if use_chinese:
                camera_keywords.append("环绕镜头")
            else:
                camera_keywords.append("orbital shot, rotating camera")
        elif "上移" in camera_desc or "上扬" in camera_desc:
            if use_chinese:
                camera_keywords.append("上摇")
            else:
                camera_keywords.append("tilt up, camera tilt up")
        elif "下移" in camera_desc or "下推" in camera_desc:
            if use_chinese:
                camera_keywords.append("下摇")
            else:
                camera_keywords.append("tilt down, camera tilt down")
        elif "横移" in camera_desc or "平移" in camera_desc:
            if use_chinese:
                camera_keywords.append("横移")
            else:
                camera_keywords.append("pan shot, lateral movement")
        elif "定格" in camera_desc or "静止" in camera_desc:
            if use_chinese:
                camera_keywords.append("静止镜头")
            else:
                camera_keywords.append("static shot, still frame")
        
        # 镜头角度
        if "俯拍" in camera_desc or "俯视" in camera_desc:
            if use_chinese:
                camera_keywords.append("俯视，鸟瞰")
            else:
                camera_keywords.append("aerial view, top down, bird's eye view")
        elif "仰拍" in camera_desc or "仰视" in camera_desc:
            if use_chinese:
                camera_keywords.append("仰视，低角度")
            else:
                camera_keywords.append("low angle, worm's eye view")
        elif "侧拍" in camera_desc or "侧面" in camera_desc:
            if use_chinese:
                camera_keywords.append("侧面视角")
            else:
                camera_keywords.append("side view, profile shot")
        elif "背后" in camera_desc or "背影" in camera_desc:
            if use_chinese:
                camera_keywords.append("背后视角")
            else:
                camera_keywords.append("back view, from behind")
        elif "正面" in camera_desc:
            if use_chinese:
                camera_keywords.append("正面视角")
            else:
                camera_keywords.append("front view, face forward")
        
        # 特殊效果
        if "抖动" in camera_desc or "震动" in camera_desc:
            if use_chinese:
                camera_keywords.append("镜头抖动")
            else:
                camera_keywords.append("shaky camera, camera shake")
        elif "慢动作" in camera_desc or "慢镜" in camera_desc:
            if use_chinese:
                camera_keywords.append("慢动作")
            else:
                camera_keywords.append("slow motion")
        elif "快速" in camera_desc or "急速" in camera_desc:
            if use_chinese:
                camera_keywords.append("快速运动")
            else:
                camera_keywords.append("fast movement, rapid camera")
        elif "缓缓" in camera_desc or "缓慢" in camera_desc:
            if use_chinese:
                camera_keywords.append("缓慢运动")
            else:
                camera_keywords.append("slow movement, gentle camera")
        
        # 如果没有任何匹配，尝试直接翻译关键词
        if not camera_keywords:
            if use_chinese:
                camera_keywords.append("电影级镜头")
            else:
                camera_keywords.append("cinematic shot")
        
        return ", ".join(camera_keywords)
    
    def _convert_motion_to_prompt(self, motion: Any) -> str:
        """将 visual.motion 转换为 prompt 描述"""
        if not motion:
            return ""
        
        if isinstance(motion, str):
            return motion
        
        if isinstance(motion, dict):
            motion_type = motion.get("type", "")
            direction = motion.get("direction", "")
            speed = motion.get("speed", "medium")
            
            motion_keywords = []
            
            # 类型转换
            type_map = {
                "static": "static shot",
                "pan": "pan shot",
                "tilt": "tilt shot",
                "push_in": "push in, dolly in",
                "pull_out": "pull out, dolly out",
                "orbit": "orbital shot, rotating camera",
                "shake": "shaky camera, camera shake",
                "follow": "follow shot, tracking shot",
            }
            
            if motion_type in type_map:
                motion_keywords.append(type_map[motion_type])
            elif motion_type:
                motion_keywords.append(motion_type)
            
            # 方向转换
            if direction:
                direction_map = {
                    "left_to_right": "left to right",
                    "right_to_left": "right to left",
                    "up": "tilt up",
                    "down": "tilt down",
                    "forward": "forward",
                    "backward": "backward",
                    "around": "around subject",
                }
                if direction in direction_map:
                    motion_keywords.append(direction_map[direction])
            
            # 速度
            if speed and speed != "medium":
                motion_keywords.append(f"{speed} movement")
            
            return ", ".join(motion_keywords)
        
        return ""
    
    def _looks_like_camera_prompt(self, text: str) -> bool:
        """判断文本是否看起来像相机描述"""
        if not text:
            return False
        lowered = text.lower()
        keywords = [
            "远景", "全景", "近景", "中景", "特写", "镜头", "俯拍", "俯视", "仰拍", "仰视", "推近", "拉远",
            "跟拍", "环绕", "横移", "推镜", "镜头缓缓", "镜头快速", "slow motion", "close-up", "wide shot",
            "shot", "pan", "tilt", "dolly", "camera"
        ]
        return any(kw in lowered for kw in keywords)
    
    def _build_character_prompt(self, character_id: str = "hanli") -> str:
        """根据角色模板构建角色描述 prompt（完整版）"""
        profile = self._get_character_profile(character_id)
        if not profile:
            return ""
        
        parts = []
        
        # 发型描述（最高权重，确保发型正确）
        if profile.get("hair", {}).get("prompt_keywords"):
            parts.append(profile["hair"]["prompt_keywords"])
        
        # 服饰描述（最高权重，强调修仙风格，排除铠甲）
        if profile.get("clothes", {}).get("prompt_keywords"):
            parts.append(profile["clothes"]["prompt_keywords"])
        
        # 修仙气质特征
        parts.append("(xianxia cultivator aura:1.3)")
        parts.append("(elegant immortal style:1.2)")
        parts.append("(refined scholar-warrior appearance:1.2)")
        
        # 面部特征
        if profile.get("face_keywords"):
            parts.append(f"({profile['face_keywords']}:1.2)")
        
        # 身体特征（强调瘦削，避免过宽）
        body = profile.get("body", {})
        if body.get("build"):
            parts.append(f"({body['build']}:1.2)")
        parts.append("(slim physique, lean body:1.2)")  # 强调瘦削身材
        if body.get("posture"):
            parts.append(f"({body['posture']}:1.1)")
        
        # 角色一致性标记
        parts.append("(consistent character appearance:1.3)")
        parts.append("(correct hairstyle, correct clothing:1.3)")
        
        return ", ".join(parts)
    
    def _build_character_prompt_compact(self, character_id: str = "hanli", shot_type: Dict[str, bool] = None) -> str:
        """根据角色模板构建极简版角色描述 prompt（确保在前 77 tokens 内）"""
        profile = self._get_character_profile(character_id)
        if not profile:
            return ""
        
        use_chinese = not self.ascii_only_prompt
        parts = []
        
        # 0. 性别特征（最高优先级，从 identity 字段提取，避免生成错误性别）
        identity = profile.get("identity", "")
        if identity:
            identity_lower = identity.lower()
            if "male" in identity_lower or "男" in identity:
                if use_chinese:
                    parts.append("(男性，男:1.8)")
                else:
                    parts.append("(male, man:1.8)")
            elif "female" in identity_lower or "女" in identity:
                if use_chinese:
                    parts.append("(女性，女:1.8)")
                else:
                    parts.append("(female, woman:1.8)")
        else:
            # 向后兼容：对于韩立，默认是男性
            if character_id == "hanli":
                if use_chinese:
                    parts.append("(男性，男:1.8)")
                else:
                    parts.append("(male, man:1.8)")
        
        # 1. 发型和服饰（最高优先级，合并描述）
        if use_chinese:
            parts.append("(黑色长发，深绿道袍:1.7)")
        else:
            parts.append("(long black hair, dark green robe:1.7)")
        
        # 2. 面部特征（极简）
        if use_chinese:
            parts.append("(平静眼神:1.2)")
        else:
            parts.append("(calm eyes:1.2)")
        
        # 3. 身体特征（根据镜头类型，极简）
        if shot_type and (shot_type.get("is_medium") or shot_type.get("is_close")):
            if use_chinese:
                parts.append("(修长身材，窄肩:1.3)")
            else:
                parts.append("(slim body, narrow shoulders:1.3)")
        
        return ", ".join(parts)
    
    def _build_character_description_prompt(self, profile: Dict[str, Any], shot_type: Dict[str, bool] = None, compact: bool = False) -> str:
        """根据角色描述构建 prompt（用于根据描述生成图像，不使用参考照片）"""
        if not profile:
            return ""
        
        use_chinese = not self.ascii_only_prompt
        parts = []
        
        # 0. 身份和性别（最高优先级，从 identity 字段提取，必须包含）
        identity = profile.get("identity", "")
        character_id = profile.get("character_id", "").lower() or profile.get("id", "").lower()
        
        # 对于韩立，默认是男性（如果identity中没有明确说明）
        if "hanli" in character_id or "han li" in character_id or "韩立" in str(profile.get("character_name", "")):
            if not identity or ("male" not in identity.lower() and "女" not in identity and "female" not in identity.lower()):
                if use_chinese:
                    parts.append("(男性，男:2.0)")
                else:
                    parts.append("(male, man:2.0)")
            else:
                identity_lower = identity.lower()
                if "male" in identity_lower or "男" in identity:
                    if use_chinese:
                        parts.append("(男性，男:2.0)")
                    else:
                        parts.append("(male, man:2.0)")
                elif "female" in identity_lower or "女" in identity:
                    if use_chinese:
                        parts.append("(女性，女:2.0)")
                    else:
                        parts.append("(female, woman:2.0)")
        elif identity:
            identity_lower = identity.lower()
            if "male" in identity_lower or "男" in identity:
                if use_chinese:
                    parts.append("(男性，男:2.0)")
                else:
                    parts.append("(male, man:2.0)")
            elif "female" in identity_lower or "女" in identity:
                if use_chinese:
                    parts.append("(女性，女:2.0)")
                else:
                    parts.append("(female, woman:2.0)")
        
        # 1. 角色名称（必须包含，确保角色识别）
        character_name = profile.get("character_name", "")
        if character_name:
            parts.append(character_name)
        
        # 2. 发型描述（提高权重，确保不被优化掉）
        hair = profile.get("hair", {})
        if hair.get("prompt_keywords"):
            parts.append(hair["prompt_keywords"])
        elif hair.get("style") and hair.get("color"):
            parts.append(f"({hair['color']} {hair['style']}:1.8)")  # 从1.7提高到1.8，确保不被优化掉
        else:
            # 对于韩立，默认添加黑色长发
            if "hanli" in character_id or "han li" in character_id or "韩立" in character_name:
                if use_chinese:
                    parts.append("(黑色长发:1.8)")
                else:
                    parts.append("(long black hair:1.8)")
        
        # 3. 服饰描述（提高权重，确保不被优化掉，必须包含修仙风格）
        clothes = profile.get("clothes", {})
        if clothes.get("prompt_keywords"):
            parts.append(clothes["prompt_keywords"])
        elif clothes.get("style") and clothes.get("color"):
            parts.append(f"({clothes['color']} {clothes['style']}:1.8)")  # 从1.7提高到1.8，确保不被优化掉
        else:
            # 对于韩立，默认添加深绿道袍和修仙风格
            if "hanli" in character_id or "han li" in character_id or "韩立" in character_name:
                if use_chinese:
                    parts.append("(深绿道袍，修仙服饰:1.8)")
                else:
                    parts.append("(dark green robe, xianxia cultivator robe:1.8)")
        
        # 4. 修仙气质特征（必须包含，确保修仙风格）
        if "hanli" in character_id or "han li" in character_id or "韩立" in character_name or "xianxia" in str(profile.get("world", "")).lower():
            if use_chinese:
                parts.append("(修仙者，仙侠气质:1.5)")
            else:
                parts.append("(xianxia cultivator, immortal cultivator aura:1.5)")
        
        # 5. 面部特征
        if profile.get("face_keywords"):
            parts.append(f"({profile['face_keywords']}:1.3)")
        
        # 6. 身体特征（根据镜头类型）
        body = profile.get("body", {})
        if shot_type and (shot_type.get("is_medium") or shot_type.get("is_close")):
            if body.get("build"):
                parts.append(f"({body['build']}:1.2)")
        
        # 如果是精简版，只保留最核心的特征（性别、发型、服饰、修仙气质）
        if compact:
            # 保留：性别、角色名称、发型、服饰、修仙气质
            essential_parts = []
            for part in parts:
                if any(kw in part.lower() for kw in ["male", "female", "男", "女", "han li", "韩立", "hair", "长发", "robe", "道袍", "cultivator", "修仙"]):
                    essential_parts.append(part)
            if essential_parts:
                parts = essential_parts[:5]  # 最多保留5个核心特征
        
        return ", ".join(parts) if parts else ""
    
    def _build_scene_background_prompt_compact(self, scene: Dict[str, Any], script_data: Dict[str, Any] = None) -> str:
        """构建精简版场景背景 prompt"""
        scene_id = scene.get("id")
        is_opening_ending = scene_id in [0, 999]
        
        if is_opening_ending:
            if not self.ascii_only_prompt:
                return "(仙域天空，灵气缭绕:1.3)"
            else:
                return "(immortal realm sky, spiritual mist:1.3)"
        
        # 首先检查场景描述中的实际颜色和地形
        visual = scene.get("visual", {}) or {}
        environment = self._clean_prompt_text(visual.get("environment", "") if isinstance(visual, dict) else "")
        composition = self._clean_prompt_text(visual.get("composition", "") if isinstance(visual, dict) else "")
        description = self._clean_prompt_text(scene.get("description", ""))
        
        # 从场景描述中提取颜色信息
        scene_text = f"{environment} {composition} {description}".lower()
        
        # 检测场景描述中的颜色
        detected_colors = []
        color_keywords = {
            "gray-green": "gray-green",
            "grey-green": "gray-green",
            "gray green": "gray-green",
            "grey green": "gray-green",
            "青灰": "gray-green",
            "青灰色": "gray-green",
            "golden": "golden",
            "金色": "golden",
            "warm orange": "warm orange",
            "暖橙": "warm orange",
            "orange": "warm orange",
            "blue": "blue",
            "蓝色": "blue",
            "red": "red",
            "红色": "red"
        }
        
        for keyword, color_name in color_keywords.items():
            if keyword in scene_text:
                detected_colors.append(color_name)
                break  # 只取第一个匹配的颜色
        
        # 检测场景描述中的地形
        detected_terrain = None
        terrain_keywords = {
            "desert": "desert",
            "sand": "desert",
            "沙漠": "desert",
            "沙地": "desert",
            "gravel": "gravel",
            "沙砾": "gravel",
            "chamber": "chamber",
            "遗迹": "chamber",
            "corridor": "corridor",
            "走廊": "corridor"
        }
        
        for keyword, terrain_name in terrain_keywords.items():
            if keyword in scene_text:
                detected_terrain = terrain_name
                break
        
        profile_key = scene.get("scene_profile") or scene.get("scene_template") or scene.get("scene_key")
        scene_name = scene.get("scene_name") or scene.get("title", "")
        if not scene_name and script_data:
            scene_name = script_data.get("title", "")
        
        episode = scene.get("episode")
        if not episode and script_data:
            episode = script_data.get("episode")
        
        profile = self._get_scene_profile(scene_name, episode, profile_key=profile_key)
        
        parts = []
        
        # 优先使用场景描述中检测到的颜色，如果没有则使用模板
        if detected_colors:
            color = detected_colors[0]
            if color == "gray-green":
                parts.append("(gray-green tones:1.2)")
            elif color == "golden":
                parts.append("(golden sand:1.4)")
            elif color == "warm orange":
                parts.append("(warm orange tones:1.2)")
        elif profile and profile.get("color_palette", {}).get("prompt"):
            color_prompt = profile["color_palette"]["prompt"]
            if "golden sand" in color_prompt.lower():
                parts.append("(golden sand:1.4)")
            elif "warm orange" in color_prompt.lower():
                parts.append("(warm orange tones:1.2)")
        
        # 优先使用场景描述中检测到的地形，如果没有则使用模板
        if detected_terrain:
            if detected_terrain == "desert":
                # 根据颜色调整沙漠描述
                if detected_colors and detected_colors[0] == "gray-green":
                    parts.append("(gray-green desert:1.3)")
                else:
                    parts.append("(vast golden desert:1.3)")
            elif detected_terrain == "gravel":
                parts.append("(gray-green gravel plain:1.3)")
            elif detected_terrain == "chamber":
                parts.append("(ancient stone chamber:1.3)")
            elif detected_terrain == "corridor":
                parts.append("(stone corridor:1.3)")
        elif profile and profile.get("terrain", {}).get("prompt"):
            terrain_prompt = profile["terrain"]["prompt"]
            if "desert" in terrain_prompt.lower() or "sand" in terrain_prompt.lower():
                parts.append("(vast golden desert:1.3)")
            elif "chamber" in terrain_prompt.lower() or "corridor" in terrain_prompt.lower():
                parts.append("(ancient stone chamber:1.3)")
        
        # 默认值
        if not parts:
            if "沙漠" in scene_name or "desert" in scene_name.lower():
                if not self.ascii_only_prompt:
                    return "(金色沙漠:1.3)"
                else:
                    return "(golden desert:1.3)"
            elif "遗迹" in scene_name or "chamber" in scene_name.lower():
                if not self.ascii_only_prompt:
                    return "(古代石室:1.3)"
                else:
                    return "(ancient stone chamber:1.3)"
            else:
                if not self.ascii_only_prompt:
                    return "(背景一致:1.2)"
                else:
                    return "(consistent background:1.2)"
        
        return ", ".join(parts)
    
    def _build_scene_background_prompt(self, scene: Dict[str, Any], script_data: Dict[str, Any] = None) -> str:
        """根据场景模板构建背景描述 prompt"""
        profile_key = scene.get("scene_profile") or scene.get("scene_template") or scene.get("scene_key")
        scene_name = scene.get("scene_name") or scene.get("title", "")
        if not scene_name and script_data:
            scene_name = script_data.get("title", "")
        
        episode = scene.get("episode")
        if not episode and script_data:
            episode = script_data.get("episode")
        
        profile = self._get_scene_profile(scene_name, episode, profile_key=profile_key)
        if not profile:
            return ""
        
        parts = []
        
        # 颜色调色板
        if profile.get("color_palette", {}).get("prompt"):
            parts.append(profile["color_palette"]["prompt"])
        
        # 地形地貌
        if profile.get("terrain", {}).get("prompt"):
            parts.append(profile["terrain"]["prompt"])
        
        # 光照
        if profile.get("lighting", {}).get("prompt"):
            parts.append(profile["lighting"]["prompt"])
        
        return ", ".join(parts)

