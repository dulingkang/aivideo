"""
Prompt构建器

负责根据场景数据构建完整的Prompt，这是Prompt模块的核心组件。
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
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
        use_semantic_prompt: Optional[bool] = None,  # ⚡ 新增：是否使用语义化 prompt（FLUX 专用）
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
        
        # ⚡ 关键修复：FLUX 专用语义化 prompt 构建（wide + top_down + lying 场景）
        # ⚡ 重要：FLUX 使用 T5 tokenizer，支持 512+ tokens，不需要 77 token 限制
        if use_semantic_prompt:
            # FLUX 不需要 token 限制，直接返回完整语义化 prompt
            return self._build_semantic_prompt_for_flux(scene, intent)
        
        # 根据意图分析结果判断是否需要角色
        # ⚡ v2 格式支持：优先使用 character.present 字段
        character = scene.get("character", {}) or {}
        character_present_v2 = character.get("present", False)
        
        if include_character is None:
            # 优先使用 v2 格式的 character.present 字段
            if character_present_v2:
                include_character = True
                print(f"  ℹ v2 格式：character.present=true，需要角色")
            # 如果主要实体是角色，则需要角色
            elif intent['primary_entity'] and intent['primary_entity'].get('type') == 'character':
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
        
        # 处理 v2 格式：如果 camera 是字典，转换为字符串
        if isinstance(camera_desc, dict):
            camera_desc = self._convert_camera_v2_to_string(camera_desc)
        
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
            # 确保 camera_desc 是字符串
            if not isinstance(camera_desc, str):
                camera_desc = str(camera_desc)
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
        
        # ========== 第一部分：风格标签（根据任务类型决定）==========
        # 检查是否是科普视频（通过 script_data 或 scene 中的 category 判断）
        is_kepu_video = False
        
        # 方法1: 通过 script_data 判断
        if script_data:
            category = script_data.get('category', '')
            topic = script_data.get('topic', '')
            # 检查 category 是否是科普类别
            if category and category in ['universe', 'quantum', 'earth', 'energy', 'city', 'biology', 'ai']:
                is_kepu_video = True
            # 检查 topic 是否包含科普关键词
            elif topic and any(kw in topic.lower() for kw in ['科普', '科学', '宇宙', '量子', '地球', '能源', '城市', '生物', '人工智能']):
                is_kepu_video = True
        
        # 方法2: 通过 scene 判断
        if not is_kepu_video and scene:
            # 检查 scene 中是否有科普相关的标记
            scene_category = scene.get('category', '')
            if scene_category in ['universe', 'quantum', 'earth', 'energy', 'city', 'biology', 'ai']:
                is_kepu_video = True
            # 检查 prompt 中是否包含科普关键词
            scene_prompt = scene.get('prompt', '').lower()
            if any(kw in scene_prompt for kw in ['space', 'scientific', 'quantum', 'earth', 'energy', 'city', 'biology', 'ai', '科普', '科学', '宇宙', '量子', '地球', '能源', '城市', '生物', '人工智能', 'astronaut', 'space station', 'planet', 'satellite', 'nebula', 'black hole', 'mars', 'solar system']):
                is_kepu_video = True
        
        # 方法3: 通过 task_type 判断（如果 scene 中有 task_type 字段）
        if not is_kepu_video and scene:
            task_type = scene.get('task_type', '')
            if task_type == 'scene':
                # 场景生成通常是科普背景，但需要进一步确认
                # 如果 prompt 中没有明确的仙侠关键词，则认为是科普
                scene_prompt_lower = scene.get('prompt', '').lower()
                has_xianxia_keywords = any(kw in scene_prompt_lower for kw in ['xianxia', 'fantasy', '仙侠', '修仙', 'cultivator', 'han li', '韩立'])
                if not has_xianxia_keywords:
                    is_kepu_video = True
        
        use_chinese_prompt = not self.ascii_only_prompt
        
        # 初始化 xianxia_style（用于后续代码）
        if use_chinese_prompt:
            xianxia_style = "仙侠风格"
        else:
            xianxia_style = "xianxia fantasy"
        
        # 先识别角色（用于决定使用哪种风格）
        identified_characters = []
        if self._identify_characters:
            identified_characters = self._identify_characters(scene)
        
        # ⚡ 修复场景2：如果角色识别未检测到hanli，但prompt/composition中包含"Han Li"或"hanli"，强制识别
        if not identified_characters or "hanli" not in [c.lower() for c in identified_characters]:
            # 检查prompt、composition、description中是否包含Han Li
            scene_text = " ".join([
                str(scene.get("prompt", "")),
                str(scene.get("description", "")),
                str(scene.get("visual", {}).get("composition", "") if isinstance(scene.get("visual"), dict) else ""),
            ]).lower()
            if "han li" in scene_text or "hanli" in scene_text or "韩立" in scene_text:
                if not identified_characters:
                    identified_characters = ["hanli"]
                elif "hanli" not in [c.lower() for c in identified_characters]:
                    identified_characters.insert(0, "hanli")  # 添加到最前面
                print(f"  ✓ 强制识别：在prompt/composition中检测到Han Li，已添加hanli到角色列表")
        
        # ⚡ 核心修复：人物资产化 + 风格分离
        # 原则：人物层不使用风格词，风格只在Scene层注入
        # 不在这里添加风格标签，风格将在场景层添加（如果有角色，在角色描述之后）
        is_hanli = "hanli" in [c.lower() for c in identified_characters] if identified_characters else False
        
        if is_kepu_video:
            # 科普视频：不添加仙侠风格，使用科学/专业风格（在场景层添加）
            pass  # 风格在场景层处理
        else:
            # 仙侠视频：不在这里添加风格，风格将在场景层添加
            pass  # 风格在场景层处理
        
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
            
            # ⚡ v2 格式支持：优先使用 visual_constraints.environment
            visual_constraints = scene.get("visual_constraints", {}) or {}
            environment = self._clean_prompt_text(visual_constraints.get("environment", "") or "")
            if not environment and isinstance(visual, dict):
                environment = self._clean_prompt_text(visual.get("environment", "") or "")
            
            # ⚡ v2 格式支持：处理 visual_constraints.elements（关键物体，如卷轴）
            elements = visual_constraints.get("elements", [])
            if elements and isinstance(elements, list):
                # 将元素转换为可读描述
                element_descriptions = []
                for element in elements:
                    if isinstance(element, str):
                        element_lower = element.lower()
                        # 映射常见元素到可读描述
                        element_map = {
                            "golden_scroll": "golden scroll, prominent, clearly visible, main element, unrolling, glowing with spiritual light",
                            "scroll": "scroll, prominent, clearly visible, main element",
                            "golden_scroll_unrolling": "golden scroll unrolling, prominent, clearly visible, main element, glowing with spiritual light",
                        }
                        element_desc = element_map.get(element_lower, element.replace("_", " "))
                        element_descriptions.append(element_desc)
                
                if element_descriptions:
                    elements_text = ", ".join(element_descriptions)
                    priority_parts.append(f"({elements_text}:2.0)")
                    print(f"  ✓ 添加关键元素（最高优先级，权重2.0）: {elements_text[:60]}...")
            
            # 优先使用 environment（v2 格式），如果没有则使用 composition
            if environment:
                # 使用 environment 作为主要描述
                priority_parts.append(f"({environment}:2.0)")
                print(f"  ✓ 添加环境描述（最高优先级，权重2.0）: {environment[:60]}...")
            elif description_text:
                # 如果没有 environment，使用 description
                priority_parts.append(f"({description_text}:2.0)")
                print(f"  ✓ 添加场景描述（最高优先级，权重2.0）: {description_text[:60]}...")
            elif prompt_text:
                # 如果没有 environment 和 description，使用 prompt
                priority_parts.append(f"({prompt_text}:2.0)")
                print(f"  ✓ 添加场景 prompt（最高优先级，权重2.0）: {prompt_text[:60]}...")
            
            # 从composition中提取关键信息（作为补充）
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
            # ⚡ Prompt 优化：确保逗号分隔清晰（符合 Flux 最佳实践）
            cleaned_parts = []
            for part in priority_parts:
                if part:
                    part = part.strip().strip(',').strip()
                    if part:
                        cleaned_parts.append(part)
            priority_prompt = ", ".join(cleaned_parts)
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
            # ⚡ Prompt 优化：单人约束放在风格之后（第1位），确保风格标签在最前面
            # 用户反馈：场景5和7生成了多个人物，在所有人物场景都添加单人约束
            # 但风格标签必须在最前面（SDXL/Flux 最佳实践）
            if self.ascii_only_prompt:
                priority_parts.insert(1, "(single person:2.0)")  # 插入到第1位（风格之后）
            else:
                priority_parts.insert(1, "(单人:2.0)")  # 插入到第1位（风格之后）
            # print(f"  ✓ 人物场景：在风格之后添加单人约束（第1位，权重2.0，防止多个人物）")  # 减少日志
            # 识别场景中的所有角色
            if self._identify_characters:
                identified_characters = self._identify_characters(scene)
            else:
                identified_characters = []
            
            # ⚡ v2 格式支持：如果角色识别失败，直接从 character.id 读取
            if not identified_characters:
                character = scene.get("character", {}) or {}
                if isinstance(character, dict):
                    character_id = character.get("id", "")
                    if character_id:
                        identified_characters = [character_id]
                        print(f"  ✓ v2 格式：从 character.id 识别到角色: {character_id}")
            
            # ⚡ 修复场景2：如果角色识别未检测到hanli，但prompt/composition中包含"Han Li"，强制识别
            if not identified_characters or "hanli" not in [c.lower() for c in identified_characters]:
                # 检查prompt、composition、description中是否包含Han Li
                scene_text = " ".join([
                    str(scene.get("prompt", "")),
                    str(scene.get("description", "")),
                    str(scene.get("visual", {}).get("composition", "") if isinstance(scene.get("visual"), dict) else ""),
                ]).lower()
                if "han li" in scene_text or "hanli" in scene_text or "韩立" in scene_text:
                    if not identified_characters:
                        identified_characters = ["hanli"]
                    elif "hanli" not in [c.lower() for c in identified_characters]:
                        identified_characters.insert(0, "hanli")  # 添加到最前面
                    # print(f"  ✓ 强制识别（人物场景）：在prompt/composition中检测到Han Li，已添加hanli到角色列表")  # 减少日志
            
            # 如果识别到其他角色（不仅仅是韩立），使用角色描述生成
            if identified_characters:
                # 优先使用第一个识别的角色（通常是主要角色）
                primary_character = identified_characters[0]
                
                # ⚡ 核心修复：人物资产化 - 韩立使用Prompt模板（无风格词）
                is_hanli_char = primary_character.lower() == "hanli"
                if is_hanli_char:
                    # 加载HanLi.prompt模板（纯人物描述，无风格词）
                    hanli_prompt = self._load_character_template("HanLi")
                    if hanli_prompt:
                        character_desc = hanli_prompt.strip()
                        # 插入到第1位（约束之后）
                        if len(priority_parts) >= 1:
                            priority_parts.insert(1, character_desc)
                            insert_pos = 1
                        else:
                            priority_parts.append(character_desc)
                            insert_pos = len(priority_parts) - 1
                        # print(f"  ✓ 使用HanLi.prompt模板（人物资产，无风格词，第{insert_pos}位）")  # 减少日志
                    else:
                        # 降级到角色模板
                        character_profile = self._get_character_profile(primary_character)
                        if character_profile:
                            character_desc = self._build_character_description_prompt(character_profile, shot_type_for_prompt)
                            if character_desc:
                                if len(priority_parts) >= 1:
                                    priority_parts.insert(1, character_desc)
                                    insert_pos = 1
                                else:
                                    priority_parts.append(character_desc)
                                    insert_pos = len(priority_parts) - 1
                                print(f"  ✓ 使用角色模板（降级方案，第{insert_pos}位）")
                        else:
                            character_desc = None
                else:
                    # 其他角色：使用角色模板
                    character_profile = self._get_character_profile(primary_character)
                    if character_profile:
                        # 构建角色描述 prompt
                        character_desc = self._build_character_description_prompt(character_profile, shot_type_for_prompt)
                        if character_desc:
                            # 插入到第1位（约束之后）
                            if len(priority_parts) >= 1:
                                priority_parts.insert(1, character_desc)
                                insert_pos = 1
                            else:
                                priority_parts.append(character_desc)
                                insert_pos = len(priority_parts) - 1
                            print(f"  ✓ 应用角色描述（第{insert_pos}位）: {character_profile.get('character_name', primary_character)}")
                            print(f"  📝 角色描述内容: {character_desc[:100]}...")
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
        # ⚡ v2 格式支持：优先使用 character.pose，如果没有则使用 visual.character_pose
        character = scene.get("character", {}) or {}
        character_pose_v2 = character.get("pose", "")
        
        # 将 v2 格式的 pose 值转换为可读描述
        # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting（SDXL 对 NOT 不敏感）
        pose_map = {
            "lying_motionless": "body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position",
            "turning_head": "turning head, looking around, head movement",
            "recalling": "recalling, remembering, thoughtful expression",
            "focusing_gaze": "focusing gaze, looking intently, concentrated expression",
            "standing": "standing, upright position",
            "sitting": "sitting, seated position",
            "walking": "walking, moving forward",
        }
        
        if character_pose_v2 and character_pose_v2 in pose_map:
            character_pose_v2 = pose_map[character_pose_v2]
        
        if isinstance(visual, dict) and not use_chinese:
            # 优先使用 character.pose（v2 格式），如果没有则使用 visual.character_pose（v1 格式）
            character_pose = character_pose_v2 or self._clean_prompt_text(visual.get("character_pose") or "")
            if character_pose:
                # 检查是否包含正面朝向关键词
                pose_lower = character_pose.lower()
                has_facing = any(kw in pose_lower for kw in ["facing", "front", "正面", "面向", "forward", "toward camera", "facing camera"])
                has_back = any(kw in pose_lower for kw in ["back", "背面", "背后", "from behind", "rear"])
                
                # ⚡ 修复：检测表情相关描述（grim, dark, unpleasant等），提高权重
                has_expression_keywords = any(kw in pose_lower for kw in [
                    "grim", "dark", "unpleasant", "gloomy", "serious", "stern", "frown", "scowl",
                    "阴沉", "严肃", "不悦", "皱眉", "表情", "expression"
                ])
                pose_weight = 1.8
                if has_expression_keywords:
                    pose_weight = 2.5  # 表情描述提高权重
                    print(f"  ✓ 检测到character_pose中的表情描述，提高权重到{pose_weight:.1f}")
                
                # 基于意图分析的动作类型，动态调整权重（通用处理）
                action_type = intent['action_type']
                if action_type == 'static':
                    # 静态动作，使用较高权重确保姿势准确
                    if not has_back:  # 如果不是明确要求背面，添加正面朝向
                        priority_parts.append(f"({character_pose}, facing camera, front view:{pose_weight:.1f})")
                        print(f"  ✓ 使用 visual.character_pose（静态动作，增强正面朝向，权重{pose_weight:.1f}）: {character_pose}")
                    else:
                        priority_parts.append(f"({character_pose}:{pose_weight:.1f})")
                        print(f"  ✓ 使用 visual.character_pose（静态动作，增强权重{pose_weight:.1f}）: {character_pose}")
                else:
                    # 动态动作或其他，根据是否包含正面朝向调整权重
                    if has_facing:
                        priority_parts.append(f"({character_pose}:{pose_weight:.1f})")
                        # 额外强调正面朝向，防止被其他描述覆盖
                        priority_parts.append("(facing camera, front view, face forward, frontal view:1.8)")
                        print(f"  ✓ 使用 visual.character_pose（正面朝向，增强权重{pose_weight:.1f}）: {character_pose}")
                    elif has_back:
                        priority_parts.append(f"({character_pose}:{pose_weight:.1f})")
                        print(f"  ✓ 使用 visual.character_pose（背面朝向，权重{pose_weight:.1f}）: {character_pose}")
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
                # ⚡ 优化：如果composition包含关键动作（如"lying on"、"lying in"），提高权重
                composition_lower = composition.lower()
                if any(kw in composition_lower for kw in ["lying on", "lying in", "lying", "sitting on", "standing on"]):
                    # 包含关键动作和环境，提高权重到1.8，确保动作和环境都被正确生成
                    composition_weight = max(composition_weight, 1.8)
                    print(f"  ✓ 检测到关键动作（lying/sitting/standing on），提高composition权重到{composition_weight:.2f}")
                
                # ⚡ 特殊处理：如果composition包含表情描述（expression darkens, grim等）或recall动作，提高权重并强调
                composition_lower = composition.lower()
                has_expression_in_composition = any(kw in composition_lower for kw in [
                    "expression darkens", "expression turning grim", "grim expression", "dark expression",
                    "expression darken", "face darkens", "expression turns", "表情", "阴沉"
                ])
                has_recall_action = any(kw in composition_lower for kw in ["recall", "recalls", "recalling", "回想", "回忆"])
                
                if has_expression_in_composition:
                    composition_weight = 2.5  # 大幅提高权重
                    # 增强描述，明确表情
                    enhanced_composition = composition
                    if "expression darkens" in composition_lower or "expression turning grim" in composition_lower:
                        enhanced_composition = f"{composition}, grim expression, dark expression, serious face, stern look, unpleasant expression"
                    priority_parts.append(f"({enhanced_composition}:{composition_weight:.2f})")
                    print(f"  ✓ 检测到composition中的表情描述，大幅提高权重到{composition_weight:.2f}，强调表情")
                elif has_recall_action:
                    composition_weight = max(composition_weight, 2.0)  # 提高权重
                    # 增强描述，明确recall动作和表情
                    enhanced_composition = composition
                    if "expression darkens" in composition_lower or "expression turning grim" in composition_lower:
                        enhanced_composition = f"{composition}, grim expression, dark expression, serious face, stern look"
                    priority_parts.append(f"({enhanced_composition}:{composition_weight:.2f})")
                    print(f"  ✓ 检测到recall动作，提高composition权重到{composition_weight:.2f}，强调回想和表情")
                else:
                    # ⚡ 使用通用的prompt增强方法（基于语义分析，而不是硬编码关键词）
                    enhanced_composition = self.optimizer.enhance_prompt_part(composition, "composition")
                    
                    # 如果被增强了，提取新的权重（如果有）
                    import re
                    weight_match = re.search(r':(\d+\.?\d*)', enhanced_composition)
                    if weight_match:
                        composition_weight = float(weight_match.group(1))
                    
                    priority_parts.append(f"({enhanced_composition}:{composition_weight:.2f})" if not enhanced_composition.startswith("(") else enhanced_composition)
                    if enhanced_composition != composition:
                        print(f"  ✓ 使用 visual.composition（已增强，权重{composition_weight:.2f}）: {enhanced_composition[:80]}...")
                    else:
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
        
        # ⚡ 竖屏模式优化：如果没有明确指定镜头类型，默认使用中景（避免过近的镜头）
        # 但需要检查camera字段是否包含明确的镜头描述
        has_explicit_shot_type = (
            shot_type_for_prompt["is_wide"] or 
            shot_type_for_prompt["is_medium"] or 
            shot_type_for_prompt["is_close"] or 
            shot_type_for_prompt["is_full_body"]
        )
        
        # 检查camera字段是否包含明确的镜头关键词（即使shot_type_for_prompt没有标记）
        camera_has_shot_type = False
        if camera_desc:
            camera_lower = camera_desc.lower()
            # 检查是否包含明确的镜头类型关键词
            if any(kw in camera_lower for kw in [
                "wide shot", "wide pan", "long shot", "extreme wide", "establishing shot",
                "medium shot", "mid shot", "中景",
                "close-up", "closeup", "close up", "特写", "近景",
                "full body", "全身",
                "top-down", "俯视", "bird's eye",
                "eye close-up", "extreme eye", "眼睛特写"
            ]):
                camera_has_shot_type = True
        
        # 如果没有明确指定镜头类型，且camera字段也没有明确的镜头描述，默认使用中景
        if not has_explicit_shot_type and not camera_has_shot_type and include_character:
            # 竖屏模式默认中景，避免镜头过近
            shot_type_for_prompt["is_medium"] = True
            print(f"  ✓ 竖屏模式优化：未指定镜头类型，默认使用中景（避免过近的镜头）")
        elif has_explicit_shot_type or camera_has_shot_type:
            print(f"  ✓ 检测到明确的镜头类型，保持原始镜头描述（不强制转换为中景）")
        
        if shot_type_for_prompt["is_wide"] or shot_type_for_prompt["is_full_body"]:
            # 远景场景：强制添加正面朝向和排除背影，避免人物太小和背影
            # ⚡ 修复镜头太近：远景场景明确添加"distant view"确保镜头距离
            if use_chinese:
                priority_parts.append("(单人，正面视角，面向镜头，远景，远距离:1.8)")
                priority_parts.append("(正确宽高比，自然面部比例:1.3)")  # 保护宽高比，防止瘦长脸
            else:
                priority_parts.append("(single person, front view, facing camera, distant view, far away, wide shot:1.8)")
                priority_parts.append("(correct aspect ratio, natural face proportions, no stretch:1.3)")  # 保护宽高比，防止瘦长脸
        elif shot_type_for_prompt["is_medium"]:
            # 中景场景：强制添加正面朝向，避免背影和镜头过近
            if use_chinese:
                priority_parts.append("(中景，正面视角，面向镜头，自然身体比例，适当距离:1.8)")  # 提高权重，强调正面和自然比例，明确适当距离
                priority_parts.append("(修长身材，窄肩，自然姿势:1.3)")  # 强调自然姿势
                priority_parts.append("(避免过近镜头，保持适当距离:1.2)")  # 明确排除过近镜头
            else:
                priority_parts.append("(medium shot, front view, facing camera, natural body proportions, appropriate distance:1.8)")  # 提高权重，强调正面和自然比例，明确适当距离
                priority_parts.append("(slim body, narrow shoulders, natural pose:1.3)")  # 强调自然姿势
                priority_parts.append("(avoid too close, maintain appropriate distance:1.2)")  # 明确排除过近镜头
        elif shot_type_for_prompt["is_close"]:
            # 检查是否是眼睛特写或面部特写场景（需要保持特写，不转换为中景）
            is_eye_closeup = shot_type_for_prompt.get("is_eye_closeup", False)
            camera_desc_check = scene.get("camera") if scene else ""
            # 处理 v2 格式：如果 camera 是字典，转换为字符串
            if isinstance(camera_desc_check, dict):
                camera_desc_check = self._convert_camera_v2_to_string(camera_desc_check)
            if not isinstance(camera_desc_check, str):
                camera_desc_check = str(camera_desc_check) if camera_desc_check else ""
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
                # 其他特写场景：避免太近的镜头，使用中景描述（竖屏模式优化）
                if use_chinese:
                    priority_parts.append("(中景，适当距离:1.5)")  # 提高权重，明确适当距离
                    priority_parts.append("(修长身材，窄肩:1.3)")
                    priority_parts.append("(避免过近镜头:1.2)")  # 明确排除过近镜头
                else:
                    priority_parts.append("(medium shot, appropriate distance:1.5)")  # 提高权重，明确适当距离
                    priority_parts.append("(slim body, narrow shoulders:1.3)")
                    priority_parts.append("(avoid too close, maintain distance:1.2)")  # 明确排除过近镜头
                print(f"  ⚠ 检测到特写镜头，已转换为中景以避免身体过宽和模糊（竖屏模式优化：明确适当距离）")
        
        # ========== 第三部分：场景背景（增强版，保留完整细节）==========
        # 如果已经使用了中文 description，就不再添加 visual.environment（避免重复和混用中英文）
        # 如果还没有添加 description，才考虑使用 visual.environment
        # ⚡ v2 格式支持：优先使用 visual_constraints.environment，如果没有则使用 visual.environment
        if not use_chinese:
            # 优先从 visual_constraints.environment 读取（v2 格式）
            visual_constraints = scene.get("visual_constraints", {}) or {}
            environment_visual = self._clean_prompt_text(visual_constraints.get("environment") or "")
            
            # 如果没有 visual_constraints.environment，则使用 visual.environment（v1 格式）
            if not environment_visual and isinstance(visual, dict):
                environment_visual = self._clean_prompt_text(visual.get("environment") or "")
            
            if environment_visual:
                # 不再过度精简，保留完整的环境描述以增强场景表现
                # 环境描述包含场景中的物体、地形、天气等重要信息
                # 提高权重从1.4到1.8，确保环境场景（如沙漠）被正确生成
                # ⚡ 优化：对于远景场景，进一步提高环境权重到2.0，确保背景清晰可见
                env_weight = 1.8
                if shot_type_for_prompt.get("is_wide") or shot_type_for_prompt.get("is_full_body"):
                    env_weight = 2.0
                    print(f"  ✓ 远景场景：提高环境描述权重到{env_weight:.1f}，确保背景清晰可见")
                
                # ⚡ 特殊处理：如果环境描述包含"three suns"或"lunar phantoms"，大幅提高权重并强调可见性
                env_lower = environment_visual.lower()
                if "three" in env_lower and ("sun" in env_lower or "lunar" in env_lower or "moon" in env_lower):
                    env_weight = 2.5  # 大幅提高权重
                    # 增强描述，强调太阳和月亮的可见性和数量
                    enhanced_env = environment_visual
                    if "three dazzling suns" in env_lower:
                        enhanced_env = f"Three large and prominent dazzling suns, clearly visible and bright, dominating the sky, {environment_visual}"
                    elif "three" in env_lower and "sun" in env_lower:
                        enhanced_env = f"Three large and prominent dazzling suns, clearly visible and bright, dominating the sky, {environment_visual}"
                    if "four" in env_lower and ("lunar" in env_lower or "moon" in env_lower):
                        enhanced_env = f"{enhanced_env}, four faint but clearly visible lunar phantoms, clearly distinguishable in the sky, not just one sun"
                    priority_parts.append(f"({enhanced_env}:{env_weight:.1f})")
                    print(f"  ✓ 检测到天空场景（太阳/月亮），大幅提高权重到{env_weight:.1f}，强调可见性和数量")
                else:
                    # ⚡ 使用通用的prompt增强方法（基于语义分析，而不是硬编码关键词）
                    enhanced_env = self.optimizer.enhance_prompt_part(environment_visual, "environment")
                    
                    # 如果被增强了，提取新的权重（如果有）
                    import re
                    weight_match = re.search(r':(\d+\.?\d*)', enhanced_env)
                    if weight_match:
                        env_weight = float(weight_match.group(1))
                    
                    priority_parts.append(f"({enhanced_env}:{env_weight:.1f})" if not enhanced_env.startswith("(") else enhanced_env)
                    if enhanced_env != environment_visual:
                        print(f"  ✓ 使用 visual.environment（已增强，权重{env_weight:.1f}）: {enhanced_env[:80]}...")
                # print(f"  ✓ 使用 visual.environment（完整版，权重{env_weight:.1f}）: {environment_visual}")  # 减少日志
        
        # ========== 添加原始场景 prompt（关键信息，优先处理）==========
        # ⚡ Prompt 优化：原始场景 prompt 应该在风格和角色之后，环境之前
        # 顺序：风格(0) -> 约束(1) -> 角色(2) -> 场景prompt(3) -> 环境/背景 -> 其他
        # 注意：prompt_text 在第 461 行已定义
        if prompt_text and not use_chinese:
            # 检查是否已经包含在 priority_parts 中（避免重复）
            prompt_already_included = any(
                prompt_text.lower() in part.lower() or 
                part.lower() in prompt_text.lower() or
                any(keyword in part.lower() for keyword in prompt_text.lower().split()[:3])  # 检查前3个关键词
                for part in priority_parts
            )
            if not prompt_already_included:
                # ⚡ 优化：原始场景 prompt 插入到第3位（风格、约束、角色之后）
                # 这是场景的核心内容，应该在风格和角色之后立即出现
                insert_pos = min(3, len(priority_parts))  # 最多插入到第3位
                # 如果已经有风格、约束、角色，插入到第3位；否则插入到合适位置
                if len(priority_parts) >= 3:
                    insert_pos = 3
                elif len(priority_parts) >= 2:
                    insert_pos = 2
                elif len(priority_parts) >= 1:
                    insert_pos = 1
                else:
                    insert_pos = 0
                priority_parts.insert(insert_pos, prompt_text)
                # print(f"  ✓ 添加原始场景 prompt（核心内容，第{insert_pos}位，风格和角色之后）: {prompt_text[:80]}...")  # 减少日志
        
        # ========== 添加场景背景描述（确保有背景，即使有角色）==========
        # ⚡ Prompt 优化：场景背景应该在原始场景 prompt 之后
        # 顺序：风格(0) -> 约束(1) -> 角色(2) -> 场景prompt(3) -> 背景(4) -> 其他
        # 对于科普视频，即使有角色，也需要场景背景
        scene_bg_compact = self._build_scene_background_prompt_compact(scene, script_data)
        if scene_bg_compact:
            # 将背景描述添加到 priority_parts（在原始场景 prompt 之后）
            # 找到原始场景 prompt 的位置，在其后插入
            insert_pos = len(priority_parts)
            if prompt_text:
                for i, part in enumerate(priority_parts):
                    if prompt_text.lower() in part.lower():
                        insert_pos = i + 1  # 在原始场景 prompt 之后
                        break
            # 如果没找到原始场景 prompt，插入到角色之后（第3位）
            if insert_pos == len(priority_parts):
                insert_pos = min(4, len(priority_parts))  # 默认插入到第4位（风格、约束、角色、场景prompt之后）
            priority_parts.insert(insert_pos, scene_bg_compact)
            # print(f"  ✓ 应用场景背景模板（精简版，第{insert_pos}位，场景prompt之后，确保有背景）: {scene_bg_compact}")  # 减少日志
        
        # ========== 第五部分：动作描述（智能综合权重调整）==========
        # 使用综合权重调整后的动作权重
        weight_adjustments = intent.get('weight_adjustments', {})
        action_weight = weight_adjustments.get('action_weight', 1.2)
        
        # 如果已经有character_pose，检查是否需要补充动作信息
        use_chinese = not self.ascii_only_prompt
        # ⚡ v2 格式支持：优先使用 character.pose
        character = scene.get("character", {}) or {}
        character_pose_v2 = character.get("pose", "")
        
        # 将 v2 格式的 pose 值转换为可读描述
        pose_map = {
            "lying_motionless": "lying motionless",
            "turning_head": "turning head",
            "recalling": "recalling",
            "focusing_gaze": "focusing gaze",
        }
        if character_pose_v2 and character_pose_v2 in pose_map:
            character_pose_v2 = pose_map[character_pose_v2]
        
        # 优先使用 character.pose（v2 格式），如果没有则使用 visual.character_pose（v1 格式）
        character_pose_from_visual = visual.get("character_pose", "") if isinstance(visual, dict) else ""
        character_pose_combined = character_pose_v2 or character_pose_from_visual
        
        if character_pose_combined and not use_chinese:
            # 如果 character_pose 存在但不够详细，可以补充 action（仅英文模式）
            character_pose_text = character_pose_combined.lower()
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
                        # ⚡ 优化：对于lying动作，提高权重并明确"lying on ground/sand"
                        # 检查composition或environment中是否有"on sand/ground/desert"
                        composition_text = str(visual.get("composition", "")).lower() if isinstance(visual, dict) else ""
                        environment_text = str(visual.get("environment", "")).lower() if isinstance(visual, dict) else ""
                        
                        # 如果composition或environment中包含"sand/ground/desert"，明确"lying on"
                        if "sand" in composition_text or "sand" in environment_text or "desert" in composition_text or "desert" in environment_text:
                            # ⚡ 修复：大幅提高权重到2.5，确保"lying on sand"被正确生成
                            lying_weight = max(action_weight + 0.8, 2.8)  # ⚡ 修复：提高到2.8，确保高优先级
                            # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting（SDXL 对 NOT 不敏感）
                            lying_text = f"(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:{lying_weight:.2f})"
                            # ⚡ 使用通用的prompt增强方法（自动添加NOT standing等排除词）
                            enhanced_lying = self.optimizer.enhance_prompt_part(lying_text, "action")
                            priority_parts.append(enhanced_lying)
                            print(f"  ✓ 检测到lying动作和sand/desert环境，强调'lying on sand/ground/desert'，权重{lying_weight:.2f}（高优先级，排除standing和sitting）")
                        else:
                            # ⚡ 修复：即使没有明确环境，也添加排除词和提高权重
                            lying_weight = max(action_weight + 0.5, 2.5)  # 至少2.5
                            # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting
                            lying_text = f"(body fully on the ground, legs fully extended, arms lying flat, no bent knees, horizontal position:{lying_weight:.2f})"
                            # ⚡ 使用通用的prompt增强方法
                            enhanced_lying = self.optimizer.enhance_prompt_part(lying_text, "action")
                            priority_parts.append(enhanced_lying)
                            print(f"  ✓ 检测到lying动作，强调'lying down'，权重{lying_weight:.2f}（排除standing和sitting）")
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
                    # ⚡ 优化：对于lying动作，提高权重并明确"lying on ground/sand"
                    # 检查composition或environment中是否有"on sand/ground/desert"
                    composition_text = str(visual.get("composition", "")).lower() if isinstance(visual, dict) else ""
                    environment_text = str(visual.get("environment", "")).lower() if isinstance(visual, dict) else ""
                    
                    # 如果composition或environment中包含"sand/ground/desert"，明确"lying on"
                    if "sand" in composition_text or "sand" in environment_text or "desert" in composition_text or "desert" in environment_text:
                        # ⚡ 修复：大幅提高权重到2.5，确保"lying on sand"被正确生成
                        lying_weight = max(action_weight + 0.8, 2.8)  # ⚡ 修复：提高到2.8，确保高优先级
                        lying_text = f"(lying on sand, lying on ground, lying on desert, NOT standing, NOT sitting, horizontal position, prone, supine:{lying_weight:.2f})"
                        # ⚡ 使用通用的prompt增强方法（自动添加NOT standing等排除词）
                        enhanced_lying = self.optimizer.enhance_prompt_part(lying_text, "action")
                        priority_parts.append(enhanced_lying)
                        print(f"  ✓ 检测到lying动作和sand/desert环境，强调'lying on sand/ground/desert'，权重{lying_weight:.2f}（高优先级，排除standing和sitting）")
                    else:
                        lying_text = f"(lying, lying down:{action_weight:.2f})"
                        # ⚡ 使用通用的prompt增强方法
                        enhanced_lying = self.optimizer.enhance_prompt_part(lying_text, "action")
                        priority_parts.append(enhanced_lying)
                elif "use" in action_simple or "cast" in action_simple:
                    priority_parts.append(f"({action_simple}:{action_weight:.2f})")
        
        # ========== 第六部分：风格注入（Scene层，无权重标记）==========
        # ⚡ 核心修复：风格只在Scene层注入，不在人物层
        # 风格描述简洁，无权重标记，避免干扰
        use_chinese = not self.ascii_only_prompt
        scene_style_text = None
        if not is_kepu_video:
            # 仙侠风格：简洁描述，无权重标记，但更明确强调动漫风格
            scene_style_text = "Chinese xianxia anime illustration, 3D rendered anime, anime cinematic style, cinematic lighting" if not use_chinese else "中国仙侠动漫插画，3D渲染动漫，动漫电影风格，电影级光照"
            # 添加到场景描述之后（如果有角色，在角色之后）
            priority_parts.append(scene_style_text)
            # print(f"  ✓ Scene层风格注入（无权重标记，增强动漫风格）: {scene_style_text}")  # 减少日志
        
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
        
        # ⚡ 核心修复：移除过多风格标签，风格已在Scene层注入（第845行）
        # secondary_parts中的风格标签已移除，避免与Scene层风格冲突
        # 风格只在Scene层通过priority_parts注入，保持一致性
        
        # 合并：只使用优先部分，确保关键信息在前 77 tokens 内
        # 使用更准确的 token 估算（考虑括号和权重标记）
        # ⚡ Prompt 优化：确保逗号分隔清晰（符合 Flux/SDXL 最佳实践）
        # 清理每个部分，移除多余的逗号和空格
        cleaned_parts = []
        for part in priority_parts:
            if part:
                # 移除开头和结尾的逗号和空格
                part = part.strip().strip(',').strip()
                if part:
                    cleaned_parts.append(part)
        priority_prompt = ", ".join(cleaned_parts)
        
        # 尝试使用CLIP tokenizer进行准确计算，如果不可用则使用保守估算
        estimated_tokens = self.token_estimator.estimate(priority_prompt)
        
        # ⚡ 核心修复：通用保护机制 - 保护所有高权重和关键内容不被优化器移除
        # 在优化前保存关键内容（基于权重和重要性，而非硬编码关键词）
        protected_contents = []  # 存储需要保护的内容及其元数据
        
        for i, part in enumerate(priority_parts):
            part_lower = part.lower()
            
            # 1. 保护角色模板内容（通过特征词检测）
            if any(keyword in part_lower for keyword in ["young male cultivator", "chinese xianxia novel", "slim but resilient", "sharp calm eyes", "dark simple cultivator robe", "dark green simple"]):
                protected_contents.append({
                    "content": part,
                    "type": "character_template",
                    "priority": 1,  # 最高优先级
                    "keywords": ["young male cultivator", "chinese xianxia novel", "slim but resilient"]
                })
                # print(f"  🛡️ 检测到角色模板内容（位置{i}），将在优化后检查并保护")  # 减少日志
            
            # 2. 保护高权重动作描述（权重 >= 1.8 或包含关键动作+环境组合）
            # 提取权重（如果存在）
            import re
            weight_match = re.search(r':([\d.]+)\)', part)
            weight = float(weight_match.group(1)) if weight_match else 1.0
            
            # 检测关键动作+环境组合（如"lying on sand/desert/ground"）
            key_action_patterns = [
                r"lying\s+on\s+(sand|desert|ground|floor|earth)",
                r"sitting\s+on\s+(sand|desert|ground|floor|rock|stone)",
                r"standing\s+on\s+(sand|desert|ground|floor|rock|stone|mountain)",
                r"walking\s+(in|on|through)\s+(sand|desert|forest|mountain|valley)",
            ]
            has_key_action_env = any(re.search(pattern, part_lower) for pattern in key_action_patterns)
            
            if weight >= 1.8 or has_key_action_env:
                protected_contents.append({
                    "content": part,
                    "type": "high_weight_action",
                    "priority": 2 if weight >= 2.0 else 3,
                    "keywords": [part_lower[:50]]  # 使用内容的前50个字符作为关键词
                })
                # print(f"  🛡️ 检测到高权重动作描述（位置{i}，权重{weight:.2f}），将在优化后检查并保护")  # 减少日志
            
            # 3. 保护高权重环境描述（权重 >= 1.8 或包含关键环境词）
            key_environment_keywords = ["desert", "sand", "forest", "mountain", "valley", "ocean", "sea", "river", "lake", "cave", "temple", "palace"]
            has_key_env = any(kw in part_lower for kw in key_environment_keywords)
            
            if (weight >= 1.8 or has_key_env) and "action" not in part_lower[:20]:  # 排除动作描述（已在上面处理）
                # 检查是否已经作为动作+环境组合被保护
                is_already_protected = any(part == pc["content"] for pc in protected_contents)
                if not is_already_protected:
                    protected_contents.append({
                        "content": part,
                        "type": "high_weight_environment",
                        "priority": 2 if weight >= 2.0 else 3,
                        "keywords": [kw for kw in key_environment_keywords if kw in part_lower]
                    })
                    # print(f"  🛡️ 检测到高权重环境描述（位置{i}，权重{weight:.2f}），将在优化后检查并保护")  # 减少日志
        
        # 如果估算超过 60 tokens（留出安全边界，确保不超过77），使用智能优化
        # 从70降低到60，因为实际tokenizer计算可能比估算值高，需要更多安全边界
        if estimated_tokens > 60:
            # 尝试使用智能优化（基于语义重要性）
            # print(f"  🧠 Prompt 过长 ({estimated_tokens} tokens)，尝试智能优化...")  # 减少日志
            optimized_parts = self.optimizer.optimize(priority_parts, max_tokens=60)
            if len(optimized_parts) < len(priority_parts):
                # 检查所有保护的内容是否仍然存在
                for protected in protected_contents:
                    content = protected["content"]
                    keywords = protected["keywords"]
                    content_type = protected["type"]
                    priority = protected["priority"]
                    
                    # 检查是否仍然存在（完全匹配或包含关键词）
                    still_present = any(
                        content == part or 
                        any(kw in part.lower() for kw in keywords)
                        for part in optimized_parts
                    )
                    
                    if not still_present:
                        # 根据优先级决定插入位置
                        if priority == 1:  # 角色模板：插入到前面
                            insert_pos = min(1, len(optimized_parts)) if len(optimized_parts) > 0 else 0
                            optimized_parts.insert(insert_pos, content)
                            # print(f"  ⚠ {content_type}被优化器移除，已强制加回（位置{insert_pos}，优先级{priority}）")  # 减少日志
                        elif priority == 2:  # 高优先级：插入到前面
                            insert_pos = min(2, len(optimized_parts)) if len(optimized_parts) > 0 else 0
                            optimized_parts.insert(insert_pos, content)
                            # print(f"  ⚠ {content_type}被优化器移除，已强制加回（位置{insert_pos}，优先级{priority}）")  # 减少日志
                        else:  # 普通优先级：追加到后面
                            optimized_parts.append(content)
                            # print(f"  ⚠ {content_type}被优化器移除，已强制加回（优先级{priority}）")  # 减少日志
                
                # 保护Scene层风格
                style_still_present = scene_style_text and any(scene_style_text in part or "xianxia anime" in part.lower() for part in optimized_parts)
                if not style_still_present and scene_style_text:
                    optimized_parts.append(scene_style_text)
                    # print(f"  ⚠ Scene层风格被优化器移除，已强制加回")  # 减少日志
                
                priority_parts = optimized_parts
                priority_prompt = ", ".join(filter(None, priority_parts))
                estimated_tokens = self.token_estimator.estimate(priority_prompt)
                # print(f"  ✓ 智能优化完成: {len(optimized_parts)} 个部分，{estimated_tokens} tokens（关键内容已保护）")  # 减少日志
            else:
                # 如果智能优化没有效果，使用传统精简方法
                print(f"  ⚠ 智能优化未达到预期，使用传统精简方法...")
        
        # 确保仙侠风格描述不会被优化阶段剔除（仅对非科普视频）
        # 但只添加简单的风格关键词，完整的Scene层风格应该在前面
        if not is_kepu_video:
            has_any_style = any(self._has_xianxia_keyword(part) or (scene_style_text and scene_style_text in part) for part in priority_parts)
            if not has_any_style:
                # 如果没有完整的Scene层风格，至少添加简单风格关键词
                simple_style = "xianxia fantasy" if not use_chinese else "仙侠风格"
                priority_parts.insert(0, simple_style)
                priority_prompt = ", ".join(filter(None, priority_parts))
                estimated_tokens = self.token_estimator.estimate(priority_prompt)
                print("  ✓ 智能优化后补回仙侠风格提示（简单关键词），确保风格一致")
        
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
        
        # ⚡ 关键修复：如果仍然超过77 tokens，进行强制精简
        if final_estimated > 77:
            print(f"  ⚠ 警告: Prompt 最终长度 ({final_estimated} tokens) 超过 77 tokens 限制，进行强制精简...")
            # 强制精简：只保留最关键的部分
            # 1. 保留角色模板（如果存在）
            # 2. 保留高权重动作+环境组合（权重 >= 2.0）
            # 3. 保留高权重环境描述（权重 >= 2.0）
            # 4. 保留风格描述（简化版）
            essential_parts = []
            style_part = None
            single_person_part = None
            
            # 提取并保留最关键的内容
            for part in priority_parts:
                part_lower = part.lower()
                import re
                weight_match = re.search(r':([\d.]+)\)', part)
                weight = float(weight_match.group(1)) if weight_match else 1.0
                
                # ⚡ 关键修复：优先保留风格描述（最高优先级，确保风格正确）
                if any(kw in part_lower for kw in ["xianxia", "anime", "cinematic", "illustration", "仙侠", "动漫"]):
                    if style_part is None:  # 只保留第一个风格描述
                        style_part = part
                        print(f"  ✓ 保留风格描述: {part[:60]}...")
                    continue
                
                # ⚡ 关键修复：优先保留single person约束（第二优先级，确保单人）
                if "single person" in part_lower or "only one" in part_lower or "单人" in part_lower:
                    if single_person_part is None:  # 只保留第一个single person约束
                        single_person_part = part
                        print(f"  ✓ 保留single person约束: {part[:60]}...")
                    continue
                
                # 保留角色模板
                if any(kw in part_lower for kw in ["young male cultivator", "chinese xianxia novel", "slim but resilient", "cultivator", "robe", "dark green"]):
                    essential_parts.append(part)
                    continue
                
                # ⚡ 关键修复：优先保留关键动作（lying/sitting/standing），无论是否有环境描述
                # 检查是否包含关键动作关键词
                has_lying = any(kw in part_lower for kw in ["lying", "lie", "躺", "horizontal position", "prone", "supine"])
                has_sitting = any(kw in part_lower for kw in ["sitting", "sit", "坐", "seated"])
                has_standing = any(kw in part_lower for kw in ["standing", "stand", "站", "upright"])
                
                # 如果包含关键动作，必须保留（降低权重阈值到1.5，确保lying描述被保留）
                if has_lying or has_sitting or has_standing:
                    if weight >= 1.5:  # 降低阈值，确保lying描述被保留
                        essential_parts.append(part)
                        print(f"  ✓ 保留关键动作描述（权重{weight:.2f}）: {part[:60]}...")
                        continue
                
                # 保留高权重动作+环境组合（权重 >= 2.0）
                key_action_patterns = [
                    r"lying\s+on\s+(sand|desert|ground)",
                    r"sitting\s+on\s+(sand|desert|ground|rock)",
                    r"standing\s+on\s+(sand|desert|ground|rock|mountain)",
                ]
                has_key_action_env = any(re.search(pattern, part_lower) for pattern in key_action_patterns)
                if has_key_action_env and weight >= 2.0:
                    essential_parts.append(part)
                    continue
                
                # 保留高权重环境描述（权重 >= 2.0）
                key_env_keywords = ["desert", "sand", "forest", "mountain", "valley"]
                has_key_env = any(kw in part_lower for kw in key_env_keywords)
                if has_key_env and weight >= 2.0:
                    essential_parts.append(part)
                    continue
                
                # 其他部分继续处理（single person已在前面处理）
                continue
            
            # ⚡ 关键修复：确保风格和single person约束在最前面（按正确顺序）
            # 1. 风格描述（第0位）
            if style_part:
                essential_parts.insert(0, style_part)
            elif not any("xianxia" in p.lower() or "anime" in p.lower() for p in essential_parts):
                essential_parts.insert(0, "Chinese xianxia anime illustration, anime cinematic style")
            
            # 2. single person约束（第1位，风格之后）
            if single_person_part:
                essential_parts.insert(1, single_person_part)
            elif include_character:  # 如果是人物场景但没有single person约束，强制添加
                essential_parts.insert(1, "(single person:2.5)")
                print(f"  ✓ 强制添加single person约束（确保单人）")
            
            # 重新组合
            priority_prompt = ", ".join(filter(None, essential_parts))
            final_estimated = self.token_estimator.estimate(priority_prompt)
            
            # 如果仍然超过，进一步精简角色模板
            if final_estimated > 77:
                print(f"  ⚠ 强制精简后仍超过限制 ({final_estimated} tokens)，进一步精简角色模板...")
                # 精简角色模板：保留关键特征（服饰、发型、角色类型）
                simplified_parts = []
                style_part_simplified = None
                single_person_part_simplified = None
                
                for part in essential_parts:
                    part_lower = part.lower()
                    # ⚡ 关键修复：保留风格描述（不精简，放在最前面）
                    if any(kw in part_lower for kw in ["xianxia", "anime", "cinematic", "illustration"]):
                        if style_part_simplified is None:
                            style_part_simplified = part
                        continue
                    
                    # ⚡ 关键修复：保留single person约束（不精简，放在第二位）
                    if "single person" in part_lower or "only one" in part_lower or "单人" in part_lower:
                        if single_person_part_simplified is None:
                            single_person_part_simplified = part
                        continue
                    
                    if any(kw in part_lower for kw in ["young male cultivator", "chinese xianxia novel", "slim but resilient"]):
                        # ⚡ 关键修复：保留更多关键特征，确保风格正确
                        # 保留：角色类型、服饰、发型、基本特征
                        simplified_parts.append("young male cultivator, (dark green simple cultivator robe:2.0), long black hair, slim build, calm expression")
                    else:
                        simplified_parts.append(part)
                
                # ⚡ 确保风格和single person约束在最前面（按正确顺序）
                # 1. 风格描述（第0位）
                if style_part_simplified:
                    simplified_parts.insert(0, style_part_simplified)
                elif not any("xianxia" in p.lower() or "anime" in p.lower() for p in simplified_parts):
                    simplified_parts.insert(0, "Chinese xianxia anime illustration, anime cinematic style")
                
                # 2. single person约束（第1位，风格之后）
                if single_person_part_simplified:
                    simplified_parts.insert(1, single_person_part_simplified)
                elif include_character:  # 如果是人物场景但没有single person约束，强制添加
                    simplified_parts.insert(1, "(single person:2.5)")
                
                priority_prompt = ", ".join(filter(None, simplified_parts))
                final_estimated = self.token_estimator.estimate(priority_prompt)
            
            # ⚡ 关键修复：如果仍然超过77，进行最终强制精简，确保不超过77
            if final_estimated > 77:
                print(f"  ⚠ 最终精简后仍超过限制 ({final_estimated} tokens)，进行最终强制精简...")
                # 最终精简：只保留最核心的内容，确保不超过77 tokens
                final_parts = []
                
                # 1. 必须保留：风格描述（最高优先级，放在最前面）
                style_found = False
                for part in simplified_parts if 'simplified_parts' in locals() else essential_parts:
                    if any(kw in part.lower() for kw in ["xianxia", "anime", "cinematic", "illustration"]) and not style_found:
                        # 精简为最短形式
                        final_parts.insert(0, "Chinese xianxia anime illustration, anime cinematic style")
                        style_found = True
                        continue
                
                # 如果没有找到风格，强制添加
                if not style_found:
                    final_parts.insert(0, "Chinese xianxia anime illustration, anime cinematic style")
                
                # 2. 必须保留：single person约束（第二优先级，放在风格之后）
                single_person_found = False
                for part in simplified_parts if 'simplified_parts' in locals() else essential_parts:
                    if "single person" in part.lower() and not single_person_found:
                        # 精简为最短形式，但保持高权重
                        final_parts.insert(1, "(single person:2.5)")  # 提高权重到2.5，放在风格之后
                        single_person_found = True
                        continue
                
                # 如果没有找到single person，强制添加
                if not single_person_found:
                    final_parts.insert(1, "(single person:2.5)")
                
                # 3. ⚡ 关键修复：优先保留关键动作（lying/sitting/standing），无论权重
                # 检查是否有lying/sitting/standing等关键动作
                has_lying_action = False
                lying_action_text = None
                for part in simplified_parts if 'simplified_parts' in locals() else essential_parts:
                    part_lower = part.lower()
                    import re
                    # 检查是否包含关键动作
                    if any(kw in part_lower for kw in ["lying", "lie", "躺", "horizontal position", "prone", "supine"]):
                        has_lying_action = True
                        # 提取lying相关的描述
                        if "lying on" in part_lower or "lie on" in part_lower:
                            # 提取完整的lying描述
                            lying_match = re.search(r'(lying\s+on\s+[^,\)]+|lie\s+on\s+[^,\)]+)', part_lower)
                            if lying_match:
                                lying_action_text = f"(lying on desert sand:2.5)"  # 使用高权重确保保留
                            else:
                                lying_action_text = f"(lying on desert sand:2.5)"
                        else:
                            # ⚡ 关键修复：即使没有"lying on"格式，只要有"lying"关键词，也要添加lying描述
                            # 检查环境描述，确定lying的位置
                            has_desert = any("desert" in p.lower() or "sand" in p.lower() for p in (simplified_parts if 'simplified_parts' in locals() else essential_parts))
                            if has_desert:
                                # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting
                                lying_action_text = "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)"
                            else:
                                # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting
                                lying_action_text = "(body fully on the ground, legs fully extended, arms lying flat, no bent knees, horizontal position:3.0)"
                        break
                
                # ⚡ 修复：如果有lying动作，必须添加到final_parts的最前面（最高优先级）
                if has_lying_action:
                    # 检查是否已经存在lying描述
                    has_lying_in_final = any("lying" in str(p).lower() or "lie" in str(p).lower() for p in final_parts)
                    if not has_lying_in_final:
                        # 插入到最前面，确保最高优先级
                        # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting
                        final_parts.insert(0, "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)")
                        print(f"  ✓ 强制在prompt最前面添加lying描述（权重3.0，最高优先级）")
                    else:
                        # 如果已存在，检查权重是否足够高
                        for i, part in enumerate(final_parts):
                            if "lying" in str(part).lower() or "lie" in str(part).lower():
                                # 提取权重
                                import re
                                weight_match = re.search(r':([\d.]+)\)', str(part))
                                if weight_match:
                                    weight = float(weight_match.group(1))
                                    if weight < 3.0:
                                        # 替换为高权重版本
                                        # ⚡ 关键修复：使用物理接触描述而不是 NOT sitting
                                        final_parts[i] = "(body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0)"
                                        print(f"  ✓ 提升lying描述权重到3.0（最高优先级）")
                                break
                
                # 3. 保留：高权重动作+环境组合（权重 >= 1.8，降低阈值以保留更多关键信息）
                for part in simplified_parts if 'simplified_parts' in locals() else essential_parts:
                    part_lower = part.lower()
                    import re
                    weight_match = re.search(r':([\d.]+)\)', part)
                    weight = float(weight_match.group(1)) if weight_match else 1.0
                    
                    # 跳过single person（已处理）
                    if "single person" in part_lower:
                        continue
                    
                    # 跳过lying（已处理）
                    if "lying" in part_lower or "lie" in part_lower:
                        continue
                    
                    # 保留高权重动作+环境组合（降低阈值到1.8）
                    key_action_patterns = [
                        r"sitting\s+on\s+(sand|desert|ground)",
                        r"standing\s+on\s+(sand|desert|ground)",
                    ]
                    has_key_action_env = any(re.search(pattern, part_lower) for pattern in key_action_patterns)
                    if has_key_action_env and weight >= 1.8:
                        # 精简为最短形式
                        if "(sitting on" not in str(final_parts) and "(standing on" not in str(final_parts):
                            if "sitting" in part_lower:
                                final_parts.append("(sitting on desert sand:2.0)")
                            elif "standing" in part_lower:
                                final_parts.append("(standing on desert sand:2.0)")
                        continue
                    
                    # 保留高权重环境描述（权重 >= 1.8，降低阈值）
                    if any(kw in part_lower for kw in ["desert", "sand"]) and weight >= 1.8:
                        # 精简为最短形式
                        if "desert" in part_lower and "(desert" not in str(final_parts) and "(gray-green desert" not in str(final_parts):
                            final_parts.append("(gray-green desert:2.0)")
                        continue
                
                # 4. ⚡ 关键修复：必须保留角色描述（包括服饰），确保不会生成光着上身的图像
                # 检查是否已有角色描述
                has_character_desc = any("cultivator" in str(p).lower() or "robe" in str(p).lower() for p in final_parts)
                if not has_character_desc:
                    # 必须添加角色描述（包括服饰），这是核心特征
                    final_parts.append("young male cultivator, (dark green simple cultivator robe:2.0), long black hair")
                else:
                    # 如果已有角色描述，确保包含服饰
                    for i, part in enumerate(final_parts):
                        if "cultivator" in part.lower() and "robe" not in part.lower():
                            # 在角色描述中添加服饰
                            final_parts[i] = part + ", (dark green simple cultivator robe:2.0)"
                            break
                
                priority_prompt = ", ".join(filter(None, final_parts))
                final_estimated = self.token_estimator.estimate(priority_prompt)
                
                # 如果仍然超过，只保留最核心的（但必须包含风格和服饰）
                if final_estimated > 77:
                    print(f"  ⚠ 最终精简后仍超过限制 ({final_estimated} tokens)，只保留最核心内容（必须包含风格和服饰）...")
                    # 最精简版本：必须包含风格、角色、服饰、动作、环境
                    # ⚡ 关键修复：增强风格描述和单人约束（wide + top_down + lying 场景没有 LoRA，需要更强的风格和约束）
                    # ⚡ 关键修复：使用物理接触描述而不是 "lying on desert sand"（SDXL 对物理接触描述更敏感）
                    priority_prompt = "Chinese xianxia anime illustration, 3D rendered anime, anime cinematic style, cinematic lighting, (single person:3.0), (only one person:3.0), young male cultivator, (dark green simple cultivator robe:2.0), (body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0), (gray-green desert:2.0)"
                    final_estimated = self.token_estimator.estimate(priority_prompt)
                    
                    # 如果仍然超过，进一步精简（但保留风格和服饰）
                    if final_estimated > 77:
                        print(f"  ⚠ 最精简版本仍超过限制 ({final_estimated} tokens)，使用极简版本（保留风格和服饰）...")
                        # ⚡ 关键修复：增强风格描述和单人约束
                        # ⚡ 关键修复：使用物理接触描述而不是 "lying on desert sand"
                        priority_prompt = "Chinese xianxia anime illustration, 3D rendered anime, anime cinematic style, (single person:3.0), (only one person:3.0), young male cultivator, (dark green robe:2.0), (body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0), (desert:2.0)"
                        final_estimated = self.token_estimator.estimate(priority_prompt)
                        
                        # 如果仍然超过，使用最极简版本（但必须包含风格和服饰）
                        if final_estimated > 77:
                            print(f"  ⚠ 极简版本仍超过限制 ({final_estimated} tokens)，使用最极简版本（保留风格和服饰）...")
                            # ⚡ 关键修复：增强风格描述和单人约束
                            # ⚡ 关键修复：使用物理接触描述而不是 "lying on desert sand"
                            priority_prompt = "Chinese xianxia anime illustration, 3D rendered anime, anime cinematic style, (single person:3.0), (only one person:3.0), (dark green robe:2.0), (body fully on the ground, back touching the sand, legs fully extended on the ground, arms lying flat on the sand, no bent knees, horizontal position:3.0), (desert:2.0)"
                            final_estimated = self.token_estimator.estimate(priority_prompt)
            
            # print(f"  ✓ 强制精简完成: {final_estimated} tokens（保留最关键信息）")  # 减少日志
        
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
    
    def _build_semantic_prompt_for_flux(self, scene: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """
        FLUX 专用语义化 prompt 构建（wide + top_down + lying 场景）
        
        使用自然语言句子而不是权重标记，FLUX 对语义理解更强
        ⚡ 关键修复：FLUX 使用 T5 tokenizer，支持 512+ tokens，不需要 77 token 限制
        ⚡ 关键修复：简化 prompt，让 IP-Adapter 的参考图发挥主要作用
        """
        character = scene.get("character", {}) or {}
        character_pose = character.get("pose", "")
        visual_constraints = scene.get("visual_constraints", {}) or {}
        environment = visual_constraints.get("environment", "")
        camera = scene.get("camera", {}) or {}
        
        # ⚡ 关键修复：简化 prompt，让 IP-Adapter 的参考图发挥主要作用
        # FLUX IP-Adapter 会从参考图中提取形象特征，prompt 只需要描述场景和姿态
        prompt_parts = []
        
        # 1. 姿态描述（使用物理接触描述，最重要）
        if character_pose in ["lying_motionless", "lying"]:
            prompt_parts.append("lies motionless on a vast desert")
            prompt_parts.append("body fully on the ground, back touching the sand, legs fully extended, arms lying flat, no bent knees, horizontal position")
        
        # 2. 环境描述
        if environment:
            prompt_parts.append(f"on {environment}")
        else:
            prompt_parts.append("on a vast gray-green desert")
        
        # 3. 镜头描述
        camera_shot = camera.get("shot", "wide")
        camera_angle = camera.get("angle", "top_down")
        if camera_shot == "wide" and camera_angle == "top_down":
            prompt_parts.append("Wide top-down cinematic shot")
        
        # 4. 风格描述（简化）
        prompt_parts.append("Chinese xianxia anime style")
        
        # 5. 单人约束（自然语言）
        prompt_parts.append("one person only, single character")
        
        # ⚡ 关键修复：不添加"形象一致性"提示，让 IP-Adapter 的参考图发挥主要作用
        # IP-Adapter 会自动从参考图中提取形象特征，prompt 只需要描述场景
        
        # 组合成完整的语义化 prompt
        semantic_prompt = ". ".join(prompt_parts) + "."
        
        print(f"  ✓ FLUX 语义化 prompt 构建完成（简化版，让 IP-Adapter 参考图发挥主要作用）")
        print(f"  📝 语义化 Prompt: {semantic_prompt}")
        print(f"  ℹ FLUX 使用 T5 tokenizer，支持 512+ tokens，当前 prompt 长度: {len(semantic_prompt.split())} words")
        
        return semantic_prompt
    
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
    
    def _load_character_template(self, template_name: str) -> Optional[str]:
        """加载角色Prompt模板文件（无风格词，纯人物描述）
        
        Args:
            template_name: 模板文件名（不含.prompt扩展名），如"HanLi"
            
        Returns:
            模板内容字符串，如果文件不存在则返回None
        """
        try:
            # 查找模板文件路径（相对于prompt模块目录）
            current_file = Path(__file__)
            template_dir = current_file.parent / "templates"
            template_path = template_dir / f"{template_name}.prompt"
            
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    print(f"  ✓ 加载角色模板文件: {template_path}")
                    return content
            else:
                print(f"  ⚠ 角色模板文件不存在: {template_path}")
                return None
        except Exception as e:
            print(f"  ⚠ 加载角色模板失败: {e}")
            return None
    
    def _get_character_profile(self, character_id: str = "hanli") -> Dict[str, Any]:
        """获取角色模板"""
        # 处理科普主持人的映射
        if character_id in ["kepu_gege", "科普哥哥"]:
            character_id = "kepu_gege"
        elif character_id in ["weilai_jiejie", "未来姐姐"]:
            character_id = "weilai_jiejie"
        
        return self.character_profiles.get(character_id, {})
    
    def _get_scene_profile(
        self,
        scene_name: str = None,
        episode: int = None,
        profile_key: str = None,
        is_kepu_video: bool = False,
    ) -> Dict[str, Any]:
        """根据场景 key、名称或集数获取场景模板
        
        Args:
            scene_name: 场景名称
            episode: 集数
            profile_key: 场景模板 key
            is_kepu_video: 是否为科普视频（科普视频不使用场景模板）
        """
        # 科普视频不使用场景模板，直接返回空字典
        if is_kepu_video:
            return {}
        
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
    
    def _convert_camera_v2_to_string(self, camera_dict: Dict[str, Any]) -> str:
        """将 v2 格式的 camera 字典转换为字符串描述"""
        if not camera_dict or not isinstance(camera_dict, dict):
            return ""
        
        parts = []
        
        # shot 字段映射
        shot_map = {
            "wide": "远景",
            "medium": "中景",
            "close_up": "特写",
            "closeup": "特写",
            "extreme_close": "极近特写",
            "full_body": "全身",
            "long": "长镜头",
        }
        shot = camera_dict.get("shot", "")
        if shot:
            shot_str = shot_map.get(shot.lower(), shot)
            parts.append(shot_str)
        
        # angle 字段映射
        angle_map = {
            "eye_level": "平视",
            "top_down": "俯拍",
            "bird_eye": "鸟瞰",
            "low_angle": "仰拍",
            "worm_eye": "极低角度",
            "side": "侧拍",
            "front": "正面",
            "back": "背后",
        }
        angle = camera_dict.get("angle", "")
        if angle:
            angle_str = angle_map.get(angle.lower(), angle)
            parts.append(angle_str)
        
        # movement 字段映射
        # ⚡ 关键修复：单帧生成时，去掉视频语义（pan/tilt/push_in/pull_out），改为 static
        # 原因：SDXL 会当成"人物动态姿态"，导致姿态错误
        movement_map = {
            "static": "静止",
            "pan": "静止",  # 单帧生成时，pan 改为静止
            "tilt": "静止",  # 单帧生成时，tilt 改为静止
            "push_in": "静止",  # 单帧生成时，push_in 改为静止
            "pull_out": "静止",  # 单帧生成时，pull_out 改为静止
            "orbit": "静止",  # 单帧生成时，orbit 改为静止
            "follow": "静止",  # 单帧生成时，follow 改为静止
            "shake": "静止",  # 单帧生成时，shake 改为静止
        }
        movement = camera_dict.get("movement", "")
        if movement:
            movement_str = movement_map.get(movement.lower(), "静止")  # 默认改为静止
            if movement_str == "静止":
                # 只在非静止时才添加，避免重复
                if "静止" not in " ".join(parts):
                    parts.append(movement_str)
            else:
                parts.append(movement_str)
        
        return " ".join(parts) if parts else ""
    
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
            # ⚡ 关键修复：单帧生成时，去掉视频语义（pan/lateral），改为 static
            # 原因：SDXL 会当成"人物动态姿态"，导致姿态错误
            if use_chinese:
                camera_keywords.append("静止镜头")  # 改为静止
            else:
                camera_keywords.append("static shot, still frame")  # 改为静止
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
                # ⚡ 修复性别错误：提高权重到2.5，确保性别正确
                if use_chinese:
                    parts.append("(男性，男，男人:2.5)")
                else:
                    parts.append("(male, man, masculine:2.5)")
            elif "female" in identity_lower or "女" in identity:
                if use_chinese:
                    parts.append("(女性，女:1.8)")
                else:
                    parts.append("(female, woman:1.8)")
        else:
            # 向后兼容：对于韩立，默认是男性
            # ⚡ 修复性别错误：提高权重到2.5，确保性别正确
            if character_id == "hanli":
                if use_chinese:
                    parts.append("(男性，男，男人:2.5)")
                else:
                    parts.append("(male, man, masculine:2.5)")
        
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
        
        # 0. 身份和性别（精简版，避免与single person重复）
        # 注意：如果已经有single person约束，就不需要再强调male/man（避免重复）
        # 对于科普主持人，性别信息已经在角色名称中体现，不需要额外添加
        identity = profile.get("identity", "")
        character_id = profile.get("character_id", "").lower() or profile.get("id", "").lower()
        character_name = str(profile.get("character_name", "")).lower()
        
        # 对于科普主持人，不添加性别标记（角色名称已体现）
        if "kepu" in character_id or "weilai" in character_id or "科普" in character_name or "未来" in character_name:
            # 科普主持人不需要额外性别标记
            pass
        elif identity:
            # 其他角色：只使用一个词，避免重复
            identity_lower = identity.lower()
            if "male" in identity_lower or "男" in identity:
                # ⚡ 修复性别错误：提高权重到2.5，确保性别正确
                if use_chinese:
                    parts.append("(男性，男，男人:2.5)")
                else:
                    parts.append("(male, man, masculine:2.5)")
            elif "female" in identity_lower or "女" in identity:
                if use_chinese:
                    parts.append("(女性:1.5)")
                else:
                    parts.append("(female:1.5)")
        
        # 1. 角色名称（必须包含，确保角色识别）
        character_name = profile.get("character_name", "")
        if character_name:
            parts.append(character_name)
        
        # 2. 发型描述（精简版，只保留核心描述）
        hair = profile.get("hair", {})
        if hair.get("prompt_keywords"):
            # 大幅简化：只提取第一个核心描述
            import re
            hair_keywords = hair["prompt_keywords"]
            matches = re.findall(r'\(([^)]+)\)', hair_keywords)
            if matches:
                # 只使用第一个描述，简化权重
                core_desc = matches[0].split(':')[0].strip()
                parts.append(f"({core_desc}:1.5)")
            else:
                # 如果没有括号，简化权重
                hair_keywords = re.sub(r':\d+\.\d+', ':1.5', hair_keywords)
                parts.append(hair_keywords)
        elif hair.get("style"):
            # 只使用style，不添加color（减少token）
            parts.append(f"({hair.get('style')}:1.5)")
        
        # 3. 服饰描述（精简版，只保留核心描述）
        clothes = profile.get("clothes", {})
        if clothes.get("prompt_keywords"):
            # 大幅简化：只提取第一个核心描述
            import re
            clothes_keywords = clothes["prompt_keywords"]
            matches = re.findall(r'\(([^)]+)\)', clothes_keywords)
            if matches:
                # 只使用第一个描述（最重要的），进一步精简：只保留前3个关键词
                core_desc = matches[0].split(':')[0].strip()
                core_words = core_desc.split(',')[:3]
                parts.append(f"({', '.join(core_words)}:1.6)")
            else:
                # 如果没有括号，简化权重，只保留前50个字符
                clothes_keywords = re.sub(r':\d+\.\d+', ':1.6', clothes_keywords)
                if len(clothes_keywords) > 50:
                    clothes_keywords = clothes_keywords[:50] + "..."
                parts.append(clothes_keywords)
        elif clothes.get("style"):
            # 只使用style，不添加color（减少token）
            parts.append(f"({clothes.get('style')}:1.6)")
        
        # 4. 面部特征（精简版，只保留前2个关键词）
        if profile.get("face_keywords"):
            face_keywords = profile["face_keywords"]
            # 大幅简化：只保留前2个关键词
            face_parts = [p.strip() for p in face_keywords.split(",")][:2]
            if face_parts:
                parts.append(f"({', '.join(face_parts)}:1.3)")
        
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
        # v2 兼容：优先 scene_id，其次 id
        scene_id = scene.get("scene_id", scene.get("id"))
        is_opening_ending = scene_id in [0, 999]
        
        if is_opening_ending:
            if not self.ascii_only_prompt:
                return "(仙域天空，灵气缭绕:1.3)"
            else:
                return "(immortal realm sky, spiritual mist:1.3)"
        
        # 首先检查场景描述中的实际颜色和地形
        # ⚡ v2 格式支持：优先使用 visual_constraints，如果没有则使用 visual
        visual_constraints = scene.get("visual_constraints", {}) or {}
        visual = scene.get("visual", {}) or {}
        
        # 优先从 visual_constraints 读取（v2 格式）
        environment = self._clean_prompt_text(visual_constraints.get("environment", "") or "")
        if not environment:
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
        
        # 检查是否为科普视频（科普视频不使用场景模板）
        is_kepu_video = False
        if script_data:
            category = script_data.get("category", "")
            if category in ["universe", "quantum", "earth", "energy", "city", "biology", "ai"]:
                is_kepu_video = True
        
        profile = self._get_scene_profile(scene_name, episode, profile_key=profile_key, is_kepu_video=is_kepu_video)
        
        # 如果是科普视频，不使用场景模板，直接返回空字符串
        if is_kepu_video:
            return ""
        
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
        # 检查是否为科普视频（科普视频不使用场景模板）
        is_kepu_video = False
        if script_data:
            category = script_data.get("category", "")
            if category in ["universe", "quantum", "earth", "energy", "city", "biology", "ai"]:
                is_kepu_video = True
        
        # 如果是科普视频，直接返回空字符串
        if is_kepu_video:
            return ""
        
        profile_key = scene.get("scene_profile") or scene.get("scene_template") or scene.get("scene_key")
        scene_name = scene.get("scene_name") or scene.get("title", "")
        if not scene_name and script_data:
            scene_name = script_data.get("title", "")
        
        episode = scene.get("episode")
        if not episode and script_data:
            episode = script_data.get("episode")
        
        profile = self._get_scene_profile(scene_name, episode, profile_key=profile_key, is_kepu_video=is_kepu_video)
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

