#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说推文视频生成脚本
使用 Flux 生成图片，然后用 HunyuanVideo 生成视频
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import yaml
import json
import re

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from image_generator import ImageGenerator
from video_generator import VideoGenerator
from PIL import Image


class NovelVideoGenerator:
    """小说推文视频生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化生成器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (project_root / self.config_path).resolve()
        
        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化图像生成器（使用 Flux）
        print("=" * 60)
        print("初始化图像生成器（Flux）...")
        self.image_generator = ImageGenerator(str(self.config_path))
        
        # 初始化视频生成器（使用 HunyuanVideo）
        print("初始化视频生成器（HunyuanVideo）...")
        self.video_generator = VideoGenerator(str(self.config_path))

        # M6 增强视频生成器（懒加载：仅在启用身份验证时初始化）
        self._m6_video_generator = None
        
        # 确保使用正确的模型
        self._ensure_model_config()
        
        print("=" * 60)
        print("✅ 初始化完成")
        print("=" * 60)

    @staticmethod
    def _infer_character_id_from_text(text: str) -> Optional[str]:
        """
        从文本中推断角色ID（当前只做“韩立”识别，避免误伤纯场景推文）。
        - 命中 “韩立”/“Han Li”/“hanli” 等 → hanli
        """
        if not text:
            return None
        t = str(text)
        if "韩立" in t:
            return "hanli"
        tl = t.lower()
        if "hanli" in tl:
            return "hanli"
        if re.search(r"\bhan\s*li\b", tl):
            return "hanli"
        return None

    def _resolve_character_and_m6(
        self,
        prompt: str,
        scene: Optional[Dict[str, Any]],
        include_character: Optional[bool],
        character_id: Optional[str],
        enable_m6_identity: Optional[bool],
        auto_character: bool,
        auto_m6_identity: bool,
        force_scene: bool,
    ) -> Tuple[bool, Optional[str], bool]:
        """
        统一决策：
        - 是否包含韩立（以及角色ID）
        - 是否启用 M6 身份验证
        """
        if force_scene:
            return False, None, False

        inferred_id = self._infer_character_id_from_text(prompt) if auto_character else None

        # scene 中显式 character.id 优先
        if scene and isinstance(scene, dict):
            c = scene.get("character")
            if isinstance(c, dict):
                cid = c.get("id")
                if cid:
                    inferred_id = str(cid)

        effective_character_id = str(character_id) if character_id else inferred_id

        if include_character is None:
            effective_include_character = bool(effective_character_id)
        else:
            effective_include_character = bool(include_character)

        # M6 开关：显式优先；否则自动（仅对韩立场景打开）
        if enable_m6_identity is None:
            effective_enable_m6 = bool(auto_m6_identity and effective_include_character and effective_character_id == "hanli")
        else:
            effective_enable_m6 = bool(enable_m6_identity and effective_include_character)

        return effective_include_character, effective_character_id, effective_enable_m6
    
    def _ensure_model_config(self):
        """确保配置使用 Flux + HunyuanVideo"""
        # 修改配置，确保使用 Flux 生成图像
        image_config = self.config.get('image', {})
        if image_config.get('engine') != 'flux-instantid':
            print("  ⚠ 警告: image.engine 不是 flux-instantid，建议修改配置")
        
        # 修改配置，确保使用 HunyuanVideo 生成视频
        video_config = self.config.get('video', {})
        if video_config.get('model_type') != 'hunyuanvideo':
            print("  ⚠ 警告: video.model_type 不是 hunyuanvideo，建议修改配置")
            print("  ℹ 临时修改配置为 hunyuanvideo")
            video_config['model_type'] = 'hunyuanvideo'
            self.video_generator.video_config['model_type'] = 'hunyuanvideo'
    
    def generate(
        self,
        prompt: str = None,
        output_dir: Optional[Path] = None,
        image_output_path: Optional[Path] = None,
        video_output_path: Optional[Path] = None,
        width: int = 1280,
        height: int = 768,
        num_frames: int = 120,
        fps: int = 24,
        scene: Optional[Dict[str, Any]] = None,
        use_v21_exec: bool = False,  # v2.1-exec模式开关
        # === 角色一致（图片端）===
        include_character: Optional[bool] = None,
        character_id: Optional[str] = None,
        auto_character: bool = True,
        force_scene: bool = False,
        image_model_engine: Optional[str] = None,
        # === 视频一致（M6 身份验证+重试）===
        enable_m6_identity: Optional[bool] = None,
        auto_m6_identity: bool = True,
        reference_image_path: Optional[str] = None,
        shot_type: str = "medium",
        motion_intensity: str = "moderate",
        m6_max_retries: Optional[int] = None,
        m6_quick: bool = False,
    ) -> Dict[str, Path]:
        """
        生成小说推文视频
        
        Args:
            prompt: 文本提示词（小说场景描述）
            output_dir: 输出目录
            image_output_path: 图像输出路径（可选）
            video_output_path: 视频输出路径（可选）
            width: 图像宽度
            height: 图像高度
            num_frames: 视频帧数
            fps: 视频帧率
            scene: 场景配置（可选）
            include_character: 是否生成带角色的画面（启用后会使用现有“角色一致”系统）
            character_id: 角色ID（默认 hanli）
            auto_character: 是否自动从 prompt/scene 推断是否包含韩立（默认 True，仅识别韩立，避免误伤纯场景）
            force_scene: 强制按纯场景生成（忽略自动推断/手动角色）
            image_model_engine: 覆盖图片引擎（例如 auto / flux-instantid / pulid / flux1 等；不传则按模式选择默认）
            enable_m6_identity: 是否启用 M6 身份验证 + 重试（仅在 include_character=True 时强烈建议开启）
            auto_m6_identity: 是否自动对“韩立场景”启用 M6（默认 True）
            reference_image_path: 身份验证参考图（不传则自动按 character_id 选择，找不到则用生成图）
            shot_type: 镜头类型（影响阈值容忍度）
            motion_intensity: 运动强度（会传入 scene，供生成/重试策略参考）
            m6_max_retries: 覆盖最大重试次数（None=用 config.yaml）
            m6_quick: 快速模式（更少步数/更少重试，适合冒烟）
        
        Returns:
            dict: 包含 'image' 和 'video' 路径的字典
        """
        print("=" * 60)
        print("开始生成小说推文视频")
        print("=" * 60)
        
        # ⚡ v2.1-exec模式：如果scene是v2.1-exec格式，使用Execution Executor
        if use_v21_exec and scene and scene.get("version", "").startswith("v2.1"):
            return self._generate_v21_exec(scene, output_dir, width, height, num_frames, fps)
        
        # 兼容模式：如果scene是v2格式，自动转换为v2.1-exec（可选）
        if scene and scene.get("version") == "v2" and use_v21_exec:
            print("  ℹ 检测到v2格式，自动转换为v2.1-exec")
            from utils.json_v2_to_v21_converter import JSONV2ToV21Converter
            converter = JSONV2ToV21Converter()
            scene = converter.convert_scene(scene)
            return self._generate_v21_exec(scene, output_dir, width, height, num_frames, fps)
        
        # 原有流程（兼容）
        if prompt is None:
            prompt = scene.get("prompt", {}).get("positive_core", "") if scene else ""
        
        print(f"提示词: {prompt}")
        print()

        # ⚡ 关键修复：根据配音时长（duration_sec）计算帧数
        # 如果 scene 中有 duration_sec，优先使用它来计算帧数
        if scene and isinstance(scene, dict):
            duration_sec = scene.get('duration_sec')
            if duration_sec:
                # 根据配音时长计算帧数：帧数 = 时长(秒) × 帧率
                calculated_frames = int(duration_sec * fps)
                if calculated_frames != num_frames:
                    print(f"  ℹ 根据配音时长调整帧数: {duration_sec}秒 × {fps}fps = {calculated_frames}帧 (原值: {num_frames}帧)")
                    num_frames = calculated_frames
                else:
                    print(f"  ℹ 帧数已匹配配音时长: {num_frames}帧 = {duration_sec}秒 × {fps}fps")

        # 自动推断是否有韩立，并据此决定是否启用 M6
        effective_include_character, effective_character_id, effective_enable_m6 = self._resolve_character_and_m6(
            prompt=prompt,
            scene=scene,
            include_character=include_character,
            character_id=character_id,
            enable_m6_identity=enable_m6_identity,
            auto_character=auto_character,
            auto_m6_identity=auto_m6_identity,
            force_scene=force_scene,
        )
        if effective_character_id:
            print(f"  ℹ 角色推断: character_id={effective_character_id}, include_character={effective_include_character}")
        else:
            print(f"  ℹ 角色推断: 无韩立（按纯场景生成）")
        print(f"  ℹ M6 身份验证: {'启用' if effective_enable_m6 else '关闭'}")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = project_root / "outputs" / "novel_videos"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 步骤1: 使用 Flux 生成图像
        print("=" * 60)
        print("步骤1: 使用 Flux 生成图像")
        print("=" * 60)
        
        if image_output_path is None:
            image_output_path = output_dir / "novel_image.png"
        
        try:
            # 构建scene字典（包含width和height）
            image_scene = scene.copy() if scene else {}
            image_scene['width'] = width
            image_scene['height'] = height
            
            # 角色模式：让 ImageGenerator/EnhancedImageGenerator 接管“角色一致”
            if effective_include_character:
                # 给下游一个明确的角色信号（ImageGenerator 内部会识别 character.id）
                image_scene.setdefault("character", {})
                if isinstance(image_scene.get("character"), dict):
                    if effective_character_id:
                        image_scene["character"].setdefault("id", effective_character_id)
                # 运动强度也写入（Prompt Engine / 生成器可按需使用）
                image_scene.setdefault("motion_intensity", motion_intensity)
            
            # 生成图像：
            # - 默认（include_character=False）：走“场景图”逻辑（无人物）
            # - 角色模式（include_character=True）：走“角色一致”逻辑（人物+场景）
            print(f"  [DEBUG] 原始prompt: {prompt}")
            print(f"  [DEBUG] scene: {image_scene}")
            if effective_include_character:
                print(f"  [DEBUG] character_id: {effective_character_id}")
            
            # 使用 Prompt Engine V2 优化提示词（完全本地模式，无需LLM API）
            print(f"  🔧 开始优化提示词（使用 Prompt Engine V2 本地模式）...")
            original_prompt = prompt
            negative_prompt = None
            optimized_prompt = None
            
            try:
                from utils.prompt_engine_v2 import PromptEngine, UserRequest
                
                # 创建 Prompt Engine V2（默认本地模式，无需LLM API）
                prompt_engine_v2 = PromptEngine()
                
                # 创建用户请求（图像生成阶段）
                req = UserRequest(
                    text=original_prompt,
                    scene_type="novel",  # 小说推文场景
                    style="novel",  # 使用novel风格模板
                    target_model="flux",  # 图像生成使用Flux
                    params={"width": width, "height": height}
                )
                
                # 执行处理
                pkg = prompt_engine_v2.run(req)
                
                # 获取优化后的prompt和negative prompt
                optimized_prompt = pkg.final_prompt
                negative_prompt = pkg.negative
                
                # ⚡ 工程级优化：移除 HF token 统计（LLM 已返回正确数量，且可能阻塞）
                # 如果确实需要 token 统计，可以使用 LLM 返回的信息或简单估算
                # 不再使用 T5Tokenizer（可能阻塞或加载慢）
                
                # 添加场景强化关键词（确保是场景而非人物）
                # 角色模式下不要加 no people
                scene_enhancers = "landscape, nature" if effective_include_character else "landscape, nature, no people"
                optimized_prompt = f"{optimized_prompt}, {scene_enhancers}"
                
                # 增强负面提示词（确保排除人物）
                if not effective_include_character:
                    additional_negatives = [
                        "faces, portraits, black faces, dark faces, human faces, person faces, character faces",
                        "people in image, humans in scene, any people, any persons, any characters, any human figures",
                    ]
                    negative_prompt = f"{negative_prompt}, {', '.join(additional_negatives)}"
                
                # ⚡ 关键修复：如果包含角色，强制添加角色描述（特别是服饰描述和性别），确保不被优化掉
                if effective_include_character and effective_character_id:
                    try:
                        # ⚡ 修复：不要在这里重新导入 Path，使用文件顶部已导入的 Path
                        # from pathlib import Path  # 删除这行，因为文件顶部已经导入了
                        # 读取角色档案
                        profile_path = Path(__file__).parent / "character_profiles.yaml"
                        if profile_path.exists():
                            with open(profile_path, 'r', encoding='utf-8') as f:
                                profiles = yaml.safe_load(f) or {}
                            character_profile = profiles.get("characters", {}).get(effective_character_id, {})
                            
                            if character_profile:
                                # 构建角色描述（特别是服饰描述和性别）
                                character_parts = []
                                
                                # ⚡ 关键修复：添加性别描述（从 identity 字段提取）
                                # ⚡ 注意：Flux 使用 T5 编码器，不支持权重语法 (xxx:1.5)
                                # 使用自然语言描述，通过重复和位置来强调重要性
                                identity = character_profile.get("identity", "")
                                if identity:
                                    # 提取性别（Male/Female）
                                    identity_lower = identity.lower()
                                    if "male" in identity_lower:
                                        character_parts.append("Male, male character, male person")  # 通过重复强调
                                    elif "female" in identity_lower:
                                        character_parts.append("Female, female character, female person")
                
                                # 服饰描述（最高优先级，确保不被优化掉）
                                # ⚡ 注意：character_profiles.yaml 中的 prompt_keywords 可能包含权重语法
                                # 需要移除权重语法，只保留描述内容
                                clothes = character_profile.get("clothes", {})
                                clothes_keywords = clothes.get("prompt_keywords", "")
                                if clothes_keywords:
                                    # 移除权重语法 (xxx:1.5)，只保留描述内容
                                    import re
                                    clothes_clean = re.sub(r'\(([^:]+):[\d.]+\)', r'\1', clothes_keywords)
                                    clothes_clean = re.sub(r'\(([^)]+)\)', r'\1', clothes_clean)  # 移除普通括号
                                    character_parts.append(clothes_clean)
                                
                                # 发型描述
                                hair = character_profile.get("hair", {})
                                hair_keywords = hair.get("prompt_keywords", "")
                                if hair_keywords:
                                    # 移除权重语法
                                    import re
                                    hair_clean = re.sub(r'\(([^:]+):[\d.]+\)', r'\1', hair_keywords)
                                    hair_clean = re.sub(r'\(([^)]+)\)', r'\1', hair_clean)
                                    character_parts.append(hair_clean)
                                
                                # 如果构建了角色描述，添加到 prompt 最前面（最高优先级）
                                if character_parts:
                                    # ⚡ 关键修复：使用去重工具，避免角色描述与 prompt 重复
                                    try:
                                        from utils.prompt_deduplicator import filter_duplicates, merge_prompt_parts
                                        
                                        # 检查角色描述是否与 prompt 重复
                                        filtered_character_parts = filter_duplicates(
                                            new_descriptions=character_parts,
                                            existing_texts=[optimized_prompt],
                                            threshold=0.5  # 50% 重叠认为是重复（角色描述更严格）
                                        )
                                        
                                        if filtered_character_parts:
                                            # 合并角色描述和 prompt
                                            all_parts = filtered_character_parts + [optimized_prompt]
                                            optimized_prompt = merge_prompt_parts(all_parts)
                                            print(f"  ✓ 已添加角色描述（性别+服饰+发型）到 prompt 最前面，已去重")
                                        else:
                                            print(f"  ℹ 角色描述与 prompt 重复，已跳过")
                                    except ImportError:
                                        # 如果去重工具不可用，直接合并
                                        character_desc = ", ".join(character_parts)
                                        optimized_prompt = f"{character_desc}, {optimized_prompt}"
                                        print(f"  ✓ 已强制添加角色描述（性别+服饰+发型）到 prompt 最前面，确保不被优化掉")
                    except Exception as e:
                        print(f"  ⚠ 添加角色描述时出错: {e}")
                
                # ⚡ 关键修复：场景增强描述由 ExecutionPlannerV3 的场景分析器统一处理
                # 这里不再重复添加，避免 prompt 重复
                # ExecutionPlannerV3 会使用场景分析器进行更智能的分析，并自动添加增强描述
                
                print(f"  ✓ Prompt Engine V2 处理完成")
                print(f"  ℹ 原始提示词: {original_prompt[:80]}...")
                print(f"  ℹ 优化后提示词: {optimized_prompt[:100]}...")
                print(f"  ℹ QA评分: {pkg.metadata.get('qa_score', 0)}/{pkg.metadata.get('qa_max_score', 0)}")
                
            except Exception as e:
                print(f"  ⚠ Prompt Engine V2 处理失败: {e}，使用备用方案")
                import traceback
                traceback.print_exc()
                
                # 备用方案：使用原始提示词+场景强化
                # ⚡ 修复：Flux 使用 T5，支持 512 tokens，不需要 77 token 限制
                scene_enhancers = "landscape, nature" if effective_include_character else "landscape, nature, no people"
                optimized_prompt = f"{original_prompt}, {scene_enhancers}"
                
                # ⚡ 工程级优化：移除 HF token 统计（LLM 已返回正确数量，且可能阻塞）
                # 不再使用 T5Tokenizer（可能阻塞或加载慢）
                # 如果需要 token 统计，可以使用简单估算或 LLM 返回的信息
                if effective_include_character:
                    # ⚡ 关键修复：增强负面提示词，特别是排除"站立"、"直立"等姿态
                    # 检查场景分析结果，如果是"lying"动作，添加更强的负面提示
                    # ⚡ 关键修复：排除耳坠、饰品等不需要的装饰
                    negative_prompt = "low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark, flickering, jittery, unstable, abrupt changes, worst quality, unrealistic details, earrings, earring, jewelry, accessories, decorative ornaments, decorative items, unnecessary decorations"
                    
                    # ⚡ 关键优化：优先使用 LLM 返回的姿态负面提示词（如果可用）
                    # 如果 LLM 已经返回了精确的姿态负面提示词，直接使用，不需要再次调用 PostureController
                    try:
                        # 尝试从场景分析结果中获取姿态负面提示词
                        from utils.scene_analyzer import analyze_scene
                        prompt_engine_config = self.config.get("prompt_engine", {})
                        use_llm = prompt_engine_config.get("scene_analyzer_mode", "local") in ["llm", "hybrid"]
                        
                        if use_llm:
                            llm_client = None
                            try:
                                llm_api_config = prompt_engine_config.get("llm_api", {})
                                if llm_api_config.get("api_key"):
                                    from utils.scene_analyzer import OpenAILLMClient
                                    llm_client = OpenAILLMClient(
                                        api_key=llm_api_config.get("api_key"),
                                        model=llm_api_config.get("model", "gpt-4o-mini"),
                                        base_url=llm_api_config.get("base_url")
                                    )
                            except Exception as e:
                                print(f"  ⚠ LLM 客户端创建失败: {e}，使用本地模式")
                                use_llm = False
                            
                            analysis_result = analyze_scene(
                                prompt=original_prompt,
                                current_shot_type=scene.get('shot_type', 'medium') if scene else 'medium',
                                use_llm=use_llm,
                                llm_client=llm_client
                            )
                            
                            if analysis_result and analysis_result.posture_negative:
                                # LLM 已经返回了精确的姿态负面提示词，直接使用
                                negative_prompt = f"{analysis_result.posture_negative}, {negative_prompt}"
                                print(f"  ✓ LLM 已返回姿态负面提示词: {analysis_result.posture_type}")
                        else:
                            # 使用 PostureController 作为回退
                            from utils.posture_controller import PostureController
                            posture_controller = PostureController()
                            
                            # 检测姿态
                            posture = posture_controller.detect_posture(original_prompt)
                            if posture:
                                posture_prompt = posture_controller.get_posture_prompt(posture, use_chinese=False)
                                if posture_prompt["negative"]:
                                    # 添加姿态相关的负面提示词
                                    negative_prompt = f"{posture_prompt['negative']}, {negative_prompt}"
                                    print(f"  ✓ PostureController 检测到姿态: {posture}，已添加姿态负面提示词")
                    except ImportError:
                        # 回退到原有逻辑
                        try:
                            from utils.scene_analyzer import analyze_scene
                            # ⚡ 关键修复：读取配置，决定是否使用 LLM
                            prompt_engine_config = self.config.get("prompt_engine", {})
                            use_llm = prompt_engine_config.get("scene_analyzer_mode", "local") in ["llm", "hybrid"]
                            
                            # 如果使用 LLM，需要创建 LLM 客户端
                            llm_client = None
                            if use_llm:
                                try:
                                    llm_api_config = prompt_engine_config.get("llm_api", {})
                                    if llm_api_config.get("api_key"):
                                        from utils.scene_analyzer import OpenAILLMClient
                                        llm_client = OpenAILLMClient(
                                            api_key=llm_api_config.get("api_key"),
                                            model=llm_api_config.get("model", "gpt-4o-mini"),
                                            base_url=llm_api_config.get("base_url")
                                        )
                                except Exception as e:
                                    print(f"  ⚠ LLM 客户端创建失败: {e}，使用本地模式")
                                    use_llm = False
                            
                            analysis_result = analyze_scene(
                                prompt=original_prompt,
                                current_shot_type=scene.get('shot_type', 'medium') if scene else 'medium',
                                use_llm=use_llm,
                                llm_client=llm_client
                            )
                            if analysis_result and analysis_result.action_type == "lying":
                                # ⚡ 关键修复：增强负面提示词，强烈排除站立和直立姿态
                                # 注意：Flux 对负面描述不够敏感，主要依赖正面描述（在 prompt 中强调"躺下"）
                                # 负面提示词只作为辅助，不要添加太多"不要xx"，避免 prompt 过长
                                negative_prompt = "standing, upright, vertical position, person standing, person upright, standing pose, upright pose, vertical pose, " + negative_prompt
                                print(f"  ✓ 检测到'lying'动作，已增强负面提示词（排除站立）")
                        except Exception as e:
                            # 如果场景分析失败，忽略
                            print(f"  ⚠ 场景分析失败: {e}")
                            pass
                else:
                    # 非角色模式：使用纯场景的负面提示词
                    # ⚡ 关键修复：不要排除 anime/cartoon 风格（因为 quality_target.style 可能是 xianxia_anime）
                    # 只排除人物，保留风格灵活性
                    negative_prompt = "characters, people, persons, human figures, faces, portraits, black faces, dark faces, human faces, person faces, character faces, people in image, humans in scene, any people, any persons, any characters, any human figures, low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark, worst quality, distorted proportions, unrealistic details"
            
            print(f"  ✅ 提示词优化完成:")
            print(f"     原始: {original_prompt}")
            print(f"     优化后: {optimized_prompt[:150]}...")
            print(f"     负面提示词: {negative_prompt[:150]}...")
            
            prompt = optimized_prompt
            negative_prompt = negative_prompt
            
            # 非角色模式：确保 scene 中不包含角色信息，避免误识别为人物生成
            if (not effective_include_character) and image_scene:
                image_scene.pop('character', None)
                image_scene.pop('characters', None)
                image_scene.pop('primary_character', None)
                image_scene.pop('face_reference_image_path', None)
                image_scene.pop('reference_image_path', None)
                print(f"  [DEBUG] 已清理scene中的角色相关字段，确保生成场景图像")

            # 选择图片引擎/任务类型
            if image_model_engine is None:
                # 默认策略：场景=flux1；角色=auto（走你现有"角色一致"路由）
                image_model_engine = "auto" if effective_include_character else "flux1"
            image_task_type = "character" if effective_include_character else "scene"
            
            # ⚡ 关键修复：为角色模式查找并传递参考图路径
            face_ref_path = None
            if effective_include_character and effective_character_id:
                # 优先级 1：用户显式指定的参考图
                if reference_image_path:
                    ref_p = Path(reference_image_path)
                    if not ref_p.is_absolute():
                        ref_p = (project_root / ref_p).resolve()
                    if ref_p.exists():
                        face_ref_path = ref_p
                        print(f"  ✓ 使用用户指定的参考图: {face_ref_path.name}")
                
                # 优先级 2：自动查找 reference_image/{character_id}_mid.jpg
                if face_ref_path is None:
                    candidate = (project_root / "reference_image" / f"{effective_character_id}_mid.jpg").resolve()
                    if candidate.exists():
                        face_ref_path = candidate
                        print(f"  ✓ 自动找到参考图: {face_ref_path.name}")
                    else:
                        # 尝试 .png
                        candidate = (project_root / "reference_image" / f"{effective_character_id}_mid.png").resolve()
                        if candidate.exists():
                            face_ref_path = candidate
                            print(f"  ✓ 自动找到参考图: {face_ref_path.name}")
                
                # 优先级 3：使用 ImageGenerator 的自动查找逻辑（通过 scene 中的 character_id）
                # 这会在 image_generator.generate_image 内部调用 _select_face_reference_image
                if face_ref_path is None:
                    print(f"  ⚠ 未找到显式参考图，将使用 ImageGenerator 的自动查找逻辑")
            
            image_path = self.image_generator.generate_image(
                prompt=prompt,
                output_path=image_output_path,
                scene=image_scene,
                model_engine=image_model_engine,
                task_type=image_task_type,
                negative_prompt=negative_prompt,  # 使用优化后的负面提示词
                face_reference_image_path=face_ref_path,  # ⚡ 关键修复：传递参考图路径
            )
            print(f"✅ 图像生成成功: {image_path}")
            
            # 读取生成图像的实际分辨率，确保视频使用相同的分辨率
            from PIL import Image as PILImage
            generated_image = PILImage.open(image_path)
            actual_image_width, actual_image_height = generated_image.size
            image_aspect_ratio = actual_image_width / actual_image_height
            print(f"  ℹ 生成图像实际分辨率: {actual_image_width}x{actual_image_height} (宽高比: {image_aspect_ratio:.2f})")
            
            # 更新width和height为图像的实际分辨率
            width = actual_image_width
            height = actual_image_height
        except Exception as e:
            print(f"❌ 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 清理图像生成器的模型和显存，为视频生成释放显存
        print()
        print("=" * 60)
        print("清理图像生成器模型，释放显存")
        print("=" * 60)
        try:
            import torch
            import gc
            
            # 记录清理前的显存状态
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
                print(f"  ℹ 清理前显存: 已分配={allocated_before:.2f}GB, 已保留={reserved_before:.2f}GB")
            
            # 清理所有可能的pipeline引用（先调用unload，再删除引用）
            pipelines_to_clean = [
                'pipeline',
                'flux_pipeline',
                'flux1_pipeline',  # Flux.1 pipeline
                'flux2_pipeline',  # Flux.2 pipeline
                'sdxl_pipeline',
                'instantid_pipeline',
                'kolors_pipeline',
                'hunyuan_dit_pipeline',
            ]
            
            for pipeline_name in pipelines_to_clean:
                if hasattr(self.image_generator, pipeline_name):
                    pipeline = getattr(self.image_generator, pipeline_name)
                    if pipeline is not None:
                        try:
                            # 先尝试调用unload方法（如果存在）
                            if hasattr(pipeline, 'unload'):
                                pipeline.unload()
                                print(f"  ✓ 已卸载 {pipeline_name} (通过unload方法)")
                            elif hasattr(pipeline, 'pipe'):
                                # 如果是diffusers pipeline，手动移动到CPU并删除
                                pipe = pipeline.pipe
                                try:
                                    # 移动到CPU
                                    if hasattr(pipe, 'to'):
                                        pipe.to('cpu')
                                    # 删除所有组件
                                    components = ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']
                                    for comp_name in components:
                                        if hasattr(pipe, comp_name):
                                            comp = getattr(pipe, comp_name)
                                            if comp is not None:
                                                try:
                                                    if hasattr(comp, 'to'):
                                                        comp.to('cpu')
                                                    del comp
                                                except:
                                                    pass
                                    # 删除pipe
                                    del pipe
                                    print(f"  ✓ 已卸载 {pipeline_name} (手动清理diffusers pipeline)")
                                except Exception as e:
                                    print(f"  ⚠ 手动清理 {pipeline_name} 时出错: {e}")
                        except Exception as e:
                            print(f"  ⚠ 卸载 {pipeline_name} 时出错: {e}")
                        finally:
                            # 删除引用
                            try:
                                delattr(self.image_generator, pipeline_name)
                                setattr(self.image_generator, pipeline_name, None)
                            except:
                                pass
            
            # ⚡ 关键修复：清理 EnhancedImageGenerator 的 PuLID 引擎和融合引擎
            # 先检查 enhanced_generator（如果存在）
            if hasattr(self.image_generator, 'enhanced_generator') and self.image_generator.enhanced_generator is not None:
                try:
                    # 清理 enhanced_generator 的 PuLID 引擎
                    if hasattr(self.image_generator.enhanced_generator, 'pulid_engine') and self.image_generator.enhanced_generator.pulid_engine is not None:
                        try:
                            self.image_generator.enhanced_generator.pulid_engine.unload()
                            self.image_generator.enhanced_generator.pulid_engine = None
                            print("  ✓ 已卸载 enhanced_generator 的 PuLID 引擎")
                        except Exception as e:
                            print(f"  ⚠ 卸载 enhanced_generator PuLID 引擎时出错: {e}")
                    
                    # 清理 enhanced_generator 的融合引擎
                    if hasattr(self.image_generator.enhanced_generator, 'fusion_engine') and self.image_generator.enhanced_generator.fusion_engine is not None:
                        try:
                            if hasattr(self.image_generator.enhanced_generator.fusion_engine, 'unload'):
                                self.image_generator.enhanced_generator.fusion_engine.unload()
                            self.image_generator.enhanced_generator.fusion_engine = None
                            print("  ✓ 已卸载 enhanced_generator 的融合引擎")
                        except Exception as e:
                            print(f"  ⚠ 卸载 enhanced_generator 融合引擎时出错: {e}")
                    
                    # 清理 enhanced_generator 的 flux_pipeline
                    if hasattr(self.image_generator.enhanced_generator, 'flux_pipeline') and self.image_generator.enhanced_generator.flux_pipeline is not None:
                        try:
                            if hasattr(self.image_generator.enhanced_generator.flux_pipeline, 'unload'):
                                self.image_generator.enhanced_generator.flux_pipeline.unload()
                            del self.image_generator.enhanced_generator.flux_pipeline
                            self.image_generator.enhanced_generator.flux_pipeline = None
                            print("  ✓ 已卸载 enhanced_generator 的 flux_pipeline")
                        except Exception as e:
                            print(f"  ⚠ 卸载 enhanced_generator flux_pipeline 时出错: {e}")
                    
                    # 调用 enhanced_generator 的 unload_all
                    if hasattr(self.image_generator.enhanced_generator, 'unload_all'):
                        try:
                            self.image_generator.enhanced_generator.unload_all()
                            print("  ✓ 已调用 enhanced_generator.unload_all()")
                        except Exception as e:
                            print(f"  ⚠ 调用 enhanced_generator.unload_all 时出错: {e}")
                    
                    # ⚡ 关键修复：删除 enhanced_generator 对象本身，确保所有引用都被清理
                    try:
                        del self.image_generator.enhanced_generator
                        self.image_generator.enhanced_generator = None
                        print("  ✓ 已删除 enhanced_generator 对象")
                    except Exception as e:
                        print(f"  ⚠ 删除 enhanced_generator 对象时出错: {e}")
                except Exception as e:
                    print(f"  ⚠ 清理 enhanced_generator 时出错: {e}")
            
            # 清理 ImageGenerator 自己的 PuLID 引擎和融合引擎（如果存在）
            if hasattr(self.image_generator, 'pulid_engine') and self.image_generator.pulid_engine is not None:
                try:
                    self.image_generator.pulid_engine.unload()
                    self.image_generator.pulid_engine = None
                    print("  ✓ 已卸载 ImageGenerator 的 PuLID 引擎")
                except Exception as e:
                    print(f"  ⚠ 卸载 ImageGenerator PuLID 引擎时出错: {e}")
            
            if hasattr(self.image_generator, 'fusion_engine') and self.image_generator.fusion_engine is not None:
                try:
                    if hasattr(self.image_generator.fusion_engine, 'unload'):
                        self.image_generator.fusion_engine.unload()
                    self.image_generator.fusion_engine = None
                    print("  ✓ 已卸载 ImageGenerator 的融合引擎")
                except Exception as e:
                    print(f"  ⚠ 卸载 ImageGenerator 融合引擎时出错: {e}")
            
            # ⚡ 关键修复：清理 planner 的 LLM 客户端（如果存在）
            if hasattr(self.image_generator, 'planner') and self.image_generator.planner is not None:
                try:
                    if hasattr(self.image_generator.planner, 'llm_client') and self.image_generator.planner.llm_client is not None:
                        # LLM 客户端通常不占用显存，但清理引用有助于垃圾回收
                        self.image_generator.planner.llm_client = None
                        print("  ✓ 已清理 planner 的 LLM 客户端")
                except Exception as e:
                    print(f"  ⚠ 清理 planner LLM 客户端时出错: {e}")
            
            # 如果 EnhancedImageGenerator 有 unload_all 方法，调用它
            if hasattr(self.image_generator, 'unload_all'):
                try:
                    self.image_generator.unload_all()
                    print("  ✓ 已调用 EnhancedImageGenerator.unload_all()")
                except Exception as e:
                    print(f"  ⚠ 调用 unload_all 时出错: {e}")
            
            # 清理ModelManager（如果使用）
            if hasattr(self.image_generator, 'model_manager') and self.image_generator.model_manager is not None:
                try:
                    if hasattr(self.image_generator.model_manager, 'unload_all'):
                        self.image_generator.model_manager.unload_all(include_critical=False)
                        print("  ✓ 已卸载ModelManager所有模型")
                    elif hasattr(self.image_generator.model_manager, 'unload'):
                        self.image_generator.model_manager.unload()
                        print("  ✓ 已卸载ModelManager")
                except Exception as e:
                    print(f"  ⚠ 卸载ModelManager时出错: {e}")
            
            # ⚡ 关键修复：清理 quality_analyzer（如果存在，可能持有 InsightFace 模型）
            if hasattr(self.image_generator, 'quality_analyzer') and self.image_generator.quality_analyzer is not None:
                try:
                    # InsightFace 模型可能占用显存
                    if hasattr(self.image_generator.quality_analyzer, 'face_analyzer'):
                        self.image_generator.quality_analyzer.face_analyzer = None
                    self.image_generator.quality_analyzer = None
                    print("  ✓ 已清理 quality_analyzer")
                except Exception as e:
                    print(f"  ⚠ 清理 quality_analyzer 时出错: {e}")
            
            # ⚡ 关键修复：强制清理所有CUDA缓存，每几步清理一次
            if torch.cuda.is_available():
                # 同步所有 CUDA 操作
                torch.cuda.synchronize()
                
                # 多次清理，每几步清理一次（模拟之前优化的方式）
                for i in range(20):  # 增加到20次，更彻底
                    if i % 3 == 0:  # 每3次同步一次
                        torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 最终同步和清理
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                
                # 等待一小段时间让显存真正释放
                import time
                time.sleep(1.0)  # 增加到1秒，让显存有更多时间释放
                
                # 再次清理
                for i in range(10):
                    torch.cuda.empty_cache()
                    gc.collect()
                    if i % 2 == 0:
                        torch.cuda.synchronize()
                
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                freed = allocated_before - allocated_after if torch.cuda.is_available() else 0
                print(f"  ℹ 清理后显存: 已分配={allocated_after:.2f}GB, 已保留={reserved_after:.2f}GB")
                if freed > 0:
                    print(f"  ✓ 已释放显存: {freed:.2f}GB")
                else:
                    print(f"  ⚠ 警告：显存未释放，可能仍有模型占用显存")
                
                # 检查可用显存是否足够
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - reserved_after
                print(f"  ℹ 可用显存: {free:.2f}GB / {total:.2f}GB")
                if free < 20:
                    print(f"  ⚠ 警告: 可用显存较少 ({free:.2f}GB)，视频生成可能会失败")
            
        except Exception as e:
            print(f"  ⚠ 清理显存时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 步骤2: 使用 HunyuanVideo 生成视频
        print()
        print("=" * 60)
        print("步骤2: 使用 HunyuanVideo 生成视频")
        print("=" * 60)
        
        # ⚡ 关键修复：视频生成前再次彻底清理显存
        print("  🔧 视频生成前最后一次清理显存...")
        try:
            import torch
            import gc
            
            if torch.cuda.is_available():
                allocated_before_video = torch.cuda.memory_allocated() / 1024**3
                reserved_before_video = torch.cuda.memory_reserved() / 1024**3
                print(f"  ℹ 视频生成前显存: 已分配={allocated_before_video:.2f}GB, 已保留={reserved_before_video:.2f}GB")
                
                # 多次彻底清理
                for i in range(10):
                    if i % 2 == 0:
                        torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 最终同步和清理
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                
                # 等待显存真正释放
                import time
                time.sleep(0.3)
                
                # 再次清理
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                
                allocated_after_cleanup = torch.cuda.memory_allocated() / 1024**3
                reserved_after_cleanup = torch.cuda.memory_reserved() / 1024**3
                freed = allocated_before_video - allocated_after_cleanup
                print(f"  ℹ 清理后显存: 已分配={allocated_after_cleanup:.2f}GB, 已保留={reserved_after_cleanup:.2f}GB")
                if freed > 0:
                    print(f"  ✓ 已释放显存: {freed:.2f}GB")
                
                # 检查可用显存
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - reserved_after_cleanup
                print(f"  ℹ 可用显存: {free:.2f}GB / {total:.2f}GB")
                if free < 15:
                    print(f"  ⚠ 警告: 可用显存较少 ({free:.2f}GB)，视频生成可能会失败")
        except Exception as e:
            print(f"  ⚠ 视频生成前清理显存时出错: {e}")
        
        if video_output_path is None:
            video_output_path = output_dir / "novel_video.mp4"
        
        try:
            # 构建视频生成提示词（可以更详细，描述运动方式）
            video_prompt = self._build_video_prompt(prompt, scene)
            
            # 构建scene字典（包含prompt信息和分辨率）
            video_scene = scene.copy() if scene else {}
            video_scene['description'] = video_prompt
            video_scene['prompt'] = video_prompt  # 也添加到prompt字段
            video_scene['motion_intensity'] = motion_intensity
            # 重要：确保视频使用与图像相同的分辨率，保持长宽比一致
            # width和height已经在图像生成后更新为实际分辨率
            video_scene['width'] = width  # 使用图像的实际宽度
            video_scene['height'] = height  # 使用图像的实际高度
            print(f"  ℹ 视频将使用分辨率: {width}x{height} (与图像一致，保持长宽比 {width/height:.2f})")
            
            # 生成视频：
            # - 默认：VideoGenerator（纯生成）
            # - 启用 enable_m6_identity：EnhancedVideoGeneratorM6（验证 + 重试 + 产出 report）
            if effective_enable_m6:
                if not effective_include_character:
                    print("  ⚠ 警告：enable_m6_identity=True 但 include_character=False（无人物场景通常无法做人脸验证），将退回普通视频生成")
                    effective_enable_m6 = False

            identity_report_path: Optional[Path] = None
            if effective_enable_m6:
                from enhanced_video_generator_m6 import EnhancedVideoGeneratorM6
                if self._m6_video_generator is None:
                    print("初始化 M6 增强视频生成器（身份验证+重试）...")
                    self._m6_video_generator = EnhancedVideoGeneratorM6(str(self.config_path))

                # 选择参考图：优先用户显式传入；否则尝试按 character_id 找 reference_image/<id>_mid.jpg；否则用生成图
                ref = None
                if reference_image_path:
                    rp = Path(reference_image_path)
                    if not rp.is_absolute():
                        rp = (project_root / rp).resolve()
                    if rp.exists():
                        ref = str(rp)
                if ref is None and effective_character_id:
                    candidate = (project_root / "reference_image" / f"{effective_character_id}_mid.jpg").resolve()
                    if candidate.exists():
                        ref = str(candidate)
                if ref is None:
                    ref = str(image_path)

                # quick 模式：减少步数（保守默认 8）并将重试设为 0（除非用户显式传）
                if m6_quick:
                    self._m6_video_generator.video_config.setdefault("hunyuanvideo", {})
                    hv = self._m6_video_generator.video_config["hunyuanvideo"]
                    hv["num_inference_steps"] = min(int(hv.get("num_inference_steps", 25)), 8)
                    if m6_max_retries is None:
                        m6_max_retries = 0

                vp, result = self._m6_video_generator.generate_video_with_identity_check(
                    image_path=str(image_path),
                    output_path=str(video_output_path),
                    reference_image=ref,
                    scene=video_scene,
                    shot_type=shot_type,
                    enable_verification=True,
                    max_retries=m6_max_retries,
                    num_frames=num_frames,
                    fps=fps,
                )
                video_path = vp

                # 写一个轻量 report（便于后续批量统计/归档）
                identity_report_path = output_dir / "novel_video_identity.json"
                payload = {
                    "passed": bool(result.passed) if result else False,
                    "avg_similarity": float(result.avg_similarity) if result else 0.0,
                    "min_similarity": float(result.min_similarity) if result else 0.0,
                    "drift_ratio": float(result.drift_ratio) if result else 1.0,
                    "face_detect_ratio": float(result.face_detect_ratio) if result else 0.0,
                    "issues": list(result.issues or []) if result else ["result=None"],
                    "reference_image": ref,
                    "video_path": str(video_path),
                    "character_id": effective_character_id,
                }
                identity_report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"✅ M6 身份验证报告: {identity_report_path}")
            else:
                # 非 M6 模式：使用普通视频生成器
                video_path = self.video_generator.generate_video(
                    image_path=str(image_path),
                    output_path=str(video_output_path),
                    num_frames=num_frames,
                    fps=fps,
                    scene=video_scene,
                )

            print(f"✅ 视频生成成功: {video_path}")
            
            # ⚡ 关键修复：视频生成后彻底清理显存
            print()
            print("  🔧 视频生成后清理显存...")
            try:
                import torch
                import gc
                
                if torch.cuda.is_available():
                    # 多次清理，确保彻底释放
                    for i in range(10):
                        if i % 2 == 0:
                            torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # 最终同步和清理
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 等待显存真正释放
                    import time
                    time.sleep(0.2)
                    
                    # 再次清理
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    
                    allocated_after = torch.cuda.memory_allocated() / 1024**3
                    reserved_after = torch.cuda.memory_reserved() / 1024**3
                    print(f"  ℹ 视频生成后显存: 已分配={allocated_after:.2f}GB, 已保留={reserved_after:.2f}GB")
            except Exception as e:
                print(f"  ⚠ 视频生成后清理显存时出错: {e}")
                
        except Exception as e:
            print(f"❌ 视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print()
        print("=" * 60)
        print("✅ 小说推文视频生成完成")
        print("=" * 60)
        print(f"图像: {image_path}")
        print(f"视频: {video_path}")
        
        return {
            'image': image_path,
            'video': video_path,
            **({"identity_report": identity_report_path} if effective_enable_m6 and identity_report_path else {}),
        }
    
    def _generate_v21_exec(
        self,
        scene: Dict[str, Any],
        output_dir: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int
    ) -> Dict[str, Path]:
        """
        使用v2.1-exec格式生成（Execution Executor模式）
        
        Args:
            scene: v2.1-exec格式的场景JSON
            output_dir: 输出目录
            width: 图像宽度
            height: 图像高度
            num_frames: 视频帧数
            fps: 视频帧率
            
        Returns:
            dict: 包含 'image' 和 'video' 路径的字典
        """
        print("=" * 60)
        print("使用v2.1-exec模式生成")
        print("=" * 60)
        
        try:
            from utils.execution_executor_v21 import (
                ExecutionExecutorV21,
                ExecutionConfig,
                ExecutionMode
            )
            from utils.execution_validator import ExecutionValidator
            
            # 1. 校验JSON
            validator = ExecutionValidator()
            validation_result = validator.validate_scene(scene)
            if not validation_result.is_valid:
                print(f"  ✗ JSON校验失败: {validation_result.errors_count} 个错误")
                raise ValueError("场景JSON校验失败")
            
            print(f"  ✓ JSON校验通过")
            
            # 2. 创建Execution Executor
            config = ExecutionConfig(mode=ExecutionMode.STRICT)
            executor = ExecutionExecutorV21(
                config=config,
                image_generator=self.image_generator,
                video_generator=self.video_generator,
                tts_generator=None  # TTS可以后续添加
            )
            
            # 3. 执行场景生成
            result = executor.execute_scene(scene, str(output_dir))
            
            if result.success:
                print(f"  ✓ 场景 {scene.get('scene_id')} 生成成功")
                return {
                    "image": Path(result.image_path) if result.image_path else None,
                    "video": Path(result.video_path) if result.video_path else None
                }
            else:
                print(f"  ✗ 场景 {scene.get('scene_id')} 生成失败: {result.error_message}")
                raise RuntimeError(f"生成失败: {result.error_message}")
                
        except ImportError as e:
            print(f"  ⚠ v2.1-exec模块未找到: {e}")
            print("  回退到原有流程")
            # 回退到原有流程
            prompt = scene.get("prompt", {}).get("positive_core", "")
            return self.generate(
                prompt=prompt,
                output_dir=output_dir,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                scene=scene,
                use_v21_exec=False
            )
        except Exception as e:
            print(f"  ✗ v2.1-exec生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_video_prompt(self, image_prompt: str, scene: Optional[Dict[str, Any]] = None) -> str:
        """
        构建视频生成提示词（使用 Prompt Engine V2）
        
        Args:
            image_prompt: 图像生成时的提示词
            scene: 场景配置
            
        Returns:
            优化后的视频生成提示词
        """
        def _extract_scene_motion(prompt_text: str) -> list:
            """
            从prompt中提取场景元素，并生成相应的运动描述
            
            Args:
                prompt_text: 提示词文本
                
            Returns:
                场景元素的运动描述列表
            """
            motion_keywords = {
                # 水相关 - 使用更强烈的运动描述
                '瀑布': ['waterfall continuously flowing down', 'water cascading and rushing', 'waterfall in motion'],
                '瀑布流': ['waterfall continuously flowing', 'water cascading'],
                'waterfall': ['waterfall continuously flowing down', 'water cascading'],
                '河流': ['river flowing and streaming', 'water continuously moving'],
                'river': ['river flowing and streaming'],
                '溪流': ['stream flowing and trickling', 'water moving'],
                'stream': ['stream flowing and trickling'],
                '水': ['water rippling and flowing', 'water in motion'],
                'water': ['water rippling and flowing'],
                '湖': ['lake rippling with waves', 'water gently moving'],
                'lake': ['lake rippling with waves'],
                '海': ['waves rolling and crashing', 'ocean waves in motion'],
                'sea': ['waves rolling and crashing'],
                'ocean': ['waves rolling and crashing'],
                
                # 天空相关 - 使用更明确的运动描述
                '云': ['clouds slowly drifting across the sky', 'clouds moving in the wind'],
                '云彩': ['clouds slowly drifting across the sky', 'clouds moving'],
                'cloud': ['clouds slowly drifting', 'clouds moving'],
                'clouds': ['clouds slowly drifting across the sky', 'clouds moving'],
                '彩虹': ['rainbow shimmering and glowing', 'rainbow light effects in motion'],
                'rainbow': ['rainbow shimmering and glowing'],
                '阳光': ['sunlight shifting and moving', 'light rays in motion'],
                'sunlight': ['sunlight shifting and moving'],
                '光线': ['light rays moving and shifting', 'light in motion'],
                'light': ['light rays moving and shifting'],
                
                # 植物相关 - 强调运动
                '树': ['leaves swaying in the wind', 'trees gently moving'],
                '树叶': ['leaves swaying and rustling', 'leaves in motion'],
                'tree': ['leaves swaying in the wind'],
                'leaves': ['leaves swaying and rustling'],
                '草': ['grass swaying in the breeze', 'grass moving'],
                'grass': ['grass swaying in the breeze'],
                '花': ['flowers swaying gently', 'flowers moving'],
                'flower': ['flowers swaying gently'],
                'flowers': ['flowers swaying gently'],
                
                # 风相关
                '风': ['wind blowing and moving', 'breeze in motion'],
                'wind': ['wind blowing and moving'],
                'breeze': ['wind blowing and moving'],
                
                # 雾气相关
                '雾': ['mist rising and drifting', 'fog moving'],
                '雾气': ['mist rising and drifting'],
                'mist': ['mist rising and drifting'],
                'fog': ['mist rising and drifting'],
                
                # 火相关
                '火': ['flames flickering and dancing', 'fire in motion'],
                '火焰': ['flames flickering and dancing'],
                'fire': ['flames flickering and dancing'],
                'flame': ['flames flickering and dancing'],
                
                # 雪相关
                '雪': ['snow falling and drifting', 'snowflakes in motion'],
                'snow': ['snow falling and drifting'],
                'snowflake': ['snow falling and drifting'],
                
                # 鸟相关
                '鸟': ['birds flying and soaring', 'birds in motion'],
                'bird': ['birds flying and soaring'],
                'birds': ['birds flying and soaring'],
            }
            
            scene_motions = []
            prompt_lower = prompt_text.lower()
            
            # 检查每个关键词
            for keyword, motions in motion_keywords.items():
                if keyword.lower() in prompt_lower:
                    # 使用第一个运动描述（最常用）
                    scene_motions.append(motions[0])
            
            return scene_motions
        
        try:
            from utils.prompt_engine_v2 import PromptEngine, UserRequest
            
            # 创建 Prompt Engine V2（本地模式）
            prompt_engine_v2 = PromptEngine()
            
            # 创建用户请求（视频生成阶段）
            req = UserRequest(
                text=image_prompt,
                scene_type="novel",  # 小说推文场景
                style="novel",  # 使用novel风格模板
                target_model="hunyuanvideo",  # 视频生成使用HunyuanVideo
                params=scene.get('params', {}) if scene else {}
            )
            
            # 执行处理
            pkg = prompt_engine_v2.run(req)
            
            # 获取优化后的prompt
            video_prompt = pkg.final_prompt
            
            # 提取场景元素的运动描述（关键：添加物体运动，而不仅仅是相机运动）
            scene_motions = _extract_scene_motion(image_prompt)
            
            # 关键修复：将运动描述直接融入到prompt中，而不是作为后缀
            # HunyuanVideo需要运动描述直接融入到场景描述中
            if scene_motions:
                print(f"  ℹ 检测到场景元素运动: {', '.join(scene_motions)}")
                
                # 将运动描述直接插入到prompt的前面部分（在主体描述之后）
                # 格式：主体描述 + 运动描述 + 其他描述
                prompt_parts = video_prompt.split('.')
                if len(prompt_parts) > 1:
                    # 在第一个句号后插入运动描述
                    enhanced_prompt = prompt_parts[0] + ". " + ", ".join(scene_motions) + ". " + ". ".join(prompt_parts[1:])
                    video_prompt = enhanced_prompt
                else:
                    # 如果没有句号，直接添加到前面
                    video_prompt = ", ".join(scene_motions) + ". " + video_prompt
            
            # 添加运动描述（增强版，确保物体运动）
            motion_descriptions = []
            
            # 1. 再次强调场景元素的运动（使用更强烈的描述）
            if scene_motions:
                # 使用更强烈的运动描述
                strong_motions = []
                for motion in scene_motions:
                    if 'flowing' in motion:
                        strong_motions.append("water continuously flowing, dynamic water movement")
                    elif 'drifting' in motion:
                        strong_motions.append("clouds slowly drifting, sky in motion")
                    elif 'shimmering' in motion:
                        strong_motions.append("rainbow shimmering and glowing, light effects in motion")
                    elif 'swaying' in motion:
                        strong_motions.append("leaves gently swaying, natural wind movement")
                    else:
                        strong_motions.append(motion + ", motion visible")
                motion_descriptions.extend(strong_motions)
            
            # 2. 添加场景配置中的运动强度
            if scene and isinstance(scene, dict):
                motion_intensity = scene.get('motion_intensity', 'moderate')
                camera_motion = scene.get('camera_motion', {})
                
                if motion_intensity == 'dynamic':
                    motion_descriptions.append("dynamic movement, active motion, objects in motion")
                elif motion_intensity == 'moderate':
                    motion_descriptions.append("moderate movement, natural motion, elements moving")
                else:
                    motion_descriptions.append("gentle movement, subtle motion, natural flow")
                
                # 3. 添加相机运动（次要，避免只有相机运动）
                if isinstance(camera_motion, dict):
                    camera_type = camera_motion.get('type', 'static')
                    if camera_type == 'pan':
                        motion_descriptions.append("smooth camera pan")
                    elif camera_type == 'zoom':
                        motion_descriptions.append("smooth camera zoom")
                    elif camera_type == 'dolly':
                        motion_descriptions.append("smooth camera dolly")
            
            # 如果没有检测到场景运动，添加默认的自然运动描述
            if not scene_motions:
                motion_descriptions.append("natural movement, subtle motion, elements in motion")
                print(f"  ℹ 未检测到特定场景元素，添加默认自然运动描述")
            
            # 组合运动描述（添加到prompt末尾，作为补充）
            if motion_descriptions:
                video_prompt += ". " + ", ".join(motion_descriptions)
            
            # 添加视频质量描述
            video_prompt += ". High quality, cinematic, smooth motion, natural movement, objects in motion"
            
            print(f"  ✓ 视频提示词已使用 Prompt Engine V2 优化")
            print(f"  ℹ QA评分: {pkg.metadata.get('qa_score', 0)}/{pkg.metadata.get('qa_max_score', 0)}")
            
            return video_prompt
            
        except Exception as e:
            print(f"  ⚠ Prompt Engine V2 处理失败: {e}，使用基础方案")
            import traceback
            traceback.print_exc()
            
            # 备用方案：基础提示词构建
            video_prompt = image_prompt
            
            # 提取场景元素的运动描述
            scene_motions = _extract_scene_motion(image_prompt)
            
            # 关键修复：将运动描述直接融入到prompt中
            if scene_motions:
                print(f"  ℹ 检测到场景元素运动: {', '.join(scene_motions)}")
                
                # 将运动描述直接插入到prompt的前面部分
                prompt_parts = video_prompt.split('.')
                if len(prompt_parts) > 1:
                    enhanced_prompt = prompt_parts[0] + ". " + ", ".join(scene_motions) + ". " + ". ".join(prompt_parts[1:])
                    video_prompt = enhanced_prompt
                else:
                    video_prompt = ", ".join(scene_motions) + ". " + video_prompt
            
            # 添加运动描述（增强版）
            motion_descriptions = []
            
            # 1. 再次强调场景元素的运动（使用更强烈的描述）
            if scene_motions:
                strong_motions = []
                for motion in scene_motions:
                    if 'flowing' in motion:
                        strong_motions.append("water continuously flowing, dynamic water movement")
                    elif 'drifting' in motion:
                        strong_motions.append("clouds slowly drifting, sky in motion")
                    elif 'shimmering' in motion:
                        strong_motions.append("rainbow shimmering and glowing, light effects in motion")
                    elif 'swaying' in motion:
                        strong_motions.append("leaves gently swaying, natural wind movement")
                    else:
                        strong_motions.append(motion + ", motion visible")
                motion_descriptions.extend(strong_motions)
            
            # 2. 添加场景配置中的运动强度
            if scene and isinstance(scene, dict):
                motion_intensity = scene.get('motion_intensity', 'moderate')
                camera_motion = scene.get('camera_motion', {})
                
                if motion_intensity == 'dynamic':
                    motion_descriptions.append("dynamic movement, active motion, objects in motion")
                elif motion_intensity == 'moderate':
                    motion_descriptions.append("moderate movement, natural motion, elements moving")
                else:
                    motion_descriptions.append("gentle movement, subtle motion, natural flow")
                
                # 3. 添加相机运动
                if isinstance(camera_motion, dict):
                    camera_type = camera_motion.get('type', 'static')
                    if camera_type == 'pan':
                        motion_descriptions.append("smooth camera pan")
                    elif camera_type == 'zoom':
                        motion_descriptions.append("smooth camera zoom")
                    elif camera_type == 'dolly':
                        motion_descriptions.append("smooth camera dolly")
            
            # 如果没有检测到场景运动，添加默认的自然运动描述
            if not scene_motions:
                motion_descriptions.append("natural movement, subtle motion, elements in motion")
                print(f"  ℹ 未检测到特定场景元素，添加默认自然运动描述")
            
            # 组合运动描述
            if motion_descriptions:
                video_prompt += ". " + ", ".join(motion_descriptions)
            
            # 添加质量描述
            video_prompt += ". High quality, cinematic, smooth motion, natural movement, objects in motion"
            
            return video_prompt


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成小说推文视频")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示词（小说场景描述）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度")
    parser.add_argument("--height", type=int, default=768, help="图像高度")
    parser.add_argument("--num-frames", type=int, default=120, help="视频帧数")
    parser.add_argument("--fps", type=int, default=24, help="视频帧率")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    # 角色一致（图片端）
    parser.add_argument("--include-character", action="store_true", help="强制启用角色模式（人物出镜，走角色一致系统）")
    parser.add_argument("--force-scene", action="store_true", help="强制纯场景模式（忽略自动推断/手动角色）")
    parser.add_argument("--auto-character", action=argparse.BooleanOptionalAction, default=True, help="是否自动识别是否包含韩立（默认开启）")
    parser.add_argument("--character-id", type=str, default=None, help="角色ID（可选，覆盖自动推断）")
    parser.add_argument("--image-model-engine", type=str, default=None, help="覆盖图片引擎（auto/flux-instantid/pulid/flux1...）")

    # 视频一致（M6）
    parser.add_argument("--enable-m6-identity", action="store_true", help="强制启用 M6 身份验证+重试（仅在检测到韩立/角色模式时生效）")
    parser.add_argument("--disable-m6-identity", action="store_true", help="强制关闭 M6（即使检测到韩立）")
    parser.add_argument("--auto-m6-identity", action=argparse.BooleanOptionalAction, default=True, help="是否对韩立场景自动启用 M6（默认开启）")
    parser.add_argument("--reference-image-path", type=str, default=None, help="身份验证参考图（不传则按 character-id 自动找 *_mid.jpg，否则用生成图）")
    parser.add_argument("--shot-type", type=str, default="medium", choices=["wide", "medium", "medium_close", "close", "extreme_close"], help="镜头类型")
    parser.add_argument("--motion-intensity", type=str, default="moderate", choices=["gentle", "moderate", "dynamic"], help="运动强度")
    parser.add_argument("--m6-max-retries", type=int, default=None, help="覆盖 M6 最大重试次数（0=不重试）")
    parser.add_argument("--m6-quick", action="store_true", help="M6 快速模式（更少步数/默认不重试，适合冒烟）")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = NovelVideoGenerator(config_path=args.config)
    
    # M6 显式开关优先级：disable > enable > auto(None)
    enable_m6_identity = None
    if args.disable_m6_identity:
        enable_m6_identity = False
    elif args.enable_m6_identity:
        enable_m6_identity = True
    
    # 生成视频
    result = generator.generate(
        prompt=args.prompt,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        include_character=True if args.include_character else None,
        character_id=args.character_id,
        auto_character=bool(args.auto_character),
        force_scene=bool(args.force_scene),
        image_model_engine=args.image_model_engine,
        enable_m6_identity=enable_m6_identity,
        auto_m6_identity=bool(args.auto_m6_identity),
        reference_image_path=args.reference_image_path,
        shot_type=args.shot_type,
        motion_intensity=args.motion_intensity,
        m6_max_retries=args.m6_max_retries,
        m6_quick=bool(args.m6_quick),
    )
    
    print("\n生成完成！")
    print(f"图像: {result['image']}")
    print(f"视频: {result['video']}")


if __name__ == "__main__":
    main()

