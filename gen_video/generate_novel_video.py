#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说推文视频生成脚本
使用 Flux 生成图片，然后用 HunyuanVideo 生成视频
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

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
        
        # 确保使用正确的模型
        self._ensure_model_config()
        
        print("=" * 60)
        print("✅ 初始化完成")
        print("=" * 60)
    
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
        prompt: str,
        output_dir: Optional[Path] = None,
        image_output_path: Optional[Path] = None,
        video_output_path: Optional[Path] = None,
        width: int = 1280,
        height: int = 768,
        num_frames: int = 120,
        fps: int = 24,
        scene: Optional[Dict[str, Any]] = None,
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
        
        Returns:
            dict: 包含 'image' 和 'video' 路径的字典
        """
        print("=" * 60)
        print("开始生成小说推文视频")
        print("=" * 60)
        print(f"提示词: {prompt}")
        print()
        
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
            
            # 生成图像（使用纯 Flux 生成场景，不使用 InstantID）
            # 对于小说推文，应该生成场景图像，而不是人物图像
            print(f"  [DEBUG] 原始prompt: {prompt}")
            print(f"  [DEBUG] scene: {image_scene}")
            print(f"  [DEBUG] model_engine: flux1")
            print(f"  [DEBUG] task_type: scene")
            
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
                
                # 检查并限制提示词长度（CLIP限制77 tokens）
                def count_tokens(text: str) -> int:
                    """估算token数量（简单方法）"""
                    try:
                        from transformers import CLIPTokenizer
                        tokenizer = CLIPTokenizer.from_pretrained(
                            "openai/clip-vit-large-patch14"
                        )
                        tokens = tokenizer(text, truncation=False, return_tensors="pt")
                        return tokens.input_ids.shape[1]
                    except Exception:
                        # 如果无法加载tokenizer，使用简单估算
                        # 中文约1.5 tokens/字，英文约1.3 tokens/词
                        chinese_chars = sum(1 for c in text if ord(c) > 127)
                        english_words = len([w for w in text.split() if not any(ord(c) > 127 for c in w)])
                        return int(chinese_chars * 1.5 + english_words * 1.3)
                
                def truncate_prompt(prompt: str, max_tokens: int = 77) -> str:
                    """截断prompt到指定token数"""
                    current_tokens = count_tokens(prompt)
                    if current_tokens <= max_tokens:
                        return prompt
                    
                    # 如果超过限制，逐步移除后面的部分
                    parts = [p.strip() for p in prompt.split(',')]
                    truncated_parts = []
                    truncated_prompt = ""
                    
                    for part in parts:
                        test_prompt = truncated_prompt + (", " if truncated_prompt else "") + part
                        if count_tokens(test_prompt) <= max_tokens:
                            truncated_parts.append(part)
                            truncated_prompt = test_prompt
                        else:
                            break
                    
                    if not truncated_parts:
                        # 如果第一部分就超过，直接截断字符串
                        return prompt[:int(len(prompt) * max_tokens / current_tokens)]
                    
                    return ", ".join(truncated_parts)
                
                # 先检查优化后的prompt长度
                optimized_tokens = count_tokens(optimized_prompt)
                print(f"  ℹ 优化后prompt token数: {optimized_tokens}")
                
                # 添加场景强化关键词（确保是场景而非人物）
                # 使用更简洁的scene_enhancers，避免超过token限制
                scene_enhancers = "landscape, nature, no people"
                
                # 检查添加scene_enhancers后是否会超过限制
                test_prompt = f"{optimized_prompt}, {scene_enhancers}"
                test_tokens = count_tokens(test_prompt)
                
                if test_tokens > 77:
                    print(f"  ⚠ 添加scene_enhancers后会超过77 tokens ({test_tokens})，先截断optimized_prompt")
                    # 预留空间给scene_enhancers（约5 tokens）
                    optimized_prompt = truncate_prompt(optimized_prompt, max_tokens=72)
                    optimized_tokens = count_tokens(optimized_prompt)
                    print(f"  ℹ 截断后prompt token数: {optimized_tokens}")
                
                optimized_prompt = f"{optimized_prompt}, {scene_enhancers}"
                final_tokens = count_tokens(optimized_prompt)
                print(f"  ℹ 最终prompt token数: {final_tokens}")
                
                if final_tokens > 77:
                    print(f"  ⚠ 最终prompt仍然超过77 tokens ({final_tokens})，进行截断")
                    optimized_prompt = truncate_prompt(optimized_prompt, max_tokens=77)
                    final_tokens = count_tokens(optimized_prompt)
                    print(f"  ℹ 截断后最终prompt token数: {final_tokens}")
                
                # 增强负面提示词（确保排除人物）
                additional_negatives = [
                    "faces, portraits, black faces, dark faces, human faces, person faces, character faces",
                    "people in image, humans in scene, any people, any persons, any characters, any human figures"
                ]
                negative_prompt = f"{negative_prompt}, {', '.join(additional_negatives)}"
                
                print(f"  ✓ Prompt Engine V2 处理完成")
                print(f"  ℹ 原始提示词: {original_prompt[:80]}...")
                print(f"  ℹ 优化后提示词: {optimized_prompt[:100]}...")
                print(f"  ℹ QA评分: {pkg.metadata.get('qa_score', 0)}/{pkg.metadata.get('qa_max_score', 0)}")
                
            except Exception as e:
                print(f"  ⚠ Prompt Engine V2 处理失败: {e}，使用备用方案")
                import traceback
                traceback.print_exc()
                
                # 备用方案：使用原始提示词+场景强化（简化版本，避免超过token限制）
                scene_enhancers = "landscape, nature, no people"
                optimized_prompt = f"{original_prompt}, {scene_enhancers}"
                
                # 检查token数
                try:
                    from transformers import CLIPTokenizer
                    tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14"
                    )
                    tokens = tokenizer(optimized_prompt, truncation=False, return_tensors="pt")
                    token_count = tokens.input_ids.shape[1]
                    if token_count > 77:
                        print(f"  ⚠ 备用方案prompt超过77 tokens ({token_count})，将被CLIP截断")
                except Exception:
                    pass
                negative_prompt = "anime, cartoon, characters, people, persons, human figures, anime style, cartoon style, faces, portraits, black faces, dark faces, human faces, person faces, character faces, people in image, humans in scene, any people, any persons, any characters, any human figures, low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark, flickering, jittery, unstable, sudden movement, abrupt changes, low quality, worst quality, distorted proportions, unrealistic details"
            
            print(f"  ✅ 提示词优化完成:")
            print(f"     原始: {original_prompt}")
            print(f"     优化后: {optimized_prompt[:150]}...")
            print(f"     负面提示词: {negative_prompt[:150]}...")
            
            prompt = optimized_prompt
            negative_prompt = negative_prompt
            
            # 确保scene中不包含角色信息，避免被误识别为人物生成
            if image_scene:
                # 移除可能触发角色检测的字段
                image_scene.pop('character', None)
                image_scene.pop('characters', None)
                image_scene.pop('primary_character', None)
                image_scene.pop('face_reference_image_path', None)
                image_scene.pop('reference_image_path', None)
                print(f"  [DEBUG] 已清理scene中的角色相关字段，确保生成场景图像")
            
            image_path = self.image_generator.generate_image(
                prompt=prompt,
                output_path=image_output_path,
                scene=image_scene,
                model_engine="flux1",  # 使用纯 Flux 1，不包含 InstantID（用于场景生成）
                task_type="scene",  # 明确指定为场景生成任务
                character_lora=None,  # 明确不使用角色LoRA
                use_lora=False,  # 明确不使用LoRA
                face_reference_image_path=None,  # 明确不使用面部参考图
                reference_image_path=None,  # 明确不使用参考图
                negative_prompt=negative_prompt,  # 使用优化后的负面提示词
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
            
            # 清理ModelManager（如果使用）
            if hasattr(self.image_generator, 'model_manager') and self.image_generator.model_manager is not None:
                try:
                    if hasattr(self.image_generator.model_manager, 'unload'):
                        self.image_generator.model_manager.unload()
                        print("  ✓ 已卸载ModelManager所有模型")
                except Exception as e:
                    print(f"  ⚠ 卸载ModelManager时出错: {e}")
            
            # 强制清理所有CUDA缓存
            if torch.cuda.is_available():
                # 多次清理，确保彻底释放
                for i in range(10):  # 增加到10次
                    torch.cuda.empty_cache()
                    gc.collect()
                torch.cuda.synchronize()
                
                # 再次清理
                torch.cuda.empty_cache()
                gc.collect()
                
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                freed = allocated_before - allocated_after if torch.cuda.is_available() else 0
                print(f"  ℹ 清理后显存: 已分配={allocated_after:.2f}GB, 已保留={reserved_after:.2f}GB")
                if freed > 0:
                    print(f"  ✓ 已释放显存: {freed:.2f}GB")
                else:
                    print(f"  ⚠ 警告: 显存未释放（可能被其他进程占用）")
            
        except Exception as e:
            print(f"  ⚠ 清理显存时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 步骤2: 使用 HunyuanVideo 生成视频
        print()
        print("=" * 60)
        print("步骤2: 使用 HunyuanVideo 生成视频")
        print("=" * 60)
        
        if video_output_path is None:
            video_output_path = output_dir / "novel_video.mp4"
        
        try:
            # 构建视频生成提示词（可以更详细，描述运动方式）
            video_prompt = self._build_video_prompt(prompt, scene)
            
            # 构建scene字典（包含prompt信息和分辨率）
            video_scene = scene.copy() if scene else {}
            video_scene['description'] = video_prompt
            video_scene['prompt'] = video_prompt  # 也添加到prompt字段
            # 重要：确保视频使用与图像相同的分辨率，保持长宽比一致
            # width和height已经在图像生成后更新为实际分辨率
            video_scene['width'] = width  # 使用图像的实际宽度
            video_scene['height'] = height  # 使用图像的实际高度
            print(f"  ℹ 视频将使用分辨率: {width}x{height} (与图像一致，保持长宽比 {width/height:.2f})")
            
            # 生成视频
            video_path = self.video_generator.generate_video(
                image_path=str(image_path),
                output_path=str(video_output_path),
                num_frames=num_frames,
                fps=fps,
                scene=video_scene,
            )
            print(f"✅ 视频生成成功: {video_path}")
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
        }
    
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
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = NovelVideoGenerator(config_path=args.config)
    
    # 生成视频
    result = generator.generate(
        prompt=args.prompt,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
    )
    
    print("\n生成完成！")
    print(f"图像: {result['image']}")
    print(f"视频: {result['video']}")


if __name__ == "__main__":
    main()

