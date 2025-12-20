#!/usr/bin/env python3
# ⚡ 关键修复：设置 PyTorch CUDA allocator 为可扩展段模式（解决显存碎片化问题）
# 这必须在导入任何 torch 模块之前设置
import os
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# -*- coding: utf-8 -*-
"""
小说推文批量生成工具

功能：
1. 批量处理 JSON 场景文件
2. 支持多场景并行/串行生成
3. 自动错误重试
4. 生成详细报告
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback
import yaml
import gc

# 尝试导入 torch（如果可用）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generate_novel_video import NovelVideoGenerator


class BatchNovelGenerator:
    """批量小说推文生成器
    
    三阶段流程：
    1. 阶段1：批量生成所有图片
    2. 阶段2：批量生成所有配音，并获取实际时长
    3. 阶段3：根据配音时长批量生成所有视频
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化批量生成器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 解析配置文件路径
        if config_path is None:
            config_path = project_root / "config.yaml"
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化生成器
        self.generator = NovelVideoGenerator(str(self.config_path))
        
        # 初始化 TTS 生成器（用于阶段2：配音生成）
        self.tts_generator = None
        try:
            from tts_generator import TTSGenerator
            self.tts_generator = TTSGenerator(str(self.config_path))
            print("  ✓ TTS 生成器已加载")
        except Exception as e:
            print(f"  ⚠ TTS 生成器加载失败: {e}，将跳过配音生成")
        
        self.results = []
        self.errors = []
        
    def load_scenes_from_json(self, json_path: Path, auto_convert_v21: bool = True) -> List[Dict[str, Any]]:
        """
        从 JSON 文件加载场景列表
        
        Args:
            json_path: JSON 文件路径
            auto_convert_v21: 是否自动将v2格式转换为v2.1-exec
        
        Returns:
            场景列表
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenes = data.get('scenes', [])
        print(f"  ✓ 从 {json_path} 加载了 {len(scenes)} 个场景")
        
        # ⚡ v2.1-exec支持：检测并转换v2格式
        converted_count = 0
        for i, scene in enumerate(scenes):
            scene_version = scene.get('version', '')
            
            # 如果是v2格式且启用自动转换
            if scene_version == 'v2' and auto_convert_v21:
                try:
                    from utils.json_v2_to_v21_converter import JSONV2ToV21Converter
                    converter = JSONV2ToV21Converter()
                    scenes[i] = converter.convert_scene(scene)
                    converted_count += 1
                    print(f"  ℹ 场景 {scene.get('scene_id', i)}: v2 → v2.1-exec 转换完成")
                except Exception as e:
                    print(f"  ⚠ 场景 {scene.get('scene_id', i)} 转换失败: {e}，使用原始格式")
            # 如果已经是v2.1-exec格式，直接使用
            elif scene_version.startswith('v2.1'):
                print(f"  ✓ 场景 {scene.get('scene_id', i)}: 已是v2.1-exec格式")
        
        if converted_count > 0:
            print(f"  ✓ 共转换 {converted_count} 个场景为v2.1-exec格式")
        
        return scenes
    
    def extract_prompt_from_scene(self, scene: Dict[str, Any]) -> str:
        """
        从场景字典中提取 prompt
        
        Args:
            scene: 场景字典
        
        Returns:
            提示词字符串
        """
        # ⚡ 关键修复：确保包含 character.pose 信息（v2 格式）
        # 如果场景中有 character.pose，需要将其包含在 prompt 中，以便 LLM 正确识别姿态
        character = scene.get('character', {})
        character_pose = character.get('pose', '')
        
        # 尝试多种方式提取 prompt
        prompt_parts = []
        
        # 1. 从 visual_constraints 提取
        visual = scene.get('visual_constraints', {})
        if isinstance(visual, dict):
            environment = visual.get('environment', '')
            if environment:
                prompt_parts.append(environment)
        
        # 2. 从 narration 提取（⚠️ 注意：不要直接使用旁白文本，避免在图像中渲染文字）
        # narration 是语音旁白，不应该出现在视觉 prompt 中
        # 如果需要从旁白中提取视觉描述，应该使用更智能的提取方式
        # 暂时跳过 narration，避免文字出现在图像中
        
        # 3. 从 character 提取
        character = scene.get('character', {})
        if character.get('present', False):
            character_id = character.get('id', '')
            if character_id == 'hanli':
                prompt_parts.insert(0, "韩立")
            
            # ⚡ 关键修复：包含 character.pose 信息，确保 LLM 能识别姿态
            character_pose = character.get('pose', '')
            if character_pose:
                # 将 pose 转换为自然语言描述
                pose_descriptions = {
                    'lying_motionless': 'lying motionless on the ground',
                    'lying': 'lying on the ground',
                    'sitting': 'sitting',
                    'standing': 'standing',
                    'walking': 'walking',
                    'running': 'running',
                }
                pose_desc = pose_descriptions.get(character_pose.lower(), character_pose)
                # 将姿态描述添加到 prompt 中（放在角色名之后）
                if character_id == 'hanli' and len(prompt_parts) > 0:
                    # 如果已经有"韩立"，在它后面添加姿态描述
                    prompt_parts[0] = f"韩立, {pose_desc}"
                else:
                    prompt_parts.append(pose_desc)
        
        # 4. 从其他字段提取
        if not prompt_parts:
            # 尝试从其他字段提取
            description = scene.get('description', '')
            if description:
                prompt_parts.append(description)
            else:
                prompt_parts.append("一个仙侠场景")
        
        return ", ".join(prompt_parts) if prompt_parts else "一个仙侠场景"
    
    def generate_scene(
        self,
        scene: Dict[str, Any],
        output_base_dir: Path,
        scene_index: int,
        total_scenes: int,
        enable_m6: bool = True,
        quick_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        生成单个场景
        
        Args:
            scene: 场景字典
            output_base_dir: 输出基础目录
            scene_index: 场景索引
            total_scenes: 总场景数
            enable_m6: 是否启用 M6 身份验证
            quick_mode: 快速模式（减少帧数）
        
        Returns:
            生成结果字典
        """
        scene_id = scene.get('scene_id', scene_index)
        print(f"\n{'='*60}")
        print(f"生成场景 {scene_index + 1}/{total_scenes} (ID: {scene_id})")
        print(f"{'='*60}")
        
        # 提取 prompt
        prompt = self.extract_prompt_from_scene(scene)
        print(f"  提示词: {prompt[:100]}...")
        
        # 提取场景参数
        character = scene.get('character', {})
        character_present = character.get('present', False)
        character_id = character.get('id') if character_present else None
        
        camera = scene.get('camera', {})
        shot_type = camera.get('shot', 'medium')
        
        quality_target = scene.get('quality_target', {})
        motion_intensity = quality_target.get('motion_intensity', 'moderate')
        
        # 构建输出目录
        scene_output_dir = output_base_dir / f"scene_{scene_id:03d}"
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成参数
        width = scene.get('width', 768)
        height = scene.get('height', 1152)
        fps = scene.get('target_fps', 24) or 24
        
        # ⚡ 关键修复：根据配音时长（duration_sec）计算帧数
        # 优先级：duration_sec > num_frames > 默认值
        duration_sec = scene.get('duration_sec')
        if duration_sec:
            # 根据配音时长计算帧数：帧数 = 时长(秒) × 帧率
            calculated_frames = int(duration_sec * fps)
            if quick_mode:
                # 快速模式：至少24帧，但不超过计算值
                num_frames = max(24, min(calculated_frames, 60))  # 快速模式最多60帧
            else:
                num_frames = calculated_frames
            print(f"  ℹ 根据配音时长计算: {duration_sec}秒 × {fps}fps = {num_frames}帧")
        else:
            # 如果没有 duration_sec，使用 num_frames 或默认值
            num_frames = 24 if quick_mode else scene.get('num_frames', 120)
            print(f"  ⚠ 未找到 duration_sec，使用默认帧数: {num_frames}帧")
        
        print(f"  参数: {width}x{height}, {num_frames}帧, {fps}fps (时长: {num_frames/fps:.2f}秒)")
        print(f"  镜头: {shot_type}, 运动强度: {motion_intensity}")
        if character_present:
            print(f"  角色: {character_id} (M6: {'启用' if enable_m6 else '禁用'})")
        
        try:
            # 生成视频
            result = self.generator.generate(
                prompt=prompt,
                output_dir=scene_output_dir,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                scene=scene,
                include_character=character_present,
                character_id=character_id,
                auto_character=True,
                enable_m6_identity=enable_m6 if character_present else False,
                auto_m6_identity=enable_m6,
                shot_type=shot_type,
                motion_intensity=motion_intensity,
                m6_quick=quick_mode,
            )
            
            print(f"  ✅ 生成成功!")
            print(f"     图片: {result.get('image')}")
            if 'video' in result:
                print(f"     视频: {result.get('video')}")
            
            return {
                'scene_id': scene_id,
                'scene_index': scene_index,
                'status': 'success',
                'prompt': prompt,
                'result': result,
                'error': None,
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ 生成失败: {error_msg}")
            traceback.print_exc()
            
            return {
                'scene_id': scene_id,
                'scene_index': scene_index,
                'status': 'error',
                'prompt': prompt,
                'result': None,
                'error': error_msg,
            }
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        try:
            from video_composer import VideoComposer
            composer = VideoComposer(str(self.config_path))
            return composer.get_media_duration(audio_path)
        except Exception as e:
            print(f"  ⚠ 无法获取音频时长 {audio_path}: {e}")
            return 0.0
    
    def generate_batch(
        self,
        json_path: Path,
        output_dir: Path,
        enable_m6: bool = True,
        quick_mode: bool = False,
        max_retries: int = 2,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        批量生成场景（三阶段流程）
        
        阶段1：批量生成所有图片
        阶段2：批量生成所有配音，并获取实际时长
        阶段3：根据配音时长批量生成所有视频
        
        Args:
            json_path: JSON 场景文件路径
            output_dir: 输出目录
            enable_m6: 是否启用 M6 身份验证
            quick_mode: 快速模式
            max_retries: 最大重试次数
            start_index: 开始索引（用于断点续传）
            end_index: 结束索引（用于分批处理）
        
        Returns:
            批量生成结果
        """
        print("="*60)
        print("小说推文批量生成（三阶段流程）")
        print("="*60)
        
        # 加载场景
        scenes = self.load_scenes_from_json(json_path)
        
        # 过滤场景范围
        if end_index is None:
            end_index = len(scenes)
        scenes = scenes[start_index:end_index]
        
        print(f"\n生成范围: {start_index} - {end_index-1} (共 {len(scenes)} 个场景)")
        print(f"输出目录: {output_dir}")
        print(f"M6 身份验证: {'启用' if enable_m6 else '禁用'}")
        print(f"快速模式: {'是' if quick_mode else '否'}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        audios_dir = output_dir / "audios"
        audios_dir.mkdir(parents=True, exist_ok=True)
        
        # ==========================================
        # 阶段1：批量生成所有图片
        # ==========================================
        print("\n" + "="*60)
        print("阶段1：批量生成所有图片")
        print("="*60)
        
        image_results = []
        for i, scene in enumerate(scenes):
            scene_index = start_index + i
            scene_id = scene.get('scene_id', scene_index)
            scene_output_dir = output_dir / f"scene_{scene_id:03d}"
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[阶段1] 生成图片 {i+1}/{len(scenes)} (场景ID: {scene_id})")
            
            # 提取 prompt
            prompt = self.extract_prompt_from_scene(scene)
            
            # 提取场景参数
            character = scene.get('character', {})
            character_present = character.get('present', False)
            character_id = character.get('id') if character_present else None
            
            camera = scene.get('camera', {})
            shot_type = camera.get('shot', 'medium')
            
            quality_target = scene.get('quality_target', {})
            motion_intensity = quality_target.get('motion_intensity', 'moderate')
            
            width = scene.get('width', 768)
            height = scene.get('height', 1152)
            
            try:
                # 阶段1：只生成图片（不生成视频）
                # 直接使用 image_generator，避免生成视频
                image_output_path = scene_output_dir / "novel_image.png"
                
                # 构建 scene 字典用于图片生成
                image_scene = scene.copy() if scene else {}
                image_scene['width'] = width
                image_scene['height'] = height
                if character_present:
                    image_scene.setdefault("character", {})
                    if isinstance(image_scene.get("character"), dict):
                        if character_id:
                            image_scene["character"].setdefault("id", character_id)
                image_scene.setdefault("motion_intensity", motion_intensity)
                
                # ⚡ v2.1-exec支持：如果scene是v2.1-exec格式，使用v2.1流程
                scene_version = scene.get('version', '')
                if scene_version.startswith('v2.1'):
                    # 使用v2.1-exec流程
                    print(f"  ℹ 使用v2.1-exec模式生成")
                    try:
                        result = self.generator.generate(
                            prompt=prompt,
                            output_dir=scene_output_dir,
                            width=width,
                            height=height,
                            num_frames=24,  # 阶段1只生成图片，帧数不重要
                            fps=24,
                            scene=scene,  # 传入完整的v2.1-exec格式scene
                            use_v21_exec=True,  # 启用v2.1-exec模式
                        )
                        if result and result.get('image'):
                            image_path = result['image']
                        else:
                            raise ValueError("v2.1-exec模式生成失败")
                    except Exception as e:
                        print(f"  ⚠ v2.1-exec模式失败: {e}，回退到原有流程")
                        # 回退到原有流程
                        image_path = self.generator.image_generator.generate_image(
                            prompt=prompt,
                            output_path=image_output_path,
                            scene=image_scene,
                        )
                else:
                    # 原有流程
                    image_path = self.generator.image_generator.generate_image(
                        prompt=prompt,
                        output_path=image_output_path,
                        scene=image_scene,
                    )
                
                if image_path and Path(image_path).exists():
                    print(f"  ✅ 图片生成成功: {image_path}")
                    image_results.append({
                        'scene_id': scene_id,
                        'scene_index': scene_index,
                        'image_path': image_path,
                        'status': 'success'
                    })
                else:
                    print(f"  ❌ 图片生成失败")
                    image_results.append({
                        'scene_id': scene_id,
                        'scene_index': scene_index,
                        'image_path': None,
                        'status': 'error'
                    })
                
                # ⚡ 关键修复：每张图片生成后清理显存，避免第二张图片卡住
                print(f"  🧹 清理显存...")
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    # 多次清理，确保显存真正释放
                    for _ in range(3):
                        torch.cuda.empty_cache()
                        gc.collect()
                    torch.cuda.synchronize()
                    
                    # 显示清理后的显存状态
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"  ℹ 清理后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
                
                print(f"  ✓ 显存清理完成")
                
            except Exception as e:
                print(f"  ❌ 图片生成异常: {e}")
                traceback.print_exc()
                image_results.append({
                    'scene_id': scene_id,
                    'scene_index': scene_index,
                    'image_path': None,
                    'status': 'error',
                    'error': str(e)
                })
                
                # 即使失败也要清理显存
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # ==========================================
        # 阶段2：批量生成所有配音，并获取实际时长
        # ==========================================
        print("\n" + "="*60)
        print("阶段2：批量生成所有配音，并获取实际时长")
        print("="*60)
        
        audio_durations = {}
        if self.tts_generator is None:
            print("  ⚠ TTS 生成器未加载，跳过配音生成")
            print("  ⚠ 将使用 JSON 中的 duration_sec（如果存在）")
        else:
            for i, scene in enumerate(scenes):
                scene_index = start_index + i
                scene_id = scene.get('scene_id', scene_index)
                
                print(f"\n[阶段2] 生成配音 {i+1}/{len(scenes)} (场景ID: {scene_id})")
                
                # 提取旁白文本
                narration = scene.get('narration', {})
                if isinstance(narration, dict):
                    narration_text = narration.get('text', '')
                else:
                    narration_text = str(narration) if narration else ''
                
                if not narration_text:
                    print(f"  ⚠ 无旁白文本，跳过")
                    continue
                
                # 生成配音
                audio_path = audios_dir / f"audio_scene_{scene_id:03d}.wav"
                try:
                    self.tts_generator.generate(narration_text, str(audio_path))
                    print(f"  ✅ 配音生成成功: {audio_path}")
                    
                    # 获取实际音频时长
                    duration = self._get_audio_duration(str(audio_path))
                    if duration > 0:
                        audio_durations[scene_id] = duration
                        print(f"  ✅ 音频时长: {duration:.3f}秒")
                    else:
                        print(f"  ⚠ 无法获取音频时长，使用 JSON 中的 duration_sec")
                except Exception as e:
                    print(f"  ❌ 配音生成失败: {e}")
                    traceback.print_exc()
        
        # 阶段2完成统计
        print(f"\n[阶段2完成] 配音生成统计:")
        print(f"  成功: {len(audio_durations)} 个场景")
        print(f"  失败: {len(scenes) - len(audio_durations)} 个场景")
        if audio_durations:
            total_duration = sum(audio_durations.values())
            avg_duration = total_duration / len(audio_durations)
            print(f"  总时长: {total_duration:.2f}秒")
            print(f"  平均时长: {avg_duration:.2f}秒")
        
        # ==========================================
        # 阶段3：根据配音时长批量生成所有视频
        # ==========================================
        print("\n" + "="*60)
        print("阶段3：根据配音时长批量生成所有视频")
        print("="*60)
        
        results = []
        for i, scene in enumerate(scenes):
            scene_index = start_index + i
            scene_id = scene.get('scene_id', scene_index)
            
            print(f"\n[阶段3] 生成视频 {i+1}/{len(scenes)} (场景ID: {scene_id})")
            
            # 检查图片是否生成成功
            image_result = image_results[i] if i < len(image_results) else None
            if not image_result or image_result['status'] != 'success' or not image_result.get('image_path'):
                print(f"  ⚠ 图片未生成，跳过视频生成")
                results.append({
                    'scene_id': scene_id,
                    'scene_index': scene_index,
                    'status': 'error',
                    'error': '图片未生成'
                })
                continue
            
            image_path = image_result['image_path']
            
            # 获取配音时长（优先级：实际音频时长 > JSON duration_sec > 默认值）
            fps = scene.get('target_fps', 24) or 24
            duration_sec = None
            
            if scene_id in audio_durations:
                duration_sec = audio_durations[scene_id]
                print(f"  ℹ 使用实际配音时长: {duration_sec:.3f}秒")
            elif scene.get('duration_sec'):
                duration_sec = scene.get('duration_sec')
                print(f"  ℹ 使用 JSON 中的 duration_sec: {duration_sec}秒")
            else:
                print(f"  ⚠ 未找到配音时长，使用默认值")
            
            # 计算帧数
            if duration_sec:
                calculated_frames = int(duration_sec * fps)
                if quick_mode:
                    num_frames = max(24, min(calculated_frames, 60))
                else:
                    num_frames = calculated_frames
                print(f"  ℹ 计算帧数: {duration_sec:.3f}秒 × {fps}fps = {num_frames}帧")
            else:
                num_frames = 24 if quick_mode else scene.get('num_frames', 120)
                print(f"  ⚠ 使用默认帧数: {num_frames}帧")
            
            # 提取场景参数
            character = scene.get('character', {})
            character_present = character.get('present', False)
            character_id = character.get('id') if character_present else None
            
            camera = scene.get('camera', {})
            shot_type = camera.get('shot', 'medium')
            
            quality_target = scene.get('quality_target', {})
            motion_intensity = quality_target.get('motion_intensity', 'moderate')
            
            width = scene.get('width', 768)
            height = scene.get('height', 1152)
            
            scene_output_dir = output_dir / f"scene_{scene_id:03d}"
            
            # 生成视频（带重试）
            result = None
            for retry in range(max_retries + 1):
                if retry > 0:
                    print(f"  🔄 重试 {retry}/{max_retries}...")
                
                try:
                    # ⚡ 关键修复：在视频生成前，清理图片生成器留下的模型（SDXL pipeline）
                    # 避免显存碎片化导致 HunyuanVideo 加载失败
                    if hasattr(self.generator, 'image_generator') and self.generator.image_generator is not None:
                        print("  🔧 清理图片生成器模型以释放显存...")
                        try:
                            # 清理 SDXL pipeline
                            if hasattr(self.generator.image_generator, 'pipeline') and self.generator.image_generator.pipeline is not None:
                                try:
                                    self.generator.image_generator.pipeline.to("cpu")
                                    del self.generator.image_generator.pipeline
                                    self.generator.image_generator.pipeline = None
                                except:
                                    pass
                            if hasattr(self.generator.image_generator, 'sdxl_pipeline') and self.generator.image_generator.sdxl_pipeline is not None:
                                try:
                                    self.generator.image_generator.sdxl_pipeline.to("cpu")
                                    del self.generator.image_generator.sdxl_pipeline
                                    self.generator.image_generator.sdxl_pipeline = None
                                except:
                                    pass
                            # 清理增强生成器
                            if hasattr(self.generator.image_generator, 'enhanced_generator') and self.generator.image_generator.enhanced_generator is not None:
                                try:
                                    if hasattr(self.generator.image_generator.enhanced_generator, '_unload_all_models'):
                                        self.generator.image_generator.enhanced_generator._unload_all_models()
                                except:
                                    pass
                            # 清理 GPU 缓存
                            if TORCH_AVAILABLE:
                                # ⚡ 关键修复：torch 已在文件顶部导入，不需要再次导入
                                # 如果 TORCH_AVAILABLE 为 True，torch 已经全局可用
                                for _ in range(3):
                                    torch.cuda.empty_cache()
                                    gc.collect()
                            print("  ✓ 图片生成器模型已清理")
                        except Exception as e:
                            print(f"  ⚠ 清理图片生成器模型时出错: {e}")
                    
                    # 阶段3：使用已生成的图片生成视频
                    # 传入已生成的图片路径，避免重新生成
                    video_output_path = scene_output_dir / "novel_video.mp4"
                    
                    # 构建视频场景参数
                    video_scene = scene.copy() if scene else {}
                    video_scene['width'] = width
                    video_scene['height'] = height
                    video_scene['motion_intensity'] = motion_intensity
                    
                    # 使用视频生成器直接生成视频（图片已存在）
                    if character_present and enable_m6:
                        # 使用 M6 视频生成器
                        if self.generator._m6_video_generator is None:
                            from enhanced_video_generator_m6 import EnhancedVideoGeneratorM6
                            self.generator._m6_video_generator = EnhancedVideoGeneratorM6(str(self.generator.config_path))
                        
                        # 查找参考图
                        reference_image = None
                        if character_id == 'hanli':
                            ref_candidates = [
                                project_root / "reference_image" / "hanli_mid.jpg",
                                project_root / "reference_image" / "hanli_mid.png",
                            ]
                            for ref_candidate in ref_candidates:
                                if ref_candidate.exists():
                                    reference_image = str(ref_candidate)
                                    break
                        
                        if not reference_image:
                            reference_image = image_path  # 使用生成的图片作为参考
                        
                        # 从配置中获取 M6 最大重试次数
                        m6_max_retries_config = self.config.get('identity_verification', {}).get('max_retries', 3)
                        # ⚡ 关键修复：确保 image_path 是字符串类型
                        image_path_str = str(image_path) if image_path else None
                        video_path, m6_result = self.generator._m6_video_generator.generate_video_with_identity_check(
                            image_path=image_path_str,
                            output_path=str(video_output_path),
                            reference_image=reference_image,
                            scene=video_scene,
                            shot_type=shot_type,
                            enable_verification=True,
                            max_retries=m6_max_retries_config,
                            num_frames=num_frames,
                            fps=fps,
                        )
                    else:
                        # 使用普通视频生成器
                        # ⚡ 关键修复：确保 image_path 是字符串类型
                        image_path_str = str(image_path) if image_path else None
                        video_path = self.generator.video_generator.generate_video(
                            image_path=image_path_str,
                            output_path=str(video_output_path),
                            num_frames=num_frames,
                            fps=fps,
                            scene=video_scene,
                        )
                    
                    # 检查视频是否生成成功
                    if video_path and Path(video_path).exists():
                        print(f"  ✅ 视频生成成功: {video_path}")
                        result = {
                            'scene_id': scene_id,
                            'scene_index': scene_index,
                            'status': 'success',
                            'image': str(image_path) if image_path else None,  # ⚡ 修复：转换为字符串
                            'video': str(video_path) if video_path else None,  # ⚡ 修复：转换为字符串
                            'audio_duration': duration_sec,
                            'num_frames': num_frames,
                        }
                        break  # 成功，退出重试循环
                    else:
                        print(f"  ❌ 视频生成失败：文件不存在")
                        result = {
                            'scene_id': scene_id,
                            'scene_index': scene_index,
                            'status': 'error',
                            'error': '视频文件未生成'
                        }
                except Exception as e:
                    print(f"  ❌ 视频生成异常: {e}")
                    traceback.print_exc()
                    result = {
                        'scene_id': scene_id,
                        'scene_index': scene_index,
                        'status': 'error',
                        'error': str(e)
                    }
            
            results.append(result)
            
            # 保存中间结果
            if (i + 1) % 5 == 0:
                self._save_progress(output_dir, results, scenes)
        
        # 保存最终结果
        self._save_progress(output_dir, results, scenes)
        
        # 生成报告
        report = self._generate_report(results, output_dir)
        
        return {
            'results': results,
            'report': report,
        }
    
    def _save_progress(self, output_dir: Path, results: List[Dict], scenes: List[Dict]):
        """保存进度"""
        progress_file = output_dir / "progress.json"
        
        # ⚡ 关键修复：将 results 中的 PosixPath 转换为字符串
        def convert_paths(obj):
            """递归转换 PosixPath 为字符串"""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_paths(results)
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_scenes': len(scenes),
                'completed': len(results),
                'results': serializable_results,
            }, f, ensure_ascii=False, indent=2)
    
    def _generate_report(self, results: List[Dict], output_dir: Path) -> Dict[str, Any]:
        """生成报告"""
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        # 统计错误
        error_details = []
        for r in results:
            if r['status'] == 'error':
                error_details.append({
                    'scene_id': r.get('scene_id', 'unknown'),
                    'prompt': r.get('prompt', 'N/A')[:50] + '...' if r.get('prompt') else 'N/A',
                    'error': r['error'],
                })
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'success': success,
                'errors': errors,
                'success_rate': f"{success_rate:.1f}%",
            },
            'errors': error_details,
        }
        
        # 保存报告
        report_file = output_dir / "batch_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成 Markdown 报告
        md_report = self._generate_markdown_report(report, results)
        md_file = output_dir / "batch_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"\n{'='*60}")
        print("批量生成完成")
        print(f"{'='*60}")
        print(f"总计: {total}")
        print(f"成功: {success} ({success_rate:.1f}%)")
        print(f"失败: {errors}")
        print(f"\n报告已保存:")
        print(f"  JSON: {report_file}")
        print(f"  Markdown: {md_file}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict, results: List[Dict]) -> str:
        """生成 Markdown 格式报告"""
        md = f"""# 小说推文批量生成报告

生成时间: {report['timestamp']}

## 摘要

- **总计**: {report['summary']['total']} 个场景
- **成功**: {report['summary']['success']} 个
- **失败**: {report['summary']['errors']} 个
- **成功率**: {report['summary']['success_rate']}

## 失败场景详情

"""
        if report['errors']:
            for error in report['errors']:
                md += f"### 场景 {error['scene_id']}\n\n"
                md += f"- **提示词**: {error['prompt']}\n"
                md += f"- **错误**: {error['error']}\n\n"
        else:
            md += "无失败场景 ✅\n"
        
        md += "\n## 成功场景列表\n\n"
        for r in results:
            if r['status'] == 'success':
                md += f"- 场景 {r['scene_id']}: {r['prompt'][:50]}...\n"
        
        return md


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="小说推文批量生成工具")
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='JSON 场景文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认: outputs/batch_novel_<timestamp>）'
    )
    parser.add_argument(
        '--enable-m6',
        action='store_true',
        default=True,
        help='启用 M6 身份验证（默认: 启用）'
    )
    parser.add_argument(
        '--disable-m6',
        action='store_true',
        help='禁用 M6 身份验证'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式（减少帧数，用于测试）'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='开始索引（用于断点续传）'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='结束索引（用于分批处理）'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='最大重试次数（默认: 2）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    # 如果是在 gen_video 目录下执行，相对路径应该相对于 gen_video 目录
    json_path_str = args.json
    json_path = Path(json_path_str)
    
    if not json_path.is_absolute():
        # 处理相对路径
        # 如果路径以 ../ 开头，从 gen_video 目录向上查找
        # 否则，相对于 gen_video 目录
        if json_path_str.startswith('../'):
            # 去掉 ../ 前缀，然后从 fanren 目录开始
            relative_path = json_path_str[3:]  # 去掉 '../'
            json_path = project_root.parent / relative_path
        else:
            # 相对于 gen_video 目录
            json_path = project_root / json_path
        
        # 规范化路径（处理 .. 和 .）
        json_path = json_path.resolve()
    
    # 解析输出目录路径
    if args.output_dir:
        output_dir_str = args.output_dir
        output_dir = Path(output_dir_str)
        if not output_dir.is_absolute():
            # 处理相对路径
            if output_dir_str.startswith('../'):
                # 去掉 ../ 前缀，然后从 fanren 目录开始
                relative_path = output_dir_str[3:]  # 去掉 '../'
                output_dir = project_root.parent / relative_path
            else:
                # 相对于 gen_video 目录
                output_dir = project_root / output_dir
            output_dir = output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / f"batch_novel_{timestamp}"
    
    # M6 设置
    enable_m6 = args.enable_m6 and not args.disable_m6
    
    # 创建生成器
    generator = BatchNovelGenerator()
    
    # 批量生成
    result = generator.generate_batch(
        json_path=json_path,
        output_dir=output_dir,
        enable_m6=enable_m6,
        quick_mode=args.quick,
        max_retries=args.max_retries,
        start_index=args.start,
        end_index=args.end,
    )
    
    # 返回状态码
    success_count = result['report']['summary']['success']
    total_count = result['report']['summary']['total']
    
    if success_count == total_count:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

