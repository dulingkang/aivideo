#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成器
使用 Stable Video Diffusion (SVD) 从图像生成视频
支持静态图像动画（Ken Burns效果）和SVD视频生成
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import imageio
import json

# 优先使用项目中的diffusers（支持HunyuanVideo 1.5）
# diffusers现在在项目根目录（与gen_video平级）
_diffusers_path = Path(__file__).parent.parent / "diffusers" / "src"
if _diffusers_path.exists():
    sys.path.insert(0, str(_diffusers_path))

try:
    from scene_motion_analyzer import SceneMotionAnalyzer
    from deforum_camera_motion import DeforumCameraMotion
    from svd_parameter_generator import SVDParameterGenerator
    from scene_type_classifier import SceneTypeClassifier
    from utils.model_router import ModelRouter
except ImportError:
    # 如果导入失败，创建占位类
    class SceneMotionAnalyzer:
        def analyze(self, scene): return {}
    class DeforumCameraMotion:
        def generate_motion_params_from_scene(self, scene): return {}
        def apply_camera_motion(self, image, num_frames, motion_params, curve): return []
    class SVDParameterGenerator:
        pass
    class SceneTypeClassifier:
        pass
    class ModelRouter:
        def __init__(self, config): pass
        def get_model_for_user(self, user_tier, scene_type): return None


class VideoGenerator:
    """视频生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化视频生成器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path.cwd() / self.config_path).resolve()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.video_config = self.config.get('video', {})
        self.config_dir = self.config_path.parent
        
        # 初始化智能分析器
        self.motion_analyzer = SceneMotionAnalyzer()
        self.deforum_motion = DeforumCameraMotion()
        self.svd_param_generator = SVDParameterGenerator()
        self.scene_classifier = SceneTypeClassifier()
        
        # 初始化Prompt Engine（专业级Prompt工程系统）
        try:
            from utils.prompt_engine import PromptEngine
            prompt_engine_config = self.video_config.get('prompt_engine', {})
            use_llm = prompt_engine_config.get('use_llm_rewriter', False)
            self.prompt_engine = PromptEngine(
                config_path=None,  # 使用默认配置
                use_llm_rewriter=use_llm,
                llm_api=None  # TODO: 如果需要LLM，可以在这里传入API
            )
            print("  ✓ Prompt Engine已初始化")
        except ImportError as e:
            print(f"  ⚠ Prompt Engine导入失败: {e}，将使用基础prompt构建")
            self.prompt_engine = None
        
        # 初始化模型路由器（用于自动选择模型）
        try:
            self.model_router = ModelRouter(self.video_config)
            print("  ✓ 模型路由器已初始化")
        except Exception as e:
            print(f"  ⚠ 模型路由器初始化失败: {e}，将使用配置中的model_type")
            self.model_router = None
        
        # 模型相关
        self.pipeline = None
        self.hunyuanvideo_pipeline = None  # HunyuanVideo pipeline
        self.cogvideox_pipeline = None  # CogVideoX pipeline
        self.model_loaded = False
        
        # RIFE 插帧相关
        self.rife_enabled = self.video_config.get('rife', {}).get('enabled', False)
        self.rife_interpolation_scale = self.video_config.get('rife', {}).get('interpolation_scale', 2.0)
        self._rife_model = None
    
    def unload_model(self):
        """卸载模型，释放显存"""
        import torch
        import gc
        
        print("  ℹ 卸载模型，释放显存...")
        
        # 卸载HunyuanVideo pipeline
        if self.hunyuanvideo_pipeline is not None:
            try:
                # 尝试调用unload方法（如果存在）
                if hasattr(self.hunyuanvideo_pipeline, 'unload'):
                    self.hunyuanvideo_pipeline.unload()
                # 删除引用
                del self.hunyuanvideo_pipeline
                self.hunyuanvideo_pipeline = None
                print("  ✓ HunyuanVideo pipeline已卸载")
            except Exception as e:
                print(f"  ⚠ 卸载HunyuanVideo pipeline失败: {e}")
        
        # 卸载CogVideoX pipeline
        if self.cogvideox_pipeline is not None:
            try:
                # 尝试调用unload方法（如果存在）
                if hasattr(self.cogvideox_pipeline, 'unload'):
                    self.cogvideox_pipeline.unload()
                # 删除引用
                del self.cogvideox_pipeline
                self.cogvideox_pipeline = None
                print("  ✓ CogVideoX pipeline已卸载")
            except Exception as e:
                print(f"  ⚠ 卸载CogVideoX pipeline失败: {e}")
        
        # 卸载SVD/AnimateDiff pipeline
        if self.pipeline is not None:
            try:
                # 尝试调用unload方法（如果存在）
                if hasattr(self.pipeline, 'unload'):
                    self.pipeline.unload()
                # 删除引用
                del self.pipeline
                self.pipeline = None
                print("  ✓ SVD/AnimateDiff pipeline已卸载")
            except Exception as e:
                print(f"  ⚠ 卸载pipeline失败: {e}")
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  ℹ 卸载后显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
        
        self.model_loaded = False
    
    def load_model(self):
        """加载视频生成模型（支持SVD、AnimateDiff、HunyuanVideo和CogVideoX）"""
        import torch
        import gc
        
        # 如果已有模型加载，先卸载释放显存
        if self.model_loaded:
            print("  ℹ 检测到已有模型，先卸载释放显存...")
            self.unload_model()
        
        model_type = self.video_config.get('model_type', 'svd-xt')
        
        print(f"  加载视频生成模型: {model_type}")
        
        try:
            import torch
            
            if model_type == 'animatediff' or model_type == 'animatediff-sdxl':
                # 加载AnimateDiff模型（文生视频）
                model_path = self.video_config.get('model_path')
                if not model_path:
                    raise ValueError("video.model_path 未配置")
                self._load_animatediff_model(model_path)
            elif model_type in ['svd', 'svd-xt', 'svd-image-to-video']:
                # 加载SVD模型（图生视频）
                model_path = self.video_config.get('model_path')
                if not model_path:
                    raise ValueError("video.model_path 未配置")
                self._load_svd_model(model_path)
            elif model_type == 'hunyuanvideo':
                # 加载HunyuanVideo模型（图生视频）
                hunyuan_config = self.video_config.get('hunyuanvideo', {})
                model_path = hunyuan_config.get('model_path')
                if not model_path:
                    # 尝试从HuggingFace下载
                    model_path = "Tencent-Hunyuan/HunyuanVideo-ImageToVideo"
                    print(f"  ℹ 未配置本地路径，将使用HuggingFace模型: {model_path}")
                self._load_hunyuanvideo_model(model_path)
            elif model_type == 'cogvideox':
                # 加载CogVideoX模型（图生视频）
                cogvideox_config = self.video_config.get('cogvideox', {})
                model_path = cogvideox_config.get('model_path')
                if not model_path:
                    # 尝试从HuggingFace下载
                    model_path = "THUDM/CogVideoX-5b-I2V"
                    print(f"  ℹ 未配置本地路径，将使用HuggingFace模型: {model_path}")
                self._load_cogvideox_model(model_path)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.model_loaded = True
            print("  ✓ 模型加载成功")
            
        except Exception as e:
            print(f"  ✗ 模型加载失败: {e}")
            raise
    
    def _load_svd_model(self, model_path: str):
        """加载SVD模型"""
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image
        import torch
        
        # 加载SVD pipeline
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        # 移动到GPU
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.enable_model_cpu_offload()
    
    def _load_hunyuanvideo_model(self, model_path: str):
        """加载HunyuanVideo模型（图生视频）
        
        支持两个版本：
        - HunyuanVideo 1.5（推荐）：使用 HunyuanVideo15ImageToVideoPipeline
        - HunyuanVideo（原版）：使用 HunyuanVideoImageToVideoPipeline
        """
        import torch

        # 兼容补丁：某些版本的 transformer.forward 不接受 timestep_r（diffusers pipeline 可能会传入）
        # 该补丁尽量在 pipeline 加载后立刻注入，避免运行期 TypeError。
        def _patch_timestep_r_compat() -> None:
            try:
                import inspect

                pipe = self.hunyuanvideo_pipeline
                if pipe is None:
                    return
                transformer = getattr(pipe, "transformer", None)
                if transformer is None or not hasattr(transformer, "forward"):
                    return
                if getattr(transformer, "_timestep_r_compat_patched", False):
                    return

                # 若 forward 已显式支持 timestep_r，就不打补丁（避免引入额外开销）
                try:
                    sig = inspect.signature(transformer.forward)
                    if "timestep_r" in sig.parameters:
                        return
                except Exception:
                    # 签名不可用时，继续走“捕获 TypeError 再回退”的兜底补丁
                    pass

                orig_forward = transformer.forward

                def forward_compat(*args, **kwargs):
                    if "timestep_r" not in kwargs:
                        return orig_forward(*args, **kwargs)
                    try:
                        return orig_forward(*args, **kwargs)
                    except TypeError as e:
                        msg = str(e)
                        if "timestep_r" in msg and "unexpected keyword argument" in msg:
                            kwargs.pop("timestep_r", None)
                            return orig_forward(*args, **kwargs)
                        raise

                transformer.forward = forward_compat  # type: ignore[assignment]
                transformer._timestep_r_compat_patched = True  # type: ignore[attr-defined]
                print("  ✓ timestep_r 兼容补丁：transformer.forward 不支持时将自动忽略 timestep_r")
            except Exception as e:
                print(f"  ⚠ timestep_r 兼容补丁注入失败: {e}")
        
        # 检查是1.5版本还是原版
        hunyuan_config = self.video_config.get('hunyuanvideo', {})
        use_v15 = hunyuan_config.get('use_v15', True)  # 默认使用1.5版本
        
        # 从模型路径判断（如果包含"1.5"或"1_5"或"tencent/HunyuanVideo-1.5"，自动使用1.5版本）
        if ("1.5" in str(model_path) or "1_5" in str(model_path) or 
            "HunyuanVideo-1.5" in str(model_path) or 
            "tencent/HunyuanVideo-1.5" in str(model_path)):
            use_v15 = True
        
        # 如果是本地路径，检查model_index.json判断版本
        if Path(model_path).exists() and Path(model_path).is_dir():
            model_index_path = Path(model_path) / "model_index.json"
            if model_index_path.exists():
                try:
                    import json
                    with open(model_index_path, 'r') as f:
                        model_index = json.load(f)
                    class_name = model_index.get('_class_name', '')
                    # 检查是否是1.5版本
                    if 'HunyuanVideo15' in class_name or 'HunyuanVideo-1.5' in class_name:
                        use_v15 = True
                        print(f"  ℹ 从model_index.json检测到：1.5版本")
                    elif 'HunyuanVideo' in class_name:
                        use_v15 = False
                        print(f"  ℹ 从model_index.json检测到：原版")
                except Exception as e:
                    print(f"  ⚠ 无法读取model_index.json: {e}，使用配置的版本")
        
        print(f"  加载HunyuanVideo模型: {model_path}")
        print(f"  版本: {'1.5（推荐）' if use_v15 else '原版'}")
        
        try:
            # 检查diffusers是否支持1.5版本
            try:
                from diffusers import HunyuanVideo15ImageToVideoPipeline
                has_v15_support = True
            except ImportError:
                has_v15_support = False
                print("  ℹ diffusers版本不支持HunyuanVideo 1.5，将使用原版")
            
            if use_v15 and has_v15_support:
                # 使用HunyuanVideo 1.5（推荐）
                # 检查是否指定了transformer子目录（仅用于官方格式模型）
                transformer_subfolder = hunyuan_config.get('transformer_subfolder', None)
                
                # 判断是HuggingFace模型ID还是本地路径
                is_hf_model_id = not Path(model_path).exists() and "/" in str(model_path)
                
                if transformer_subfolder and not is_hf_model_id:
                    # 官方格式模型：需要指定transformer子目录
                    print(f"  ℹ 使用transformer子目录: {transformer_subfolder}")
                    from diffusers import HunyuanVideo15Transformer3DModel
                    # 重要：transformer也在CPU上加载，避免GPU显存分配
                    print(f"  ℹ 使用低显存模式加载transformer（避免一次性分配）...")
                    # 临时设置CUDA设备为CPU，确保transformer在CPU上加载
                    import os
                    old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                    try:
                        # 临时隐藏GPU，强制在CPU上加载
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                            model_path,
                            subfolder=transformer_subfolder,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True  # 降低CPU内存使用
                        )
                    finally:
                        # 恢复CUDA设置
                        if old_cuda_visible is None:
                            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                        else:
                            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
                    # 加载完整pipeline，使用指定的transformer
                    # 重要：使用low_cpu_mem_usage避免一次性分配显存，不指定device让模型在CPU上加载
                    print(f"  ℹ 使用低显存模式加载模型（避免一次性分配）...")
                    # 临时设置CUDA设备为CPU，确保模型在CPU上加载
                    import os
                    old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                    try:
                        # 临时隐藏GPU，强制在CPU上加载
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        try:
                            self.hunyuanvideo_pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                                model_path,
                                transformer=transformer,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True  # 降低CPU内存使用
                            )
                        except (ValueError, OSError) as e:
                            print(f"  ⚠ 加载pipeline失败: {e}")
                            print(f"  ℹ 尝试不使用transformer参数...")
                            self.hunyuanvideo_pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True  # 降低CPU内存使用
                            )
                            self.hunyuanvideo_pipeline.transformer = transformer
                    finally:
                        # 恢复CUDA设置
                        if old_cuda_visible is None:
                            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                        else:
                            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
                else:
                    # diffusers格式模型：直接加载完整pipeline（推荐）
                    if is_hf_model_id:
                        print(f"  ℹ 使用HuggingFace模型: {model_path}")
                        print(f"  ℹ 将自动下载所有必需组件...")
                    # 重要：使用low_cpu_mem_usage避免一次性分配显存，不指定device让模型在CPU上加载
                    print(f"  ℹ 使用低显存模式加载模型（避免一次性分配）...")
                    # 智能加载：如果有大量显存，直接加载到GPU；否则先到CPU再offload
                    has_large_vram = False
                    if torch.cuda.is_available():
                        try:
                            # 获取真实可用显存 (free, total)
                            free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info()
                            free_vram_gb = free_memory_bytes / 1024**3
                            total_vram_gb = total_memory_bytes / 1024**3
                            # HunyuanVideo 1.5 大约需要 35GB 显存 (fp16) 才能舒适地运行在GPU上
                            has_large_vram = free_vram_gb > 35
                            print(f"  ℹ 检测到显存: 总计{total_vram_gb:.2f}GB, 可用{free_vram_gb:.2f}GB (大显存模式: {has_large_vram})")
                        except Exception as e:
                            print(f"  ⚠ 显存检测失败: {e}, 默认使用保守模式")
                            has_large_vram = False

                    if has_large_vram:
                        self.hunyuanvideo_pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                        ).to("cuda")
                        print("  ✓ 已直接加载到GPU (高性能模式)")
                    else:
                        # 临时设置CUDA设备为CPU，确保模型在CPU上加载
                        import os
                        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                        try:
                            # 临时隐藏GPU，强制在CPU上加载
                            os.environ['CUDA_VISIBLE_DEVICES'] = ''
                            self.hunyuanvideo_pipeline = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True  # 降低CPU内存使用
                            )
                        finally:
                            # 恢复CUDA设置
                            if old_cuda_visible is None:
                                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                            else:
                                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
                print("  ✓ 使用HunyuanVideo 1.5（推荐版本）")
                _patch_timestep_r_compat()
            else:
                # 使用原版HunyuanVideo
                from diffusers import HunyuanVideoImageToVideoPipeline
                has_large_vram = False  # 默认假设
                # 尝试加载fp16变体，如果不存在则使用默认
                try:
                    self.hunyuanvideo_pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        variant="fp16"
                    )
                except (ValueError, OSError):
                    # 如果没有fp16变体，使用默认
                    self.hunyuanvideo_pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16
                    )
                print("  ✓ 使用HunyuanVideo（原版）")
                _patch_timestep_r_compat()

            # 使用激进的显存优化策略，避免一次性分配完显存
            # 重要：模型已经在CPU上加载（device_map="cpu"），现在配置按需加载到GPU
            if torch.cuda.is_available():
                # 设置显存限制（类似JAX，防止一次性分配完显存）
                # 注意：在多租户环境下，不要过分限制，否则可能导致OOM（如果PyTorch认为配额已满）
                max_memory_fraction = hunyuan_config.get('max_memory_fraction', 1.0) # 默认为不限制
                try:
                    # 只有在确实需要限制时才设置
                     if max_memory_fraction < 1.0:
                        torch.cuda.set_per_process_memory_fraction(max_memory_fraction, device=0)
                        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                        print(f"  ✓ 已设置显存限制: {max_memory_fraction*100:.0f}%")
                except Exception as e:
                    print(f"  ⚠ 设置显存限制失败: {e}")

                # 检查显存是否足够
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free_memory = total_memory - reserved

                print(f"  ℹ 模型加载后显存状态: 总计={total_memory:.2f}GB, 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 可用={free_memory:.2f}GB")

                # 检查是否有其他进程占用显存
                # 如果可用显存少于20GB，强制使用sequential CPU offload（更激进，只在需要时加载）
                force_cpu_offload = hunyuan_config.get('force_cpu_offload', False)
                # 由于模型已经在CPU上，使用model CPU offload来按需加载（比sequential更稳定）
                if not has_large_vram:  # 只有在显存不足时才使用CPU offload
                    print(f"  ℹ 模型已在CPU上加载，配置按需GPU加载（避免一次性分配显存）...")

                    # 使用标准的model CPU offload（比sequential更稳定，避免meta device问题）
                    try:
                        self.hunyuanvideo_pipeline.enable_model_cpu_offload()
                        print(f"  ✓ 已启用CPU offload（组件按需加载到GPU）")
                    except Exception as e:
                        print(f"  ⚠ CPU offload启用失败: {e}")
                        # 即使失败也不移到GPU，保持CPU状态
                        print(f"  ℹ 保持模型在CPU，运行时按需加载")
                
                # Attention优化（小说推文优化：优先使用FlashAttention2/SDPA，获得20-40%速度提升）
                try:
                    # 优先尝试启用FlashAttention2/SDPA（PyTorch 2.1+）
                    if torch.__version__ >= "2.1.0":
                        try:
                            # 启用Flash Attention SDPA（自动选择最优实现）
                            torch.backends.cuda.enable_flash_sdp(True)
                            torch.backends.cuda.enable_mem_efficient_sdp(True)
                            print("  ✓ 已启用FlashAttention2/SDPA优化（20-40%速度提升）")
                        except Exception as e:
                            print(f"  ⚠ 启用FlashAttention2/SDPA失败: {e}，回退到attention slicing")
                    
                    # 如果没有FlashAttention2，使用attention slicing作为备选
                    # 但在大显存模式下，尽量不要slice，除非显存真的很满
                    if hasattr(self.hunyuanvideo_pipeline, 'enable_attention_slicing'):
                        # 使用更激进的切片（使用数字1而不是"max"），最小化显存占用
                        # slice_size=1 表示每次只处理1个head，最省显存但最慢
                        # slice_size="max" 表示尽可能多的head，平衡显存和速度
                        # 对于显存紧张的情况，使用数字1更安全
                        # 只有在小显存模式下才开启slicing
                        if not has_large_vram:
                            slice_size = hunyuan_config.get('attention_slice_size', 8)  # 默认使用8，平衡速度和显存
                            if slice_size == "max":
                                # 如果设置为"max"，改为使用1（最激进的切片）
                                slice_size = 1
                            self.hunyuanvideo_pipeline.enable_attention_slicing(slice_size=slice_size)
                            print(f"  ✓ 已启用attention slicing（slice_size={slice_size}，减少attention计算显存占用）")
                except Exception as e:
                    print(f"  ⚠ Attention优化设置失败: {e}")
                    
                    # 尝试启用gradient checkpointing（如果支持）
                    try:
                        if hasattr(self.hunyuanvideo_pipeline, 'transformer') and hasattr(self.hunyuanvideo_pipeline.transformer, 'enable_gradient_checkpointing'):
                            self.hunyuanvideo_pipeline.transformer.enable_gradient_checkpointing()
                            print(f"  ✓ 已启用gradient checkpointing（进一步减少显存占用）")
                    except Exception as e:
                        # gradient checkpointing可能不支持，忽略错误
                        pass
                else:
                    # HunyuanVideo 1.5 更轻量，显存需求更低
                    if use_v15:
                        # 1.5版本：消费级GPU可跑，显存需求更低
                        if total_memory < 16:
                            print(f"  ⚠ 显存不足16GB ({total_memory:.1f}GB)，启用CPU offload")
                            self.hunyuanvideo_pipeline.enable_model_cpu_offload()
                        else:
                            self.hunyuanvideo_pipeline = self.hunyuanvideo_pipeline.to("cuda")
                            print(f"  ✓ 模型已加载到GPU ({total_memory:.1f}GB显存)")
                    else:
                        # 原版：显存需求较高
                        if total_memory < 24:
                            print(f"  ⚠ 显存不足24GB ({total_memory:.1f}GB)，启用CPU offload")
                            self.hunyuanvideo_pipeline.enable_model_cpu_offload()
                        else:
                            self.hunyuanvideo_pipeline = self.hunyuanvideo_pipeline.to("cuda")
                            print(f"  ✓ 模型已加载到GPU ({total_memory:.1f}GB显存)")
                
                # 启用VAE优化以降低显存占用（720p模型在VAE解码时显存占用很大）
                if hasattr(self.hunyuanvideo_pipeline, 'vae') and self.hunyuanvideo_pipeline.vae is not None:
                    # 同时启用tiling和slicing，最大化显存节省
                    try:
                        if hasattr(self.hunyuanvideo_pipeline.vae, 'enable_tiling'):
                            self.hunyuanvideo_pipeline.vae.enable_tiling()
                            print(f"  ✓ 已启用VAE tiling（降低VAE解码时的显存占用）")
                    except Exception as e:
                        print(f"  ⚠ 启用VAE tiling失败: {e}")
                    
                    try:
                        if hasattr(self.hunyuanvideo_pipeline.vae, 'enable_slicing'):
                            self.hunyuanvideo_pipeline.vae.enable_slicing()
                            print(f"  ✓ 已启用VAE slicing（进一步降低VAE解码时的显存占用）")
                    except Exception as e:
                        print(f"  ⚠ 启用VAE slicing失败: {e}")
            
            print("  ✓ HunyuanVideo模型加载成功")
            
        except Exception as e:
            print(f"  ✗ HunyuanVideo模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_cogvideox_model(self, model_path: str):
        """加载CogVideoX模型（图生视频）"""
        import torch
        from pathlib import Path
        
        print(f"  加载CogVideoX模型: {model_path}")
        
        try:
            from diffusers import CogVideoXImageToVideoPipeline
            from diffusers.utils import export_to_video
            
            # 获取CogVideoX配置
            cogvideox_config = self.video_config.get('cogvideox', {})
            
            # 判断是HuggingFace模型ID还是本地路径
            is_hf_model_id = not Path(model_path).exists() and "/" in str(model_path)
            
            if is_hf_model_id:
                print(f"  ℹ 使用HuggingFace模型: {model_path}")
                print(f"  ℹ 将自动下载所有必需组件...")
            
            # 检查是否启用CPU offload
            use_cpu_offload = cogvideox_config.get('enable_model_cpu_offload', True)
            
            # CogVideoX只支持balanced和cuda策略，不支持cpu
            # 如果启用CPU offload，使用balanced策略；否则使用cuda
            if use_cpu_offload:
                device_map_strategy = "balanced"
                print("  ℹ 使用balanced策略加载模型（支持CPU offload）...")
            else:
                device_map_strategy = "cuda"
                print("  ℹ 使用cuda策略加载模型...")
            
            # 加载CogVideoX pipeline
            # CogVideoX-5B 推荐使用 BF16 精度（根据官方文档）
            # BF16 可以提供更好的视频质量，避免纯色问题
            self.cogvideox_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # 从 float16 改为 bfloat16（CogVideoX-5B 推荐）
                low_cpu_mem_usage=True,  # 降低CPU内存使用
                device_map=device_map_strategy  # 使用balanced或cuda策略
            )
            print(f"  ✓ CogVideoX pipeline已加载（策略: {device_map_strategy}）")
            
            # 显存优化：设置显存限制
            if torch.cuda.is_available():
                max_memory_fraction = cogvideox_config.get('max_memory_fraction', 0.3)
                torch.cuda.set_per_process_memory_fraction(max_memory_fraction, device=0)
                print(f"  ✓ 已设置显存限制: {max_memory_fraction * 100}%")
            
            # 显存优化：启用CPU offload（如果配置启用）
            # 注意：当使用 device_map="cuda" 时，模型已经自动在 GPU 上，不能再使用 .to("cuda")
            if cogvideox_config.get('enable_model_cpu_offload', True):
                try:
                    # 如果使用 device_map="balanced"，可以启用 CPU offload
                    if device_map_strategy == "balanced":
                        self.cogvideox_pipeline.enable_model_cpu_offload()
                        print("  ✓ 已启用CPU offload（降低显存占用）")
                    else:
                        print("  ℹ 使用 device_map='cuda'，模型已在 GPU 上，无需 CPU offload")
                except Exception as e:
                    print(f"  ⚠ 启用CPU offload失败: {e}")
            else:
                # 使用 device_map="cuda" 时，模型已经自动在 GPU 上，无需手动移动
                if device_map_strategy == "cuda":
                    print("  ✓ 模型已通过 device_map='cuda' 自动加载到 GPU")
                else:
                    print("  ℹ 模型已通过 device_map 策略加载")
            
            # 显存优化：启用VAE tiling（如果配置启用）
            if cogvideox_config.get('enable_tiling', True):
                try:
                    if hasattr(self.cogvideox_pipeline, 'vae') and self.cogvideox_pipeline.vae is not None:
                        if hasattr(self.cogvideox_pipeline.vae, 'enable_tiling'):
                            self.cogvideox_pipeline.vae.enable_tiling()
                            print("  ✓ 已启用VAE tiling（降低VAE解码时的显存占用）")
                except Exception as e:
                    print(f"  ⚠ 启用VAE tiling失败: {e}")
            
            print("  ✓ CogVideoX模型加载成功")
            
        except Exception as e:
            print(f"  ✗ CogVideoX模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_animatediff_model(self, model_path: str):
        """加载AnimateDiff模型（文生视频）"""
        import torch
        from pathlib import Path
        
        # 注意：diffusers的AnimateDiff只支持SD1.5，不支持SDXL
        # 但我们可以尝试使用SDXL的motion module配合SD1.5 base model
        print("  ⚠ 注意：diffusers的AnimateDiff只支持SD1.5，不支持SDXL")
        print("  ℹ 将使用SD1.5 base model + SDXL motion module")
        
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            from diffusers.utils import export_to_video
            
            # 检查motion module路径
            motion_module_path = Path(model_path) / "mm_sdxl_v10_beta.ckpt"
            if not motion_module_path.exists():
                # 尝试其他可能的路径
                motion_module_path = Path(model_path) / "mm_sd_v15_v2.ckpt"
                if not motion_module_path.exists():
                    raise FileNotFoundError(f"未找到motion module: {motion_module_path}")
            
            print(f"  使用motion module: {motion_module_path}")
            
            # 加载motion adapter
            # 注意：SDXL的motion module可能与SD1.5不兼容，优先使用SD1.5的motion module
            try:
                # 优先尝试使用SD1.5的motion module（更稳定）
                print("  ℹ 尝试加载SD1.5 motion adapter...")
                adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
                print("  ✓ 使用SD1.5 motion adapter")
            except Exception as e:
                print(f"  ⚠ 无法从HuggingFace加载motion adapter: {e}")
                # 尝试加载本地SDXL motion module（可能不兼容，但可以尝试）
                try:
                    print(f"  ℹ 尝试加载本地motion module: {motion_module_path}")
                    adapter = MotionAdapter.from_single_file(str(motion_module_path))
                    print("  ✓ 使用本地SDXL motion module（可能不兼容，请测试）")
                except Exception as e2:
                    print(f"  ✗ 无法加载motion module: {e2}")
                    raise
            
            # 加载SD1.5 base model（因为diffusers不支持SDXL）
            base_model = "runwayml/stable-diffusion-v1-5"
            print(f"  使用base model: {base_model}")
            
            # 加载AnimateDiff pipeline
            print(f"  加载AnimateDiff pipeline...")
            # 根据分析，关键问题可能是：
            # 1. motion_scale未设置或设置错误
            # 2. FP16/FP32不匹配
            # 3. 必须使用AnimateDiffPipeline（已确认）
            
            # 优先使用float16（如果GPU支持），否则使用float32
            use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
            dtype = torch.float16 if use_fp16 else torch.float32
            print(f"  ℹ 使用数据类型: {dtype} (FP16支持: {use_fp16})")
            
            try:
                self.pipeline = AnimateDiffPipeline.from_pretrained(
                    base_model,
                    motion_adapter=adapter,
                    torch_dtype=dtype,
                )
                print(f"  ✓ Pipeline加载成功")
            except Exception as e:
                print(f"  ⚠ {dtype}加载失败，尝试float32: {e}")
                self.pipeline = AnimateDiffPipeline.from_pretrained(
                    base_model,
                    motion_adapter=adapter,
                    torch_dtype=torch.float32,
                )
                dtype = torch.float32
            
            # 设置scheduler（可能有助于修复绿色竖条/斑点问题）
            try:
                from diffusers import DDIMScheduler
                print(f"  ℹ 设置DDIMScheduler...")
                self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
                print(f"  ✓ Scheduler已设置")
            except Exception as e:
                print(f"  ⚠ 无法设置scheduler: {e}")
            
            # 移动到GPU
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
                # 启用内存优化（但不要启用VAE slicing/tiling，可能导致绿色竖条）
                # self.pipeline.enable_vae_slicing()  # 可能导致绿色竖条，先禁用
                # self.pipeline.enable_vae_tiling()   # 可能导致绿色竖条，先禁用
                if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    self.pipeline.enable_model_cpu_offload()
            
            # 设置scheduler（可能有助于修复绿色竖条/斑点问题）
            try:
                from diffusers import DDIMScheduler
                print(f"  ℹ 设置DDIMScheduler...")
                self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
                print(f"  ✓ Scheduler已设置")
            except Exception as e:
                print(f"  ⚠ 无法设置scheduler: {e}")
            
            # 关键修复：设置motion_scale（根据专业分析，这是导致绿色的最常见原因）
            # motion_scale应该在0.5-1.0之间，默认1.0
            animatediff_config = self.video_config.get('animatediff', {})
            if isinstance(animatediff_config, dict):
                motion_scale = float(animatediff_config.get('motion_scale', 1.0))
            else:
                motion_scale = 1.0
            
            # 检查pipeline是否有motion_scale属性
            if hasattr(self.pipeline, 'motion_scale'):
                self.pipeline.motion_scale = motion_scale
                print(f"  ✓ 设置motion_scale={motion_scale}（关键修复：防止latent爆掉导致全绿色）")
            elif hasattr(self.pipeline, 'motion_adapter') and hasattr(self.pipeline.motion_adapter, 'scale'):
                # 某些版本可能motion_scale在adapter上
                self.pipeline.motion_adapter.scale = motion_scale
                print(f"  ✓ 设置motion_adapter.scale={motion_scale}（关键修复：防止latent爆掉导致全绿色）")
            else:
                print(f"  ⚠ 无法设置motion_scale（pipeline可能不支持此参数，如果还是绿色，可能需要更新diffusers版本）")
            
            # 尝试修复VAE（如果可能）
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                vae = self.pipeline.vae
                # 确保VAE在正确的设备上
                if torch.cuda.is_available():
                    vae = vae.to("cuda")
                # 尝试设置VAE的scale_factor（可能有助于修复绿色竖条）
                if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
                    print(f"  ℹ VAE scaling_factor: {vae.config.scaling_factor}")
                # 确保VAE使用正确的数据类型
                if hasattr(vae, 'to'):
                    vae = vae.to(torch.float16 if torch.cuda.is_available() else torch.float32)
            
            # 验证VAE是否正确加载
            if hasattr(self.pipeline, 'vae') and self.pipeline.vae is not None:
                vae = self.pipeline.vae
                print(f"  ✓ VAE已加载: {type(vae).__name__}")
                # 检查VAE是否在正确的设备上
                if hasattr(vae, 'device'):
                    print(f"    VAE设备: {vae.device}")
            else:
                print(f"  ⚠ 警告：VAE未正确加载，可能导致绿色竖条")
            
            # 检查motion adapter
            if hasattr(self.pipeline, 'motion_adapter') and self.pipeline.motion_adapter is not None:
                print(f"  ✓ Motion adapter已加载")
            else:
                print(f"  ⚠ 警告：Motion adapter未正确加载")
            
            self.animatediff_export_to_video = export_to_video
            print("  ✓ AnimateDiff模型加载成功（文生视频模式）")
            
        except ImportError as e:
            raise ImportError(f"AnimateDiff依赖未安装: {e}\n请安装: pip install animatediff") from e
        except Exception as e:
            print(f"  ✗ AnimateDiff模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_video(
        self,
        image_path: str,
        output_path: str,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成视频
        
        Args:
            image_path: 输入图像路径
            output_path: 输出视频路径
            num_frames: 帧数（可选，默认从配置读取）
            fps: 帧率（可选，默认从配置读取）
            motion_bucket_id: 运动参数（可选）
            noise_aug_strength: 噪声参数（可选）
            scene: 场景JSON数据（用于智能分析）
            
        Returns:
            输出视频路径
        """
        print(f"\n生成视频: {image_path} -> {output_path}")
        
        # ========== 1. 智能分析场景 ==========
        analysis = None
        if scene:
            analysis = self.motion_analyzer.analyze(scene)
            print(f"  ℹ 智能分析结果:")
            print(f"    - 物体运动: {analysis.get('has_object_motion', False)} ({analysis.get('object_motion_type', 'unknown')})")
            print(f"    - 镜头运动: {analysis.get('camera_motion_type', 'static')}")
            print(f"    - 运动强度: {analysis.get('motion_intensity', 'moderate')}")
            print(f"    - 使用SVD: {analysis.get('use_svd', False)}")
            
            # 根据分析结果覆盖参数
            if analysis.get('motion_bucket_id_override') is not None:
                motion_bucket_id = analysis['motion_bucket_id_override']
                print(f"    - 覆盖 motion_bucket_id: {motion_bucket_id}")
            if analysis.get('noise_aug_strength_override') is not None:
                noise_aug_strength = analysis['noise_aug_strength_override']
                print(f"    - 覆盖 noise_aug_strength: {noise_aug_strength}")
        
        # ========== 2. 确定使用哪种生成方式 ==========
        use_svd = True
        if analysis and not analysis['use_svd']:
            use_svd = False
            print(f"  ℹ 检测到完全静态场景，使用静态图像动画")
        elif scene:
            # 检查场景描述中是否有物体运动关键词
            description = (scene.get("description") or "").lower()
            action = (scene.get("action") or "").lower()
            visual = scene.get("visual") or {}
            composition = (visual.get("composition", "") or "").lower() if isinstance(visual, dict) else ""
            fx = (visual.get("fx", "") or "").lower() if isinstance(visual, dict) else ""
            
            all_text = f"{description} {action} {composition} {fx}".lower()
            object_motion_keywords = [
                "unfurling", "unfold", "unfolding", "展开", "舒展开", "展开来", "缓缓展开",
                "open", "opening", "打开", "开启", "张开",
                "rotate", "rotating", "spin", "spinning", "旋转", "转动", "翻转",
                "float", "floating", "drift", "drifting", "飘动", "漂浮", "流动"
            ]
            has_object_motion = any(keyword in all_text for keyword in object_motion_keywords)
            
            if has_object_motion:
                use_svd = True
                print(f"  ℹ 检测到物体运动，使用SVD生成视频")
            elif any(keyword in all_text for keyword in ["still", "motionless", "静止", "不动"]):
                use_svd = False
                print(f"  ℹ 检测到完全静态场景，使用静态图像动画")
        
        # ========== 3. 获取参数并根据场景类型优化 ==========
        # 记录是否显式传递了num_frames参数（用于后续判断是否允许场景分析覆盖）
        num_frames_explicit = num_frames is not None
        
        if num_frames is None:
            num_frames = self.video_config.get('num_frames', 120)
        if fps is None:
            fps = self.video_config.get('fps', 24)
        if motion_bucket_id is None:
            motion_bucket_id = self.video_config.get('motion_bucket_id', 1.5)
        if noise_aug_strength is None:
            noise_aug_strength = self.video_config.get('noise_aug_strength', 0.00025)
        
        # 根据场景类型进一步优化参数（确保人物动作自然流畅，镜头移动明显，物体运动明显）
        if analysis and use_svd:
            motion_intensity = analysis.get('motion_intensity', 'gentle')
            camera_motion_type = analysis.get('camera_motion_type', 'static')
            has_object_motion = analysis.get('has_object_motion', False)
            object_motion_type = analysis.get('object_motion_type')
            
            # 如果有物体运动（如卷轴展开），确保运动参数适中，减少闪动
            if has_object_motion:
                # 物体运动需要适中的运动参数，确保运动明显但减少闪动
                if motion_bucket_id > 1.8:
                    motion_bucket_id = 1.8  # 限制到1.8，减少闪动
                    print(f"  ℹ 检测到物体运动（{object_motion_type}），限制 motion_bucket_id 至 {motion_bucket_id} 减少闪动")
                if noise_aug_strength > 0.0003:
                    noise_aug_strength = 0.0003  # 限制到0.0003，减少闪动
                    print(f"  ℹ 物体运动场景，限制 noise_aug_strength 至 {noise_aug_strength} 减少闪动")
            
            # 如果有镜头运动，确保运动参数适中，减少闪动
            elif camera_motion_type != 'static':
                # 镜头移动需要适中的运动参数，确保移动明显但减少闪动
                if motion_bucket_id > 1.8:
                    motion_bucket_id = 1.8  # 限制到1.8，减少闪动
                    print(f"  ℹ 检测到镜头运动（{camera_motion_type}），限制 motion_bucket_id 至 {motion_bucket_id} 减少闪动")
                if noise_aug_strength > 0.0003:
                    noise_aug_strength = 0.0003  # 限制到0.0003，减少闪动
                    print(f"  ℹ 镜头运动场景，限制 noise_aug_strength 至 {noise_aug_strength} 减少闪动")
            
            # 对于人物动作场景，确保有适中的运动参数，减少闪动
            elif motion_intensity in ['dynamic', 'moderate']:
                # 确保motion_bucket_id适中，使动作明显但减少闪动
                if motion_bucket_id > 1.8:
                    motion_bucket_id = 1.8  # 限制到1.8，减少闪动
                    print(f"  ℹ 人物动作场景，限制 motion_bucket_id 至 {motion_bucket_id} 减少闪动")
                
                # 确保noise_aug_strength适中，使动作自然流畅但减少闪动
                if noise_aug_strength > 0.0003:
                    noise_aug_strength = 0.0003  # 限制到0.0003，减少闪动
                    print(f"  ℹ 人物动作场景，限制 noise_aug_strength 至 {noise_aug_strength} 减少闪动")
            
            # 对于轻微动作场景，使用适中的参数
            elif motion_intensity == 'gentle':
                # 确保有轻微运动，但不过度
                if motion_bucket_id < 1.8:
                    motion_bucket_id = 1.8
                if noise_aug_strength < 0.00025:
                    noise_aug_strength = 0.0003
                print(f"  ℹ 轻微动作场景，使用适中参数（motion_bucket_id={motion_bucket_id}, noise_aug_strength={noise_aug_strength}）")
        
        # 限制最大帧数
        max_frames = self.video_config.get('max_frames', 384)
        if num_frames > max_frames:
            num_frames = max_frames
            print(f"  ⚠ 帧数超过最大值，限制为 {max_frames}")
        
        # 确保num_frames足够，使动作流畅（对于动作场景）
        # 但如果显式传递了num_frames参数，则尊重用户设置，不覆盖
        if analysis and analysis.get('motion_intensity') in ['dynamic', 'moderate'] and not num_frames_explicit:
            # 对于动作场景，确保有足够的帧数（仅在未显式传递num_frames时）
            min_frames_for_action = int(fps * 4)  # 至少4秒的帧数
            if num_frames < min_frames_for_action:
                num_frames = min_frames_for_action
                print(f"  ℹ 动作场景，确保足够帧数（{num_frames}帧）使动作流畅")
        
        # ========== 4. 生成视频 ==========
        model_type = self.video_config.get('model_type', 'svd-xt')
        
        # 如果配置了模型路由，根据场景自动选择模型
        if model_type == 'auto' and self.model_router is not None:
            # 使用模型路由器自动选择
            user_tier = scene.get('user_tier', 'basic') if scene else 'basic'
            selected_model = self.model_router.select_model(
                scene=scene,
                user_tier=user_tier,
                force_model=None,
                available_memory=None  # 自动检测
            )
            print(f"  ℹ 模型路由选择: {selected_model}")
            model_type = selected_model
        
        # 如果使用CogVideoX，使用CogVideoX生成
        if model_type == 'cogvideox':
            # 获取CogVideoX配置
            cogvideox_config = self.video_config.get('cogvideox', {})
            # 获取CogVideoX特定的参数
            cogvideox_num_frames = num_frames if num_frames else cogvideox_config.get('num_frames', 81)
            cogvideox_fps = fps if fps else cogvideox_config.get('fps', 16)
            
            # 构建prompt和negative_prompt
            prompt = ""
            negative_prompt = cogvideox_config.get('negative_prompt', '')
            if scene:
                # 从scene中提取prompt信息
                from diffusers.utils import load_image
                image = load_image(image_path)
                prompt = self._build_detailed_prompt(image_path, image, scene, model_type="cogvideox")
            
            return self._generate_video_cogvideox(
                image_path=image_path,
                output_path=output_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=cogvideox_num_frames,
                fps=cogvideox_fps,
                scene=scene,
            )
        # 如果使用HunyuanVideo，使用HunyuanVideo生成
        elif model_type == 'hunyuanvideo':
            # HunyuanVideo 专用参数：优先使用 video.hunyuanvideo 配置（否则 quick 模式设置的 hv.num_frames/hv.fps 无效）
            if not num_frames_explicit:
                hunyuan_config = self.video_config.get('hunyuanvideo', {})
                if hunyuan_config:
                    num_frames = hunyuan_config.get('num_frames', num_frames)
                    fps = hunyuan_config.get('fps', fps)

            # 从scene中提取prompt（如果有）
            video_prompt = ""
            video_negative_prompt = ""
            if scene:
                # 优先使用scene中的prompt或description
                video_prompt = scene.get('prompt') or scene.get('description') or ""
                # 读取 negative prompt（如果提供）
                video_negative_prompt = scene.get('negative_prompt') or scene.get('neg_prompt') or ""
                # 可以添加运动、风格等描述
                if video_prompt:
                    # 如果场景有运动信息，可以添加到prompt中
                    motion_intensity = scene.get('motion_intensity', '')
                    if motion_intensity == 'dynamic':
                        video_prompt = f"{video_prompt}, dynamic motion, fast movement, energetic"
                    elif motion_intensity == 'moderate':
                        video_prompt = f"{video_prompt}, smooth motion, natural movement"
                    elif motion_intensity == 'gentle':
                        video_prompt = f"{video_prompt}, gentle motion, slow movement, peaceful"
            
            return self._generate_video_hunyuanvideo(
                image_path,
                output_path,
                prompt=video_prompt,
                negative_prompt=video_negative_prompt,
                num_frames=num_frames,
                fps=fps,
                scene=scene,  # 传递scene以便构建详细prompt
            )
        # 如果使用AnimateDiff，使用文生视频模式
        elif model_type == 'animatediff' or model_type == 'animatediff-sdxl':
            # AnimateDiff是文生视频，需要从scene中提取prompt
            prompt = None
            if scene:
                # 从scene中提取prompt或description
                prompt = scene.get('prompt') or scene.get('description') or ""
                # 如果没有prompt，尝试从visual字段构建
                if not prompt:
                    visual = scene.get('visual', {})
                    if isinstance(visual, dict):
                        composition = visual.get('composition', '')
                        prompt = composition if composition else "scientific educational scene"
            else:
                prompt = "scientific educational scene"
            
            return self._generate_video_animatediff(
                prompt=prompt,
                output_path=output_path,
                num_frames=num_frames,
                fps=fps,
                scene=scene,
            )
        elif use_svd:
            return self._generate_video_svd(
                image_path,
                output_path,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
            )
        else:
            return self._generate_static_image_animation(
                image_path,
                output_path,
                num_frames=num_frames,
                fps=fps,
                scene=scene,
            )
    
    def _generate_video_svd(
        self,
        image_path: str,
        output_path: str,
        num_frames: int,
        fps: float,
        motion_bucket_id: float,
        noise_aug_strength: float,
    ) -> str:
        """使用SVD生成视频"""
        print(f"  使用SVD生成视频")
        print(f"    参数: num_frames={num_frames}, fps={fps}")
        print(f"          motion_bucket_id={motion_bucket_id}, noise_aug_strength={noise_aug_strength}")
        
        if not self.model_loaded:
            self.load_model()
        
        # 加载图像
        from diffusers.utils import load_image
        image = load_image(image_path)
        
        # 调整图像大小（SVD要求必须是64的倍数）
        # 保持宽高比，避免拉伸
        target_width = self.video_config.get('width', 1280)
        target_height = self.video_config.get('height', 768)
        
        # 计算缩放比例（保持宽高比）
        orig_width, orig_height = image.size
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)  # 使用较小的比例，确保图像不被拉伸
        
        # 计算新尺寸（保持宽高比）
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # 确保是64的倍数（SVD要求）
        new_width = (new_width // 64) * 64
        new_height = (new_height // 64) * 64
        
        # 如果尺寸为0，使用最小值
        if new_width == 0:
            new_width = 64
        if new_height == 0:
            new_height = 64
        
        # 调整图像大小（保持宽高比）
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 如果需要，创建目标尺寸的画布并居中放置（填充黑色或透明）
        if new_width != target_width or new_height != target_height:
            # 创建目标尺寸的画布（黑色背景）
            canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            # 计算居中位置
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            # 将图像粘贴到画布中心
            canvas.paste(image, (x_offset, y_offset))
            image = canvas
            print(f"  ℹ 图像已调整（保持宽高比）: {orig_width}x{orig_height} -> {new_width}x{new_height} (画布: {target_width}x{target_height})")
        else:
            print(f"  ℹ 图像已调整（保持宽高比）: {orig_width}x{orig_height} -> {new_width}x{new_height}")
        
        # 生成视频
        num_inference_steps = self.video_config.get('num_inference_steps', 40)
        decode_chunk_size = self.video_config.get('decode_chunk_size', 8)
        
        # 对于动作场景和镜头运动场景，增加推理步数以提高稳定性和减少闪动
        if motion_bucket_id >= 1.7:
            # 动作场景和镜头运动需要更多步数以确保稳定性和减少闪动
            num_inference_steps = max(num_inference_steps, 50)  # 提高到50步，减少闪动
            print(f"  ℹ 动作/镜头运动场景，增加推理步数至 {num_inference_steps} 提高稳定性和减少闪动")
        
        # 对于动作场景和镜头运动场景，适当调整decode_chunk_size以提高连贯性和减少闪动
        if motion_bucket_id >= 1.7:
            # 动作场景和镜头运动需要适中的chunk size以确保帧间连贯性，减少闪动
            decode_chunk_size = min(decode_chunk_size, 8)  # 保持8，SVD-XT最稳定
            # 不要太小，避免过度平滑导致不自然
            if decode_chunk_size < 7:
                decode_chunk_size = 7  # 最小7，保持流畅度
            print(f"  ℹ 动作/镜头运动场景，调整 decode_chunk_size 至 {decode_chunk_size} 提高连贯性和减少闪动")
        
        # 限制motion_bucket_id在SVD-XT的有效范围内（最大2）
        if motion_bucket_id > 2.0:
            motion_bucket_id = 2.0
            print(f"  ⚠ motion_bucket_id 超过SVD-XT最大值，限制为 2.0")
        
        # 如果启用 RIFE 插帧，先生成关键帧（减少帧数，提高速度）
        if self.rife_enabled:
            # 计算关键帧数（目标帧数 / 插帧倍数）
            keyframe_count = max(int(num_frames / self.rife_interpolation_scale), 8)  # 至少8帧
            print(f"  ℹ RIFE 插帧已启用，先生成 {keyframe_count} 个关键帧，然后插帧到 {num_frames} 帧")
            
            frames = self.pipeline(
                image,
                decode_chunk_size=decode_chunk_size,
                num_frames=keyframe_count,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                num_inference_steps=num_inference_steps,
            ).frames[0]
            
            # 使用 RIFE 插帧
            frames = self._interpolate_frames_rife(frames, target_frames=num_frames)
            print(f"  ✓ RIFE 插帧完成：{keyframe_count} 帧 → {len(frames)} 帧")
        else:
            # 不使用插帧，直接生成目标帧数
            frames = self.pipeline(
                image,
                decode_chunk_size=decode_chunk_size,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                num_inference_steps=num_inference_steps,
            ).frames[0]
        
        # 保存视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        
        print(f"  ✓ 视频生成成功: {output_path}")
        return output_path
    
    def _generate_video_animatediff(
        self,
        prompt: str,
        output_path: str,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """使用AnimateDiff生成视频（文生视频模式）
        
        Args:
            prompt: 文本提示词
            output_path: 输出视频路径
            num_frames: 帧数（可选）
            fps: 帧率（可选）
            scene: 场景数据（可选，用于构建更详细的prompt）
        """
        print(f"  使用AnimateDiff生成视频（文生视频模式）")
        
        if not self.model_loaded:
            self.load_model()
        
        # 获取参数
        if num_frames is None:
            num_frames = self.video_config.get('num_frames', 32)
        if fps is None:
            fps = self.video_config.get('fps', 16)
        
        # AnimateDiff参数
        num_inference_steps = self.video_config.get('num_inference_steps', 50)
        guidance_scale = self.video_config.get('guidance_scale', 7.5)
        width = self.video_config.get('width', 768)
        height = self.video_config.get('height', 768)
        
        # 限制分辨率（SD1.5最大768x768）
        if width > 768:
            width = 768
            print(f"  ⚠ 分辨率限制为768（SD1.5最大分辨率）")
        if height > 768:
            height = 768
            print(f"  ⚠ 分辨率限制为768（SD1.5最大分辨率）")
        
        # 确保分辨率是8的倍数（SD要求，避免绿色竖条）
        width = (width // 8) * 8
        height = (height // 8) * 8
        if width != self.video_config.get('width', 768) or height != self.video_config.get('height', 768):
            print(f"  ℹ 分辨率调整为8的倍数: {width}x{height}")
        
        # 限制帧数（AnimateDiff通常支持最多32帧）
        if num_frames > 32:
            num_frames = 32
            print(f"  ⚠ AnimateDiff限制帧数为32，已调整")
        
        # 构建完整的prompt（结合场景信息和风格配置）
        full_prompt = prompt
        if scene:
            # 获取风格配置
            hunyuan_config = self.video_config.get('hunyuanvideo', {})
            style_templates = hunyuan_config.get('style_templates', {})
            default_style = hunyuan_config.get('default_style', 'realistic')
            
            # 从scene中获取风格
            style_name = None
            visual = scene.get('visual', {})
            if isinstance(visual, dict):
                style_name = visual.get('style', '')
            if not style_name:
                style_name = scene.get('style', default_style)
            
            # 应用风格模板
            style_template = style_templates.get(style_name, style_templates.get(default_style, {}))
            if style_template:
                style_keywords = style_template.get('keywords', [])
                if style_keywords:
                    style_prefix = ", ".join(style_keywords[:3])  # 使用前3个关键词
                    full_prompt = f"{style_prefix}, {prompt}"
                else:
                    full_prompt = f"realistic, professional, {prompt}"
            else:
                full_prompt = f"realistic, professional, {prompt}"
            # 添加角色信息（如果有）
            characters = scene.get('characters', [])
            if characters:
                char_names = [c.get('name', '') for c in characters if isinstance(c, dict)]
                if char_names:
                    full_prompt = f"{full_prompt}, {', '.join(char_names)}"
        
        # 构建negative prompt
        negative_prompt = "low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark"
        
        print(f"    Prompt: {full_prompt[:100]}...")
        print(f"    参数: num_frames={num_frames}, fps={fps}, resolution={width}x{height}")
        print(f"          num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}")
        
        try:
            import torch
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            if scene and scene.get('seed'):
                generator.manual_seed(int(scene['seed']))
            else:
                generator.manual_seed(42)
            
            # 生成视频
            print(f"  开始生成...")
            print(f"  ℹ 提示：如果出现绿色竖条，可能是VAE解码问题，建议切换回SVD")
            
            # 尝试使用不同的参数组合来避免绿色竖条/斑点
            try:
                # 方法1：标准参数（明确指定输出类型）
                print(f"  ℹ 使用标准参数生成...")
                result = self.pipeline(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="np",  # 明确指定输出numpy数组
                )
            except Exception as e:
                print(f"  ⚠ 标准参数生成失败: {e}")
                print(f"  ℹ 尝试使用备用参数（降低分辨率，增加步数）...")
                # 方法2：降低分辨率，增加步数
                width_fallback = min(512, width)
                height_fallback = min(512, height)
                width_fallback = (width_fallback // 8) * 8
                height_fallback = (height_fallback // 8) * 8
                print(f"  ℹ 使用备用分辨率: {width_fallback}x{height_fallback}")
                result = self.pipeline(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=min(16, num_frames),  # 减少帧数
                    num_inference_steps=max(60, num_inference_steps),  # 增加步数
                    guidance_scale=guidance_scale,
                    width=width_fallback,
                    height=height_fallback,
                    generator=generator,
                )
            
            # 检查生成结果
            # 注意：使用output_type="np"时，result可能是字典或对象
            if hasattr(result, 'frames'):
                # result是对象，有frames属性
                if result.frames is None:
                    raise ValueError("生成结果中frames为None，可能生成失败")
                # 检查frames是否是数组（不能直接用not判断）
                import numpy as np
                if isinstance(result.frames, np.ndarray):
                    if result.frames.size == 0:
                        raise ValueError("生成结果中frames为空数组，可能生成失败")
                    frames = result.frames
                elif isinstance(result.frames, (list, tuple)):
                    if len(result.frames) == 0:
                        raise ValueError("生成结果中frames为空列表，可能生成失败")
                    frames = result.frames[0] if len(result.frames) > 0 else result.frames
                else:
                    frames = result.frames
            elif isinstance(result, dict):
                # result是字典
                if 'frames' not in result:
                    raise ValueError("生成结果中没有frames键，可能生成失败")
                frames = result['frames']
                if frames is None or (isinstance(frames, (list, tuple)) and len(frames) == 0):
                    raise ValueError("生成结果中frames为空，可能生成失败")
                if isinstance(frames, (list, tuple)) and len(frames) > 0:
                    frames = frames[0]
            else:
                # result可能就是frames本身
                frames = result
            
            # 检查frames是否有效
            import numpy as np
            if frames is None:
                raise ValueError("生成的frames为None")
            if isinstance(frames, np.ndarray):
                if frames.size == 0:
                    raise ValueError("生成的frames为空数组")
            elif isinstance(frames, (list, tuple)):
                if len(frames) == 0:
                    raise ValueError("生成的frames为空列表")
            else:
                # 其他类型，尝试获取长度
                try:
                    if len(frames) == 0:
                        raise ValueError("生成的frames为空")
                except (TypeError, AttributeError):
                    pass  # 无法获取长度，可能是单个帧
            
            # 检查第一帧是否有效（避免绿色竖条）
            first_frame = frames[0]
            import numpy as np
            from PIL import Image
            
            # 转换PIL Image到numpy array（如果需要）
            if isinstance(first_frame, Image.Image):
                first_frame_np = np.array(first_frame)
            elif isinstance(first_frame, np.ndarray):
                first_frame_np = first_frame
            else:
                first_frame_np = None
            
            if first_frame_np is not None and len(first_frame_np.shape) >= 2:
                # 检查是否全是绿色（可能是VAE解码失败）
                if len(first_frame_np.shape) == 3 and first_frame_np.shape[2] >= 3:  # RGB
                    # 检查是否主要是绿色
                    mean_values = np.mean(first_frame_np, axis=(0, 1))
                    if len(mean_values) >= 3:
                        green_value = mean_values[1]  # G channel
                        total_mean = np.mean(mean_values)
                        green_ratio = green_value / (total_mean + 1e-6)
                        
                        # 检查是否主要是绿色竖条（绿色值很高，其他值很低）
                        if green_ratio > 0.6 and mean_values[0] < 50 and mean_values[2] < 50:
                            print(f"  ⚠ 警告：检测到绿色竖条问题！")
                            print(f"     RGB均值: R={mean_values[0]:.1f}, G={mean_values[1]:.1f}, B={mean_values[2]:.1f}")
                            print(f"     绿色比例: {green_ratio:.2f}")
                            print(f"  ✗ 可能原因：VAE解码失败或motion adapter不兼容")
                            print(f"  💡 建议：")
                            print(f"     1. 切换回SVD（推荐）：修改config.yaml中model_type为svd-xt")
                            print(f"     2. 尝试降低分辨率：width=512, height=512")
                            print(f"     3. 尝试增加推理步数：num_inference_steps=60-80")
                            
                            # 保存第一帧用于调试
                            debug_frame_path = Path(output_path).parent / "debug_first_frame.png"
                            if isinstance(first_frame, Image.Image):
                                first_frame.save(str(debug_frame_path))
                            else:
                                Image.fromarray(first_frame_np).save(str(debug_frame_path))
                            print(f"  ℹ 已保存第一帧用于调试: {debug_frame_path}")
            
            print(f"  ✓ 生成了 {len(frames)} 帧")
            
            # 导出视频
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保frames格式正确（export_to_video期望list of numpy arrays，每个是(h, w, c)）
            import numpy as np
            from PIL import Image
            
            # 先打印frames的格式用于调试
            print(f"  ℹ 调试：frames类型={type(frames)}")
            if isinstance(frames, np.ndarray):
                print(f"  ℹ 调试：frames形状={frames.shape}, dtype={frames.dtype}, min={frames.min()}, max={frames.max()}")
            elif isinstance(frames, (list, tuple)):
                print(f"  ℹ 调试：frames是列表/元组，长度={len(frames)}")
                if len(frames) > 0:
                    print(f"  ℹ 调试：第一帧类型={type(frames[0])}")
                    if isinstance(frames[0], np.ndarray):
                        print(f"  ℹ 调试：第一帧形状={frames[0].shape}")
            
            # 转换frames为正确的格式
            video_frames = []
            try:
                if isinstance(frames, np.ndarray):
                    # 如果是numpy数组，检查维度
                    if len(frames.shape) == 5:
                        # (batch, time, h, w, c) 格式，需要squeeze和reshape
                        print(f"  ℹ 检测到5D数组，形状={frames.shape}，格式可能是(batch, time, h, w, c)")
                        # 移除batch和time维度
                        frames = frames.squeeze()  # 移除大小为1的维度
                        print(f"  ℹ squeeze后形状={frames.shape}")
                        # 如果还是5D，手动处理
                        if len(frames.shape) == 5:
                            # 合并batch和time维度
                            frames = frames.reshape(-1, frames.shape[2], frames.shape[3], frames.shape[4])
                            print(f"  ℹ reshape后形状={frames.shape}")
                        # 现在应该是4D (num_frames, h, w, c) 或 3D (h, w, c)
                        if len(frames.shape) == 4:
                            # (num_frames, h, w, c) 格式
                            print(f"  ℹ 拆分为多个帧")
                            for i in range(frames.shape[0]):
                                frame = frames[i]
                                # 确保是(h, w, c)格式
                                if len(frame.shape) == 3:
                                    # 确保值在0-255范围内
                                    if frame.max() <= 1.0:
                                        frame = (frame * 255).astype(np.uint8)
                                    else:
                                        frame = frame.astype(np.uint8)
                                    video_frames.append(frame)
                                else:
                                    # 转换其他格式
                                    frame = np.array(Image.fromarray(frame).convert("RGB"))
                                    video_frames.append(frame)
                        elif len(frames.shape) == 3:
                            # 单个帧 (h, w, c)
                            if frames.max() <= 1.0:
                                frames = (frames * 255).astype(np.uint8)
                            else:
                                frames = frames.astype(np.uint8)
                            video_frames.append(frames)
                    elif len(frames.shape) == 4:
                        # (num_frames, h, w, c) 格式
                        print(f"  ℹ 检测到4D数组，形状={frames.shape}，拆分为多个帧")
                        for i in range(frames.shape[0]):
                            frame = frames[i]
                            # 确保是(h, w, c)格式
                            if len(frame.shape) == 3:
                                # 确保值在0-255范围内
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                                video_frames.append(frame)
                            else:
                                # 转换其他格式
                                frame = np.array(Image.fromarray(frame).convert("RGB"))
                                video_frames.append(frame)
                    elif len(frames.shape) == 3:
                        # 单个帧 (h, w, c)
                        print(f"  ℹ 检测到3D数组，形状={frames.shape}，作为单帧处理")
                        if frames.max() <= 1.0:
                            frames = (frames * 255).astype(np.uint8)
                        else:
                            frames = frames.astype(np.uint8)
                        video_frames.append(frames)
                    else:
                        print(f"  ⚠ 未知的数组维度：{frames.shape}，尝试squeeze和转换")
                        # 先尝试squeeze移除大小为1的维度
                        frames_squeezed = frames.squeeze()
                        print(f"  ℹ squeeze后形状={frames_squeezed.shape}")
                        # 如果squeeze后是3D或4D，递归处理
                        if len(frames_squeezed.shape) in [3, 4, 5]:
                            # 递归处理
                            if isinstance(frames_squeezed, np.ndarray):
                                if len(frames_squeezed.shape) == 4:
                                    for i in range(frames_squeezed.shape[0]):
                                        frame = frames_squeezed[i]
                                        if len(frame.shape) == 3:
                                            if frame.max() <= 1.0:
                                                frame = (frame * 255).astype(np.uint8)
                                            else:
                                                frame = frame.astype(np.uint8)
                                            video_frames.append(frame)
                                elif len(frames_squeezed.shape) == 3:
                                    if frames_squeezed.max() <= 1.0:
                                        frames_squeezed = (frames_squeezed * 255).astype(np.uint8)
                                    else:
                                        frames_squeezed = frames_squeezed.astype(np.uint8)
                                    video_frames.append(frames_squeezed)
                        else:
                            raise ValueError(f"无法处理数组维度：{frames.shape} (squeeze后: {frames_squeezed.shape})")
                elif isinstance(frames, (list, tuple)):
                    # 如果是列表，处理每个帧
                    print(f"  ℹ 检测到列表/元组，长度={len(frames)}")
                    for i, frame in enumerate(frames):
                        if isinstance(frame, Image.Image):
                            frame = np.array(frame.convert("RGB"))
                        elif isinstance(frame, np.ndarray):
                            if len(frame.shape) == 3:
                                # (h, w, c)
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                            elif len(frame.shape) == 2:
                                # (h, w) 灰度图，转换为RGB
                                frame = np.stack([frame] * 3, axis=-1)
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                            else:
                                # 其他格式，尝试转换
                                try:
                                    if frame.max() <= 1.0:
                                        frame = (frame * 255).astype(np.uint8)
                                    frame = np.array(Image.fromarray(frame).convert("RGB"))
                                except Exception as e:
                                    print(f"  ⚠ 第{i}帧转换失败: {e}")
                                    continue
                        else:
                            # 其他类型，尝试转换
                            try:
                                frame = np.array(Image.fromarray(np.array(frame)).convert("RGB"))
                            except Exception as e:
                                print(f"  ⚠ 第{i}帧转换失败: {e}")
                                continue
                        video_frames.append(frame)
                else:
                    # 其他类型，尝试转换
                    print(f"  ℹ 检测到其他类型，尝试转换")
                    if isinstance(frames, Image.Image):
                        video_frames.append(np.array(frames.convert("RGB")))
                    else:
                        try:
                            arr = np.array(frames)
                            if arr.size > 0:
                                if arr.max() <= 1.0:
                                    arr = (arr * 255).astype(np.uint8)
                                img = Image.fromarray(arr)
                                video_frames.append(np.array(img.convert("RGB")))
                        except Exception as e:
                            print(f"  ⚠ 转换失败: {e}")
                            raise
            except Exception as e:
                print(f"  ✗ frames转换过程出错: {e}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"无法转换frames为正确的格式: {e}")
            
            if len(video_frames) == 0:
                print(f"  ✗ 转换后video_frames为空")
                raise ValueError("无法转换frames为正确的格式：转换后为空")
            
            print(f"  ℹ 已转换 {len(video_frames)} 帧，格式: {video_frames[0].shape}")
            
            # 使用diffusers的export_to_video
            if hasattr(self, 'animatediff_export_to_video'):
                self.animatediff_export_to_video(video_frames, str(output_path), fps=fps)
            else:
                from diffusers.utils import export_to_video
                export_to_video(video_frames, str(output_path), fps=fps)
            
            print(f"  ✓ AnimateDiff视频生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"  ✗ AnimateDiff视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_detailed_prompt(
        self,
        image_path: str,
        image: Image.Image,
        scene: Optional[Dict[str, Any]] = None,
        model_type: str = "general"
    ) -> str:
        """
        构建详细的prompt，描述图片内容和期望的运动
        优先使用Prompt Engine（专业级Prompt工程系统），如果不可用则回退到基础方法
        
        参考HunyuanVideo官方示例格式：
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. 
        The fluffy-furred feline gazes directly at the camera with a relaxed expression. 
        Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, 
        and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, 
        as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's 
        intricate details and the refreshing atmosphere of the seaside."
        """
        # 如果Prompt Engine可用，优先使用
        if self.prompt_engine is not None:
            try:
                # 从scene中提取用户输入
                user_input = ""
                if scene:
                    user_input = scene.get('description', '') or scene.get('prompt', '') or scene.get('narration', '')
                
                # 如果没有用户输入，尝试从图片路径提取
                if not user_input:
                    from pathlib import Path
                    image_name = Path(image_path).stem
                    user_input = f"a scene from {image_name}"
                
                # 确定场景类型
                scene_type = "general"
                if scene:
                    scene_type = scene.get('type') or scene.get('scene_type', 'general')
                    # 如果没有明确指定，尝试从其他字段推断
                    if scene_type == "general":
                        visual = scene.get('visual', {})
                        if isinstance(visual, dict):
                            style = visual.get('style', '')
                            if style in ['scientific', 'novel', 'drama', 'government', 'enterprise']:
                                scene_type = style
                
                # 构建相机配置（从scene中提取）
                camera_config = None
                if scene:
                    camera_motion = scene.get('camera_motion', {})
                    if isinstance(camera_motion, dict):
                        camera_type = camera_motion.get('type', 'static')
                        # 映射到Prompt Engine的相机配置
                        camera_config = {
                            "shot_type": "wide",  # 默认wide shot
                            "movement": camera_type,  # pan, zoom, dolly, static
                            "viewpoint": "third_person",
                            "dof": "shallow",
                            "focal_length": "normal"
                        }
                        # 从scene中提取更多相机信息
                        visual = scene.get('visual', {})
                        if isinstance(visual, dict):
                            composition = visual.get('composition', '')
                            if 'close' in composition.lower() or 'close-up' in composition.lower():
                                camera_config["shot_type"] = "close"
                            elif 'medium' in composition.lower():
                                camera_config["shot_type"] = "medium"
                
                # 使用Prompt Engine处理
                result = self.prompt_engine.process(
                    user_input=user_input,
                    scene=scene,
                    model_type=model_type,
                    scene_type=scene_type,
                    camera_config=camera_config
                )
                
                prompt = result["prompt"]
                qa_result = result["qa_result"]
                
                print(f"  ℹ Prompt Engine处理完成")
                print(f"    QA评分: {qa_result['score']}/{qa_result['max_score']}")
                if qa_result.get('suggestions'):
                    print(f"    建议: {', '.join(qa_result['suggestions'][:2])}")
                
                return prompt
            except Exception as e:
                print(f"  ⚠ Prompt Engine处理失败: {e}，回退到基础方法")
                import traceback
                traceback.print_exc()
        
        # 回退到基础方法（原有逻辑）
        prompt_parts = []
        
        # 获取风格配置
        hunyuan_config = self.video_config.get('hunyuanvideo', {})
        style_templates = hunyuan_config.get('style_templates', {})
        default_style = hunyuan_config.get('default_style', 'realistic')
        
        # 1. 从scene中提取风格信息（优先使用scene中的style）
        style_name = None
        if scene:
            # 优先从scene.visual.style获取
            visual = scene.get('visual', {})
            if isinstance(visual, dict):
                style_name = visual.get('style', '')
            # 如果没有，从scene.style获取
            if not style_name:
                style_name = scene.get('style', '')
        
        # 如果仍未指定，使用默认风格
        if not style_name:
            style_name = default_style
        
        # 应用风格模板
        style_template = style_templates.get(style_name, style_templates.get(default_style, {}))
        if style_template:
            style_desc = style_template.get('description', '')
            style_keywords = style_template.get('keywords', [])
            if style_desc:
                prompt_parts.append(style_desc)
            elif style_keywords:
                prompt_parts.append(", ".join(style_keywords[:3]))  # 使用前3个关键词
        
        # 2. 从scene中提取主体描述
        if scene:
            # 提取主体描述（优先使用description，然后是prompt）
            description = scene.get('description', '') or scene.get('prompt', '')
            if description:
                # 如果description太短，尝试补充更多细节
                if len(description.split()) < 10:
                    # 描述太短，尝试从其他字段补充
                    narration = scene.get('narration', '')
                    if narration:
                        # 从旁白中提取关键信息（简化版）
                        description = f"{description}. {narration[:100]}"  # 限制长度
                prompt_parts.append(description)
            
            # 提取视觉信息（如果之前没有从visual.style获取风格，现在提取composition）
            visual = scene.get('visual', {})
            if isinstance(visual, dict):
                composition = visual.get('composition', '')
                if composition:
                    prompt_parts.append(composition)
                
                # 提取其他视觉细节
                lighting = visual.get('lighting', '')
                if lighting:
                    prompt_parts.append(f"lighting: {lighting}")
                
                mood = visual.get('mood', '')
                if mood:
                    prompt_parts.append(f"mood: {mood}")
        
        # 2. 从图片路径提取信息（如果scene信息不足）
        if not prompt_parts:
            from pathlib import Path
            image_name = Path(image_path).stem
            # 尝试从文件名提取关键词
            if 'scene' in image_name.lower():
                prompt_parts.append("a detailed scene")
            elif 'face' in image_name.lower() or 'portrait' in image_name.lower():
                prompt_parts.append("a detailed portrait")
            else:
                prompt_parts.append("a detailed image")
        
        # 3. 添加运动描述（根据scene的motion_intensity）
        motion_intensity = scene.get('motion_intensity', 'moderate') if scene else 'moderate'
        camera_motion = scene.get('camera_motion', {}) if scene else {}
        camera_motion_type = camera_motion.get('type', 'static') if isinstance(camera_motion, dict) else 'static'
        
        # 构建运动描述
        motion_description = []
        if motion_intensity == 'dynamic':
            motion_description.append("dynamic motion")
            motion_description.append("fast movement")
            motion_description.append("energetic action")
        elif motion_intensity == 'moderate':
            motion_description.append("smooth motion")
            motion_description.append("natural movement")
        elif motion_intensity == 'gentle':
            motion_description.append("gentle motion")
            motion_description.append("slow movement")
            motion_description.append("peaceful atmosphere")
        
        # 添加镜头运动描述
        if camera_motion_type == 'pan':
            motion_description.append("smooth camera pan")
        elif camera_motion_type == 'zoom':
            motion_description.append("gentle camera zoom")
        elif camera_motion_type == 'dolly':
            motion_description.append("smooth camera dolly movement")
        elif camera_motion_type == 'static':
            motion_description.append("stable camera")
        
        # 4. 添加细节描述
        detail_parts = []
        detail_parts.append("high quality")
        detail_parts.append("cinematic")
        detail_parts.append("detailed")
        
        # 5. 组合所有部分（优化格式，使其更符合HunyuanVideo的要求）
        # 格式：[风格] style, [主体详细描述]. [背景/构图]. [运动描述]. [质量描述]
        full_prompt = ""
        
        # 第一部分：风格和主体
        if prompt_parts:
            full_prompt = ", ".join(prompt_parts)
        
        # 第二部分：运动描述（用句号分隔，更清晰）
        if motion_description:
            if full_prompt:
                full_prompt += ". "
            full_prompt += ", ".join(motion_description)
        
        # 第三部分：质量描述（用句号分隔）
        if detail_parts:
            if full_prompt:
                full_prompt += ". "
            full_prompt += ", ".join(detail_parts)
        
        # 确保prompt足够详细（HunyuanVideo需要详细描述）
        if len(full_prompt.split()) < 20:
            # 如果prompt太短，添加通用描述
            full_prompt += ". The scene is rich in detail with excellent composition and visual appeal"
        
        return full_prompt
    
    def _generate_video_hunyuanvideo(
        self,
        image_path: str,
        output_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        num_frames: int = 120,
        fps: int = 24,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """使用HunyuanVideo生成视频（图生视频）"""
        # 导入必要的模块（在函数开始处）
        from diffusers.utils import load_image
        from PIL import Image as PILImage  # 导入PIL Image，避免与局部变量冲突
        
        print(f"  使用HunyuanVideo生成视频")
        print(f"    参数: num_frames={num_frames}, fps={fps}")
        
        if not self.model_loaded or self.hunyuanvideo_pipeline is None:
            self.load_model()
        
        # 生成前清理显存（释放之前可能残留的显存）
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 加载图像
        image = load_image(image_path)
        
        # 获取HunyuanVideo配置
        hunyuan_config = self.video_config.get('hunyuanvideo', {})
        use_v15 = hunyuan_config.get('use_v15', True)  # 默认使用1.5版本
        
        # 优先从scene中获取分辨率（确保与图像一致），否则使用配置
        if scene and 'width' in scene and 'height' in scene:
            width = scene['width']
            height = scene['height']
            print(f"  ℹ 从scene获取分辨率: {width}x{height} (与图像一致)")
        else:
            width = hunyuan_config.get('width', self.video_config.get('width', 1280))
            height = hunyuan_config.get('height', self.video_config.get('height', 768))
            print(f"  ℹ 使用配置分辨率: {width}x{height}")
        
        # 确保分辨率是8的倍数（HunyuanVideo要求）
        # 重要：调整时保持长宽比，避免变形
        original_width, original_height = width, height
        original_aspect = width / height
        
        # 先调整宽度到8的倍数
        width = (width // 8) * 8
        # 根据原始长宽比计算高度（保持长宽比）
        height = int(width / original_aspect)
        # 再调整高度到8的倍数
        height = (height // 8) * 8
        # 重新计算宽度，确保长宽比一致
        width = int(height * original_aspect)
        width = (width // 8) * 8
        
        if width != original_width or height != original_height:
            new_aspect = width / height
            print(f"  ℹ 分辨率已调整为8的倍数: {width}x{height} (原始: {original_width}x{original_height})")
            print(f"  ℹ 长宽比: 原始={original_aspect:.3f}, 调整后={new_aspect:.3f} (差异: {abs(original_aspect - new_aspect):.3f})")
            if abs(original_aspect - new_aspect) > 0.01:
                print(f"  ⚠ 警告: 长宽比略有变化（由于8的倍数限制），但已尽量保持接近")
        num_inference_steps = hunyuan_config.get('num_inference_steps', 50)
        guidance_scale = hunyuan_config.get('guidance_scale', 7.5)
        # 色彩调整参数（用于修复过暗、色彩过浓的问题）
        saturation_factor = hunyuan_config.get('saturation_factor', 1.0)  # 饱和度调整因子
        brightness_factor = hunyuan_config.get('brightness_factor', 1.0)  # 亮度调整因子
        contrast_factor = hunyuan_config.get('contrast_factor', 1.0)  # 对比度调整因子
        
        # 获取prompt和negative_prompt（优先使用传入的参数，否则从配置读取）
        config_prompt = hunyuan_config.get('prompt', '')
        config_negative_prompt = hunyuan_config.get('negative_prompt', '')
        
        # 构建详细的prompt（参考HunyuanVideo官方示例格式）
        # 如果传入的prompt为空，尝试构建详细的prompt
        if not prompt:
            if config_prompt:
                prompt = config_prompt
            else:
                # 从图片路径提取信息，构建基础prompt
                # 注意：scene参数需要从generate_video传递下来
                prompt = self._build_detailed_prompt(image_path, image, scene, model_type="hunyuanvideo")
        
        # 如果prompt仍然为空或太简单，使用默认值
        if not prompt or len(prompt) < 20:
            prompt = "high quality video, smooth motion, cinematic, natural movement, stable camera"
        
        # 如果传入的negative_prompt为空，使用配置中的negative_prompt
        if not negative_prompt and config_negative_prompt:
            negative_prompt = config_negative_prompt
        # 如果仍然为空，使用默认值
        if not negative_prompt:
            negative_prompt = "low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark, static, frozen, no motion, still image"
        
        print(f"    Prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")
        print(f"    Negative Prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
        
        # 显存优化：根据可用显存动态调整分辨率
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free_memory = total_memory - reserved
            
            print(f"  ℹ 显存状态: 总计={total_memory:.2f}GB, 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 可用={free_memory:.2f}GB")
            
            # 根据可用显存动态调整分辨率
            # attention计算需要约22GB显存，如果可用显存不足，需要大幅降低分辨率
            if free_memory < 25:
                # 显存不足，大幅降低分辨率
                print(f"  ⚠ 显存不足 ({free_memory:.1f}GB可用)，大幅降低分辨率以减少显存占用")
                if free_memory < 15:
                    # 显存严重不足，降低到最小分辨率
                    width = min(width, 384)
                    height = min(height, 256)
                    print(f"  ⚠ 显存严重不足，降低到最小分辨率")
                elif free_memory < 20:
                    # 显存不足，降低到较小分辨率
                    width = min(width, 512)
                    height = min(height, 320)
                else:
                    # 显存略不足，降低到中等分辨率
                    width = min(width, 640)
                    height = min(height, 384)
                
                # 确保是8的倍数
                width = (width // 8) * 8
                height = (height // 8) * 8
                print(f"  ℹ 分辨率已调整为: {width}x{height}")
        
        # 调整图像大小（如果需要）
        # 重要：保持长宽比，避免横向压缩或变形
        if image.size != (width, height):
            image_aspect = image.size[0] / image.size[1]
            target_aspect = width / height
            
            if abs(image_aspect - target_aspect) > 0.01:  # 长宽比不一致（误差>1%）
                print(f"  ⚠ 警告: 图像长宽比 ({image_aspect:.3f}) 与目标长宽比 ({target_aspect:.3f}) 不一致")
                print(f"  ℹ 图像尺寸: {image.size[0]}x{image.size[1]}")
                print(f"  ℹ 目标尺寸: {width}x{height}")
                
                # 使用保持长宽比的方式调整（避免变形）
                # 方法：先resize到目标尺寸的某个维度，然后裁剪或填充
                if image_aspect > target_aspect:
                    # 图像更宽，先调整高度，然后裁剪宽度
                    new_height = height
                    new_width = int(image.size[0] * (height / image.size[1]))
                    resized_image = image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                    # 居中裁剪
                    left = (new_width - width) // 2
                    image = resized_image.crop((left, 0, left + width, height))
                    print(f"  ℹ 图像已调整（保持长宽比，居中裁剪）: {image.size[0]}x{image.size[1]} -> {width}x{height}")
                else:
                    # 图像更高，先调整宽度，然后裁剪高度
                    new_width = width
                    new_height = int(image.size[1] * (width / image.size[0]))
                    resized_image = image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                    # 居中裁剪
                    top = (new_height - height) // 2
                    image = resized_image.crop((0, top, width, top + height))
                    print(f"  ℹ 图像已调整（保持长宽比，居中裁剪）: {image.size[0]}x{image.size[1]} -> {width}x{height}")
            else:
                # 长宽比一致，直接resize
                image = image.resize((width, height), PILImage.Resampling.LANCZOS)
                print(f"  ℹ 图像已调整（长宽比一致）: {image.size[0]}x{image.size[1]} -> {width}x{height}")
        else:
            print(f"  ℹ 图像分辨率与目标一致: {width}x{height}，无需调整")
        
        # 生成视频前清理显存
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            print(f"  ℹ 生成前显存占用: {allocated_before:.2f}GB")
        
        print(f"  开始生成视频（HunyuanVideo {'1.5' if use_v15 else '原版'}）...")
        try:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            # 允许从 scene / 配置读取 seed（用于重试时生成不同结果）
            seed = None
            try:
                if scene and scene.get("seed") is not None:
                    seed = int(scene["seed"])
                else:
                    seed = hunyuan_config.get("seed", None)
                    if seed is not None:
                        seed = int(seed)
            except Exception:
                seed = None

            if seed is None:
                seed = 42

            generator.manual_seed(seed)
            print(f"  ℹ 使用随机种子 seed={seed}")
            
            # 显存优化：根据可用显存动态减少num_frames
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free_memory = total_memory - reserved
                
                if free_memory < 25:
                    # 显存不足，减少帧数（attention计算需要大量显存）
                    original_num_frames = num_frames
                    if free_memory < 15:
                        # 显存严重不足，限制为15帧
                        num_frames = min(num_frames, 15)
                    elif free_memory < 20:
                        # 显存不足，限制为20帧
                        num_frames = min(num_frames, 20)
                    else:
                        # 显存略不足，限制为25帧
                        num_frames = min(num_frames, 25)
                    
                    if num_frames != original_num_frames:
                        print(f"  ⚠ 显存不足 ({free_memory:.1f}GB可用)，减少帧数: {original_num_frames} -> {num_frames}")
            
            # 调用HunyuanVideo pipeline
            # 1.5版本：不接受height/width参数，会自动从image计算
            # 原版：接受height/width参数
            use_v15 = hasattr(self, 'hunyuanvideo_pipeline') and hasattr(self.hunyuanvideo_pipeline, 'target_size')
            
            # 如果使用了sequential CPU offload，需要确保在调用前清理meta device
            if hasattr(self.hunyuanvideo_pipeline, '_sequential_offload'):
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 生成前再次检查显存，并在生成过程中监控
            if torch.cuda.is_available():
                # 更彻底地清理显存，释放碎片
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                allocated_before_gen = torch.cuda.memory_allocated() / 1024**3
                reserved_before_gen = torch.cuda.memory_reserved() / 1024**3
                max_allocatable = torch.cuda.get_device_properties(0).total_memory / 1024**3 * hunyuan_config.get('max_memory_fraction', 0.3)
                available_in_limit = max_allocatable - allocated_before_gen
                
                print(f"  ℹ 生成前显存: 已分配={allocated_before_gen:.2f}GB, 已保留={reserved_before_gen:.2f}GB")
                print(f"  ℹ 显存限制内可用: {available_in_limit:.2f}GB (限制={max_allocatable:.2f}GB)")
                
                # 如果可用显存不足，进一步减少帧数
                if available_in_limit < 8:  # 如果可用显存少于8GB，进一步减少帧数
                    original_num_frames = num_frames
                    if available_in_limit < 5:
                        num_frames = min(num_frames, 10)
                    elif available_in_limit < 6:
                        num_frames = min(num_frames, 12)
                    else:
                        num_frames = min(num_frames, 15)
                    
                    if num_frames != original_num_frames:
                        print(f"  ⚠ 显存限制内可用不足 ({available_in_limit:.1f}GB)，进一步减少帧数: {original_num_frames} -> {num_frames}")
            
            # 尝试生成视频，如果显存不足则自动降级
            max_retries = 3
            retry_count = 0
            result = None
            
            while retry_count < max_retries:
                try:
                    if use_v15:
                        # HunyuanVideo 1.5: 从image自动计算尺寸，不需要height/width参数
                        result = self.hunyuanvideo_pipeline(
                            image=image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            num_frames=num_frames,
                            generator=generator,
                            output_type="np",  # 明确指定输出numpy数组，值范围[0,1]
                        )
                    else:
                        # 原版HunyuanVideo: 需要height/width参数
                        result = self.hunyuanvideo_pipeline(
                            image=image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            num_frames=num_frames,
                            generator=generator,
                        )
                    # 成功生成，跳出循环
                    break
                    
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "out of memory" in error_msg or "cuda" in error_msg and "memory" in error_msg:
                        retry_count += 1
                        if retry_count >= max_retries:
                            # 最后一次重试失败，抛出异常
                            print(f"  ✗ 显存不足，已重试{max_retries}次，生成失败")
                            raise
                        
                        # 自动降级：减少帧数和推理步数
                        print(f"  ⚠ 显存不足，自动降级参数（重试 {retry_count}/{max_retries}）...")
                        
                        # 清理显存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                            torch.cuda.synchronize()
                        
                        # 降级策略
                        original_num_frames = num_frames
                        original_num_steps = num_inference_steps
                        
                        if retry_count == 1:
                            # 第一次降级：减少帧数和步数
                            num_frames = max(10, int(num_frames * 0.6))  # 减少到60%
                            num_inference_steps = max(15, int(num_inference_steps * 0.7))  # 减少到70%
                            print(f"  ℹ 降级参数: 帧数 {original_num_frames} -> {num_frames}, 步数 {original_num_steps} -> {num_inference_steps}")
                        elif retry_count == 2:
                            # 第二次降级：更激进
                            num_frames = max(8, int(num_frames * 0.5))  # 减少到50%
                            num_inference_steps = max(10, int(num_inference_steps * 0.6))  # 减少到60%
                            # 如果可能，也降低分辨率
                            if width > 512:
                                width = 512
                                height = int(height * 512 / width)
                                width = (width // 8) * 8
                                height = (height // 8) * 8
                                image = image.resize((width, height), PILImage.Resampling.LANCZOS)
                                print(f"  ℹ 降级分辨率: {width}x{height}")
                            print(f"  ℹ 进一步降级: 帧数 {original_num_frames} -> {num_frames}, 步数 {original_num_steps} -> {num_inference_steps}")
                        
                        # 等待一下再重试
                        import time
                        time.sleep(2)
                    else:
                        # 其他错误，直接抛出
                        raise
            
            # 获取生成的帧
            # 1.5版本返回HunyuanVideo15PipelineOutput，原版返回类似结构
            import numpy as np
            import torch
            
            if hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, dict) and 'frames' in result:
                frames = result['frames']
            elif isinstance(result, tuple) and len(result) > 0:
                frames = result[0]
            else:
                frames = result
            
            # 调试信息
            print(f"  ℹ 调试：result类型={type(result)}")
            print(f"  ℹ 调试：frames类型={type(frames)}")
            if hasattr(frames, 'shape'):
                print(f"  ℹ 调试：frames形状={frames.shape}")
            elif isinstance(frames, (list, tuple)):
                print(f"  ℹ 调试：frames长度={len(frames)}")
                if len(frames) > 0:
                    print(f"  ℹ 调试：frames[0]类型={type(frames[0])}")
            
            # 转换torch.Tensor为numpy，并立即释放GPU显存
            if isinstance(frames, torch.Tensor):
                frames_np = frames.cpu().numpy()
                del frames  # 删除GPU tensor
                frames = frames_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 立即清理显存
                print(f"  ℹ 已转换torch.Tensor为numpy，形状={frames.shape}")
            
            # 1.5版本可能返回列表，需要处理
            if isinstance(frames, list) and len(frames) > 0:
                # 如果是列表，取第一个元素（通常是视频帧序列）
                if isinstance(frames[0], (list, tuple)):
                    frames = frames[0]
                # 如果列表元素是torch.Tensor，转换为numpy
                elif isinstance(frames[0], torch.Tensor):
                    frames_list = []
                    for f in frames:
                        if isinstance(f, torch.Tensor):
                            f_np = f.cpu().numpy()
                            del f  # 删除GPU tensor
                            frames_list.append(f_np)
                        else:
                            frames_list.append(f)
                    frames = frames_list
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 立即清理显存
            
            # 处理frames格式（确保是numpy数组列表）
            video_frames = []
            
            if isinstance(frames, np.ndarray):
                # 如果是numpy数组，检查维度
                if len(frames.shape) == 5:
                    # (batch, num_frames, h, w, c) 格式，取第一个batch
                    print(f"  ℹ 检测到5D数组，形状={frames.shape}，取第一个batch")
                    frames = frames[0]
                if len(frames.shape) == 4:
                    # (num_frames, h, w, c) 格式
                    print(f"  ℹ 检测到4D数组，形状={frames.shape}，拆分为多个帧")
                    for i in range(frames.shape[0]):
                        frame = frames[i]
                        # 确保是(h, w, c)格式
                        if len(frame.shape) == 3:
                            # 确保值在0-255范围内
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                            
                            # 注意：不在这里应用色彩调整，统一在插帧完成后对所有帧应用一次
                            # 这样可以避免原始帧被调整两次（提取时一次，插帧后一次）
                            video_frames.append(frame)
                        else:
                            # 转换其他格式
                            frame = np.array(Image.fromarray(frame).convert("RGB"))
                            video_frames.append(frame)
                elif len(frames.shape) == 3:
                    # 单个帧 (h, w, c)
                    print(f"  ℹ 检测到3D数组，形状={frames.shape}，作为单帧处理")
                    if frames.max() <= 1.0:
                        frames = (frames * 255).astype(np.uint8)
                    else:
                        frames = frames.astype(np.uint8)
                    
                    # 注意：不在这里应用色彩调整，统一在插帧完成后对所有帧应用一次
                    video_frames.append(frames)
                else:
                    raise ValueError(f"无法处理numpy数组维度：{frames.shape}")
            elif isinstance(frames, (list, tuple)):
                # 如果是列表，处理每个帧
                print(f"  ℹ 检测到列表/元组，长度={len(frames)}")
                for i, frame in enumerate(frames):
                    if isinstance(frame, torch.Tensor):
                        frame_np = frame.cpu().numpy()
                        del frame  # 删除GPU tensor
                        frame = frame_np
                        if i % 10 == 0 and torch.cuda.is_available():  # 每10帧清理一次
                            torch.cuda.empty_cache()
                    if isinstance(frame, Image.Image):
                        frame = np.array(frame.convert("RGB"))
                    elif isinstance(frame, np.ndarray):
                        if len(frame.shape) == 3:
                            # (h, w, c)
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                            
                            # 注意：不在这里应用色彩调整，统一在插帧完成后对所有帧应用一次
                        elif len(frame.shape) == 2:
                            # (h, w) 灰度图，转换为RGB
                            frame = np.stack([frame] * 3, axis=-1)
                            if frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                    video_frames.append(frame)
            else:
                raise ValueError(f"无法处理frames格式: {type(frames)}")
            
            # 检查video_frames是否为空
            if len(video_frames) == 0:
                raise ValueError(f"video_frames为空！frames类型={type(frames)}, frames值={frames}")
            
            print(f"  ℹ 已提取 {len(video_frames)} 帧，第一帧形状={video_frames[0].shape}")
            
            # 如果生成的帧数少于请求的帧数，使用插帧补充
            if len(video_frames) < num_frames:
                print(f"  ⚠ 生成的帧数({len(video_frames)})少于请求的帧数({num_frames})，使用插帧补充")
                video_frames = self._interpolate_frames_rife(video_frames, num_frames)
                print(f"  ✓ 插帧后帧数: {len(video_frames)} 帧")
            
            # 统一对所有帧应用色彩调整（包括原始帧和插帧生成的帧）
            # 这样可以确保所有帧都使用相同的调整参数，且只调整一次
            # 注意：如果所有调整因子都是1.0，则跳过调整（保持原始输出）
            if brightness_factor != 1.0 or contrast_factor != 1.0 or saturation_factor != 1.0:
                print(f"  🔧 对所有帧应用色彩调整（brightness={brightness_factor}, contrast={contrast_factor}, saturation={saturation_factor}）")
                
                # 添加调试信息：检查第一帧的原始值范围
                if len(video_frames) > 0:
                    first_frame = video_frames[0]
                    if len(first_frame.shape) == 3:
                        print(f"  ℹ 第一帧原始值范围: min={first_frame.min()}, max={first_frame.max()}, mean={first_frame.mean():.1f}")
                
                adjusted_frames = []
                for i, frame in enumerate(video_frames):
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        # 转换为float32进行运算
                        frame_float = frame.astype(np.float32)
                        
                        # 1. 调整亮度（线性缩放）
                        if brightness_factor != 1.0:
                            frame_float = frame_float * brightness_factor
                        
                        # 2. 调整对比度（以128为中心点，因为值已经在[0,255]范围内）
                        if contrast_factor != 1.0:
                            # 对比度调整：以128为中心，增强或减弱对比度
                            frame_float = (frame_float - 128.0) * contrast_factor + 128.0
                        
                        # 3. 限制值范围到[0, 255]
                        frame_float = np.clip(frame_float, 0, 255)
                        
                        # 4. 调整饱和度（在HSV色彩空间中）
                        if saturation_factor != 1.0:
                            from PIL import Image as PILImage
                            # 转换为PIL Image进行HSV转换
                            frame_pil = PILImage.fromarray(frame_float.astype(np.uint8))
                            # 转换为HSV色彩空间
                            frame_hsv = np.array(frame_pil.convert('HSV'))
                            # 调整饱和度（S通道，范围0-255）
                            frame_hsv[:, :, 1] = np.clip(
                                (frame_hsv[:, :, 1].astype(np.float32) * saturation_factor),
                                0, 255
                            ).astype(np.uint8)
                            # 转回RGB
                            frame_corrected = PILImage.fromarray(frame_hsv, mode='HSV').convert('RGB')
                            frame = np.array(frame_corrected)
                        else:
                            frame = frame_float.astype(np.uint8)
                        
                        # 添加调试信息：检查调整后的值范围（仅第一帧）
                        if i == 0:
                            print(f"  ℹ 第一帧调整后值范围: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                    else:
                        frame = frame  # 保持原样
                    
                    adjusted_frames.append(frame)
                video_frames = adjusted_frames
                print(f"  ✓ 色彩调整完成（共{len(video_frames)}帧）")
            else:
                print(f"  ℹ 跳过色彩调整（所有调整因子均为1.0，保持原始输出）")
            
            # 保存视频
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            from diffusers.utils import export_to_video
            from PIL import Image as PILImage
            
            # 重要修复：export_to_video 会假设所有 np.ndarray 帧都在 [0,1] 范围内并乘以255
            # 但我们的帧已经是 [0,255] 范围的 uint8，需要转换为 PIL Image 避免重复处理
            # 或者确保帧的 dtype 和值范围正确
            export_frames = []
            for frame in video_frames:
                # 确保帧是 uint8 类型，值在 [0, 255] 范围内
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                    else:
                        # 已经是 uint8，确保值在范围内
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    # 转换为 PIL Image，这样 export_to_video 就不会再次乘以255
                    frame_pil = PILImage.fromarray(frame, 'RGB')
                    export_frames.append(frame_pil)
                elif isinstance(frame, PILImage.Image):
                    export_frames.append(frame)
                else:
                    # 其他类型，尝试转换
                    frame_array = np.array(frame)
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
                    export_frames.append(PILImage.fromarray(frame_array, 'RGB'))
            
            export_to_video(export_frames, str(output_path), fps=fps)
            
            # 清理中间变量和显存
            del video_frames, frames, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                print(f"  ℹ 生成后显存占用: {allocated_after:.2f}GB")
            
            print(f"  ✓ HunyuanVideo视频生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            # 发生错误时也清理显存
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print(f"  ✗ HunyuanVideo视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_video_cogvideox(
        self,
        image_path: str,
        output_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        num_frames: int = 81,
        fps: int = 16,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """使用CogVideoX生成视频（图生视频）"""
        print(f"  使用CogVideoX生成视频")
        print(f"    参数: num_frames={num_frames}, fps={fps}")
        
        if not self.model_loaded or self.cogvideox_pipeline is None:
            self.load_model()
        
        # 生成前清理显存
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 加载图像
        from diffusers.utils import load_image
        image = load_image(image_path)
        
        # 获取CogVideoX配置
        cogvideox_config = self.video_config.get('cogvideox', {})
        
        # 检查可用显存，如果不足则降低参数
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            free_memory = available_memory - allocated_memory
            
            print(f"  ℹ GPU显存状态: 总计={available_memory:.2f}GB, 已分配={allocated_memory:.2f}GB, 可用={free_memory:.2f}GB")
            
            # 如果可用显存少于15GB，降低分辨率
            if free_memory < 15:
                print(f"  ⚠ 可用显存不足，降低分辨率以节省显存")
                width = 1024  # 降低到1024
                height = 576  # 降低到576（保持16:9比例）
            else:
                width = cogvideox_config.get('width', self.video_config.get('width', 1360))
                height = cogvideox_config.get('height', self.video_config.get('height', 768))
            
            # 如果可用显存少于10GB，进一步降低帧数
            if free_memory < 10:
                print(f"  ⚠ 可用显存严重不足，降低帧数以节省显存")
                num_frames = min(num_frames, 49)  # 降低到49帧
        else:
            width = cogvideox_config.get('width', self.video_config.get('width', 1360))
            height = cogvideox_config.get('height', self.video_config.get('height', 768))
        
        num_inference_steps = cogvideox_config.get('num_inference_steps', 50)
        guidance_scale = cogvideox_config.get('guidance_scale', 6.0)
        use_dynamic_cfg = cogvideox_config.get('use_dynamic_cfg', True)
        
        # 获取prompt和negative_prompt（优先使用传入的参数，否则从配置读取）
        config_prompt = cogvideox_config.get('prompt', '')
        config_negative_prompt = cogvideox_config.get('negative_prompt', '')
        
        # 构建详细的prompt（如果传入的prompt为空）
        if not prompt:
            if config_prompt:
                prompt = config_prompt
            else:
                # 从图片路径提取信息，构建基础prompt
                prompt = self._build_detailed_prompt(image_path, image, scene, model_type="cogvideox")
        
        # 如果prompt仍然为空或太简单，使用默认值
        if not prompt or len(prompt) < 20:
            prompt = "high quality video, smooth motion, cinematic, natural movement, stable camera"
        
        # 如果传入的negative_prompt为空，使用配置中的negative_prompt
        if not negative_prompt and config_negative_prompt:
            negative_prompt = config_negative_prompt
        # 如果仍然为空，使用默认值
        if not negative_prompt:
            negative_prompt = "low quality, blurry, distorted, deformed, bad anatomy, bad hands, text, watermark, static, frozen, no motion, still image"
        
        print(f"    Prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")
        print(f"    Negative Prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
        
        # 确保分辨率是vae_scale_factor_spatial的倍数（通常是8）
        # 获取pipeline的vae_scale_factor_spatial（需要在模型加载后）
        if self.cogvideox_pipeline is not None and hasattr(self.cogvideox_pipeline, 'vae_scale_factor_spatial'):
            vae_scale_factor = self.cogvideox_pipeline.vae_scale_factor_spatial
        else:
            vae_scale_factor = 8  # 默认值
        
        # 调整分辨率使其是vae_scale_factor的倍数
        width = (width // vae_scale_factor) * vae_scale_factor
        height = (height // vae_scale_factor) * vae_scale_factor
        
        # 确保帧数符合要求（必须是vae_scale_factor_temporal的倍数+1）
        if self.cogvideox_pipeline is not None and hasattr(self.cogvideox_pipeline, 'vae_scale_factor_temporal'):
            vae_scale_factor_temporal = self.cogvideox_pipeline.vae_scale_factor_temporal
        else:
            vae_scale_factor_temporal = 4  # 默认值
        
        # 调整帧数使其符合要求
        # CogVideoX要求: (num_frames - 1) 必须是 vae_scale_factor_temporal 的倍数
        if (num_frames - 1) % vae_scale_factor_temporal != 0:
            # 调整到最近的符合要求的帧数
            num_frames = ((num_frames - 1) // vae_scale_factor_temporal + 1) * vae_scale_factor_temporal + 1
            print(f"  ℹ 帧数已调整为符合要求: {num_frames} (vae_scale_factor_temporal={vae_scale_factor_temporal})")
        
        print(f"  ℹ 最终参数: 分辨率={width}x{height} (vae_scale_factor={vae_scale_factor}), 帧数={num_frames} (vae_scale_factor_temporal={vae_scale_factor_temporal})")
        
        # 注意：不要在这里调整图像大小！
        # CogVideoX pipeline 会在内部通过 video_processor.preprocess 处理图像
        # 如果我们提前调整，可能导致尺寸不匹配
        # pipeline 会确保图像尺寸与 height/width 参数匹配
        
        # 生成视频前彻底清理显存
        if torch.cuda.is_available():
            # 多次清理以确保释放所有可释放的显存
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
            
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            print(f"  ℹ 生成前显存: 已分配={allocated_before:.2f}GB, 已保留={reserved_before:.2f}GB")
            
            # 如果保留的显存过多，尝试释放
            if reserved_before > allocated_before * 1.5:
                print(f"  ℹ 检测到显存碎片，尝试释放...")
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"  开始生成视频（CogVideoX）...")
        print(f"    分辨率: {width}x{height}, 帧数: {num_frames}, 步数: {num_inference_steps}")
        
        try:
            # 确保使用CPU offload（如果启用）
            if cogvideox_config.get('enable_model_cpu_offload', True):
                # 确保pipeline已启用CPU offload
                if hasattr(self.cogvideox_pipeline, 'enable_model_cpu_offload'):
                    try:
                        self.cogvideox_pipeline.enable_model_cpu_offload()
                    except:
                        pass  # 如果已经启用，忽略错误
            
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(42)
            
            # 调用CogVideoX pipeline
            # 注意：必须传递height和width参数，确保pipeline使用正确的尺寸
            # 重要：使用output_type="np"让pipeline自动处理值范围转换（[-1,1] -> [0,255]）
            result = self.cogvideox_pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                height=height,  # 传递调整后的高度
                width=width,    # 传递调整后的宽度
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                generator=generator,
                output_type="np",  # 指定输出numpy数组，pipeline会自动处理值范围
            )
            
            # 提取frames（CogVideoX返回CogVideoXPipelineOutput）
            if hasattr(result, 'frames'):
                frames = result.frames
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                frames = result[0]
            else:
                frames = result
            
            # 添加调试信息
            print(f"  [DEBUG] result类型: {type(result)}")
            print(f"  [DEBUG] frames类型: {type(frames)}")
            if hasattr(frames, 'shape'):
                print(f"  [DEBUG] frames形状: {frames.shape}")
            elif isinstance(frames, (list, tuple)):
                print(f"  [DEBUG] frames是列表/元组，长度: {len(frames)}")
                if len(frames) > 0:
                    print(f"  [DEBUG] 第一帧类型: {type(frames[0])}, 形状: {getattr(frames[0], 'shape', 'N/A')}")
            
            # 转换frames为numpy数组列表
            import numpy as np
            video_frames = []
            
            if isinstance(frames, torch.Tensor):
                frames_np = frames.cpu().numpy()
                print(f"  [DEBUG] Tensor转numpy后形状: {frames_np.shape}")
                del frames
                frames = frames_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if isinstance(frames, np.ndarray):
                print(f"  [DEBUG] frames是numpy数组，形状: {frames.shape}, dtype: {frames.dtype}")
                if len(frames.shape) == 5:
                    # [B, F, H, W, C] -> [F, H, W, C]
                    frames = frames[0]
                    print(f"  [DEBUG] 去除batch维度后形状: {frames.shape}")
                if len(frames.shape) == 4:
                    # [F, H, W, C] 或 [F, C, H, W]
                    print(f"  [DEBUG] 4D数组，检查通道维度位置...")
                    # 检查通道维度：如果最后一维是3或4，说明是 [F, H, W, C]
                    # 如果第二维是3或4，说明是 [F, C, H, W]
                    if frames.shape[-1] in [1, 2, 3, 4]:
                        # [F, H, W, C] 格式
                        print(f"  [DEBUG] 格式: [F, H, W, C]，帧数={frames.shape[0]}")
                        for i in range(frames.shape[0]):
                            frame = frames[i]
                            # 确保是 [H, W, C] 格式
                            if len(frame.shape) == 3:
                                # 添加调试信息
                                if i < 3:  # 只打印前3帧的详细信息
                                    print(f"  [DEBUG] 帧{i}转换前: dtype={frame.dtype}, min={frame.min():.4f}, max={frame.max():.4f}, mean={frame.mean():.4f}")
                                
                                # 处理值范围：pipeline的postprocess_video将[-1,1]转换为[0,1]
                                # 需要将[0,1]转换为[0,255]
                                if frame.max() <= 1.0 and frame.min() >= 0.0:
                                    # 值在[0,1]范围内，直接转换为[0,255]
                                    # 不要进行对比度增强，直接使用pipeline返回的结果
                                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                                    
                                    if i < 3:
                                        print(f"  [DEBUG] 帧{i}转换后([0,1]->[0,255]): dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}, range={frame.max()-frame.min()}")
                                        # 检查是否有异常值
                                        unique_values = np.unique(frame)
                                        if len(unique_values) < 10:
                                            print(f"  [WARNING] 帧{i}只有{len(unique_values)}个不同的值，可能有问题")
                                        # 检查通道分布
                                        for c in range(3):
                                            channel = frame[:, :, c]
                                            print(f"  [DEBUG] 帧{i}通道{c}: min={channel.min()}, max={channel.max()}, mean={channel.mean():.1f}, std={channel.std():.1f}")
                                elif frame.min() >= -1.0 and frame.max() <= 1.0:
                                    # 值在[-1,1]范围内，转换为[0,255]
                                    frame = ((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                                    if i < 3:
                                        print(f"  [DEBUG] 帧{i}转换后([-1,1]->[0,255]): dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                                elif frame.max() > 255 or frame.min() < 0:
                                    # 值超出范围，先clip再转换
                                    frame = frame.clip(0, 255).astype(np.uint8)
                                    if i < 3:
                                        print(f"  [DEBUG] 帧{i}转换后(clip->[0,255]): dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                                else:
                                    # 值已经在[0,255]范围内，直接转换
                                    frame = frame.astype(np.uint8)
                                    if i < 3:
                                        print(f"  [DEBUG] 帧{i}转换后(直接转换): dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                                video_frames.append(frame)
                            else:
                                video_frames.append(frame)
                    elif frames.shape[1] in [1, 2, 3, 4]:
                        # [F, C, H, W] 格式，需要转换为 [F, H, W, C]
                        print(f"  [DEBUG] 格式: [F, C, H, W]，转换为 [F, H, W, C]，帧数={frames.shape[0]}")
                        frames = np.transpose(frames, (0, 2, 3, 1))
                        for i in range(frames.shape[0]):
                            frame = frames[i]
                            # 处理值范围：pipeline的postprocess_video将[-1,1]转换为[0,1]
                            # 需要将[0,1]转换为[0,255]
                            if frame.max() <= 1.0 and frame.min() >= 0.0:
                                # 值在[0,1]范围内，转换为[0,255]
                                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                            elif frame.min() >= -1.0 and frame.max() <= 1.0:
                                # 值在[-1,1]范围内，转换为[0,255]
                                frame = ((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                            elif frame.max() > 255 or frame.min() < 0:
                                # 值超出范围，先clip再转换
                                frame = frame.clip(0, 255).astype(np.uint8)
                            else:
                                # 值已经在[0,255]范围内，直接转换
                                frame = frame.astype(np.uint8)
                            video_frames.append(frame)
                    else:
                        # 不确定格式，尝试按第一维处理
                        print(f"  [DEBUG] 不确定格式，按第一维处理，帧数={frames.shape[0]}")
                        for i in range(frames.shape[0]):
                            video_frames.append(frames[i])
                elif len(frames.shape) == 3:
                    # [H, W, C] 单帧
                    print(f"  [DEBUG] 3D数组（单帧），形状: {frames.shape}")
                    # 确保值在 [0, 255] 范围内
                    if frames.max() <= 1.0:
                        frames = (frames * 255).astype(np.uint8)
                    else:
                        frames = frames.astype(np.uint8)
                    video_frames.append(frames)
                else:
                    print(f"  [DEBUG] 其他维度，形状: {frames.shape}")
                    video_frames = [frames]
            elif isinstance(frames, (list, tuple)):
                print(f"  [DEBUG] frames是列表/元组，长度: {len(frames)}")
                for i, f in enumerate(frames):
                    if isinstance(f, torch.Tensor):
                        f_np = f.cpu().numpy()
                        print(f"  [DEBUG] 帧{i}: Tensor转numpy，形状: {f_np.shape}")
                        # 处理多帧数组：如果第一维是帧数，需要拆分
                        if len(f_np.shape) == 4 and f_np.shape[0] > 1 and f_np.shape[-1] in [1, 2, 3, 4]:
                            # [F, H, W, C] 格式，需要拆分成多帧
                            print(f"  [DEBUG] 检测到多帧数组 [F, H, W, C]，帧数={f_np.shape[0]}，拆分成单独帧")
                            for frame_idx in range(f_np.shape[0]):
                                frame = f_np[frame_idx]
                                # 确保值在 [0, 255] 范围内
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                                video_frames.append(frame)
                        else:
                            # 单帧处理
                            if len(f_np.shape) == 4:
                                f_np = f_np[0]  # 去除batch维度
                            if len(f_np.shape) == 3 and f_np.shape[0] in [1, 2, 3, 4]:
                                # [C, H, W] -> [H, W, C]
                                f_np = np.transpose(f_np, (1, 2, 0))
                            # 确保值在 [0, 255] 范围内
                            if f_np.max() <= 1.0:
                                f_np = (f_np * 255).astype(np.uint8)
                            else:
                                f_np = f_np.astype(np.uint8)
                            video_frames.append(f_np)
                        del f
                    elif isinstance(f, np.ndarray):
                        print(f"  [DEBUG] 帧{i}: numpy数组，形状: {f.shape}")
                        # 处理多帧数组：如果第一维是帧数，需要拆分
                        if len(f.shape) == 4 and f.shape[0] > 1 and f.shape[-1] in [1, 2, 3, 4]:
                            # [F, H, W, C] 格式，需要拆分成多帧
                            print(f"  [DEBUG] 检测到多帧数组 [F, H, W, C]，帧数={f.shape[0]}，拆分成单独帧")
                            for frame_idx in range(f.shape[0]):
                                frame = f[frame_idx]
                                # 确保值在 [0, 255] 范围内
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                                video_frames.append(frame)
                        else:
                            # 单帧处理
                            if len(f.shape) == 4:
                                f = f[0]  # 去除batch维度
                            if len(f.shape) == 3 and f.shape[0] in [1, 2, 3, 4]:
                                # [C, H, W] -> [H, W, C]
                                f = np.transpose(f, (1, 2, 0))
                            # 确保值在 [0, 255] 范围内
                            if f.max() <= 1.0:
                                f = (f * 255).astype(np.uint8)
                            else:
                                f = f.astype(np.uint8)
                            video_frames.append(f)
                    else:
                        # 假设是PIL Image或列表
                        f_np = np.array(f)
                        print(f"  [DEBUG] 帧{i}: 转换为numpy，形状: {f_np.shape}, dtype: {f_np.dtype}")
                        # 处理多帧数组
                        if len(f_np.shape) == 4 and f_np.shape[0] > 1 and f_np.shape[-1] in [1, 2, 3, 4]:
                            # [F, H, W, C] 格式，需要拆分成多帧
                            print(f"  [DEBUG] 检测到多帧数组 [F, H, W, C]，帧数={f_np.shape[0]}，拆分成单独帧")
                            for frame_idx in range(f_np.shape[0]):
                                frame = f_np[frame_idx].copy()  # 使用copy避免视图问题
                                # 确保是 [H, W, C] 格式
                                if len(frame.shape) != 3 or frame.shape[2] not in [1, 2, 3, 4]:
                                    print(f"  [DEBUG] 警告：帧{frame_idx}形状异常: {frame.shape}")
                                # 确保值在 [0, 255] 范围内
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                elif frame.dtype != np.uint8:
                                    frame = frame.astype(np.uint8)
                                video_frames.append(frame)
                            print(f"  [DEBUG] 已拆分 {f_np.shape[0]} 帧")
                        elif len(f_np.shape) == 3:
                            # 单帧 [H, W, C] 格式
                            print(f"  [DEBUG] 单帧格式 [H, W, C]")
                            # 确保值在 [0, 255] 范围内
                            if f_np.max() <= 1.0:
                                f_np = (f_np * 255).astype(np.uint8)
                            elif f_np.dtype != np.uint8:
                                f_np = f_np.astype(np.uint8)
                            video_frames.append(f_np)
                        else:
                            print(f"  [DEBUG] 其他格式，直接添加: {f_np.shape}")
                            video_frames.append(f_np)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # 假设是PIL Image或其他类型
                print(f"  [DEBUG] frames是其他类型，假设是PIL Image或数组")
                f_np = np.array(frames)
                print(f"  [DEBUG] 转换后形状: {f_np.shape}, dtype: {f_np.dtype}")
                # 处理多帧数组
                if len(f_np.shape) == 4 and f_np.shape[0] > 1 and f_np.shape[-1] in [1, 2, 3, 4]:
                    # [F, H, W, C] 格式，需要拆分成多帧
                    print(f"  [DEBUG] 检测到多帧数组 [F, H, W, C]，帧数={f_np.shape[0]}，拆分成单独帧")
                    for frame_idx in range(f_np.shape[0]):
                        frame = f_np[frame_idx].copy()  # 使用copy避免视图问题
                        # 确保是 [H, W, C] 格式
                        if len(frame.shape) != 3 or frame.shape[2] not in [1, 2, 3, 4]:
                            print(f"  [DEBUG] 警告：帧{frame_idx}形状异常: {frame.shape}")
                        # 确保值在 [0, 255] 范围内
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        elif frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        video_frames.append(frame)
                    print(f"  [DEBUG] 已拆分 {f_np.shape[0]} 帧")
                elif len(f_np.shape) == 3:
                    # 单帧 [H, W, C] 格式
                    print(f"  [DEBUG] 单帧格式 [H, W, C]")
                    # 确保值在 [0, 255] 范围内
                    if f_np.max() <= 1.0:
                        f_np = (f_np * 255).astype(np.uint8)
                    elif f_np.dtype != np.uint8:
                        f_np = f_np.astype(np.uint8)
                    video_frames = [f_np]
                else:
                    print(f"  [DEBUG] 其他格式，直接添加: {f_np.shape}")
                    video_frames = [f_np]
            
            if not video_frames:
                raise ValueError("生成的视频帧为空")
            
            # 验证每帧的格式
            print(f"  [DEBUG] 最终video_frames数量: {len(video_frames)}")
            for i, frame in enumerate(video_frames[:3]):  # 只检查前3帧
                print(f"  [DEBUG] 帧{i}: 形状={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
            
            print(f"  ✓ 生成完成，共 {len(video_frames)} 帧")
            
            # 保存第一帧作为调试图像
            if len(video_frames) > 0:
                debug_frame_path = output_path.replace('.mp4', '_frame0_debug.png')
                try:
                    from PIL import Image
                    debug_frame = Image.fromarray(video_frames[0], 'RGB')
                    debug_frame.save(debug_frame_path)
                    print(f"  ✓ 调试图像已保存: {debug_frame_path}")
                except Exception as e:
                    print(f"  ⚠ 保存调试图像失败: {e}")
            
            # 导出视频
            from diffusers.utils import export_to_video
            export_to_video(video_frames, output_path, fps=fps)
            print(f"  ✓ 视频已保存: {output_path}")
            
            # 清理显存
            del video_frames, frames, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return output_path
            
        except Exception as e:
            print(f"  ✗ CogVideoX视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_static_image_animation(
        self,
        image_path: str,
        output_path: str,
        num_frames: int,
        fps: float,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成静态图像动画（Deforum风格相机运动）"""
        print(f"  使用Deforum风格静态图像动画")
        
        # 加载图像
        image = Image.open(image_path)
        width, height = image.size
        
        # 根据场景生成相机运动参数
        if scene:
            motion_params = self.deforum_motion.generate_motion_params_from_scene(scene)
            print(f"  ℹ 根据场景生成相机运动参数:")
            print(f"    - Zoom: {motion_params['zoom']['start']:.2f} → {motion_params['zoom']['end']:.2f}")
            print(f"    - Pan X: {motion_params['pan_x']['start']:.2f} → {motion_params['pan_x']['end']:.2f}")
            print(f"    - Pan Y: {motion_params['pan_y']['start']:.2f} → {motion_params['pan_y']['end']:.2f}")
            print(f"    - Rotate: {motion_params['rotate']['start']:.1f}° → {motion_params['rotate']['end']:.1f}°")
        else:
            # 默认参数（轻微zoom）
            motion_params = {
                "zoom": {"start": 1.0, "end": 1.1},
                "pan_x": {"start": 0.0, "end": 0.0},
                "pan_y": {"start": 0.0, "end": 0.0},
                "rotate": {"start": 0.0, "end": 0.0},
            }
        
        # 应用Deforum风格相机运动
        frames = self.deforum_motion.apply_camera_motion(
            image=image,
            num_frames=num_frames,
            motion_params=motion_params,
            curve="ease_in_out",  # 使用缓入缓出曲线，更自然
        )
        
        # 保存视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        
        print(f"  ✓ Deforum风格静态图像动画生成成功: {output_path}")
        return output_path
    
    def _load_rife_model(self):
        """加载插帧模型（懒加载，支持多种方法，不抛出异常）"""
        if self._rife_model is not None:
            return self._rife_model
        
        # 方法1：尝试使用官方 RIFE 实现（如果已安装）
        try:
            import sys
            import torch
            rife_path = Path(__file__).parent.parent / "RIFE"
            if rife_path.exists():
                sys.path.insert(0, str(rife_path))
                
                # 尝试加载不同版本的 RIFE 模型（按优先级）
                model_loaded = False
                
                # 尝试 v3 HD 模型（从 train_log 目录导入）
                try:
                    # 先尝试从 train_log 目录导入（解压后的位置）
                    train_log_path = rife_path / "train_log" / "train_log"
                    if train_log_path.exists():
                        sys.path.insert(0, str(train_log_path.parent))
                        from train_log.RIFE_HDv3 import Model
                        self._rife_model = Model()
                        self._rife_model.load_model(str(train_log_path), -1)
                        self._rife_type = "official"
                        print("  ✓ RIFE 模型加载成功（使用官方实现 v3 HD）")
                        model_loaded = True
                except Exception as e:
                    pass
                
                # 尝试 v2 HD 模型
                if not model_loaded:
                    try:
                        from model.RIFE_HDv2 import Model
                        self._rife_model = Model()
                        train_log_path = rife_path / "train_log" / "train_log"
                        if not train_log_path.exists():
                            train_log_path = rife_path / "train_log"
                        self._rife_model.load_model(str(train_log_path), -1)
                        self._rife_type = "official"
                        print("  ✓ RIFE 模型加载成功（使用官方实现 v2 HD）")
                        model_loaded = True
                    except Exception as e:
                        pass
                
                # 尝试标准模型
                if not model_loaded:
                    try:
                        from model.RIFE import Model
                        self._rife_model = Model()
                        train_log_path = rife_path / "train_log" / "train_log"
                        if not train_log_path.exists():
                            train_log_path = rife_path / "train_log"
                        self._rife_model.load_model(str(train_log_path), -1)
                        self._rife_type = "official"
                        print("  ✓ RIFE 模型加载成功（使用官方实现标准版）")
                        model_loaded = True
                    except Exception as e:
                        pass
                
                if model_loaded:
                    # 设置为评估模式
                    self._rife_model.eval()
                    if torch.cuda.is_available():
                        self._rife_model.device()
                    return self._rife_model
                else:
                    raise ImportError("无法加载任何 RIFE 模型版本")
                    
        except Exception as e:
            print(f"  ℹ RIFE 官方实现不可用: {e}")
        
        # 方法2：使用 OpenCV 光流插帧（简单但有效，自动可用）
        # 注意：这是降级方案，如果安装了 RIFE 官方实现，会优先使用 RIFE
        try:
            import cv2
            self._rife_model = "opencv"  # 标记使用 OpenCV
            self._rife_type = "opencv"
            print("  ℹ 使用 OpenCV 光流插帧（降级方案，如需最佳效果请安装 RIFE 官方实现）")
            print("  ℹ 提示：安装 RIFE 官方实现可提升插帧质量")
            print("  ℹ 安装方法：git clone https://github.com/hzwer/arXiv2020-RIFE.git RIFE")
            return self._rife_model
        except ImportError:
            pass
        
        # 方法3：使用简单线性插值（降级方案，保证可用）
        self._rife_model = "linear"
        self._rife_type = "linear"
        print("  ℹ 使用简单线性插值（降级方案）")
        return self._rife_model
    
    def _interpolate_frames_rife(
        self,
        frames: List[np.ndarray],
        target_frames: int,
    ) -> List[np.ndarray]:
        """
        使用 RIFE 插帧增加帧数
        
        Args:
            frames: 输入帧列表（numpy 数组，形状为 (h, w, 3)，值范围 0-255）
            target_frames: 目标帧数
        
        Returns:
            插帧后的帧列表
        """
        if len(frames) >= target_frames:
            print(f"  ℹ 当前帧数 ({len(frames)}) 已满足目标 ({target_frames})，跳过插帧")
            return frames
        
        # 计算需要插帧的倍数
        current_frames = len(frames)
        if current_frames == 0:
            print(f"  ⚠ 输入帧数为0，跳过插帧")
            return frames
        
        scale = target_frames / current_frames if current_frames > 0 else 1.0
        
        try:
            # 加载插帧模型（不抛出异常，总是返回一个可用的方法）
            if self._rife_model is None:
                self._load_rife_model()
            
            print(f"  ℹ 开始插帧: {current_frames} 帧 → {target_frames} 帧 (倍数: {scale:.2f}x)")
            
            # 使用 RIFE 插帧
            interpolated_frames = []
            
            # 确保帧格式正确（PIL Image 或 numpy array）
            # 转换为 numpy array，确保格式为 (h, w, 3)，值范围 0-255
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame.convert("RGB"))
                elif isinstance(frame, np.ndarray):
                    # 确保是 uint8 格式
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    # 确保是 RGB 格式
                    if len(frame.shape) == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    elif frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                processed_frames.append(frame)
            
            # 方法1：使用官方 RIFE 实现
            if getattr(self, '_rife_type', None) == 'official':
                import torch
                # 官方实现接口：inference(I0, I1) -> mid
                # RIFE 的 inference 方法需要 torch tensor 输入
                for i in range(len(processed_frames) - 1):
                    interpolated_frames.append(processed_frames[i])
                    
                    num_intermediate = max(1, int(scale) - 1)
                    prev_frame = processed_frames[i]
                    next_frame = processed_frames[i + 1]
                    
                    for _ in range(num_intermediate):
                        try:
                            # 转换为 torch tensor（RIFE 需要的格式）
                            # RIFE 期望输入是 (1, 3, H, W) 格式的 tensor，值范围 0-1
                            # 确保是连续的内存布局
                            prev_frame_np = np.ascontiguousarray(prev_frame)
                            next_frame_np = np.ascontiguousarray(next_frame)
                            
                            I0 = torch.from_numpy(prev_frame_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            I1 = torch.from_numpy(next_frame_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            
                            if torch.cuda.is_available():
                                I0 = I0.cuda()
                                I1 = I1.cuda()
                            
                            # 调用 RIFE inference
                            # RIFE_HDv3 的 inference 方法签名：inference(img0, img1, scale=1.0)
                            # 标准 RIFE 的 inference 方法签名：inference(img0, img1, scale=1, scale_list=None, TTA=False, timestep=0.5)
                            with torch.no_grad():
                                # 尝试使用 timestep 参数（标准 RIFE）
                                try:
                                    mid_frame_tensor = self._rife_model.inference(I0, I1, timestep=0.5)
                                except TypeError:
                                    # 如果不支持 timestep，使用 scale 参数（RIFE_HDv3）
                                    mid_frame_tensor = self._rife_model.inference(I0, I1, scale=1.0)
                            
                            # 转换回 numpy array
                            if torch.cuda.is_available():
                                mid_frame_tensor = mid_frame_tensor.cpu()
                            mid_frame_np = mid_frame_tensor.squeeze(0).permute(1, 2, 0).numpy()
                            # 确保值范围在 0-1，然后转换为 0-255
                            # 注意：RIFE可能输出超出[0,1]范围的值，需要clip
                            mid_frame_np = np.clip(mid_frame_np, 0, 1)
                            mid_frame = (mid_frame_np * 255.0).astype(np.uint8)
                            
                            # RIFE插帧后的帧可能色彩过浓，这里先不做调整
                            # 统一在插帧完成后对所有帧（包括插帧生成的）应用色彩调整
                            interpolated_frames.append(mid_frame)
                            prev_frame = mid_frame
                        except Exception as e:
                            print(f"  ⚠ RIFE 插值失败，使用线性插值: {e}")
                            mid_frame = ((prev_frame.astype(np.float32) + next_frame.astype(np.float32)) / 2).astype(np.uint8)
                            interpolated_frames.append(mid_frame)
                            prev_frame = mid_frame
                
                interpolated_frames.append(processed_frames[-1])
            
            # 方法2：使用 OpenCV 光流插帧
            elif getattr(self, '_rife_type', None) == 'opencv':
                import cv2
                for i in range(len(processed_frames) - 1):
                    interpolated_frames.append(processed_frames[i])
                    
                    num_intermediate = max(1, int(scale) - 1)
                    prev_frame = processed_frames[i]
                    next_frame = processed_frames[i + 1]
                    
                    # 使用 OpenCV 光流插值
                    for j in range(num_intermediate):
                        t = (j + 1) / (num_intermediate + 1)
                        # 使用加权平均（简单但有效）
                        mid_frame = cv2.addWeighted(
                            prev_frame, 1.0 - t,
                            next_frame, t,
                            0
                        ).astype(np.uint8)
                        interpolated_frames.append(mid_frame)
                
                interpolated_frames.append(processed_frames[-1])
            
            # 方法3：使用简单线性插值（降级方案）
            else:
                print(f"  ℹ 使用线性插值")
                for i in range(len(processed_frames) - 1):
                    interpolated_frames.append(processed_frames[i])
                    num_intermediate = max(1, int(scale) - 1)
                    for j in range(num_intermediate):
                        t = (j + 1) / (num_intermediate + 1)
                        mid_frame = ((processed_frames[i].astype(np.float32) * (1.0 - t) + 
                                     processed_frames[i+1].astype(np.float32) * t)).astype(np.uint8)
                        interpolated_frames.append(mid_frame)
                interpolated_frames.append(processed_frames[-1])
            
            # 如果插帧后帧数仍然不足，使用线性插值补充
            if len(interpolated_frames) < target_frames:
                print(f"  ℹ RIFE 插帧后帧数 ({len(interpolated_frames)}) 仍不足，使用线性插值补充到 {target_frames} 帧")
                # 使用简单的线性插值补充
                while len(interpolated_frames) < target_frames:
                    # 在最后两帧之间插值
                    if len(interpolated_frames) >= 2:
                        last_frame = interpolated_frames[-1]
                        prev_frame = interpolated_frames[-2]
                        # 简单平均
                        mid_frame = ((prev_frame.astype(np.float32) + last_frame.astype(np.float32)) / 2).astype(np.uint8)
                        interpolated_frames.insert(-1, mid_frame)
                    else:
                        break
            
            # 截断到目标帧数
            if len(interpolated_frames) > target_frames:
                interpolated_frames = interpolated_frames[:target_frames]
            
            print(f"  ✓ 插帧完成: {len(frames)} 帧 → {len(interpolated_frames)} 帧")
            return interpolated_frames
        except Exception as e:
            print(f"  ⚠ 插帧过程出错: {e}")
            print(f"  ℹ 使用简单线性插值作为降级方案")
            import traceback
            traceback.print_exc()
            
            # 降级到简单线性插值
            if current_frames >= target_frames:
                return frames[:target_frames]
            
            interpolated_frames = []
            
            # 确保帧格式正确
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame.convert("RGB"))
                elif isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                processed_frames.append(frame)
            
            # 简单线性插值
            for i in range(len(processed_frames) - 1):
                interpolated_frames.append(processed_frames[i])
                num_intermediate = max(1, int(scale) - 1)
                for j in range(num_intermediate):
                    t = (j + 1) / (num_intermediate + 1)
                    mid_frame = ((processed_frames[i].astype(np.float32) * (1.0 - t) + 
                                 processed_frames[i+1].astype(np.float32) * t)).astype(np.uint8)
                    interpolated_frames.append(mid_frame)
            interpolated_frames.append(processed_frames[-1])
            
            # 截断到目标帧数
            if len(interpolated_frames) > target_frames:
                interpolated_frames = interpolated_frames[:target_frames]
            
            print(f"  ✓ 降级插帧完成: {len(frames)} 帧 → {len(interpolated_frames)} 帧")
            return interpolated_frames
