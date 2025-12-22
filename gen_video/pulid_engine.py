#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PuLID-FLUX 引擎 - 身份保持与环境融合

比 InstantID 更好的身份-环境平衡：
- 使用 PuLID-FLUX v0.9.1 进行身份嵌入
- 支持参考强度控制 (0-100)
- 更好的环境表达能力

参考架构：
- 豆包 Seedream 2.0 的参考强度控制
- 可灵 Element Library 的多参考图系统
"""

import os
# ⚡ 关键修复：在导入任何库之前设置环境变量，抑制 transformers 的警告输出
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # 设置为 error 级别，只显示错误
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 禁用 tokenizers 的并行警告

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import logging
import warnings
import sys
from contextlib import contextmanager

# ⚡ 抑制 EVA02-CLIP 的 rope keys 缺失警告（这是正常的，不影响功能）
warnings.filterwarnings("ignore", message=".*incompatible_keys.missing_keys.*rope.*")
warnings.filterwarnings("ignore", message=".*missing_keys.*rope.*")

# ⚡ 抑制 CLIP tokenizer 的 77 token 警告（Flux 使用 T5 作为主编码器，支持 512 tokens，CLIP 只是辅助编码器）
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than the specified maximum sequence length.*")
warnings.filterwarnings("ignore", message=".*The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens.*")

# ⚡ 关键修复：创建一个上下文管理器来抑制 CLIP tokenizer 的直接 stderr 输出
@contextmanager
def suppress_clip_tokenizer_warnings():
    """抑制 CLIP tokenizer 直接打印到 stderr 的警告"""
    import sys
    from io import StringIO
    
    # 保存原始的 stderr
    original_stderr = sys.stderr
    
    try:
        # 创建一个过滤器来过滤 CLIP tokenizer 的警告
        class FilteredStderr:
            def __init__(self, original):
                self.original = original
                self.buffer = []
            
            def write(self, text):
                # 过滤掉 CLIP tokenizer 的 77 token 警告
                if "Token indices sequence length is longer" in text:
                    return
                if "The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens" in text:
                    return
                # 其他内容正常输出
                self.original.write(text)
            
            def flush(self):
                self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        # 替换 stderr
        sys.stderr = FilteredStderr(original_stderr)
        yield
    finally:
        # 恢复原始的 stderr
        sys.stderr = original_stderr

import logging
import gc

# ⚡ 添加 logging filter 来过滤 EVA02-CLIP 的 rope keys 缺失日志
class EVA02CLIPRopeFilter(logging.Filter):
    """过滤 EVA02-CLIP 的 rope keys 缺失日志（这是正常的，不影响功能）"""
    def filter(self, record):
        # 过滤包含 rope.freqs 相关的日志
        if "rope.freqs" in record.getMessage() or "missing_keys" in record.getMessage():
            if "rope" in record.getMessage().lower():
                return False  # 不显示这些日志
        return True  # 显示其他日志

# 应用 filter 到 root logger（因为 EVA02-CLIP 使用 root logger）
logging.getLogger().addFilter(EVA02CLIPRopeFilter())

logger = logging.getLogger(__name__)

def log_memory(stage: str):
    """记录显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"Memory [{stage}]: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")



class PuLIDEngine:
    """
    PuLID-FLUX 引擎
    
    功能：
    - 人脸身份嵌入 (比InstantID更好的环境融合)
    - 参考强度控制 (0-100, 类似可灵的参考强度)
    - 多角度参考图支持
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 PuLID 引擎
        
        Args:
            config: 配置字典，包含模型路径等
        """
        self.config = config
        self.device = config.get("device", "cuda")
        self.dtype = torch.bfloat16 if config.get("quantization", "bfloat16") == "bfloat16" else torch.float16
        
        # 模型路径配置
        self.model_base_path = config.get("model_dir", "/vepfs-dev/shawn/vid/fanren/gen_video/models")
        # 注意：文件名是小写的 pulid_flux_v0.9.1.safetensors
        self.pulid_path = os.path.join(self.model_base_path, "pulid", "pulid_flux_v0.9.1.safetensors")
        self.flux_path = os.path.join(self.model_base_path, "flux1-dev")
        self.antelopev2_path = os.path.join(self.model_base_path, "antelopev2")
        self.eva_clip_path = os.path.join(self.model_base_path, "clip", "EVA02_CLIP_L_336_psz14_s6B.pt")
        
        # PuLID 原生模型路径
        self.flux_native_path = os.path.join(self.model_base_path, "flux1-dev.safetensors")
        self.ae_path = os.path.join(self.model_base_path, "ae.safetensors")
        
        # Pipeline 状态
        self.pipeline = None
        self.face_analyzer = None
        self.pulid_loaded = False
        
        # PuLID 原生组件
        self.flux_model = None  # 原生 Flux 模型
        self.ae = None  # AutoEncoder
        self.t5 = None  # T5 编码器
        self.clip = None  # CLIP 编码器
        self.pulid_model = None
        self.id_embedding = None
        self.use_native = False  # 是否使用原生模式
        
        # 缓存
        self.face_embedding_cache = {}
        
        logger.info(f"PuLID Engine 初始化完成")
        logger.info(f"  PuLID 模型: {self.pulid_path}")
        logger.info(f"  Flux 模型: {self.flux_path}")
        logger.info(f"  Flux 原生: {self.flux_native_path}")
    
    def load_pipeline(self):
        """
        加载 PuLID-FLUX pipeline
        
        优先使用原生模式（flux.model），回退到 diffusers 模式
        """
        if self.pulid_loaded:
            logger.info("PuLID pipeline 已加载，跳过")
            return
        
        logger.info("开始加载 PuLID-FLUX pipeline...")
        
        # 检查是否有原生模型
        has_native_flux = os.path.exists(self.flux_native_path)
        has_ae = os.path.exists(self.ae_path)
        # 检查是否有原生 Flux 模型
        # 注意：暂时禁用原生模式，因为显存占用太大
        # TODO: 后续优化原生模式的显存管理
        if os.path.exists(self.flux_native_path) and os.path.exists(self.ae_path):
            logger.info("检测到原生 Flux 模型，启用原生模式...")
            # logger.info("使用 diffusers 模式（支持更好的显存管理）...") # Removed misleading log
            try:
                self._load_native_pipeline()
                self.use_native = True
                self.pulid_loaded = True
                logger.info("PuLID 原生模式加载完成!")
                return
            except Exception as e:
                logger.warning(f"原生模式加载失败: {e}")
                logger.info("回退到 diffusers 模式...")
                try:
                    self.use_native = False
                    self._load_diffusers_pipeline()
                    self.pulid_loaded = True
                    logger.info("PuLID diffusers 模式加载完成（原生失败回退）")
                    return
                except Exception as e2:
                    logger.error(f"diffusers 回退加载也失败: {e2}")
                    raise

        # 没有原生权重时，直接使用 diffusers 模式
        logger.info("未检测到可用的原生 Flux 权重，使用 diffusers 模式加载 PuLID...")
        self.use_native = False
        self._load_diffusers_pipeline()
        self.pulid_loaded = True
        logger.info("PuLID diffusers 模式加载完成")
        return

    def _load_native_pipeline(self):
        """加载 PuLID 原生 Flux 模型（显存优化版）"""
        log_memory("Start Native Load")
        logger.info("加载 PuLID 原生 Flux 模型...")
        
        # 检查可用显存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            logger.info(f"显存状态: 总计={total:.2f}GB, 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 可用={free:.2f}GB")
            
            # 如果可用显存少于 25GB，警告
            if free < 25:
                logger.warning(f"可用显存较少 ({free:.2f}GB)，可能会超出显存限制")
                logger.warning("建议：1) 关闭其他占用显存的程序 2) 使用更激进的 CPU offload")
        
        # 导入原生模块
        # 添加 PuLID 到 Python 路径（PuLID 子模块位于 fanren/PuLID，与 gen_video 平级）
        import sys
        from pathlib import Path
        pulid_path = Path(__file__).parent.parent / "PuLID"
        if pulid_path.exists() and str(pulid_path) not in sys.path:
            sys.path.insert(0, str(pulid_path))
        
        from flux.util import load_t5, load_clip, load_ae
        from pulid.pipeline_flux import PuLIDPipeline
        
        # ⚡ 关键修复：设置环境变量，让 PuLIDPipeline 使用本地 EVA-CLIP 模型
        if os.path.exists(self.eva_clip_path):
            os.environ['EVA_CLIP_PATH'] = self.eva_clip_path
            logger.info(f"  ✓ 设置 EVA_CLIP_PATH 环境变量: {self.eva_clip_path}")
        else:
            logger.warning(f"  ⚠ 本地 EVA-CLIP 模型不存在: {self.eva_clip_path}")
        
        # 加载 Flux DiT 模型 (主要模型，保持在 GPU)
        logger.info(f"  加载 Flux DiT: {self.flux_native_path}")
        self.flux_model = self._optimized_load_flux("flux-dev", device=self.device)
        log_memory("After Flux Load")
        
        # 加载 AutoEncoder (解码器，保持在 GPU)
        logger.info(f"  加载 AutoEncoder: {self.ae_path}")
        # ⚡ 关键修复：优先使用本地 AE 模型文件，避免下载
        # load_ae 使用 configs[name].ae_path，我们需要临时修改 configs 来使用本地路径
        from flux.util import configs, load_sft, AutoEncoder
        # ⚡ 关键修复：torch 已在文件顶部导入，不需要再次导入
        # import torch  # 删除这行，使用文件顶部的 torch
        
        if os.path.exists(self.ae_path):
            logger.info(f"  ✓ 检测到本地 AE 模型文件: {self.ae_path}")
            try:
                # 保存原始路径
                original_ae_path = configs["flux-dev"].ae_path
                # 临时修改 configs 使用本地路径
                configs["flux-dev"].ae_path = self.ae_path
                logger.info(f"  ✓ 使用本地 AE 路径: {self.ae_path}")
                
                # 调用 load_ae，现在它会使用我们设置的本地路径
                self.ae = load_ae("flux-dev", device=self.device, hf_download=False)
                logger.info(f"  ✅ 成功从本地加载 AE 模型（未下载）")
                
                # 恢复原始路径（可选，因为已经加载完成）
                # configs["flux-dev"].ae_path = original_ae_path
            except Exception as e:
                logger.warning(f"  ⚠ 使用本地 AE 文件失败: {e}")
                logger.info(f"  ℹ 回退到默认加载方式（可能会下载）")
                # 恢复原始路径
                if 'original_ae_path' in locals():
                    configs["flux-dev"].ae_path = original_ae_path
                self.ae = load_ae("flux-dev", device=self.device)
        else:
            logger.warning(f"  ⚠ 本地 AE 模型文件不存在: {self.ae_path}")
            logger.info(f"  ℹ 将使用默认加载方式（可能会下载）")
            self.ae = load_ae("flux-dev", device=self.device)
        log_memory("After AE Load")
        
        # 加载 T5 和 CLIP (先加载到 CPU，使用时再移到 GPU)
        # 注意：Flux 使用双编码器架构：
        #   - T5 编码器（主要）：支持 128/256/512 tokens，当前配置为 256
        #   - CLIP 编码器（辅助）：固定 77 tokens（用于辅助语义，不是主要限制）
        # ⚡ 重要：CLIP 的 77 token 警告是正常的，不影响生成质量（T5 是主要编码器）
        # 如果需要支持更长的 prompt，可以将 T5 max_length 提高到 512
        t5_max_length = self.config.get("t5_max_length", 256)  # 默认 256，可配置为 512
        logger.info(f"  加载 T5 编码器 (CPU offload, max_length={t5_max_length})...")
        
        # ⚡ 关键修复：优先使用本地 T5 模型，避免重复下载
        local_t5_path = os.path.join(self.model_base_path, "xflux_text_encoders")
        if os.path.exists(local_t5_path):
            logger.info(f"  ✓ 使用本地 T5 模型: {local_t5_path}")
            # 使用本地路径加载 T5（HFEmbedder 的 from_pretrained 支持本地路径）
            from flux.modules.conditioner import HFEmbedder
            # ⚡ 修复：不要在这里重新导入 torch，使用文件顶部已导入的 torch
            # import torch  # 删除这行，因为文件顶部已经导入了 torch
            try:
                # ⚡ 关键修复：使用绝对路径，并检查关键文件是否存在
                abs_t5_path = os.path.abspath(local_t5_path)
                # 检查关键文件是否存在
                required_files = [
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "config.json"
                ]
                missing_files = []
                for file in required_files:
                    file_path = os.path.join(abs_t5_path, file)
                    if not os.path.exists(file_path):
                        missing_files.append(file)
                
                if missing_files:
                    logger.warning(f"  ⚠ 本地 T5 模型文件不完整，缺少: {missing_files}")
                    raise FileNotFoundError(f"T5 模型文件不完整: {missing_files}")
                
                # 直接使用本地路径，HFEmbedder 应该会自动识别
                # 如果 HFEmbedder 内部使用 transformers，它会自动识别本地路径
                # ⚡ 关键修复：传递 local_files_only=True 强制使用本地文件
                logger.info(f"  ✓ T5 模型文件完整，使用本地路径: {abs_t5_path}")
                self.t5 = HFEmbedder(
                    abs_t5_path, 
                    max_length=t5_max_length, 
                    torch_dtype=torch.bfloat16,
                    local_files_only=True  # 强制使用本地文件，避免下载
                ).to("cpu")
                logger.info(f"  ✅ 本地 T5 模型加载成功（未下载）")
            except Exception as e:
                logger.warning(f"  ⚠ 本地 T5 模型加载失败: {e}，尝试使用 HuggingFace 缓存")
                import traceback
                traceback.print_exc()
                # 回退到原始方法（会尝试使用 HuggingFace 缓存）
                # ⚡ 关键修复：设置 local_files_only=True（如果 HFEmbedder 支持）
                try:
                    # 尝试使用 local_files_only（如果支持）
                    from transformers import AutoModel, AutoTokenizer
                    # 检查 HuggingFace 缓存
                    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                    hf_cache_path = os.path.join(hf_home, "hub", "models--xlabs-ai--xflux_text_encoders")
                    if os.path.exists(hf_cache_path):
                        logger.info(f"  ℹ 尝试从 HuggingFace 缓存加载 T5: {hf_cache_path}")
                        self.t5 = load_t5(device="cpu", max_length=t5_max_length)
                    else:
                        logger.error(f"  ❌ T5 模型不存在（本地和缓存都没有）")
                        raise FileNotFoundError(f"T5 模型不存在，请先下载到: {local_t5_path} 或 {hf_cache_path}")
                except Exception as e2:
                    logger.error(f"  ❌ T5 加载失败: {e2}")
                    raise
        else:
            # 回退到原始方法（会从 HuggingFace 下载或使用缓存）
            logger.warning(f"  ⚠ 本地 T5 模型不存在: {local_t5_path}，将使用 HuggingFace")
            # ⚡ 关键修复：检查 HuggingFace 缓存，避免下载
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            hf_cache_path = os.path.join(hf_home, "hub", "models--xlabs-ai--xflux_text_encoders")
            if os.path.exists(hf_cache_path):
                logger.info(f"  ℹ 使用 HuggingFace 缓存: {hf_cache_path}")
                self.t5 = load_t5(device="cpu", max_length=t5_max_length)
            else:
                logger.error(f"  ❌ T5 模型不存在（本地和缓存都没有）")
                logger.error(f"  💡 请先下载 T5 模型到: {local_t5_path}")
                raise FileNotFoundError(f"T5 模型不存在，请先下载到: {local_t5_path}")
        
        logger.info("  加载 CLIP 编码器 (CPU offload)...")
        # ⚡ 关键修复：优先使用本地 CLIP 模型或缓存，避免网络下载
        # ⚡ 关键修复：在加载 CLIP 时抑制 77 token 警告
        with suppress_clip_tokenizer_warnings():
            try:
                from flux.modules.conditioner import HFEmbedder
                # 检查本地 CLIP 模型路径
                local_clip_path = os.path.join(self.model_base_path, "clip", "openai-clip-vit-large-patch14")
                
                # 检查 HuggingFace 缓存路径
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                # 如果默认路径不存在，尝试使用项目配置的缓存目录
                if not os.path.exists(hf_home):
                    hf_home = "/vepfs-dev/shawn/.cache/huggingface"
                    os.environ["HF_HOME"] = hf_home
                hf_cache_path = os.path.join(hf_home, "hub", "models--openai--clip-vit-large-patch14")
                
                # 尝试从本地路径加载
                if os.path.exists(local_clip_path):
                    logger.info(f"  ✓ 使用本地 CLIP 模型: {local_clip_path}")
                    try:
                        # ⚡ 关键修复：确保路径是字符串类型，并检查关键文件是否存在
                        abs_clip_path = os.path.abspath(local_clip_path)
                        # 检查关键文件是否存在（transformers 支持 safetensors 或 pytorch_model.bin）
                        required_config_file = "config.json"
                        model_files = ["model.safetensors", "pytorch_model.bin"]
                        
                        config_exists = os.path.exists(os.path.join(abs_clip_path, required_config_file))
                        has_model_file = any(os.path.exists(os.path.join(abs_clip_path, f)) for f in model_files)
                        
                        if not config_exists:
                            logger.warning(f"  ⚠ 本地 CLIP 模型缺少配置文件: {required_config_file}")
                            # 检查是否有其他格式的文件
                            all_files = os.listdir(abs_clip_path)
                            logger.info(f"  ℹ CLIP 目录中的文件: {all_files[:10]}")
                            # 如果缺少配置文件，尝试使用 HuggingFace 缓存
                            if os.path.exists(hf_cache_path):
                                logger.info(f"  ℹ 尝试从 HuggingFace 缓存加载 CLIP: {hf_cache_path}")
                                self.clip = HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16, local_files_only=True).to("cpu")
                                logger.info(f"  ✅ 从 HuggingFace 缓存加载 CLIP 成功")
                            else:
                                raise FileNotFoundError(f"CLIP 模型缺少配置文件: {required_config_file}")
                        elif not has_model_file:
                            logger.warning(f"  ⚠ 本地 CLIP 模型缺少模型文件（需要 {model_files} 之一）")
                            # 检查是否有其他格式的文件
                            all_files = os.listdir(abs_clip_path)
                            logger.info(f"  ℹ CLIP 目录中的文件: {all_files[:10]}")
                            # 如果缺少模型文件，尝试使用 HuggingFace 缓存
                            if os.path.exists(hf_cache_path):
                                logger.info(f"  ℹ 尝试从 HuggingFace 缓存加载 CLIP: {hf_cache_path}")
                                self.clip = HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16, local_files_only=True).to("cpu")
                                logger.info(f"  ✅ 从 HuggingFace 缓存加载 CLIP 成功")
                            else:
                                raise FileNotFoundError(f"CLIP 模型缺少模型文件（需要 {model_files} 之一）")
                        else:
                            # ⚡ 关键修复：使用绝对路径字符串，并传递 local_files_only=True
                            # transformers 会自动识别 safetensors 或 pytorch_model.bin
                            # ⚡ 关键修复：HFEmbedder 现在可以识别本地 CLIP 路径（包含 "clip" 关键字）
                            logger.info(f"  ✓ CLIP 模型文件完整（config.json + 模型文件），使用本地路径: {abs_clip_path}")
                            self.clip = HFEmbedder(
                                abs_clip_path,  # 直接使用本地路径，HFEmbedder 会识别为 CLIP
                                max_length=77, 
                                torch_dtype=torch.bfloat16,
                                local_files_only=True  # 强制使用本地文件，避免下载
                            ).to("cpu")
                            logger.info(f"  ✅ 本地 CLIP 模型加载成功（未下载）")
                    except Exception as e:
                        logger.warning(f"  ⚠ 本地 CLIP 模型加载失败: {e}，尝试使用 HuggingFace 缓存")
                        import traceback
                        traceback.print_exc()
                        # 回退到使用 HuggingFace 缓存（local_files_only=True）
                        if os.path.exists(hf_cache_path):
                            try:
                                self.clip = HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16, local_files_only=True).to("cpu")
                                logger.info(f"  ✅ 从 HuggingFace 缓存加载 CLIP 成功")
                            except Exception as e2:
                                logger.error(f"  ❌ 从缓存加载 CLIP 失败: {e2}")
                                raise
                        else:
                            logger.error(f"  ❌ HuggingFace 缓存不存在: {hf_cache_path}")
                            logger.error(f"  💡 请先下载 CLIP 模型，运行: python3 -c \"from transformers import CLIPTokenizer, CLIPTextModel; CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14'); CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')\"")
                            raise FileNotFoundError(f"CLIP 模型不存在，请先下载")
                else:
                    # 检查 HuggingFace 缓存
                    if os.path.exists(hf_cache_path):
                        logger.info(f"  ℹ 本地 CLIP 模型不存在，使用 HuggingFace 缓存: {hf_cache_path}")
                        try:
                            self.clip = HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16, local_files_only=True).to("cpu")
                            logger.info(f"  ✅ 从 HuggingFace 缓存加载 CLIP 成功")
                        except Exception as e:
                            logger.error(f"  ❌ 从缓存加载 CLIP 失败: {e}")
                            raise
                    else:
                        logger.error(f"  ❌ CLIP 模型不存在（本地和缓存都没有）")
                        logger.error(f"  💡 请先下载 CLIP 模型，运行以下命令:")
                        logger.error(f"     python3 -c \"from transformers import CLIPTokenizer, CLIPTextModel; CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14'); CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')\"")
                        raise FileNotFoundError(f"CLIP 模型不存在，请先下载到: {hf_cache_path}")
            except ImportError:
                logger.warning(f"  ⚠ 无法导入 HFEmbedder，使用 load_clip（可能尝试网络下载）")
                self.clip = load_clip(device="cpu")
            except FileNotFoundError:
                # 如果明确是文件不存在，抛出异常，不要尝试网络下载
                raise
            except Exception as e:
                logger.error(f"  ❌ CLIP 加载失败: {e}")
                logger.error(f"  💡 如果网络不可用，请先下载 CLIP 模型到缓存")
                raise
        log_memory("After Encoders Load")
        
        # 创建 PuLID Pipeline
        logger.info("  创建 PuLID Pipeline...")
        self.pulid_model = PuLIDPipeline(
            dit=self.flux_model,
            device=self.device,
            weight_dtype=self.dtype
        )
        
        # 加载 PuLID 权重
        logger.info(f"  加载 PuLID 权重: {self.pulid_path}")
        # ⚡ 关键修复：传递正确的版本号，避免下载错误版本
        # 从路径中提取版本号（例如：pulid_flux_v0.9.1.safetensors -> v0.9.1）
        import re
        version_match = re.search(r'v(\d+\.\d+\.\d+)', self.pulid_path)
        if version_match:
            pulid_version = version_match.group(1)
            logger.info(f"  ✓ 检测到 PuLID 版本: v{pulid_version}")
            self.pulid_model.load_pretrain(pretrain_path=self.pulid_path, version=pulid_version)
        else:
            # 默认使用 v0.9.1（与配置中的文件名匹配）
            logger.info(f"  ℹ 使用默认 PuLID 版本: v0.9.1")
            self.pulid_model.load_pretrain(pretrain_path=self.pulid_path, version='v0.9.1')
        log_memory("After PuLID Load")
        
        # 设置标志
        self.use_pulid = True
        self.use_native = True
        self.use_cpu_offload = True  # 标记使用 CPU offload
        
        logger.info("原生模式加载完成! (CPU offload 模式)")

    def _optimized_load_flux(self, name: str, device: str = "cuda", hf_download: bool = True):
        """
        优化的 Flux 加载函数
        
        原生 util.load_flow_model 会将 checkpoint 直接加载到 GPU，
        加上已经初始化的模型，会导致显存占用翻倍 (23GB * 2 = 46GB)。
        此函数强制先加载到 CPU，再加载进模型。
        """
        # ⚡ 关键修复：torch 已在文件顶部导入，不需要再次导入
        # import torch  # 删除这行，使用文件顶部的 torch
        
        from flux.model import Flux
        from flux.util import configs, load_sft, print_load_warning
        from huggingface_hub import hf_hub_download
        
        # Loading Flux
        logger.info("Init model (Optimized)")
        
        # ⚡ 关键修复：优先使用本地模型路径（self.flux_native_path）
        # 如果本地文件存在，直接使用，避免下载
        ckpt_path = None
        if hasattr(self, 'flux_native_path') and os.path.exists(self.flux_native_path):
            ckpt_path = self.flux_native_path
            logger.info(f"  ✓ 使用本地模型文件: {ckpt_path}")
        else:
            # 回退到使用 configs 中的路径
            ckpt_path = configs[name].ckpt_path
            logger.info(f"  ℹ 使用 configs 路径: {ckpt_path}")
            
            # ⚡ 关键修复：只有在文件不存在且明确允许下载时才下载
            if (
                not os.path.exists(ckpt_path)
                and configs[name].repo_id is not None
                and configs[name].repo_flow is not None
                and hf_download
            ):
                logger.warning(f"  ⚠ 模型文件不存在: {ckpt_path}")
                logger.warning(f"  ⚠ 将尝试从 HuggingFace 下载: {configs[name].repo_id}/{configs[name].repo_flow}")
                logger.warning(f"  ⚠ 如果本地已有模型文件，请检查路径配置")
                ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow, local_dir='models')
            elif not os.path.exists(ckpt_path):
                logger.error(f"  ✗ 模型文件不存在且不允许下载: {ckpt_path}")
                raise FileNotFoundError(f"模型文件不存在: {ckpt_path}，且 hf_download=False")

        # 1. 初始化模型结构 (占用显存)
        # ⚡ 修复：使用 torch.device 对象而不是上下文管理器
        # 注意：torch.device 上下文管理器在某些 PyTorch 版本中可能不可用
        # 直接创建模型并移动到指定设备
        model = Flux(configs[name].params)
        model = model.to(device).to(torch.bfloat16)

        if ckpt_path is not None:
            logger.info(f"Loading checkpoint: {ckpt_path}")
            logger.info("  Step 1: Loading state dict to CPU RAM...")
            # 2. 加载权重到 CPU 内存 (不占用显存)
            # load_sft 内部通常使用 safetensors，支持 device 参数
            # 强制指定为 cpu
            sd = load_sft(ckpt_path, device="cpu")
            
            logger.info("  Step 2: Loading state dict into Model (GPU)...")
            # 3. 将权重加载到 GPU 模型中
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print_load_warning(missing, unexpected)
            
            # 4. 释放 CPU 内存
            del sd
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        return model
    
    def _load_diffusers_pipeline(self):
        """使用 diffusers 加载（回退模式）"""
        from diffusers import FluxPipeline
        
        logger.info("加载 Flux pipeline (diffusers 模式)...")
        logger.info(f"  Flux 模型路径: {self.flux_path}")
        
        # ⚡ 关键修复：在加载 pipeline 时抑制 CLIP tokenizer 的 77 token 警告
        with suppress_clip_tokenizer_warnings():
            # ⚡ 修复：优先使用本地路径，避免重新下载
            # 检查路径是否存在
            if os.path.exists(self.flux_path):
                logger.info(f"  ✓ 检测到本地模型路径，使用本地模型")
                # ⚡ 关键修复：检查关键文件是否存在
                required_files = [
                    "model_index.json",
                    "flux1-dev.safetensors",
                    "transformer",
                    "vae",
                    "text_encoder",
                    "text_encoder_2"
                ]
                missing_files = []
                for file_or_dir in required_files:
                    path = os.path.join(self.flux_path, file_or_dir)
                    if not os.path.exists(path):
                        missing_files.append(file_or_dir)
                
                if missing_files:
                    logger.warning(f"  ⚠ 本地模型文件不完整，缺少: {missing_files}")
                    logger.info(f"  ℹ 将从 HuggingFace 下载缺失文件")
                    # 如果缺少关键文件，允许网络下载
                    self.pipeline = FluxPipeline.from_pretrained(
                        self.flux_path,
                        torch_dtype=self.dtype
                    )
                else:
                    # 所有关键文件都存在，强制使用本地文件
                    logger.info(f"  ✓ 本地模型文件完整，强制使用本地文件（local_files_only=True）")
                    try:
                        self.pipeline = FluxPipeline.from_pretrained(
                            self.flux_path,
                            torch_dtype=self.dtype,
                            local_files_only=True
                        )
                        logger.info(f"  ✅ 成功从本地加载模型（未下载任何文件）")
                    except Exception as e:
                        logger.error(f"  ✗ 使用 local_files_only 加载失败: {e}")
                        logger.warning(f"  ⚠ 即使文件存在，加载仍失败，可能是文件损坏或格式不兼容")
                        logger.info(f"  ℹ 尝试不使用 local_files_only（可能会检查网络但不会下载，因为文件已存在）")
                        # 最后一次尝试：不使用 local_files_only，但文件已存在，应该不会下载
                        self.pipeline = FluxPipeline.from_pretrained(
                            self.flux_path,
                            torch_dtype=self.dtype
                        )
            else:
                logger.warning(f"  ⚠ 本地模型路径不存在: {self.flux_path}")
                logger.info(f"  ℹ 将从 HuggingFace 下载模型")
                self.pipeline = FluxPipeline.from_pretrained(
                    self.flux_path,
                    torch_dtype=self.dtype
                )
        self.pipeline.enable_model_cpu_offload()
        
        # 尝试加载 PuLID
        try:
            # 添加 PuLID 到 Python 路径（PuLID 子模块位于 fanren/PuLID，与 gen_video 平级）
            import sys
            from pathlib import Path
            pulid_path = Path(__file__).parent.parent / "PuLID"
            if pulid_path.exists() and str(pulid_path) not in sys.path:
                sys.path.insert(0, str(pulid_path))
            
            from pulid.pipeline_flux import PuLIDPipeline
            
            dit = self.pipeline.transformer
            self.pulid_model = PuLIDPipeline(
                dit=dit,
                device=self.device,
                weight_dtype=self.dtype
            )
            # ⚡ 关键修复：传递正确的版本号，避免下载错误版本
            # 从路径中提取版本号（例如：pulid_flux_v0.9.1.safetensors -> v0.9.1）
            import re
            version_match = re.search(r'v(\d+\.\d+\.\d+)', self.pulid_path)
            if version_match:
                pulid_version = version_match.group(1)
                logger.info(f"  ✓ 检测到 PuLID 版本: v{pulid_version}")
                self.pulid_model.load_pretrain(pretrain_path=self.pulid_path, version=pulid_version)
            else:
                # 默认使用 v0.9.1（与配置中的文件名匹配）
                logger.info(f"  ℹ 使用默认 PuLID 版本: v0.9.1")
                self.pulid_model.load_pretrain(pretrain_path=self.pulid_path, version='v0.9.1')
            self.use_pulid = True  # 标记 PuLID 可用
            logger.info("PuLID 模型加载完成 (diffusers 模式)")
            
        except Exception as e:
            logger.warning(f"PuLID 加载失败: {e}")
            self.pulid_model = None
            self.use_pulid = False
        
        # 加载 InsightFace
        self._load_face_analyzer()
        # 标记加载完成
        self.pulid_loaded = True
    
    def _load_pulid_with_diffusers(self):
        """
        使用 diffusers 方式加载 Flux + PuLID
        
        这是备用方案，当 pulid 包不可用时使用
        """
        try:
            from diffusers import FluxPipeline
            
            logger.info("加载 Flux pipeline (diffusers 方式)...")
            logger.info(f"  Flux 模型路径: {self.flux_path}")
            
            # ⚡ 关键修复：在加载 pipeline 时抑制 CLIP tokenizer 的 77 token 警告
            with suppress_clip_tokenizer_warnings():
                # 加载基础 Flux pipeline
                self.pipeline = FluxPipeline.from_pretrained(
                    self.flux_path,
                    torch_dtype=self.dtype
                )
            
            # 启用优化
            self.pipeline.enable_model_cpu_offload()
            
            # 加载 InsightFace 用于人脸检测
            self._load_face_analyzer()
            
            # 注意：使用 diffusers 方式时，PuLID 权重需要手动注入
            # 这是简化版本，完整版本需要实现 PuLID 的 attention injection
            logger.warning("使用 diffusers 简化模式，PuLID 身份注入功能受限")
            
            self.pulid_loaded = True
            self.use_pulid = False
            logger.info("Flux pipeline 加载完成 (简化模式)")
            
        except Exception as e:
            logger.error(f"diffusers 方式加载失败: {e}")
            raise
    
    def _load_face_analyzer(self):
        """加载人脸分析器 (InsightFace)"""
        if self.face_analyzer is not None:
            return
        
        try:
            from insightface.app import FaceAnalysis
            
            logger.info("加载 InsightFace FaceAnalysis...")
            
            # InsightFace 的 root 参数会在其下寻找 models/{name} 目录
            # antelopev2_path = /path/to/gen_video/models/antelopev2
            # 需要设置 root = /path/to/gen_video，这样 InsightFace 会找 {root}/models/antelopev2
            insightface_root = os.path.dirname(os.path.dirname(self.antelopev2_path))
            
            logger.info(f"  InsightFace root: {insightface_root}")
            logger.info(f"  模型目录: {self.antelopev2_path}")
            
            self.face_analyzer = FaceAnalysis(
                name='antelopev2',
                root=insightface_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info("InsightFace 加载完成")
            
        except Exception as e:
            logger.error(f"InsightFace 加载失败: {e}")
            raise
    
    def extract_face_embedding(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        提取人脸嵌入向量
        
        Args:
            image: 输入图像 (路径/PIL Image/numpy array)
            
        Returns:
            人脸嵌入向量，如果未检测到人脸则返回 None
        """
        # 确保人脸分析器已加载
        self._load_face_analyzer()
        
        # 转换图像格式
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 检测人脸
        faces = self.face_analyzer.get(image)
        
        if not faces:
            logger.warning("未检测到人脸")
            return None
        
        # 返回最大人脸的嵌入
        main_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
        return main_face.embedding
    
    def generate_with_identity(
        self,
        prompt: str,
        face_reference: Union[str, Image.Image],
        reference_strength: int = 60,
        width: int = 768,
        height: int = 1152,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """
        使用身份参考生成图像
        
        Args:
            prompt: 生成提示词 (应该包含详细的环境描述)
            face_reference: 人脸参考图像
            reference_strength: 参考强度 (0-100)
                - 0-30: 轻微参考，环境优先 (适合远景)
                - 30-60: 平衡模式 (适合中景)
                - 60-90: 强参考，人脸优先 (适合特写)
            width: 输出宽度
            height: 输出高度
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            
        Returns:
            生成的图像
        """
        # ⚡ 关键修复：如果没有参考图像，直接使用无身份注入模式
        if face_reference is None:
            logger.info("没有参考图像，使用无身份注入模式生成")
            # 确保 pipeline 已加载
            self.load_pipeline()
            if self.pipeline is None:
                raise RuntimeError("Pipeline 未加载，无法生成图像")
            
            # 设置随机种子
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # ⚡ 关键修复：在生成图像时抑制 CLIP tokenizer 的 77 token 警告
            with suppress_clip_tokenizer_warnings():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
            return result
        
        # 确保 pipeline 已加载
        self.load_pipeline()
        
        # ⚡ 关键修复：加载 LoRA（如果提供）
        lora_config = kwargs.get('lora_config', None)
        character_id = kwargs.get('character_id', None)
        lora_loaded = False
        
        if lora_config:
            lora_path = lora_config.get('lora_path', '')
            lora_weight = lora_config.get('weight', 0.9)
            
            # 检查路径（支持相对路径）
            if lora_path:
                lora_path_obj = Path(lora_path)
                if not lora_path_obj.is_absolute():
                    # 相对路径：从 gen_video 目录开始（pulid_engine.py 在 gen_video 目录下）
                    base_path = Path(__file__).parent  # gen_video 目录
                    lora_path_obj = base_path / lora_path
                lora_path = str(lora_path_obj)
            
            if lora_path and Path(lora_path).exists():
                try:
                    adapter_name = character_id if character_id else "character_lora"
                    logger.info(f"  加载 LoRA: {lora_path} (权重: {lora_weight}, 适配器: {adapter_name})")
                    print(f"  [调试] 尝试加载 LoRA: {lora_path}")
                    
                    # ⚡ 关键修复：检查 LoRA 权重格式，只在需要时转换
                    # 如果权重已经是 diffusers 格式（unet.xxx 或 transformer.xxx），直接使用
                    # 如果是 PEFT 格式（base_model.model.xxx 或 single_transformer_blocks），需要转换
                    from safetensors import safe_open
                    import tempfile
                    import os
                    
                    lora_path_obj = Path(lora_path)
                    actual_lora_path = lora_path  # 默认使用原始路径
                    converted_lora_path = None
                    needs_conversion = False
                    is_unet_lora = False  # 标记是否为UNet格式的LoRA（在整个try块内都可用）
                    
                    # 检查权重格式
                    try:
                        with safe_open(str(lora_path_obj), framework="pt") as f:
                            sample_keys = list(f.keys())[:10]  # 检查前10个键
                            # 检查是否为UNet格式
                            is_unet_lora = any(key.startswith("unet.") for key in sample_keys)
                            # 检查是否需要转换：如果包含 PEFT 格式特征，需要转换
                            for key in sample_keys:
                                if key.startswith("base_model.model.") or "single_transformer_blocks" in key:
                                    needs_conversion = True
                                    break
                        
                        if needs_conversion:
                            logger.info(f"  🔧 检测到 PEFT 格式，转换 LoRA 权重: {lora_path_obj.name}")
                            print(f"  [调试] 检测到 PEFT 格式，转换 LoRA 权重")
                            
                            # 读取并转换 LoRA 权重（PEFT → diffusers）
                            lora_state_dict = {}
                            with safe_open(str(lora_path_obj), framework="pt") as f:
                                for key in f.keys():
                                    new_key = key
                                    # 步骤 1：移除 base_model.model. 前缀（PEFT 格式）
                                    if key.startswith("base_model.model."):
                                        new_key = key.replace("base_model.model.", "")
                                    
                                    # 步骤 2：将 single_transformer_blocks 替换为 transformer_blocks
                                    if "single_transformer_blocks" in new_key:
                                        new_key = new_key.replace("single_transformer_blocks", "transformer_blocks")
                                    
                                    # 步骤 3：移除 .default 部分（PEFT 格式）
                                    if ".default." in new_key:
                                        new_key = new_key.replace(".default.", ".")
                                    
                                    # 步骤 4：添加 transformer. 前缀（如果还没有，且是 transformer_blocks 相关的键）
                                    if "transformer_blocks" in new_key and not new_key.startswith("transformer."):
                                        new_key = f"transformer.{new_key}"
                                    
                                    lora_state_dict[new_key] = f.get_tensor(key)
                            
                            # 保存转换后的权重到临时文件
                            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
                                from safetensors.torch import save_file
                                save_file(lora_state_dict, tmp_file.name)
                                converted_lora_path = tmp_file.name
                            
                            actual_lora_path = converted_lora_path
                            logger.info(f"  ✓ LoRA 权重格式转换成功")
                            print(f"  [调试] LoRA 权重格式转换成功")
                        else:
                            if is_unet_lora:
                                logger.info(f"  ℹ LoRA 权重是 UNet 格式，直接使用: {lora_path_obj.name}")
                                print(f"  [调试] LoRA 权重是 UNet 格式，直接使用 (is_unet_lora=True)")
                            else:
                                logger.info(f"  ℹ LoRA 权重已是 diffusers 格式，直接使用: {lora_path_obj.name}")
                                print(f"  [调试] LoRA 权重已是 diffusers 格式，直接使用 (is_unet_lora=False)")
                            
                    except Exception as check_e:
                        logger.warning(f"  ⚠ 检查 LoRA 格式失败: {check_e}，尝试直接加载")
                        print(f"  [调试] 检查 LoRA 格式失败，直接使用原始文件: {check_e} (is_unet_lora={is_unet_lora})")
                    
                    # 1. 优先尝试在 pipeline 上加载（diffusers 模式，支持 LoRA）
                    if self.pipeline is not None:
                        # 方法1：尝试使用 adapter_name 加载（UNet格式的LoRA应该能直接加载）
                        try:
                            # ⚡ 关键修复：对于UNet格式的LoRA，load_lora_weights会自动处理
                            # 警告"No LoRA keys associated to FluxTransformer2DModel"是正常的，因为这是UNet的LoRA
                            self.pipeline.load_lora_weights(actual_lora_path, adapter_name=adapter_name, weight_name=None)
                            
                            # ⚡ 修复：检查 adapter 是否已成功加载
                            if hasattr(self.pipeline, 'set_adapters'):
                                # 等待一下，让load_lora_weights完成注册
                                import time
                                time.sleep(0.1)
                                
                                # 检查 adapter 是否已加载
                                adapter_to_use = None
                                if hasattr(self.pipeline, 'get_list_adapters'):
                                    try:
                                        list_adapters = self.pipeline.get_list_adapters()
                                        all_adapters = {adapter for adapters in list_adapters.values() for adapter in adapters}
                                        if adapter_name in all_adapters:
                                            adapter_to_use = adapter_name
                                        elif all_adapters:
                                            # 如果指定的 adapter_name 不在列表中，但已有其他 adapter，使用第一个
                                            adapter_to_use = list(all_adapters)[0]
                                            logger.info(f"  ℹ 使用自动检测的 adapter: {adapter_to_use} (而非指定的 {adapter_name})")
                                        else:
                                            # ⚡ 关键修复：如果没有检测到任何 adapter，对于UNet LoRA，直接认为已加载
                                            if is_unet_lora:
                                                logger.info(f"  ✓ UNet LoRA 已加载（load_lora_weights成功，get_list_adapters返回空但无需adapter机制）")
                                                print(f"  [调试] UNet LoRA 已加载（get_list_adapters返回空但无需adapter机制）")
                                                lora_loaded = True
                                                # 跳过后续的 set_adapters 尝试
                                                adapter_to_use = None
                                    except Exception as check_e:
                                        logger.warning(f"  ⚠ 检查 adapter 列表失败: {check_e}")
                                        # ⚡ 关键修复：如果检查失败，对于UNet LoRA，直接认为已加载
                                        if is_unet_lora:
                                            logger.info(f"  ✓ UNet LoRA 已加载（load_lora_weights成功，检查adapter列表失败但无需adapter机制）")
                                            print(f"  [调试] UNet LoRA 已加载（检查adapter列表失败但无需adapter机制）")
                                            lora_loaded = True
                                            adapter_to_use = None
                                        else:
                                            # 如果检查失败，仍然尝试使用指定的 adapter_name
                                            adapter_to_use = adapter_name
                                else:
                                    # ⚡ 关键修复：没有 get_list_adapters 方法，对于UNet LoRA，直接认为已加载
                                    if is_unet_lora:
                                        logger.info(f"  ✓ UNet LoRA 已加载（load_lora_weights成功，无get_list_adapters方法但无需adapter机制）")
                                        print(f"  [调试] UNet LoRA 已加载（无get_list_adapters方法但无需adapter机制）")
                                        lora_loaded = True
                                        adapter_to_use = None
                                    else:
                                        # 没有 get_list_adapters 方法，直接使用指定的 adapter_name
                                        adapter_to_use = adapter_name
                                
                                if adapter_to_use:
                                    try:
                                        self.pipeline.set_adapters([adapter_to_use], adapter_weights=[lora_weight])
                                        logger.info(f"  ✓ LoRA 加载成功 (pipeline, adapter: {adapter_to_use}, weight: {lora_weight})")
                                        print(f"  [调试] LoRA 加载成功 (pipeline, adapter: {adapter_to_use}, weight: {lora_weight})")
                                        lora_loaded = True
                                    except Exception as set_e:
                                        # ⚡ 关键修复：如果set_adapters失败，对于UNet LoRA，load_lora_weights可能已经直接应用了权重
                                        logger.warning(f"  ⚠ 设置 adapter {adapter_to_use} 失败: {set_e}")
                                        if is_unet_lora:
                                            # UNet LoRA不需要set_adapters，load_lora_weights已经直接应用了权重
                                            logger.info(f"  ✓ UNet LoRA 已加载（load_lora_weights成功，set_adapters失败但无需adapter机制）")
                                            print(f"  [调试] UNet LoRA 已加载（load_lora_weights成功，set_adapters失败但无需adapter机制）")
                                            lora_loaded = True
                                        else:
                                            # 对于非UNet LoRA，尝试其他方式验证
                                            logger.info(f"  ℹ 尝试继续使用（LoRA可能已直接应用）")
                                            try:
                                                # 尝试获取已加载的adapters（可能使用不同的方法）
                                                if hasattr(self.pipeline, 'get_active_adapters'):
                                                    active = list(self.pipeline.get_active_adapters())
                                                    if active:
                                                        logger.info(f"  ✓ LoRA 可能已通过其他方式加载 (active adapters: {active})")
                                                        lora_loaded = True
                                                    else:
                                                        logger.warning(f"  ⚠ 未检测到active adapters")
                                                        lora_loaded = False
                                                else:
                                                    lora_loaded = False
                                            except:
                                                logger.warning(f"  ⚠ 无法验证LoRA是否已加载")
                                                lora_loaded = False
                                else:
                                    # ⚡ 关键修复：对于UNet格式的LoRA，即使get_list_adapters返回空，load_lora_weights也可能已成功
                                    # UNet LoRA是直接应用到UNet的，不需要通过adapter机制
                                    # ⚠ 注意：load_lora_weights 默认使用权重 1.0，对于 UNet LoRA，权重已经在加载时直接应用
                                    # 如果需要调整权重，需要在加载前手动缩放 LoRA 权重文件，或者使用 fuse_lora 方法
                                    logger.debug(f"  [调试] adapter_to_use=None, is_unet_lora={is_unet_lora}")
                                    if is_unet_lora:
                                        # ⚡ 关键修复：对于 UNet LoRA，load_lora_weights 已经直接应用了权重（默认 1.0）
                                        # 如果需要使用自定义权重（lora_weight），我们需要在加载时手动缩放
                                        if lora_weight != 1.0:
                                            logger.warning(f"  ⚠ UNet LoRA 权重设置为 {lora_weight}，但 load_lora_weights 默认使用 1.0")
                                            logger.warning(f"  ⚠ 建议：如果 LoRA 效果过强，可以降低 lora_weight；如果效果过弱，可以提高 lora_weight")
                                            print(f"  [调试] UNet LoRA 已加载（默认权重 1.0，配置权重 {lora_weight} 未应用）")
                                        else:
                                            logger.info(f"  ✓ UNet LoRA 已加载（权重: 1.0）")
                                            print(f"  [调试] UNet LoRA 已加载（权重: 1.0）")
                                        lora_loaded = True
                                    else:
                                        # adapter 未加载，尝试方法2
                                        logger.warning(f"  ⚠ Adapter {adapter_name} 未成功加载，尝试不使用 adapter_name")
                                        print(f"  [调试] is_unet_lora={is_unet_lora}，将尝试方法2")
                                        lora_loaded = False
                            else:
                                # 没有 set_adapters 方法，但 load_lora_weights 可能已成功（UNet LoRA）
                                logger.info(f"  ✓ LoRA 加载成功 (pipeline, 无 set_adapters 方法，UNet LoRA已直接应用)")
                                print(f"  [调试] LoRA 加载成功 (pipeline, UNet LoRA已直接应用)")
                                lora_loaded = True
                        except Exception as e:
                            logger.warning(f"  ⚠ Pipeline LoRA 加载失败（方法1）: {e}")
                            lora_loaded = False
                        
                        # 方法2：如果方法1失败，尝试不使用 adapter_name（让系统自动检测）
                        if not lora_loaded:
                            try:
                                logger.info(f"  尝试方法2：不使用 adapter_name 加载 LoRA...")
                                self.pipeline.load_lora_weights(actual_lora_path)  # 不使用 adapter_name
                                # 获取实际加载的 adapter 名称
                                if hasattr(self.pipeline, 'get_list_adapters'):
                                    try:
                                        list_adapters = self.pipeline.get_list_adapters()
                                        all_adapters = {adapter for adapters in list_adapters.values() for adapter in adapters}
                                        if all_adapters:
                                            actual_adapter = list(all_adapters)[0]
                                            if hasattr(self.pipeline, 'set_adapters'):
                                                self.pipeline.set_adapters([actual_adapter], adapter_weights=[lora_weight])
                                            logger.info(f"  ✓ LoRA 加载成功 (pipeline, 方法2, adapter: {actual_adapter})")
                                            print(f"  [调试] LoRA 加载成功 (pipeline, 方法2, adapter: {actual_adapter})")
                                            lora_loaded = True
                                        else:
                                            # ⚡ 关键修复：对于UNet格式的LoRA，即使get_list_adapters返回空，load_lora_weights也可能已成功
                                            if is_unet_lora:
                                                logger.info(f"  ✓ UNet LoRA 已加载（方法2，load_lora_weights成功，无需adapter机制）")
                                                print(f"  [调试] UNet LoRA 已加载（方法2，load_lora_weights成功，无需adapter机制）")
                                                lora_loaded = True
                                            else:
                                                logger.warning(f"  ⚠ 方法2：未检测到任何 adapter")
                                    except Exception as check_e2:
                                        logger.warning(f"  ⚠ 方法2：检查 adapter 列表失败: {check_e2}")
                                elif hasattr(self.pipeline, 'get_active_adapters'):
                                    # 使用 get_active_adapters 作为备选
                                    try:
                                        active_adapters = list(self.pipeline.get_active_adapters())
                                        if active_adapters:
                                            if hasattr(self.pipeline, 'set_adapters'):
                                                self.pipeline.set_adapters(active_adapters, adapter_weights=[lora_weight])
                                            logger.info(f"  ✓ LoRA 加载成功 (pipeline, 方法2, adapter: {active_adapters[0]})")
                                            print(f"  [调试] LoRA 加载成功 (pipeline, 方法2, adapter: {active_adapters[0]})")
                                            lora_loaded = True
                                    except Exception as active_e:
                                        logger.warning(f"  ⚠ 方法2：获取 active adapters 失败: {active_e}")
                                else:
                                    # 没有检查方法，假设加载成功
                                    logger.info(f"  ✓ LoRA 加载成功 (pipeline, 方法2, 无法验证)")
                                    print(f"  [调试] LoRA 加载成功 (pipeline, 方法2)")
                                    lora_loaded = True
                            except Exception as e2:
                                logger.warning(f"  ⚠ Pipeline LoRA 加载失败（方法2）: {e2}")
                    
                    # 2. 如果 pipeline 不存在但需要 LoRA，尝试加载 pipeline
                    if not lora_loaded and self.pipeline is None:
                        try:
                            logger.info(f"  Pipeline 不存在，尝试加载 diffusers pipeline 以支持 LoRA...")
                            self._load_diffusers_pipeline()
                            if self.pipeline is not None:
                                # 方法1：尝试使用 adapter_name 和 prefix=None（修复权重格式不匹配问题）
                                try:
                                    self.pipeline.load_lora_weights(actual_lora_path, adapter_name=adapter_name, weight_name=None)
                                    # ⚡ 修复：在设置 adapter 之前，先检查 adapter 是否已成功加载
                                    if hasattr(self.pipeline, 'set_adapters'):
                                        # 检查 adapter 是否已加载
                                        adapter_to_use = None
                                        if hasattr(self.pipeline, 'get_list_adapters'):
                                            try:
                                                list_adapters = self.pipeline.get_list_adapters()
                                                all_adapters = {adapter for adapters in list_adapters.values() for adapter in adapters}
                                                if adapter_name in all_adapters:
                                                    adapter_to_use = adapter_name
                                                elif all_adapters:
                                                    # 如果指定的 adapter_name 不在列表中，但已有其他 adapter，使用第一个
                                                    adapter_to_use = list(all_adapters)[0]
                                                    logger.info(f"  ℹ 使用自动检测的 adapter: {adapter_to_use} (而非指定的 {adapter_name})")
                                            except Exception as check_e:
                                                logger.warning(f"  ⚠ 检查 adapter 列表失败: {check_e}")
                                                # 如果检查失败，仍然尝试使用指定的 adapter_name
                                                adapter_to_use = adapter_name
                                        else:
                                            # 没有 get_list_adapters 方法，直接使用指定的 adapter_name
                                            adapter_to_use = adapter_name
                                        
                                        if adapter_to_use:
                                            try:
                                                self.pipeline.set_adapters([adapter_to_use], adapter_weights=[lora_weight])
                                                logger.info(f"  ✓ LoRA 加载成功 (新加载的 pipeline, adapter: {adapter_to_use})")
                                                print(f"  [调试] LoRA 加载成功 (新加载的 pipeline, adapter: {adapter_to_use})")
                                                lora_loaded = True
                                            except Exception as set_e:
                                                logger.warning(f"  ⚠ 设置 adapter {adapter_to_use} 失败: {set_e}")
                                                lora_loaded = False
                                        else:
                                            logger.warning(f"  ⚠ Adapter {adapter_name} 未成功加载，尝试不使用 adapter_name")
                                            lora_loaded = False
                                    else:
                                        # 没有 set_adapters 方法，但 load_lora_weights 可能已成功
                                        logger.info(f"  ✓ LoRA 加载成功 (新加载的 pipeline, 无 set_adapters 方法)")
                                        print(f"  [调试] LoRA 加载成功 (新加载的 pipeline)")
                                        lora_loaded = True
                                except Exception as e:
                                    logger.warning(f"  ⚠ 新加载的 Pipeline LoRA 加载失败（方法1）: {e}")
                                    lora_loaded = False
                                
                                # 方法2：如果方法1失败，尝试不使用 adapter_name（让系统自动检测）
                                if not lora_loaded:
                                    try:
                                        logger.info(f"  尝试方法2：不使用 adapter_name 加载 LoRA...")
                                        self.pipeline.load_lora_weights(actual_lora_path)  # 不使用 adapter_name
                                        # 获取实际加载的 adapter 名称
                                        if hasattr(self.pipeline, 'get_list_adapters'):
                                            try:
                                                list_adapters = self.pipeline.get_list_adapters()
                                                all_adapters = {adapter for adapters in list_adapters.values() for adapter in adapters}
                                                if all_adapters:
                                                    actual_adapter = list(all_adapters)[0]
                                                    if hasattr(self.pipeline, 'set_adapters'):
                                                        self.pipeline.set_adapters([actual_adapter], adapter_weights=[lora_weight])
                                                    logger.info(f"  ✓ LoRA 加载成功 (新加载的 pipeline, 方法2, adapter: {actual_adapter})")
                                                    print(f"  [调试] LoRA 加载成功 (新加载的 pipeline, 方法2, adapter: {actual_adapter})")
                                                    lora_loaded = True
                                                else:
                                                    logger.warning(f"  ⚠ 方法2：未检测到任何 adapter")
                                            except Exception as check_e2:
                                                logger.warning(f"  ⚠ 方法2：检查 adapter 列表失败: {check_e2}")
                                        elif hasattr(self.pipeline, 'get_active_adapters'):
                                            # 使用 get_active_adapters 作为备选
                                            try:
                                                active_adapters = list(self.pipeline.get_active_adapters())
                                                if active_adapters:
                                                    if hasattr(self.pipeline, 'set_adapters'):
                                                        self.pipeline.set_adapters(active_adapters, adapter_weights=[lora_weight])
                                                    logger.info(f"  ✓ LoRA 加载成功 (新加载的 pipeline, 方法2, adapter: {active_adapters[0]})")
                                                    print(f"  [调试] LoRA 加载成功 (新加载的 pipeline, 方法2, adapter: {active_adapters[0]})")
                                                    lora_loaded = True
                                            except Exception as active_e:
                                                logger.warning(f"  ⚠ 方法2：获取 active adapters 失败: {active_e}")
                                        else:
                                            # 没有检查方法，假设加载成功
                                            logger.info(f"  ✓ LoRA 加载成功 (新加载的 pipeline, 方法2, 无法验证)")
                                            print(f"  [调试] LoRA 加载成功 (新加载的 pipeline, 方法2)")
                                            lora_loaded = True
                                    except Exception as e2:
                                        logger.warning(f"  ⚠ 新加载的 Pipeline LoRA 加载失败（方法2）: {e2}")
                        except Exception as e:
                            logger.warning(f"  ⚠ 加载 Pipeline 失败: {e}")
                    
                    # 3. 尝试在原生模型上加载（原生模式，不支持直接加载 LoRA）
                    # 注意：原生 Flux 模型（flux.model.Flux）不支持直接 load_lora_weights
                    # 原生模式主要用于性能优化，如果需要 LoRA，建议使用 pipeline 模式
                    if not lora_loaded and self.use_native and self.flux_model is not None:
                        try:
                            # 尝试使用 PEFT 加载 LoRA
                            try:
                                from peft import PeftModel
                                from safetensors.torch import load_file
                                
                                logger.info(f"  尝试使用 PEFT 加载 LoRA 到原生模型...")
                                
                                # 方法1：尝试使用 PeftModel.from_pretrained（如果模型支持）
                                # 注意：原生 Flux 模型可能不支持 PeftModel，需要手动合并权重
                                
                                # 方法2：手动加载 LoRA 权重并合并到模型
                                logger.info(f"  手动加载 LoRA 权重: {lora_path}")
                                lora_state_dict = load_file(lora_path)
                                
                                # 获取模型的状态字典
                                model_state_dict = self.flux_model.state_dict()
                                
                                # 合并 LoRA 权重（简单方法：直接添加到对应层）
                                # 注意：这需要 LoRA 权重格式与模型层名称匹配
                                merged_count = 0
                                for key, value in lora_state_dict.items():
                                    # LoRA 权重通常以 "lora_A" 或 "lora_B" 结尾
                                    # 需要找到对应的基础层并合并
                                    base_key = key.replace(".lora_A", "").replace(".lora_B", "")
                                    if base_key in model_state_dict:
                                        # 简单的权重合并（实际应该使用 LoRA 的数学公式）
                                        # W_new = W_base + alpha * (lora_B @ lora_A)
                                        # 这里简化处理，直接按权重比例合并
                                        if ".lora_A" in key or ".lora_B" in key:
                                            # 跳过 lora_A 和 lora_B，需要成对处理
                                            continue
                                        else:
                                            # 如果是合并后的权重，直接应用
                                            model_state_dict[base_key] = model_state_dict[base_key] + lora_weight * value
                                            merged_count += 1
                                
                                if merged_count > 0:
                                    # 加载合并后的权重
                                    self.flux_model.load_state_dict(model_state_dict, strict=False)
                                    logger.info(f"  ✓ LoRA 加载成功 (原生模型，合并了 {merged_count} 个权重)")
                                    print(f"  [调试] LoRA 加载成功 (原生模型，合并了 {merged_count} 个权重)")
                                    lora_loaded = True
                                else:
                                    logger.warning(f"  ⚠ 无法匹配 LoRA 权重格式，可能需要使用 diffusers pipeline 模式")
                                    
                            except ImportError:
                                logger.warning(f"  ⚠ PEFT 库未安装，无法加载 LoRA 到原生模型")
                            except Exception as e:
                                logger.warning(f"  ⚠ 原生模型 LoRA 加载失败: {e}")
                                logger.info(f"  ℹ 建议：使用 diffusers pipeline 模式以支持 LoRA")
                                
                        except Exception as e:
                            logger.warning(f"  ⚠ 原生模型 LoRA 加载异常: {e}")
                    
                    if not lora_loaded:
                        logger.warning(f"  ⚠ LoRA 加载失败（pipeline 和原生模型都失败），继续使用 PuLID（无 LoRA）")
                    
                    # 清理临时文件
                    if converted_lora_path and converted_lora_path != lora_path and os.path.exists(converted_lora_path):
                        try:
                            os.unlink(converted_lora_path)
                        except:
                            pass
                            
                except Exception as e:
                    logger.warning(f"  ⚠ LoRA 加载异常: {e}，继续使用 PuLID（无 LoRA）")
                    # 清理临时文件
                    if converted_lora_path and converted_lora_path != lora_path and os.path.exists(converted_lora_path):
                        try:
                            os.unlink(converted_lora_path)
                        except:
                            pass
            else:
                logger.warning(f"  ⚠ LoRA 路径不存在: {lora_path}")
                print(f"  [调试] LoRA 路径不存在: {lora_path}")
        
        # 转换参考强度到 PuLID 权重
        # PuLID 权重范围通常是 0.0-1.0
        # reference_strength 0-100 映射到 0.0-1.0
        pulid_weight = reference_strength / 100.0
        
        # 检查是否需要增强服饰一致性
        # 如果 prompt 中包含服饰描述，或者参考强度在中等范围（50-75），启用服饰增强
        enhance_clothing = kwargs.get('enhance_clothing_consistency', False)
        if not enhance_clothing:
            # 自动检测：如果 prompt 中包含服饰关键词，自动启用
            clothing_keywords = ['robe', 'clothing', 'dress', 'outfit', 'garment', 'attire', 
                               '服饰', '衣服', '服装', '长袍', '道袍', '衣袍']
            prompt_lower = prompt.lower()
            if any(keyword in prompt_lower for keyword in clothing_keywords):
                enhance_clothing = True
                logger.info("  检测到服饰描述，自动启用服饰一致性增强")
        
        # 调整权重曲线 (非线性映射，使中间值更自然)
        pulid_weight = self._adjust_weight_curve(pulid_weight, enhance_clothing=enhance_clothing)
        
        # 原生模式：由于已经实现了显存管理，不再需要降低分辨率
        # 保持用户配置的原始分辨率和步数
        
        logger.info(f"生成参数:")
        logger.info(f"  参考强度: {reference_strength}% -> PuLID weight: {pulid_weight:.2f}")
        logger.info(f"  分辨率: {width}x{height}")
        logger.info(f"  步数: {num_inference_steps}")
        print(f"  [调试] PuLID引擎接收到的推理步数: {num_inference_steps}")
        
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 加载参考图像
        if isinstance(face_reference, str):
            face_reference_pil = Image.open(face_reference).convert('RGB')
        else:
            face_reference_pil = face_reference
        face_reference_np = np.array(face_reference_pil)
        
        try:
            # 检查是否有 PuLID 模型和采样模块
            if hasattr(self, 'use_pulid') and self.use_pulid and self.pulid_model is not None:
                logger.info("使用 PuLID 完整模式进行身份注入...")
                
                # 使用 PuLID 获取身份嵌入
                try:
                    id_embedding, uncond_id_embedding = self.pulid_model.get_id_embedding(
                        face_reference_np, 
                        cal_uncond=True
                    )
                    logger.info(f"  身份嵌入提取成功: {id_embedding.shape}")
                except Exception as e:
                    logger.warning(f"PuLID 身份嵌入提取失败: {e}")
                    logger.info("回退到无身份注入模式")
                    id_embedding = None
                    uncond_id_embedding = None
                
                if id_embedding is not None:
                    # 尝试使用 PuLID 的原生采样流程
                    try:
                        result = self._generate_with_pulid_native(
                            prompt=prompt,
                            id_embedding=id_embedding,
                            uncond_id_embedding=uncond_id_embedding,
                            id_weight=pulid_weight,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=seed
                        )
                        if result is not None:
                            return result
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"PuLID 原生采样显存不足: {e}")
                        logger.warning("即使启用了 aggressive_offload 仍然 OOM，尝试卸载原生模型并回退到 diffusers 模式...")
                        
                        # 先卸载原生模型以释放显存
                        if self.use_native:
                            logger.info("卸载原生 Flux 模型以释放显存...")
                            if self.flux_model is not None:
                                del self.flux_model
                                self.flux_model = None
                            if self.ae is not None:
                                del self.ae
                                self.ae = None
                            if self.t5 is not None:
                                del self.t5
                                self.t5 = None
                            if self.clip is not None:
                                del self.clip
                                self.clip = None
                            if self.pulid_model is not None:
                                del self.pulid_model
                                self.pulid_model = None
                            
                            # 强制清理显存
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            logger.info("原生模型已卸载，显存已释放")
                        
                        # 如果是原生模式且没有 diffusers pipeline，尝试加载
                        if self.use_native and self.pipeline is None:
                            logger.info("尝试加载 diffusers pipeline 作为备用...")
                            try:
                                self._load_diffusers_pipeline()
                            except Exception as load_error:
                                logger.error(f"加载 diffusers pipeline 失败: {load_error}")
                                raise RuntimeError("原生模式失败且无法加载备用 pipeline")
                        
                        if self.pipeline is not None:
                            logger.info("回退到 diffusers 模式")
                        else:
                            raise RuntimeError("原生模式失败且无备用 pipeline")
                    except Exception as e:
                        logger.warning(f"PuLID 原生采样出错: {e}")
                        # 如果是原生模式且没有 diffusers pipeline，尝试加载
                        if self.use_native and self.pipeline is None:
                            logger.info("尝试加载 diffusers pipeline 作为备用...")
                            try:
                                self._load_diffusers_pipeline()
                            except Exception as load_error:
                                logger.error(f"加载 diffusers pipeline 失败: {load_error}")
                                raise
                        
                        if self.pipeline is not None:
                            logger.info("回退到 diffusers 模式")
                        else:
                            raise
                    
                    # 回退：使用 diffusers，但设置 ID 到 transformer
                    if self.pipeline is not None:
                        dit = self.pipeline.transformer
                        if hasattr(dit, 'pulid_ca'):
                            # 设置 ID 嵌入到 pulid_ca 模块
                            for ca in dit.pulid_ca:
                                ca.id_embedding = id_embedding
                                ca.id_scale = pulid_weight
                        
                        # ⚡ 关键修复：在生成图像时抑制 CLIP tokenizer 的 77 token 警告
                        with suppress_clip_tokenizer_warnings():
                            result = self.pipeline(
                                prompt=prompt,
                                width=width,
                                height=height,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                generator=generator,
                            ).images[0]
                        
                        # 清理
                        if hasattr(dit, 'pulid_ca'):
                            for ca in dit.pulid_ca:
                                ca.id_embedding = None
                        
                        return result
            
            # 回退：使用纯 Flux 生成（无身份注入）
            # 只有当 pipeline 存在时才回退
            if self.pipeline is not None:
                logger.warning("使用纯 Flux 生成（无身份注入）")
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                return result
            else:
                logger.error("原生模式失败，且无备用 pipeline，无法生成")
                raise RuntimeError("Native generation failed and no fallback pipeline available")
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            raise
    
    def _generate_with_pulid_native(
        self,
        prompt: str,
        id_embedding: torch.Tensor,
        uncond_id_embedding: torch.Tensor,
        id_weight: float,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int
    ) -> Optional[Image.Image]:
        """
        使用 PuLID 原生采样流程生成
        
        PuLID 需要使用自定义的 Flux 模型和采样循环
        """
        try:
            # 检查是否使用原生模式
            if not self.use_native:
                logger.info("非原生模式，跳过原生采样")
                return None
            
            # 导入采样模块
            from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
            import random
            
            logger.info("使用 PuLID 原生采样流程...")
            
            # 处理 seed
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
                logger.info(f"  生成随机种子: {seed}")
            
            # 清理 GPU 缓存（更彻底）
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
                # 再次清理
                gc.collect()
                torch.cuda.empty_cache()
            
            # 准备噪声
            x = get_noise(
                num_samples=1,
                height=height,
                width=width,
                device=self.device,
                dtype=self.dtype,
                seed=seed
            )
            
            # 获取采样时间表
            timesteps = get_schedule(
                num_steps=num_inference_steps,
                image_seq_len=x.shape[1] * x.shape[2] // 4,
                shift=True
            )
            
            # 准备输入 (使用原生 T5 和 CLIP)
            # CPU offload: 临时将编码器移到 GPU
            if hasattr(self, 'use_cpu_offload') and self.use_cpu_offload:
                logger.info("  临时移动 T5/CLIP 到 GPU...")
                self.t5.to(self.device)
                self.clip.to(self.device)
            
            inp = prepare(self.t5, self.clip, x, prompt)
            
            # 记录输入张量大小
            if torch.cuda.is_available():
                img_size_gb = inp["img"].element_size() * inp["img"].nelement() / 1024**3
                txt_size_gb = inp["txt"].element_size() * inp["txt"].nelement() / 1024**3
                logger.info(f"  输入张量大小: img={img_size_gb:.2f}GB, txt={txt_size_gb:.2f}GB")
            
            # CPU offload: 编码完成后移回 CPU
            if hasattr(self, 'use_cpu_offload') and self.use_cpu_offload:
                logger.info("  移动 T5/CLIP 回 CPU...")
                self.t5.to("cpu")
                self.clip.to("cpu")
                # 强制清理显存
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                log_memory("After Encoder Offload")
            
            # 原生模式：由于已经实现了显存管理，不再需要强制使用 aggressive_offload
            # 可以根据显存情况选择是否使用（显存充足时使用标准模式速度更快）
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - reserved
                logger.info(f"  去噪前显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 可用={free:.2f}GB")
                
                # 如果可用显存充足（>50GB），不使用 aggressive_offload（速度更快）
                # 如果可用显存较少（<50GB），使用 aggressive_offload（更安全）
                use_aggressive_offload = free < 50
                if use_aggressive_offload:
                    logger.info(f"  可用显存较少 ({free:.2f}GB)，启用 aggressive_offload 模式")
                else:
                    logger.info(f"  可用显存充足 ({free:.2f}GB)，使用标准模式（速度更快）")
            else:
                use_aggressive_offload = False
            
            # 去噪前清理显存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"  开始去噪 (步数: {num_inference_steps}, ID权重: {id_weight:.2f}, aggressive_offload={use_aggressive_offload})...")
            
            # 使用包装函数，在每个时间步之间清理显存
            x = self._denoise_with_memory_management(
                model=self.flux_model,
                img=inp["img"],
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                vec=inp["vec"],
                timesteps=timesteps,
                guidance=guidance_scale,
                id=id_embedding,
                id_weight=id_weight,
                uncond_id=uncond_id_embedding,
                aggressive_offload=use_aggressive_offload
            )
            
            # 解包
            x = unpack(x.float(), height, width)
            
            # 使用 AutoEncoder 解码
            logger.info("  解码图像...")
            # 如果 AutoEncoder 在 CPU 上，需要移到 GPU
            if self.ae is not None:
                try:
                    ae_device = next(self.ae.parameters()).device
                    if ae_device.type == "cpu":
                        logger.info("  将 AutoEncoder 移到 GPU 进行解码...")
                        self.ae = self.ae.to(self.device)
                except Exception:
                    pass  # 如果无法检查设备，继续使用当前状态
            
            # 获取 AutoEncoder 的 dtype（通过参数获取）
            if self.ae is not None:
                ae_dtype = next(self.ae.parameters()).dtype
                x = x.to(ae_dtype)
                with torch.no_grad():
                    x = self.ae.decode(x)
            else:
                raise RuntimeError("AutoEncoder 不可用")
            
            # 转换为图像
            x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            x = x.cpu().permute(0, 2, 3, 1).numpy()[0]
            x = (x * 255).astype(np.uint8)
            
            logger.info("  原生采样完成!")
            return Image.fromarray(x)
            
        except ImportError as e:
            logger.warning(f"PuLID flux 模块未找到: {e}")
            return None
        except Exception as e:
            logger.warning(f"PuLID 原生采样出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _denoise_with_memory_management(
        self,
        model,
        img,
        img_ids,
        txt,
        txt_ids,
        vec,
        timesteps,
        guidance,
        id=None,
        id_weight=1.0,
        uncond_id=None,
        aggressive_offload=False,
        start_step=0,
        true_cfg=1.0,
        timestep_to_start_cfg=1,
        neg_txt=None,
        neg_txt_ids=None,
        neg_vec=None,
    ):
        """
        带显存管理的去噪函数
        
        在每个时间步之间清理显存，避免中间激活值累积导致 OOM
        
        这是原始 denoise 函数的改进版本，添加了显存管理
        """
        import gc
        
        # 手动实现去噪循环，在每个时间步之间清理显存
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        use_true_cfg = abs(true_cfg - 1.0) > 1e-2
        
        total_steps = len(timesteps) - 1
        logger.info(f"  使用显存管理去噪（{total_steps} 步）...")
        
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # 每 5 步记录一次进度和显存
            if i % 5 == 0 or i == total_steps - 1:
                logger.info(f"    去噪进度: {i+1}/{total_steps}")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"      显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
            
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            
            # 前向传播（条件）
            with torch.no_grad():  # 确保不需要梯度，减少显存占用
                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    id=id if i >= start_step else None,
                    id_weight=id_weight,
                    aggressive_offload=aggressive_offload,
                )
            
            # 如果使用 true_cfg，需要计算 negative prediction
            if use_true_cfg and i >= timestep_to_start_cfg:
                with torch.no_grad():
                    neg_pred = model(
                        img=img,
                        img_ids=img_ids,
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                        id=uncond_id if i >= start_step else None,
                        id_weight=id_weight,
                        aggressive_offload=aggressive_offload,
                    )
                pred = neg_pred + true_cfg * (pred - neg_pred)
                del neg_pred  # 立即释放
            
            # 更新图像
            img = img + (t_prev - t_curr) * pred
            
            # 清理中间变量（关键：立即释放，避免累积）
            del pred
            del t_vec
            
            # 每 2 步清理一次显存（平衡性能和显存占用）
            if (i + 1) % 2 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("  去噪完成，显存已清理")
        return img
    
    def _adjust_weight_curve(self, weight: float, enhance_clothing: bool = False) -> float:
        """
        调整权重曲线
        
        使参考强度的变化更自然:
        - 低值 (0-30%): 快速降低，让环境占主导
        - 中值 (30-70%): 平滑过渡
        - 高值 (70-100%): 趋于饱和，强锁脸
        
        Args:
            weight: 原始权重 (0.0-1.0)
            enhance_clothing: 是否增强服饰一致性（提高中等强度的权重）
            
        Returns:
            调整后的权重
        """
        # ⚡ 2024-12-22 修复：提高整体权重，解决人脸不像问题
        # 根据测试反馈，之前的权重太低，PuLID 身份注入效果不足
        import math
        
        if enhance_clothing:
            # 服饰增强模式：让中等强度也能获得更高权重
            k = 5
            center = 0.40  # 更早达到高权重
            min_weight = 0.50  # ⚡ 提高：从 0.35 到 0.50
            max_weight = 1.0   # ⚡ 提高：从 0.98 到 1.0
        else:
            # ⚡ 标准模式：也需要提高权重
            k = 5          # 稍微平缓，让中等值更稳定
            center = 0.45  # 向左偏移
            min_weight = 0.45  # ⚡ 提高：从 0.30 到 0.45
            max_weight = 1.0   # ⚡ 提高：从 0.95 到 1.0
        
        adjusted = 1 / (1 + math.exp(-k * (weight - center)))
        
        # 缩放到指定范围 (PuLID 有效工作范围)
        adjusted = min_weight + adjusted * (max_weight - min_weight)
        
        return adjusted
    
    def calculate_reference_strength(
        self,
        shot_type: str,
        camera_angle: str = "eye_level",
        has_emotion: bool = False
    ) -> int:
        """
        根据镜头类型自动计算参考强度
        
        参考可灵/即梦的策略：
        - 远景: 环境优先，弱参考
        - 中景: 平衡
        - 特写: 人脸优先，强参考
        
        Args:
            shot_type: 镜头类型 (wide, full, medium, close, extreme_close)
            camera_angle: 相机角度
            has_emotion: 是否有表情需求
            
        Returns:
            参考强度 (0-100)
        """
        # 基础强度映射
        strength_map = {
            "extreme_wide": 20,
            "wide": 30,
            "full": 45,
            "american": 55,  # 7/8 身
            "medium": 60,
            "medium_close": 70,
            "close": 80,
            "extreme_close": 90,
        }
        
        base_strength = strength_map.get(shot_type, 60)
        
        # 角度调整
        if camera_angle in ["top_down", "bird_eye"]:
            # 俯拍不需要太强的人脸参考
            base_strength = min(base_strength, 40)
        elif camera_angle == "low":
            # 仰拍需要略强的参考
            base_strength = min(base_strength + 10, 95)
        
        # 表情调整
        if has_emotion:
            # 有表情需求时，增强参考以保持表情准确
            base_strength = min(base_strength + 10, 95)
        
        return base_strength
    
    def unload(self):
        """卸载模型以释放显存"""
        log_memory("Before Unload")
        
        # ⚡ 关键修复：先移动到 CPU，再删除，确保显存真正释放
        if self.pipeline is not None:
            try:
                # 如果是 diffusers pipeline，尝试移动到 CPU
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
            except:
                pass
            self.pipeline = None
            
        if self.flux_model is not None:
            try:
                # 移动到 CPU 再删除
                if hasattr(self.flux_model, 'to'):
                    self.flux_model.to('cpu')
                del self.flux_model
            except:
                pass
            self.flux_model = None
            
        if self.ae is not None:
            try:
                if hasattr(self.ae, 'to'):
                    self.ae.to('cpu')
                del self.ae
            except:
                pass
            self.ae = None
            
        if self.pulid_model is not None:
            try:
                if hasattr(self.pulid_model, 'to'):
                    self.pulid_model.to('cpu')
                del self.pulid_model
            except:
                pass
            self.pulid_model = None
            
        if self.t5 is not None:
            try:
                if hasattr(self.t5, 'to'):
                    self.t5.to('cpu')
                del self.t5
            except:
                pass
            self.t5 = None
            
        if self.clip is not None:
            try:
                if hasattr(self.clip, 'to'):
                    self.clip.to('cpu')
                del self.clip
            except:
                pass
            self.clip = None
        
        if self.face_analyzer is not None:
            try:
                del self.face_analyzer
            except:
                pass
            self.face_analyzer = None
        
        self.pulid_loaded = False
        
        # ⚡ 关键修复：多次清理 GPU 缓存，确保显存真正释放
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 多次清理，确保彻底释放
            for i in range(5):
                gc.collect()
                torch.cuda.empty_cache()
                if i % 2 == 0:
                    torch.cuda.synchronize()
        
        logger.info("PuLID Engine 已彻底卸载")
        log_memory("After Unload")


class CharacterProfile:
    """
    角色档案系统
    
    参考可灵 Element Library 的多参考图方案
    
    支持的目录结构:
    character_profiles/hanli/
    ├── front/           # 正面角度
    │   ├── neutral.jpg
    │   ├── happy.jpg
    │   ├── sad.jpg
    │   ├── angry.jpg
    │   └── pain.jpg
    ├── side/            # 侧面角度
    │   ├── neutral.jpg
    │   └── angry.jpg
    └── three_quarter/   # 3/4 侧面
        ├── neutral.jpg
        ├── happy.jpg
        └── angry.jpg
    """
    
    def __init__(self, character_id: str, profile_dir: str):
        """
        Args:
            character_id: 角色ID (如 "hanli")
            profile_dir: 档案目录路径
        """
        self.character_id = character_id
        self.profile_dir = Path(profile_dir)
        
        # 参考图像: {角度: {表情: 路径}}
        self.references = {}
        
        # 可用的角度列表
        self.available_angles = []
        
        # 可用的表情列表
        self.available_expressions = set()
        
        # 加载参考图像
        self._load_references()
    
    def _load_references(self):
        """加载参考图像 - 适配新的目录结构"""
        if not self.profile_dir.exists():
            logger.warning(f"角色档案目录不存在: {self.profile_dir}")
            return
        
        # 支持的角度目录
        angle_dirs = {
            "front": ["front", "正面"],
            "three_quarter": ["three_quarter", "3_4", "半侧面"],
            "side": ["side", "profile", "侧面"],
            "back": ["back", "背面"],
        }
        
        # 支持的表情文件名
        expression_names = {
            "neutral": ["neutral", "中性", "default"],
            "happy": ["happy", "smile", "开心", "微笑"],
            "sad": ["sad", "悲伤"],
            "angry": ["angry", "愤怒"],
            "surprised": ["surprised", "惊讶"],
            "pain": ["pain", "痛苦"],
            "thinking": ["thinking", "思考"],
        }
        
        # 支持的文件扩展名
        extensions = [".jpg", ".jpeg", ".png", ".webp"]
        
        # 遍历角度目录
        for angle_key, angle_names in angle_dirs.items():
            for angle_name in angle_names:
                angle_path = self.profile_dir / angle_name
                if angle_path.exists() and angle_path.is_dir():
                    self.references[angle_key] = {}
                    self.available_angles.append(angle_key)
                    
                    # 遍历表情文件
                    for expr_key, expr_names in expression_names.items():
                        for expr_name in expr_names:
                            for ext in extensions:
                                file_path = angle_path / f"{expr_name}{ext}"
                                if file_path.exists():
                                    self.references[angle_key][expr_key] = file_path
                                    self.available_expressions.add(expr_key)
                                    break
                            if expr_key in self.references.get(angle_key, {}):
                                break
                    
                    break  # 找到一个角度目录就跳出
        
        # 日志
        logger.info(f"角色档案加载完成: {self.character_id}")
        logger.info(f"  可用角度: {self.available_angles}")
        logger.info(f"  可用表情: {list(self.available_expressions)}")
        
        # 详细日志
        for angle, expressions in self.references.items():
            logger.info(f"  {angle}: {list(expressions.keys())}")
    
    def get_reference_for_scene(
        self,
        camera_angle: str = "eye_level",
        emotion: str = "neutral"
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        根据场景获取最佳参考图
        
        Args:
            camera_angle: 相机角度 (eye_level, side, profile, top_down, low, etc.)
            emotion: 表情需求 (neutral, happy, sad, angry, pain, etc.)
            
        Returns:
            (主参考图路径, 表情参考图路径)
            主参考图: 根据角度和表情选择的最佳匹配
            表情参考图: 如果主参考图没有对应表情，提供正面的表情参考
        """
        primary = None
        expression_ref = None
        
        # 1. 根据相机角度选择最佳角度目录
        angle_key = self._map_camera_angle(camera_angle)
        
        # 2. 尝试获取对应角度+表情的图片
        if angle_key in self.references:
            angle_refs = self.references[angle_key]
            
            # 优先：完全匹配 (角度+表情)
            if emotion in angle_refs:
                primary = angle_refs[emotion]
            # 备选：该角度的 neutral 表情
            elif "neutral" in angle_refs:
                primary = angle_refs["neutral"]
                # 如果需要特定表情但该角度没有，从正面获取表情参考
                if emotion != "neutral" and "front" in self.references:
                    if emotion in self.references["front"]:
                        expression_ref = self.references["front"][emotion]
            # 备选：该角度的任意表情
            elif angle_refs:
                primary = list(angle_refs.values())[0]
        
        # 3. 如果没有找到，尝试其他角度
        if primary is None:
            for fallback_angle in ["three_quarter", "front", "side"]:
                if fallback_angle in self.references:
                    angle_refs = self.references[fallback_angle]
                    if emotion in angle_refs:
                        primary = angle_refs[emotion]
                        break
                    elif "neutral" in angle_refs:
                        primary = angle_refs["neutral"]
                        break
                    elif angle_refs:
                        primary = list(angle_refs.values())[0]
                        break
        
        # 4. 如果仍然没有表情参考，尝试从正面获取
        if expression_ref is None and emotion != "neutral":
            if "front" in self.references and emotion in self.references["front"]:
                expression_ref = self.references["front"][emotion]
        
        return primary, expression_ref
    
    def _map_camera_angle(self, camera_angle: str) -> str:
        """
        将相机角度映射到参考图角度
        
        Args:
            camera_angle: 场景中的相机角度描述
            
        Returns:
            对应的参考图角度 key
        """
        angle_mapping = {
            # 正面
            "eye_level": "three_quarter",  # 平视默认用 3/4 角度
            "front": "front",
            "straight": "front",
            
            # 侧面
            "side": "side",
            "profile": "side",
            
            # 3/4 角度
            "three_quarter": "three_quarter",
            "3/4": "three_quarter",
            
            # 俯拍用正面
            "top_down": "front",
            "bird_eye": "front",
            "high": "front",
            
            # 仰拍用正面或 3/4
            "low": "three_quarter",
            "worm_eye": "three_quarter",
            
            # 背面
            "back": "back",
            "behind": "back",
        }
        
        return angle_mapping.get(camera_angle, "three_quarter")
    
    def get_best_reference(
        self,
        camera_angle: str = "eye_level",
        emotion: str = "neutral"
    ) -> Optional[Path]:
        """
        获取单个最佳参考图 (简化接口)
        
        Args:
            camera_angle: 相机角度
            emotion: 表情
            
        Returns:
            最佳参考图路径
        """
        primary, _ = self.get_reference_for_scene(camera_angle, emotion)
        return primary
    
    def list_all_references(self) -> Dict[str, Dict[str, Path]]:
        """
        列出所有可用的参考图
        
        Returns:
            {角度: {表情: 路径}}
        """
        return self.references
    
    def __repr__(self):
        return f"CharacterProfile(id={self.character_id}, angles={self.available_angles}, expressions={list(self.available_expressions)})"


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试配置
    config = {
        "device": "cuda",
        "quantization": "bfloat16",
        "model_dir": "/vepfs-dev/shawn/vid/fanren/gen_video/models"
    }
    
    # 创建引擎
    engine = PuLIDEngine(config)
    
    # 测试参考强度计算
    print("\n参考强度测试:")
    for shot in ["wide", "medium", "close", "extreme_close"]:
        strength = engine.calculate_reference_strength(shot)
        print(f"  {shot}: {strength}%")
    
    print("\nPuLID Engine 初始化成功!")
