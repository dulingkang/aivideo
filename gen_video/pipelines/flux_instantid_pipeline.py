#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux + InstantID Pipeline
结合 Flux 模型和 InstantID 实现固定人脸生成
"""

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from typing import Optional
import numpy as np
from pathlib import Path
from .base_pipeline import BasePipeline

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
    
    # 修复 SCRFD 模型识别问题
    # InsightFace 的 model_zoo.get_model 可能无法识别 SCRFD 模型
    # 我们需要动态修复 get_model 方法
    try:
        # 导入必要的模块
        from insightface.model_zoo.scrfd import SCRFD
        import onnxruntime as ort
        import insightface.model_zoo.model_zoo as model_zoo_module
        
        # 保存原始的 get_model 函数
        _original_get_model = model_zoo_module.get_model
        
        def _patched_get_model(name, **kwargs):
            """修复后的 get_model，支持 SCRFD 模型"""
            # 先尝试原始方法
            model = _original_get_model(name, **kwargs)
            if model is not None:
                return model
            
            # 如果原始方法返回 None，检查是否是 SCRFD 模型
            try:
                # 检查文件是否存在
                import os.path as osp
                if not osp.exists(name) or not name.endswith('.onnx'):
                    return None
                
                # 检查文件名是否包含 scrfd
                if 'scrfd' in name.lower():
                    # 这是 SCRFD 模型，需要特殊处理
                    try:
                        # 创建 session
                        providers = kwargs.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                        session = ort.InferenceSession(name, providers=providers)
                        
                        # 使用 SCRFD 类加载
                        # 根据 scrfd.py 的源码，SCRFD 接受 model_file 和 session 参数
                        return SCRFD(model_file=name, session=session)
                    except Exception as e:
                        # 如果加载失败，返回 None
                        return None
            except Exception:
                pass
            
            return None
        
        # 替换 model_zoo 模块中的 get_model 函数
        # 注意：get_model 可能是 ModelRouter 类的方法，也可能是独立函数
        # 我们需要同时修补 ModelRouter.get_model 和独立的 get_model 函数
        
        # 1. 修补 ModelRouter 类的 get_model 方法
        if hasattr(model_zoo_module, 'ModelRouter'):
            original_router_get_model = model_zoo_module.ModelRouter.get_model
            
            def _patched_router_get_model(self, **kwargs):
                """修复后的 ModelRouter.get_model，支持 SCRFD 模型"""
                import os.path as osp
                
                # 添加调试信息
                file_name = osp.basename(self.onnx_file) if self.onnx_file else 'unknown'
                print(f'  [DEBUG] ModelRouter.get_model 被调用，文件: {file_name}')
                
                # 首先检查是否是 SCRFD 模型（通过文件名）
                if osp.exists(self.onnx_file) and self.onnx_file.endswith('.onnx'):
                    file_lower = self.onnx_file.lower()
                    print(f'  [DEBUG] 检查文件: {self.onnx_file}, 包含 scrfd: {"scrfd" in file_lower}')
                    if 'scrfd' in file_lower:
                        print(f'  [DEBUG] 识别为 SCRFD 模型，开始加载...')
                        try:
                            providers = kwargs.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                            provider_options = kwargs.get('provider_options', {})
                            print(f'  [DEBUG] 创建 ONNX Runtime session...')
                            session = ort.InferenceSession(self.onnx_file, providers=providers, provider_options=provider_options)
                            print(f'  [DEBUG] 创建 SCRFD 模型实例...')
                            scrfd_model = SCRFD(model_file=self.onnx_file, session=session)
                            # 使用与 InsightFace 标准格式一致的输出
                            input_shape_str = str(scrfd_model.input_shape) if hasattr(scrfd_model, 'input_shape') else 'None'
                            input_mean = getattr(scrfd_model, 'input_mean', 0.0)
                            input_std = getattr(scrfd_model, 'input_std', 1.0)
                            print(f'find model: {self.onnx_file} detection {input_shape_str} {input_mean} {input_std}')
                            print(f'  [DEBUG] SCRFD 模型加载成功！')
                            return scrfd_model
                        except Exception as e:
                            print(f"  ⚠️  SCRFD 加载失败: {e}")
                            import traceback
                            traceback.print_exc()
                            # 继续尝试原始方法
                
                # 尝试原始方法
                print(f'  [DEBUG] 尝试原始方法识别模型...')
                model = original_router_get_model(self, **kwargs)
                if model is not None:
                    print(f'  [DEBUG] 原始方法识别成功: {type(model).__name__}')
                    return model
                else:
                    print(f'  [DEBUG] 原始方法返回 None')
                
                return None
            
            model_zoo_module.ModelRouter.get_model = _patched_router_get_model
        
        # 2. 修补独立的 get_model 函数（如果存在）
        # 注意：model_zoo.get_model 会创建 ModelRouter 实例并调用其 get_model 方法
        # 所以主要修补 ModelRouter.get_model 就够了，但我们也修补 get_model 函数以确保完整性
        if hasattr(model_zoo_module, 'get_model') and callable(getattr(model_zoo_module, 'get_model')):
            # 检查是否是函数而不是类方法
            import inspect
            if inspect.isfunction(model_zoo_module.get_model):
                # 保存原始函数
                _original_get_model_func = model_zoo_module.get_model
                
                def _patched_get_model_func(name, **kwargs):
                    """修复后的 get_model 函数，支持 SCRFD 模型"""
                    # 如果是 .onnx 文件且包含 scrfd，直接处理
                    import os.path as osp
                    if isinstance(name, str) and name.endswith('.onnx') and 'scrfd' in name.lower():
                        print(f'  [DEBUG] get_model 函数识别到 SCRFD 模型: {name}')
                        try:
                            providers = kwargs.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                            provider_options = kwargs.get('provider_options', {})
                            session = ort.InferenceSession(name, providers=providers, provider_options=provider_options)
                            scrfd_model = SCRFD(model_file=name, session=session)
                            input_shape_str = str(scrfd_model.input_shape) if hasattr(scrfd_model, 'input_shape') else 'None'
                            input_mean = getattr(scrfd_model, 'input_mean', 0.0)
                            input_std = getattr(scrfd_model, 'input_std', 1.0)
                            print(f'find model: {name} detection {input_shape_str} {input_mean} {input_std}')
                            return scrfd_model
                        except Exception as e:
                            print(f"  ⚠️  get_model 函数中 SCRFD 加载失败: {e}")
                            # 继续使用原始方法
                    
                    # 使用原始方法
                    return _original_get_model_func(name, **kwargs)
                
                model_zoo_module.get_model = _patched_get_model_func
                print("     也修补了 get_model 函数")
        
        # 3. 修补从 __init__.py 导出的 get_model
        import insightface.model_zoo as model_zoo_pkg
        if hasattr(model_zoo_pkg, 'get_model'):
            # 使用相同的修补函数
            model_zoo_pkg.get_model = model_zoo_module.get_model
        
        print("  ✅ 已修复 SCRFD 模型识别问题")
        print(f"     修补了 ModelRouter.get_model 方法和 get_model 函数")
        
    except Exception as e:
        print(f"  ⚠️  修复 SCRFD 识别时出错: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  insightface 未安装，InstantID 功能将不可用")
    print("   安装命令: pip install insightface onnxruntime onnxruntime-gpu")


class FluxInstantIDPipeline(BasePipeline):
    """Flux + InstantID Pipeline（固定人脸生成）"""
    
    def __init__(
        self,
        model_path: str,
        instantid_path: Optional[str] = None,
        controlnet_path: Optional[str] = None,
        device: Optional[str] = None,
        model_type: str = "flux1",
        use_ip_adapter: bool = True  # 是否使用 IP-Adapter（False 时仅使用 LoRA）
    ):
        """
        初始化 Flux + InstantID Pipeline
        
        Args:
            model_path: Flux 模型路径
            instantid_path: InstantID 模型路径（包含 ip-adapter）
            controlnet_path: InstantID ControlNet 路径
            device: 设备
            model_type: 模型类型 ("flux1" 或 "flux2")
            use_ip_adapter: 是否使用 IP-Adapter（False 时仅使用 LoRA，可能效果更好）
        """
        super().__init__(model_path, device)
        self.model_type = model_type
        self.instantid_path = instantid_path
        self.controlnet_path = controlnet_path
        self.use_ip_adapter = use_ip_adapter
        self.loaded = False
        self.face_analyzer = None
        
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface 未安装，无法使用 InstantID。"
                "请安装: pip install insightface onnxruntime onnxruntime-gpu"
            )
    
    def load(self) -> None:
        """加载 Flux 模型和 InstantID 组件"""
        if self.loaded and self.pipe is not None:
            print(f"  ⏭️  Pipeline 已加载，跳过重复加载（loaded={self.loaded}, pipe={self.pipe is not None}）")
            return
        
        print(f"加载 Flux ({self.model_type}) + InstantID 模型...")
        print(f"  Flux 模型: {self.model_path}")
        
        # 加载 Flux 基础模型
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        # 加载 InstantID ControlNet（如果提供）
        if self.controlnet_path and Path(self.controlnet_path).exists():
            try:
                from diffusers import ControlNetModel
                print(f"  加载 InstantID ControlNet: {self.controlnet_path}")
                controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_path,
                    torch_dtype=torch.float16
                )
                # 注意：Flux 目前可能不支持 ControlNet，这里先记录
                print("  ⚠️  注意: Flux 模型可能不完全支持 ControlNet，InstantID 可能使用 IP-Adapter 方式")
            except Exception as e:
                print(f"  ⚠️  ControlNet 加载失败: {e}")
                print("  ℹ️  将使用 IP-Adapter 方式（不依赖 ControlNet）")
        
        # 加载 IP-Adapter（用于固定人脸）
        # 注意：InstantID 的 IP-Adapter 权重是为 SDXL 设计的，不兼容 Flux
        # Flux 需要使用 Flux 专用的 IP-Adapter 权重，例如：
        # - "XLabs-AI/flux-ip-adapter" (标准 IP-Adapter)
        # - IP-Adapter FaceID Plus for Flux (如果有)
        # 注意：如果 use_ip_adapter=False，将跳过 IP-Adapter 加载，仅使用 LoRA（可能效果更好）
        self.ip_adapter_loaded = False
        self.ip_adapter_type = None  # 'flux_standard' 或 'instantid' (不兼容)
        
        if not self.use_ip_adapter:
            print("  ℹ️  IP-Adapter 已禁用，将仅使用 LoRA 生成（可能效果更好）")
            # 注意：即使不使用 IP-Adapter，也要设置 loaded=True，否则会导致重复加载
            # 不要在这里 return，继续执行到最后的 self.loaded = True
            # 跳过 IP-Adapter 加载
            print("  ⏭️  跳过 IP-Adapter 加载（已禁用）")
        else:
            # 方法1: 尝试从本地路径加载 Flux IP-Adapter
            if self.instantid_path and Path(self.instantid_path).exists():
                try:
                    ip_adapter_path = Path(self.instantid_path)
                    print(f"  检查 IP-Adapter 路径: {ip_adapter_path}")
                    
                    # 查找 IP-Adapter 权重文件（优先查找 safetensors）
                    ip_adapter_files = list(ip_adapter_path.glob("*.safetensors")) + \
                                     list(ip_adapter_path.glob("*.bin"))
                    
                    if ip_adapter_files:
                        ip_adapter_file = ip_adapter_files[0]
                        print(f"  找到 IP-Adapter 权重文件: {ip_adapter_file.name}")
                        try:
                            # Flux 的 load_ip_adapter 需要 weight_name 参数
                            if hasattr(self.pipe, 'load_ip_adapter'):
                                # 使用目录路径和权重文件名
                                weight_name = ip_adapter_file.name
                                model_path = str(ip_adapter_path)
                                
                                print(f"  加载 Flux IP-Adapter:")
                                print(f"    路径: {model_path}")
                                print(f"    权重文件: {weight_name}")
                                
                                # Flux IP-Adapter 需要 image_encoder
                                # 默认使用 "openai/clip-vit-large-patch14"
                                try:
                                    self.pipe.load_ip_adapter(
                                        pretrained_model_name_or_path_or_dict=model_path,
                                        weight_name=weight_name,
                                        subfolder="",  # 权重文件在根目录
                                        image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
                                    )
                                    self.ip_adapter_loaded = True
                                    self.ip_adapter_type = 'flux_standard'
                                    print(f"  ✅ Flux IP-Adapter 权重已加载成功！")
                                    print(f"    使用的 image_encoder: openai/clip-vit-large-patch14")
                                except Exception as e1:
                                    # 如果指定 image_encoder 失败，尝试不指定（使用默认）
                                    print(f"  ⚠️  指定 image_encoder 失败: {e1}")
                                    print(f"  ℹ️  尝试使用默认 image_encoder...")
                                    try:
                                        self.pipe.load_ip_adapter(
                                            pretrained_model_name_or_path_or_dict=model_path,
                                            weight_name=weight_name,
                                            subfolder=""
                                        )
                                        self.ip_adapter_loaded = True
                                        self.ip_adapter_type = 'flux_standard'
                                        print(f"  ✅ Flux IP-Adapter 权重已加载成功（使用默认 image_encoder）！")
                                    except Exception as e2:
                                        raise e2
                            else:
                                print(f"  ⚠️  Pipeline 不支持 IP-Adapter 加载方法")
                        except Exception as e:
                            print(f"  ⚠️  IP-Adapter 加载失败: {e}")
                            print(f"  💡 提示: 请检查权重文件格式是否正确")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"  ⚠️  未找到 IP-Adapter 权重文件（.safetensors 或 .bin）")
                        print(f"     检查路径: {ip_adapter_path}")
                except Exception as e:
                    print(f"  ⚠️  IP-Adapter 路径检查失败: {e}")
        
        # 方法2: 如果没有本地权重，提示用户下载 Flux IP-Adapter（仅在启用时）
        if not self.use_ip_adapter:
            # IP-Adapter 已禁用，不需要提示
            pass
        elif not self.ip_adapter_loaded:
            print(f"  ℹ️  未加载 IP-Adapter 权重")
            print(f"  💡 要固定科学主持人形象，建议:")
            print(f"     1. 下载 Flux 专用的 IP-Adapter 权重")
            print(f"     2. 推荐使用: 'XLabs-AI/flux-ip-adapter' (标准 IP-Adapter)")
            print(f"     3. 或使用 IP-Adapter FaceID Plus for Flux (如果有)")
            print(f"     4. 将权重文件放到: {self.instantid_path or 'models/instantid/ip-adapter/'}")
            print(f"  ⚠️  注意: InstantID 的 IP-Adapter 权重不兼容 Flux，需要使用 Flux 专用版本")
        
        # 初始化 face analyzer（用于提取人脸特征）
        self._init_face_analyzer()
        
        # 确保 loaded 状态在最后设置（在所有初始化完成后）
        self.loaded = True
        print(f"✅ Flux ({self.model_type}) + InstantID 模型加载完成")
        print(f"  🔍 验证: loaded={self.loaded}, pipe={self.pipe is not None}")
    
    def _init_face_analyzer(self) -> None:
        """初始化 InsightFace 人脸分析器（独立方法，可在需要时重新调用）"""
        try:
            print("  初始化 InsightFace 人脸分析器...")
            
            # 确保修复代码已生效
            import insightface.model_zoo.model_zoo as model_zoo_module
            if hasattr(model_zoo_module, 'ModelRouter'):
                router_method = model_zoo_module.ModelRouter.get_model
                if hasattr(router_method, '__name__') and 'patched' in router_method.__name__.lower():
                    print("  ✅ 确认 SCRFD 修复代码已生效")
                else:
                    print("  ⚠️  警告: SCRFD 修复代码可能未生效，尝试重新应用...")
            
            # 修补 FaceAnalysis 类，允许手动添加 detection 模型
            original_face_analysis_init = insightface.app.FaceAnalysis.__init__
            
            def _patched_face_analysis_init(self, name='antelopev2', root='~/.insightface', allowed_modules=None, **kwargs):
                """修补后的 FaceAnalysis.__init__，允许缺少 detection 模型"""
                import onnxruntime
                import glob
                import os.path as osp
                # 使用绝对导入而不是相对导入
                from insightface.model_zoo import model_zoo
                from insightface.utils import ensure_available
                from insightface.app.common import Face
                
                onnxruntime.set_default_logger_severity(3)
                self.models = {}
                self.model_dir = ensure_available('models', name, root=root)
                print(f'  [DEBUG] model_dir: {self.model_dir}')
                onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
                onnx_files = sorted(onnx_files)
                print(f'  [DEBUG] 找到的 ONNX 文件数: {len(onnx_files)}')
                
                for onnx_file in onnx_files:
                    print(f'  [DEBUG] 处理 ONNX 文件: {onnx_file}')
                    model = model_zoo.get_model(onnx_file, **kwargs)
                    if model is None:
                        print('model not recognized:', onnx_file)
                    elif allowed_modules is not None and model.taskname not in allowed_modules:
                        print('model ignore:', onnx_file, model.taskname)
                        del model
                    elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                        print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                        self.models[model.taskname] = model
                    else:
                        print('duplicated model task type, ignore:', onnx_file, model.taskname)
                        del model
                
                print(f'  [DEBUG] 已识别的模型: {list(self.models.keys())}')
                
                # 如果没有 detection 模型，尝试手动添加
                if 'detection' not in self.models:
                    print('  ⚠️  未检测到 detection 模型，尝试手动添加...')
                    print(f'  [DEBUG] 检查的 ONNX 文件: {onnx_files}')
                    
                    # 如果 onnx_files 为空，尝试从已知路径查找
                    if not onnx_files:
                        print(f'  [DEBUG] ONNX 文件列表为空，尝试从 model_dir 查找: {self.model_dir}')
                        # 尝试多个可能的路径
                        possible_paths = [
                            self.model_dir,
                            osp.join(osp.expanduser(root), 'models', name, name) if root else None,
                            osp.join(osp.expanduser('~'), '.insightface', 'models', name, name),
                        ]
                        for path in possible_paths:
                            if path and osp.exists(path):
                                print(f'  [DEBUG] 尝试路径: {path}')
                                found_files = glob.glob(osp.join(path, '*.onnx'))
                                if found_files:
                                    onnx_files = sorted(found_files)
                                    print(f'  [DEBUG] 在 {path} 找到 {len(onnx_files)} 个文件')
                                    break
                    
                    scrfd_files = [f for f in onnx_files if 'scrfd' in f.lower()]
                    print(f'  [DEBUG] 找到的 SCRFD 文件: {scrfd_files}')
                    if scrfd_files:
                        scrfd_file = scrfd_files[0]
                        print(f'  [DEBUG] 尝试加载 SCRFD 文件: {scrfd_file}')
                        try:
                            from insightface.model_zoo.scrfd import SCRFD
                            import onnxruntime as ort
                            providers = kwargs.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                            # 修复 provider_options 问题 - 如果为空字典，则不传递
                            provider_options = kwargs.get('provider_options', None)
                            
                            print(f'  [DEBUG] 创建 ONNX Runtime session，providers: {providers}, provider_options: {provider_options}')
                            # 如果 provider_options 是空字典或 None，则不传递
                            if provider_options and provider_options != {}:
                                session = ort.InferenceSession(scrfd_file, providers=providers, provider_options=provider_options)
                            else:
                                session = ort.InferenceSession(scrfd_file, providers=providers)
                            print(f'  [DEBUG] 创建 SCRFD 模型实例...')
                            scrfd_model = SCRFD(model_file=scrfd_file, session=session)
                            self.models['detection'] = scrfd_model
                            input_shape_str = str(scrfd_model.input_shape) if hasattr(scrfd_model, 'input_shape') else 'None'
                            input_mean = getattr(scrfd_model, 'input_mean', 0.0)
                            input_std = getattr(scrfd_model, 'input_std', 1.0)
                            print(f'find model: {scrfd_file} detection {input_shape_str} {input_mean} {input_std}')
                            print(f'  ✅ 手动添加 detection 模型成功！')
                            
                            # 加载其他必要的模型（recognition, keypoint 等）
                            print(f'  [DEBUG] 尝试加载其他模型...')
                            for other_file in onnx_files:
                                if 'scrfd' not in other_file.lower():
                                    try:
                                        other_model = model_zoo.get_model(other_file, **kwargs)
                                        if other_model is not None and other_model.taskname not in self.models:
                                            self.models[other_model.taskname] = other_model
                                            print(f'find model: {other_file} {other_model.taskname} {other_model.input_shape} {other_model.input_mean} {other_model.input_std}')
                                    except Exception as e:
                                        print(f'  [DEBUG] 加载 {osp.basename(other_file)} 失败: {e}')
                        except Exception as e:
                            print(f'  ⚠️  手动添加 detection 模型失败: {e}')
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f'  ⚠️  未找到 SCRFD 文件！')
                        print(f'  [DEBUG] 所有 ONNX 文件: {[osp.basename(f) for f in onnx_files]}')
                        # 如果仍然找不到，尝试直接使用已知路径
                        known_scrfd_path = '/root/.insightface/models/antelopev2/antelopev2/scrfd_10g_bnkps.onnx'
                        if osp.exists(known_scrfd_path):
                            print(f'  [DEBUG] 尝试使用已知路径: {known_scrfd_path}')
                            try:
                                from insightface.model_zoo.scrfd import SCRFD
                                import onnxruntime as ort
                                providers = kwargs.get('providers', ['CUDAExecutionProvider', 'CPUExecutionProvider'])
                                provider_options = kwargs.get('provider_options', None)
                                if provider_options and provider_options != {}:
                                    session = ort.InferenceSession(known_scrfd_path, providers=providers, provider_options=provider_options)
                                else:
                                    session = ort.InferenceSession(known_scrfd_path, providers=providers)
                                scrfd_model = SCRFD(model_file=known_scrfd_path, session=session)
                                self.models['detection'] = scrfd_model
                                input_shape_str = str(scrfd_model.input_shape) if hasattr(scrfd_model, 'input_shape') else 'None'
                                input_mean = getattr(scrfd_model, 'input_mean', 0.0)
                                input_std = getattr(scrfd_model, 'input_std', 1.0)
                                print(f'find model: {known_scrfd_path} detection {input_shape_str} {input_mean} {input_std}')
                                print(f'  ✅ 使用已知路径手动添加 detection 模型成功！')
                                
                                # 尝试加载其他模型
                                known_model_dir = '/root/.insightface/models/antelopev2/antelopev2'
                                if osp.exists(known_model_dir):
                                    other_files = glob.glob(osp.join(known_model_dir, '*.onnx'))
                                    for other_file in other_files:
                                        if 'scrfd' not in other_file.lower():
                                            try:
                                                other_model = model_zoo.get_model(other_file, **kwargs)
                                                if other_model is not None and other_model.taskname not in self.models:
                                                    self.models[other_model.taskname] = other_model
                                                    print(f'find model: {other_file} {other_model.taskname} {other_model.input_shape} {other_model.input_mean} {other_model.input_std}')
                                            except Exception as e:
                                                print(f'  [DEBUG] 加载 {osp.basename(other_file)} 失败: {e}')
                            except Exception as e:
                                print(f'  ⚠️  使用已知路径也失败: {e}')
                                import traceback
                                traceback.print_exc()
                
                # 只有在仍然没有 detection 模型时才抛出异常
                if 'detection' not in self.models:
                    raise AssertionError("无法找到或加载 detection 模型")
                
                self.det_model = self.models['detection']
            
            # 应用修补
            insightface.app.FaceAnalysis.__init__ = _patched_face_analysis_init
            print("  ✅ 已修补 FaceAnalysis.__init__，允许手动添加 detection 模型")
            
            instantid_models_dir = Path(__file__).parent.parent / "models" / "instantid"
            
            # 检查本地模型文件是否存在
            # InsightFace 期望的目录结构: root/models/antelopev2/antelopev2/*.onnx
            # 当前模型位置: models/instantid/models/antelopev2/antelopev2/*.onnx
            # 所以 root 应该指向 models/instantid，这样 InsightFace 会在 models/instantid/models/antelopev2 下查找
            antelopev2_model_dir = instantid_models_dir / "models" / "antelopev2" / "antelopev2"
            antelopev2_root = instantid_models_dir  # root 应该指向 instantid 目录，不是 models 目录
            
            # 调试信息：打印路径配置
            print(f"  📁 模型目录: {antelopev2_model_dir}")
            print(f"  📁 Root 路径: {antelopev2_root}")
            print(f"  📁 InsightFace 期望查找: {antelopev2_root}/models/antelopev2/antelopev2/*.onnx")
            
            # 检查关键模型文件是否存在
            # InsightFace 需要 detection, recognition, keypoint 三个模型
            required_files = ['scrfd_10g_bnkps.onnx', '1k3d68.onnx', 'glintr100.onnx']
            has_all_files = antelopev2_model_dir.exists() and all(
                (antelopev2_model_dir / f).exists() for f in required_files
            )
            
            # 如果模型在 antelopev2/antelopev2/ 下，但 InsightFace 期望在 antelopev2/ 下
            # 检查是否需要调整路径
            antelopev2_parent = instantid_models_dir / "models" / "antelopev2"
            if not has_all_files and antelopev2_parent.exists():
                # 检查模型是否在父目录
                has_all_files = all(
                    (antelopev2_parent / f).exists() for f in required_files
                )
                if has_all_files:
                    antelopev2_model_dir = antelopev2_parent
                    print(f"  ℹ️  模型文件在父目录: {antelopev2_model_dir}")
            
            if has_all_files:
                print(f"  ✅ 检测到本地模型文件: {antelopev2_model_dir}")
                
                # 方法0: 先尝试创建符号链接到默认位置（最可靠的方法）
                import os
                default_model_dir = Path.home() / ".insightface" / "models" / "antelopev2" / "antelopev2"
                default_model_dir.parent.mkdir(parents=True, exist_ok=True)
                
                # 如果默认位置不存在模型，创建符号链接
                if not default_model_dir.exists() or len(list(default_model_dir.glob("*.onnx"))) == 0:
                    if default_model_dir.exists() and not default_model_dir.is_symlink():
                        # 如果是目录，先删除
                        import shutil
                        shutil.rmtree(default_model_dir)
                    elif default_model_dir.is_symlink():
                        default_model_dir.unlink()  # 删除旧的符号链接
                    
                    # 创建符号链接
                    try:
                        os.symlink(str(antelopev2_model_dir.absolute()), str(default_model_dir.absolute()))
                        print(f"  ✅ 已创建符号链接: {default_model_dir} -> {antelopev2_model_dir}")
                    except OSError as e:
                        if "File exists" not in str(e):
                            print(f"  ⚠️  创建符号链接失败: {e}")
                
                try:
                    # 方法1: 使用默认路径（现在应该有符号链接了）
                    # 添加调试信息
                    print(f"  🔍 尝试方法1: 使用默认路径")
                    print(f"     检查符号链接: {default_model_dir}")
                    if default_model_dir.exists():
                        if default_model_dir.is_symlink():
                            print(f"     ✅ 是符号链接，指向: {default_model_dir.readlink()}")
                        onnx_files = list(default_model_dir.glob("*.onnx"))
                        print(f"     ONNX 文件数: {len(onnx_files)}")
                        for f in onnx_files:
                            print(f"       - {f.name}")
                    
                    # 捕获 InsightFace 的输出（包括 stdout 和 stderr）
                    import io
                    import contextlib
                    import sys
                    
                    # 使用上下文管理器捕获输出，但同时也打印到控制台
                    import sys
                    from io import StringIO
                    
                    class TeeOutput:
                        """同时输出到多个目标"""
                        def __init__(self, *targets):
                            self.targets = targets
                        def write(self, obj):
                            for t in self.targets:
                                t.write(obj)
                                t.flush()
                        def flush(self):
                            for t in self.targets:
                                t.flush()
                    
                    # 保存原始输出
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    
                    # 创建捕获缓冲区
                    stdout_capture = StringIO()
                    stderr_capture = StringIO()
                    
                    # 创建同时输出到控制台和缓冲区的对象
                    tee_stdout = TeeOutput(old_stdout, stdout_capture)
                    tee_stderr = TeeOutput(old_stderr, stderr_capture)
                    
                    # 在初始化前测试修复代码是否工作
                    print("     测试修复代码是否工作...")
                    test_scrfd_file = str(default_model_dir / "scrfd_10g_bnkps.onnx")
                    if Path(test_scrfd_file).exists():
                        try:
                            import insightface.model_zoo.model_zoo as test_model_zoo
                            test_model = test_model_zoo.get_model(
                                test_scrfd_file,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                            )
                            if test_model is not None:
                                print(f"     ✅ 测试成功！SCRFD 模型可以被识别: {type(test_model).__name__}")
                            else:
                                print(f"     ⚠️  测试失败：SCRFD 模型返回 None")
                        except Exception as e:
                            print(f"     ⚠️  测试时出错: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print("     正在初始化 FaceAnalysis...")
                    try:
                        sys.stdout = tee_stdout
                        sys.stderr = tee_stderr
                        
                        self.face_analyzer = insightface.app.FaceAnalysis(
                            name='antelopev2',
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    # 获取捕获的输出
                    captured_stdout = stdout_capture.getvalue()
                    captured_stderr = stderr_capture.getvalue()
                    
                    # 打印捕获的输出（用于调试）
                    if captured_stdout:
                        print(f"  📋 捕获的 stdout 输出:")
                        for line in captured_stdout.strip().split('\n'):
                            if line.strip():
                                print(f"     {line}")
                    
                    if captured_stderr:
                        print(f"  📋 捕获的 stderr 输出:")
                        for line in captured_stderr.strip().split('\n'):
                            if line.strip():
                                print(f"     {line}")
                    
                    # 检查是否成功初始化
                    if self.face_analyzer is not None and hasattr(self.face_analyzer, 'models'):
                        detected_models = list(self.face_analyzer.models.keys())
                        print(f"  ✅ InsightFace 初始化完成（使用符号链接到默认路径）")
                        print(f"     已识别的模型: {', '.join(detected_models)}")
                        
                        # 如果没有 detection 模型，手动添加
                        if 'detection' not in detected_models:
                            print(f"  ⚠️  未检测到 detection 模型，尝试手动添加...")
                            scrfd_file = default_model_dir / "scrfd_10g_bnkps.onnx"
                            if scrfd_file.exists():
                                try:
                                    from insightface.model_zoo.scrfd import SCRFD
                                    import onnxruntime as ort
                                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                                    session = ort.InferenceSession(str(scrfd_file), providers=providers)
                                    scrfd_model = SCRFD(model_file=str(scrfd_file), session=session)
                                    self.face_analyzer.models['detection'] = scrfd_model
                                    self.face_analyzer.det_model = scrfd_model
                                    print(f"  ✅ 手动添加 detection 模型成功！")
                                except Exception as e:
                                    print(f"  ⚠️  手动添加 detection 模型失败: {e}")
                                    raise Exception("FaceAnalysis 初始化失败：无法添加 detection 模型")
                            else:
                                raise Exception("FaceAnalysis 初始化失败：找不到 SCRFD 模型文件")
                    else:
                        raise Exception("FaceAnalysis 初始化失败：models 字典为空")
                except Exception as e1:
                    print(f"  ⚠️  方法1失败: {str(e1)[:200]}")
                    # 打印更详细的错误信息
                    if 'detection' in str(e1).lower() or 'assert' in str(e1).lower():
                        print(f"  💡 检测到 detection 模型识别问题")
                        print(f"     可能原因: InsightFace 无法识别 scrfd_10g_bnkps.onnx 为 detection 模型")
                        print(f"     建议: 检查模型文件是否完整，或 InsightFace 版本是否匹配")
                    # 方法2: 尝试使用指定 root
                    try:
                        print("  ℹ️  尝试方法2: 使用指定 root 路径...")
                        self.face_analyzer = insightface.app.FaceAnalysis(
                            name='antelopev2',
                            root=str(antelopev2_root),
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                        print("  ✅ InsightFace 初始化完成（使用指定 root）")
                    except Exception as e2:
                        print(f"  ⚠️  方法2也失败: {str(e2)[:200]}")
                        if 'detection' in str(e2).lower() or 'assert' in str(e2).lower():
                            print(f"  💡 检测到 detection 模型识别问题")
                            print(f"     可能原因: InsightFace 无法识别 scrfd_10g_bnkps.onnx 为 detection 模型")
                            print(f"     检查: 模型文件是否存在且完整")
                        # 方法3: 尝试复制模型文件到默认位置（如果符号链接失败）
                        try:
                            print("  ℹ️  尝试方法3: 复制模型文件到默认位置...")
                            import shutil
                            if default_model_dir.exists() and default_model_dir.is_symlink():
                                default_model_dir.unlink()
                            
                            if not default_model_dir.exists():
                                default_model_dir.mkdir(parents=True, exist_ok=True)
                                # 复制所有模型文件
                                for onnx_file in antelopev2_model_dir.glob("*.onnx"):
                                    shutil.copy2(onnx_file, default_model_dir / onnx_file.name)
                                print(f"  ✅ 已复制模型文件到: {default_model_dir}")
                            
                            # 再次尝试使用默认路径
                            self.face_analyzer = insightface.app.FaceAnalysis(
                                name='antelopev2',
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                            )
                            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                            if self.face_analyzer is not None and hasattr(self.face_analyzer, 'models'):
                                detected_models = list(self.face_analyzer.models.keys())
                                print("  ✅ InsightFace 初始化完成（使用复制的模型文件）")
                                print(f"     已识别的模型: {', '.join(detected_models)}")
                            else:
                                raise Exception("FaceAnalysis 初始化失败：models 字典为空")
                        except Exception as e3:
                            print(f"  ⚠️  所有方法都失败: {str(e3)[:200]}")
                            import traceback
                            traceback.print_exc()
                            # 不抛出异常，允许继续运行（不使用 InstantID 人脸特征）
                            print("  ℹ️  InsightFace 初始化失败，将不使用 InstantID 人脸特征提取")
                            print("  💡 提示: 这可能是因为模型文件无法被 InsightFace 正确识别")
                            print("     建议: 检查 InsightFace 版本或重新下载 antelopev2 模型")
                            self.face_analyzer = None
                            # 不抛出异常，让流程继续
            else:
                print(f"  ℹ️  本地模型文件不完整，使用默认路径（会自动下载）")
                print(f"    期望路径: {antelopev2_model_dir}")
                try:
                    self.face_analyzer = insightface.app.FaceAnalysis(
                        name='antelopev2',
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                    print("  ✅ InsightFace 初始化完成（使用默认路径）")
                except Exception as e:
                    print(f"  ⚠️  使用默认路径也失败: {str(e)[:200]}")
                    print("  ℹ️  将不使用 InstantID 人脸特征提取")
                    self.face_analyzer = None
        except Exception as e:
            print(f"  ⚠️  InsightFace 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            print("  ℹ️  将尝试使用备用方法（不使用 InstantID 人脸特征）")
            self.face_analyzer = None
    
    def _extract_face_features(self, face_image: Image.Image) -> Optional[dict]:
        """
        提取人脸特征
        
        Args:
            face_image: 人脸参考图像
            
        Returns:
            人脸特征字典（包含 face_embed, face_keypoints 等）
        """
        if self.face_analyzer is None:
            return None
        
        try:
            # 转换为 numpy array
            face_array = np.array(face_image)
            
            # 提取人脸特征
            faces = self.face_analyzer.get(face_array)
            
            if len(faces) == 0:
                print("  ⚠️  未检测到人脸")
                return None
            
            # 使用第一个人脸
            face = faces[0]
            
            # 检查是否有 embedding（需要 recognition 模型）
            if not hasattr(face, 'embedding') or face.embedding is None:
                print("  ⚠️  人脸 embedding 为空，可能需要加载 recognition 模型")
                # 尝试手动加载 recognition 模型
                if hasattr(self.face_analyzer, 'models') and 'recognition' not in self.face_analyzer.models:
                    print("  ℹ️  尝试加载 recognition 模型...")
                    try:
                        import os.path as osp
                        import glob
                        import onnxruntime as ort
                        from insightface.model_zoo.model_zoo import get_model
                        
                        model_dir = self.face_analyzer.model_dir
                        # 查找 recognition 模型（通常是 glintr100.onnx）
                        recognition_files = glob.glob(osp.join(model_dir, '*', 'glintr100.onnx'))
                        if not recognition_files:
                            recognition_files = glob.glob(osp.join(osp.dirname(model_dir), '*', 'glintr100.onnx'))
                        
                        if recognition_files:
                            rec_file = recognition_files[0]
                            print(f"  [DEBUG] 找到 recognition 模型: {rec_file}")
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                            rec_model = get_model(rec_file, providers=providers)
                            if rec_model:
                                self.face_analyzer.models['recognition'] = rec_model
                                print(f"  ✅ 已加载 recognition 模型")
                                # 重新提取特征
                                faces = self.face_analyzer.get(face_array)
                                if len(faces) > 0:
                                    face = faces[0]
                    except Exception as e:
                        print(f"  ⚠️  加载 recognition 模型失败: {e}")
            
            # 提取特征
            face_features = {
                'face_embed': face.embedding if hasattr(face, 'embedding') and face.embedding is not None else None,
                'face_keypoints': face.kps if hasattr(face, 'kps') else None,
                'face_bbox': face.bbox if hasattr(face, 'bbox') else None,
            }
            
            if face_features['face_embed'] is not None:
                print(f"  ✅ 已提取人脸特征 (embedding shape: {face_features['face_embed'].shape})")
            else:
                print(f"  ⚠️  人脸 embedding 为空，无法使用 InstantID")
                return None
            
            return face_features
            
        except Exception as e:
            print(f"  ⚠️  人脸特征提取失败: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 18,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        face_image: Optional[Image.Image] = None,
        face_strength: float = 0.8,
        lora_path: Optional[str] = None,
        lora_alpha: float = 1.0,
        **kwargs
    ) -> Image.Image:
        """
        生成图像（带 InstantID 人脸固定）
        
        Args:
            prompt: 提示词
            negative_prompt: 负面提示词
            width: 图像宽度
            height: 图像高度
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            face_image: 人脸参考图像（PIL Image）
            face_strength: 人脸强度（0.0-1.0，默认 0.8）
            lora_path: LoRA 权重路径（可选）
            lora_alpha: LoRA 权重（0.0-1.0，默认 1.0）
            **kwargs: 其他参数
            
        Returns:
            PIL Image
        """
        if not self.loaded:
            self.load()
        
        # 提取人脸特征（如果提供人脸图像）
        face_features = None
        face_image_for_ip_adapter = None  # 保存原始图像用于 IP-Adapter
        if face_image is not None:
            print("  🔍 提取人脸特征...")
            # 检查 face_analyzer 是否已初始化
            if self.face_analyzer is None:
                print("  ⚠️  FaceAnalyzer 未初始化，尝试重新初始化...")
                try:
                    # 尝试重新初始化（如果之前失败）
                    self._init_face_analyzer()
                except Exception as e:
                    print(f"  ⚠️  重新初始化 FaceAnalyzer 失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            if self.face_analyzer is not None:
                face_features = self._extract_face_features(face_image)
                if face_features is None:
                    print("  ⚠️  人脸特征提取失败，将不使用 InstantID（仅使用 LoRA）")
                else:
                    print(f"  ✅ 人脸特征已提取，强度: {face_strength}")
                    # 保存原始图像用于 Flux IP-Adapter（Flux IP-Adapter 需要图像输入，而不是 embedding）
                    face_image_for_ip_adapter = face_image.copy()
            else:
                print("  ⚠️  FaceAnalyzer 不可用，将不使用 InstantID（仅使用 LoRA）")
        
        # 加载 LoRA（如果提供）
        if lora_path:
            from pathlib import Path
            from safetensors import safe_open
            import tempfile
            import os
            
            lora_path_obj = Path(lora_path)
            if lora_path_obj.exists():
                try:
                    print(f"  🔧 加载 LoRA: {lora_path_obj.name}")
                    
                    # 读取并转换 LoRA 权重（与 FluxPipeline 相同的逻辑）
                    lora_state_dict = {}
                    with safe_open(str(lora_path_obj), framework="pt") as f:
                        for key in f.keys():
                            new_key = key
                            if key.startswith("base_model.model."):
                                new_key = key.replace("base_model.model.", "")
                            if "single_transformer_blocks" in new_key:
                                new_key = new_key.replace("single_transformer_blocks", "transformer_blocks")
                            if ".default." in new_key:
                                new_key = new_key.replace(".default.", ".")
                            if "transformer_blocks" in new_key and not new_key.startswith("transformer."):
                                new_key = f"transformer.{new_key}"
                            lora_state_dict[new_key] = f.get_tensor(key)
                    
                    # 保存到临时文件并加载
                    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
                        from safetensors.torch import save_file
                        save_file(lora_state_dict, tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    try:
                        import time
                        # 检查适配器是否已存在，如果存在则先卸载或使用新名称
                        adapter_name = "character_lora"
                        if hasattr(self.pipe, 'get_active_adapters'):
                            try:
                                active_adapters = self.pipe.get_active_adapters()
                                if "character_lora" in active_adapters:
                                    print(f"  ℹ 检测到已存在的适配器 character_lora，先卸载...")
                                    self.pipe.set_adapters([])  # 卸载所有适配器
                                    # 或者使用不同的适配器名称
                                    adapter_name = f"character_lora_{int(time.time())}"
                                    print(f"  ℹ 使用新的适配器名称: {adapter_name}")
                            except:
                                # 如果获取失败，尝试直接卸载
                                try:
                                    self.pipe.set_adapters([])
                                except:
                                    pass
                        
                        self.pipe.load_lora_weights(tmp_path, adapter_name=adapter_name, weight_name=None)
                        self.pipe.set_adapters([adapter_name], adapter_weights=[lora_alpha])
                        print(f"  ✅ 已加载 LoRA (alpha={lora_alpha}, adapter={adapter_name})")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                except Exception as e:
                    print(f"  ⚠ LoRA 加载失败: {e}")
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 生成图像
        # 如果 use_ip_adapter=False，跳过 IP-Adapter，仅使用 LoRA（可能效果更好）
        # 如果提供了人脸图像且 IP-Adapter 已加载，使用 IP-Adapter
        if face_image_for_ip_adapter is not None and self.ip_adapter_loaded and self.use_ip_adapter:
            print("  🎨 使用 Flux IP-Adapter 生成图像（固定人脸特征）...")
            try:
                # 预处理图像：确保图像尺寸合适（Flux IP-Adapter 推荐 1024x1024）
                # 保持宽高比，但确保最小边至少 1024
                original_size = face_image_for_ip_adapter.size
                w, h = original_size
                
                # 如果图像太小，需要放大
                min_size = 1024
                if min(w, h) < min_size:
                    # 计算缩放比例
                    scale = min_size / min(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    # 确保是 64 的倍数（Flux 的要求）
                    new_w = (new_w // 64) * 64
                    new_h = (new_h // 64) * 64
                    face_image_for_ip_adapter = face_image_for_ip_adapter.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS
                    )
                    print(f"  📐 图像已调整: {original_size} -> {face_image_for_ip_adapter.size}")
                else:
                    # 如果图像足够大，也确保是 64 的倍数
                    new_w = (w // 64) * 64
                    new_h = (h // 64) * 64
                    if new_w != w or new_h != h:
                        face_image_for_ip_adapter = face_image_for_ip_adapter.resize(
                            (new_w, new_h), Image.Resampling.LANCZOS
                        )
                        print(f"  📐 图像已调整到 64 的倍数: {original_size} -> {face_image_for_ip_adapter.size}")
                
                # Flux IP-Adapter 使用图像输入，而不是 embedding
                # 图像会被自动编码为 image embeds
                kwargs['ip_adapter_image'] = face_image_for_ip_adapter
                print(f"  ✅ 已设置 IP-Adapter 图像输入 (尺寸: {face_image_for_ip_adapter.size}, 模式: {face_image_for_ip_adapter.mode})")
                
                # 设置 IP-Adapter scale（对应 face_strength）
                # Flux IP-Adapter 的 scale 通常在 0.5-1.5 之间
                if hasattr(self.pipe, 'set_ip_adapter_scale'):
                    # 将 face_strength (0.0-1.0) 映射到 IP-Adapter scale (0.5-1.5)
                    ip_scale = 0.5 + face_strength * 1.0  # 0.5 到 1.5
                    self.pipe.set_ip_adapter_scale(ip_scale)
                    print(f"  ✅ 已设置 IP-Adapter scale: {ip_scale} (face_strength={face_strength})")
                elif 'ip_adapter_scale' in kwargs or hasattr(self.pipe, 'ip_adapter_scale'):
                    ip_scale = 0.5 + face_strength * 1.0
                    kwargs['ip_adapter_scale'] = ip_scale
                    print(f"  ✅ 已设置 IP-Adapter scale: {ip_scale} (face_strength={face_strength})")
                else:
                    print(f"  ⚠️  无法设置 IP-Adapter scale，使用默认值")
                    
            except Exception as e:
                print(f"  ⚠️  设置 IP-Adapter 参数失败: {e}")
                print(f"  ℹ️  将尝试不使用 IP-Adapter，仅使用提示词生成")
                import traceback
                traceback.print_exc()
        elif face_image_for_ip_adapter is not None and not self.ip_adapter_loaded:
            if not self.use_ip_adapter:
                print("  ℹ️  IP-Adapter 已禁用，使用纯 Flux + LoRA 模式生成（推荐）")
                print("  💡 提示: 如果已加载 LoRA，将使用 LoRA 固定人脸特征")
            else:
                print("  ⚠️  提供了人脸图像，但 IP-Adapter 未加载")
                print("  ℹ️  将仅使用提示词生成，不会固定人脸特征")
        
        # 打印最终参数（用于调试）
        print(f"  📋 生成参数:")
        print(f"     图像尺寸: {width}x{height}")
        print(f"     推理步数: {num_inference_steps}")
        print(f"     引导强度: {guidance_scale}")
        if self.use_ip_adapter and face_image_for_ip_adapter is not None and self.ip_adapter_loaded:
            if 'ip_adapter_image' in kwargs:
                img = kwargs['ip_adapter_image']
                print(f"     IP-Adapter 图像: {img.size if hasattr(img, 'size') else '已设置'}")
        else:
            print(f"     IP-Adapter: 已禁用（使用纯 Flux + LoRA 模式）")
            # 确保移除所有 IP-Adapter 相关参数，避免 None 迭代错误
            # 注意：Flux pipeline 可能会检查这些参数，所以需要完全移除
            kwargs.pop('ip_adapter_image', None)
            kwargs.pop('ip_adapter_scale', None)
            kwargs.pop('ip_adapter_hidden_states', None)
            kwargs.pop('ip_adapter_image_embeds', None)
            # 同时移除可能的其他 IP-Adapter 相关参数
            for key in list(kwargs.keys()):
                if 'ip_adapter' in key.lower() or 'ipadapter' in key.lower():
                    kwargs.pop(key, None)
        
        # 防御性检查：确保 pipeline 已加载
        if self.pipe is None:
            raise RuntimeError("Pipeline 未加载，请先调用 load() 方法")
        
        # 如果 IP-Adapter 已加载但未使用，需要确保 pipeline 知道不使用它
        # 方法1: 尝试卸载 IP-Adapter（如果支持）
        if self.ip_adapter_loaded and not self.use_ip_adapter:
            if hasattr(self.pipe, 'unload_ip_adapter'):
                try:
                    self.pipe.unload_ip_adapter()
                    print(f"  ✅ 已卸载 IP-Adapter（不使用）")
                    self.ip_adapter_loaded = False  # 更新状态
                except Exception as e:
                    print(f"  ⚠️  卸载 IP-Adapter 失败: {e}")
                    # 如果卸载失败，尝试设置 scale=0
                    if hasattr(self.pipe, 'set_ip_adapter_scale'):
                        try:
                            self.pipe.set_ip_adapter_scale(0.0)
                            print(f"  ✅ 已禁用 IP-Adapter（scale=0.0）")
                        except:
                            pass
            elif hasattr(self.pipe, 'set_ip_adapter_scale'):
                # 方法2: 设置 scale=0 来禁用
                try:
                    self.pipe.set_ip_adapter_scale(0.0)
                    print(f"  ✅ 已禁用 IP-Adapter（scale=0.0）")
                except:
                    pass
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )
        
        # 检查生成结果
        if not hasattr(result, 'images') or not result.images:
            raise RuntimeError("生成失败：未返回图像")
        
        return result.images[0]

