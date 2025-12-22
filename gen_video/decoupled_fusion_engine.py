#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解耦融合引擎 - 身份与场景分离生成

核心思想：
1. 先生成完整场景 (无身份约束，环境100%表达)
2. 检测人物区域 (SAM2/YOLO)
3. 在人物区域注入身份 (PuLID/InstantID Inpainting)
4. 自然融合边缘

参考架构：
- 即梦 Dreamina 的 Flow Matching 技术
- Pika 的 Ingredients 多实体一致性
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image, ImageDraw, ImageFilter
import logging

logger = logging.getLogger(__name__)


class DecoupledFusionEngine:
    """
    解耦融合引擎
    
    解决 "人脸一致性 vs 环境丰富度" 矛盾的核心方案
    
    工作流程:
    1. Scene Generation: 使用 Flux 生成完整场景 (无身份约束)
    2. Region Detection: 使用 SAM2/YOLO 检测人物区域
    3. Identity Injection: 在检测区域注入人脸身份
    4. Edge Blending: 自然过渡，消除边界
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化解耦融合引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get("device", "cuda")
        
        # 模型路径
        self.model_base_path = config.get("model_dir", "/vepfs-dev/shawn/vid/fanren/gen_video/models")
        self.sam2_path = os.path.join(self.model_base_path, "sam2")
        
        # 模型状态
        self.sam2_predictor = None
        self.yolo_model = None
        self.sam2_loaded = False
        self.yolo_loaded = False
        
        logger.info("解耦融合引擎初始化完成")
    
    # ==========================================
    # 区域检测模块
    # ==========================================
    
    def load_sam2(self):
        """加载 SAM2 模型"""
        if self.sam2_loaded:
            return
        
        logger.info("加载 SAM2 模型...")
        
        try:
            # 尝试使用 sam2 包
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                # ⚡ 关键修复：build_sam2 使用 Hydra 加载配置，需要配置名称而不是文件路径
                # Hydra 会在 sam2 包的配置目录中查找配置（如 configs/sam2/sam2_hiera_l.yaml）
                # 配置名称格式：configs/sam2/sam2_hiera_l 或 sam2_hiera_l（Hydra 会自动查找）
                config_file = "configs/sam2/sam2_hiera_l"  # 使用 Hydra 配置路径
                logger.info(f"  ✓ 使用 SAM2 配置: {config_file} (Hydra 会自动从 sam2 包中加载)")
                
                # 查找权重文件
                checkpoint = None
                for name in ["sam2_hiera_large.pt", "model.safetensors", "sam2_hiera_l.pt"]:
                    path = os.path.join(self.sam2_path, name)
                    if os.path.exists(path):
                        checkpoint = path
                        logger.info(f"  ✓ 找到 SAM2 权重文件: {checkpoint}")
                        break
                
                if checkpoint is None:
                    logger.warning(f"  ⚠ SAM2 权重文件未找到: {self.sam2_path}")
                    logger.info("  ℹ SAM2 将尝试从 HuggingFace 下载权重（如果网络可用）")
                    # 使用模型名称，让 sam2 自动下载
                    checkpoint = "sam2_hiera_large"
                
                # 构建模型
                logger.info(f"  加载 SAM2 模型: config={config_file}, checkpoint={checkpoint}")
                sam2_model = build_sam2(config_file, checkpoint, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                
                self.sam2_loaded = True
                logger.info(f"  ✅ SAM2 加载完成")
                
            except ImportError as e:
                logger.warning(f"sam2 包未安装: {e}")
                logger.info("尝试使用 transformers 方式加载...")
                self._load_sam2_with_transformers()
                
        except Exception as e:
            logger.error(f"SAM2 加载失败: {e}")
            logger.info("将使用 YOLO 作为备用")
    
    def _load_sam2_with_transformers(self):
        """使用 transformers 加载 SAM2"""
        try:
            from transformers import Sam2Model, Sam2Processor
            
            logger.info("使用 transformers 加载 SAM2...")
            
            self.sam2_processor = Sam2Processor.from_pretrained(self.sam2_path)
            self.sam2_model = Sam2Model.from_pretrained(self.sam2_path).to(self.device)
            
            self.sam2_loaded = True
            logger.info("SAM2 (transformers) 加载完成")
            
        except Exception as e:
            logger.error(f"transformers 方式加载失败: {e}")
            raise
    
    def load_yolo(self):
        """加载 YOLO 模型 (人物检测)"""
        if self.yolo_loaded:
            return
        
        logger.info("加载 YOLO 模型...")
        
        try:
            from ultralytics import YOLO
            
            # ⚡ 关键修复：优先使用本地 YOLO 模型文件，避免重复下载
            yolo_model_name = "yolov8n.pt"
            yolo_paths = [
                yolo_model_name,  # 当前目录
                os.path.join("gen_video", yolo_model_name),  # gen_video 目录
                os.path.join(os.path.dirname(__file__), yolo_model_name),  # 脚本所在目录
            ]
            
            yolo_path = None
            for path in yolo_paths:
                if os.path.exists(path):
                    yolo_path = path
                    logger.info(f"  ✓ 找到本地 YOLO 模型: {yolo_path}")
                    break
            
            if yolo_path:
                # 使用本地文件（绝对路径，避免 YOLO 库再次下载）
                self.yolo_model = YOLO(os.path.abspath(yolo_path))
            else:
                # 如果本地文件不存在，使用默认方式（会下载）
                logger.warning(f"  ⚠ 本地 YOLO 模型不存在，将从网络下载: {yolo_model_name}")
                self.yolo_model = YOLO(yolo_model_name)
            
            self.yolo_loaded = True
            logger.info("YOLO 加载完成")
            
        except Exception as e:
            logger.error(f"YOLO 加载失败: {e}")
            raise
    
    def detect_person_region(
        self,
        image: Union[str, Image.Image, np.ndarray],
        method: str = "auto"
    ) -> Optional[np.ndarray]:
        """
        检测人物区域
        
        Args:
            image: 输入图像
            method: 检测方法 (auto, sam2, yolo)
            
        Returns:
            人物区域mask (H, W)，255=人物区域，0=背景
        """
        # 转换图像格式
        # 处理字符串路径
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 处理 Result 对象（来自 FluxWrapper）
        if hasattr(image, 'images') and isinstance(image.images, list) and len(image.images) > 0:
            image = image.images[0]
        
        # 处理 PIL Image
        if isinstance(image, Image.Image):
            image_pil = image
            image_np = np.array(image)
        # 处理 numpy 数组
        elif isinstance(image, np.ndarray):
            # numpy 数组，需要转换为 PIL Image
            if image.dtype != np.uint8:
                image_np = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            else:
                image_np = image
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_pil = Image.fromarray(image_np, mode='RGB')
            else:
                logger.error(f"不支持的图像格式: shape={image_np.shape}")
                return None
        else:
            logger.error(f"不支持的图像类型: {type(image)}")
            return None
        
        # 自动选择方法
        if method == "auto":
            # ⚡ 关键修复：优先使用 SAM2（更精确的 mask，避免方框感）
            # 如果 SAM2 不可用，回退到 YOLO
            try:
                self.load_sam2()
                if self.sam2_loaded and hasattr(self, 'sam2_predictor') and self.sam2_predictor is not None:
                    method = "sam2"  # SAM2 生成精确的 mask，避免方框感
                    logger.info("  ✓ 使用 SAM2 检测（精确 mask，避免方框感）")
                else:
                    method = "yolo"  # SAM2 不可用时使用 YOLO
                    logger.info("  ℹ SAM2 不可用，使用 YOLO 检测")
            except Exception as e:
                method = "yolo"  # 如果 SAM2 加载失败，使用 YOLO
                logger.warning(f"  ⚠ SAM2 加载失败: {e}，使用 YOLO 检测")
        
        if method == "sam2":
            return self._detect_with_sam2(image_np)
        elif method == "yolo":
            return self._detect_with_yolo(image_pil, image_np)
        else:
            raise ValueError(f"未知的检测方法: {method}")
    
    def _detect_with_yolo(self, pil_image: Image.Image, image_np: np.ndarray) -> Optional[np.ndarray]:
        """使用 YOLO 检测人物
        
        Args:
            pil_image: PIL Image 对象（用于 YOLO 检测）
            image_np: numpy 数组（用于创建 mask）
        """
        self.load_yolo()
        
        # 运行检测（YOLO 需要 PIL Image）
        try:
            results = self.yolo_model(pil_image, classes=[0], verbose=False)  # class 0 = person
        except Exception as e:
            logger.error(f"YOLO 检测失败: {e}")
            # 尝试使用 numpy 数组（如果 PIL Image 失败）
            try:
                # 将 numpy 数组保存为临时文件
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    pil_image.save(tmp_file.name)
                    results = self.yolo_model(tmp_file.name, classes=[0], verbose=False)
                    os.unlink(tmp_file.name)
            except Exception as e2:
                logger.error(f"YOLO 检测（临时文件方式）也失败: {e2}")
                return None
        
        if not results or len(results[0].boxes) == 0:
            logger.warning("YOLO 未检测到人物")
            return None
        
        # 获取所有人物框
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # 创建 mask（使用 numpy 数组的尺寸，确保匹配）
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # ⚡ 关键修复：使用圆角矩形而不是硬矩形，减少方框感
            # 创建圆角矩形 mask
            from PIL import ImageDraw
            mask_pil = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask_pil)
            # 计算圆角半径（约为框大小的 10%）
            corner_radius = int(min(x2 - x1, y2 - y1) * 0.1)
            # 绘制圆角矩形
            draw.rounded_rectangle(
                [(x1, y1), (x2, y2)],
                radius=corner_radius,
                fill=255
            )
            # 转换为 numpy 数组并合并
            mask_yolo = np.array(mask_pil)
            mask = np.maximum(mask, mask_yolo)
        
        # ⚡ 关键修复：使用更大的羽化半径，进一步减少方框感
        # 扩展 mask 边缘 (用于自然融合)
        mask = self._expand_mask(mask, padding=30)  # 从 20 增加到 30，更好的融合
        
        return mask
    
    def _detect_with_sam2(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 SAM2 检测人物"""
        # ⚡ 关键修复：优先使用原生 SAM2（sam2 包），如果不可用则回退到 YOLO
        try:
            self.load_sam2()
        except Exception as e:
            logger.warning(f"SAM2 加载失败: {e}，回退到 YOLO")
            return None
        
        # 检查是否有可用的 SAM2 predictor（原生版本）
        if hasattr(self, 'sam2_predictor') and self.sam2_predictor is not None:
            try:
                # 先用 YOLO 获取人物框作为 SAM2 的 prompt
                self.load_yolo()
                results = self.yolo_model(image, classes=[0], verbose=False)
                
                if not results or len(results[0].boxes) == 0:
                    logger.warning("未检测到人物框，SAM2 无法使用")
                    return None
                
                # 获取最大的人物框
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                main_box = boxes[np.argmax(areas)]
                
                # 使用 SAM2 生成精确 mask
                self.sam2_predictor.set_image(image)
                
                # 使用框作为 prompt
                masks, scores, _ = self.sam2_predictor.predict(
                    box=main_box,
                    multimask_output=True
                )
                
                # 选择得分最高的 mask
                best_mask = masks[np.argmax(scores)]
                
                return (best_mask * 255).astype(np.uint8)
            except Exception as e:
                logger.warning(f"SAM2 原生版本推理失败: {e}，回退到 YOLO")
                return None
        else:
            # ⚡ 关键修复：transformers 版本的 SAM2 有兼容性问题（sam2_video vs sam2），直接回退到 YOLO
            logger.warning("SAM2 原生版本不可用，transformers 版本有兼容性问题，回退到 YOLO")
            return None
    
    def _detect_with_sam2_transformers(
        self,
        image: np.ndarray,
        box: np.ndarray
    ) -> Optional[np.ndarray]:
        """使用 transformers SAM2 检测"""
        try:
            from PIL import Image as PILImage
            
            pil_image = PILImage.fromarray(image)
            
            # 准备输入
            inputs = self.sam2_processor(
                pil_image,
                input_boxes=[[box.tolist()]],
                return_tensors="pt"
            ).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.sam2_model(**inputs)
            
            # 获取 mask
            masks = self.sam2_processor.image_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )
            
            mask = masks[0][0][0].cpu().numpy()
            return (mask * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"SAM2 transformers 推理失败: {e}")
            return None
    
    def _expand_mask(
        self,
        mask: np.ndarray,
        padding: int = 20
    ) -> np.ndarray:
        """
        扩展 mask 边缘
        
        用于创建自然过渡区域
        """
        from scipy import ndimage
        
        # ⚡ 关键修复：使用高斯模糊 + 形态学操作，创建更平滑的边缘
        # 1. 先对 mask 进行轻微高斯模糊，软化边缘
        mask_float = mask.astype(float) / 255.0
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=padding * 0.3)
        
        # 2. 使用形态学膨胀扩展 mask（使用圆形核，更自然）
        # 创建圆形核而不是方形核，减少方框感
        kernel_size = padding * 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        center = kernel_size // 2
        kernel = (x - center) ** 2 + (y - center) ** 2 <= (padding) ** 2
        kernel = kernel.astype(np.uint8)
        
        # 3. 对模糊后的 mask 进行膨胀
        expanded = ndimage.binary_dilation(mask_blurred > 0.1, structure=kernel)
        
        # 4. 再次高斯模糊，创建平滑的渐变边缘
        expanded_float = expanded.astype(float)
        expanded_smooth = ndimage.gaussian_filter(expanded_float, sigma=padding * 0.2)
        
        return (expanded_smooth * 255).astype(np.uint8)
    
    # ==========================================
    # 解耦生成模块
    # ==========================================
    
    def generate_decoupled(
        self,
        prompt: str,
        face_reference: Union[str, Image.Image],
        width: int = 768,
        height: int = 1152,
        scene_generator: Any = None,
        identity_injector: Any = None,
        reference_strength: int = 60,
        **kwargs
    ) -> Image.Image:
        """
        解耦生成: 先场景后身份
        
        工作流程:
        1. 生成完整场景 (无身份约束)
        2. 检测人物区域
        3. 在人物区域注入身份
        4. 自然融合
        
        Args:
            prompt: 完整场景描述
            face_reference: 人脸参考图
            width: 输出宽度
            height: 输出高度
            scene_generator: 场景生成器 (Flux pipeline)
            identity_injector: 身份注入器 (PuLID/InstantID)
            reference_strength: 参考强度
            
        Returns:
            最终生成的图像
        """
        logger.info("开始解耦生成...")
        logger.info(f"  Prompt: {prompt[:100]}...")
        logger.info(f"  参考强度: {reference_strength}%")
        
        # 阶段1: 场景生成 (无身份约束)
        logger.info("阶段1: 场景生成 (无身份约束)...")
        
        scene_prompt = self._build_scene_prompt(prompt)
        logger.info(f"  场景 Prompt: {scene_prompt[:200]}...")
        
        if scene_generator is not None:
            # 使用提供的场景生成器
            # ⚡ 关键修复：确保传递 num_inference_steps 和 guidance_scale
            scene_kwargs = kwargs.copy()
            # ⚡ 关键修复：移除 scene_generator 不支持的参数（lora_config, character_id 等）
            # 这些参数只应该传递给 identity_injector，不应该传递给 scene_generator
            scene_kwargs.pop('lora_config', None)
            scene_kwargs.pop('character_id', None)
            # 如果 kwargs 中没有 num_inference_steps，使用默认值
            if 'num_inference_steps' not in scene_kwargs:
                scene_kwargs['num_inference_steps'] = 50  # 默认值
            if 'guidance_scale' not in scene_kwargs:
                scene_kwargs['guidance_scale'] = 7.5  # 默认值
            
            logger.info(f"场景生成参数: {width}x{height}, {scene_kwargs.get('num_inference_steps', 50)}步, guidance={scene_kwargs.get('guidance_scale', 7.5)}")
            scene_image = scene_generator(
                prompt=scene_prompt,
                width=width,
                height=height,
                **scene_kwargs
            ).images[0]
        else:
            logger.warning("未提供场景生成器，跳过场景生成")
            # 创建占位图像
            scene_image = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        # 阶段2: 检测人物区域
        logger.info("阶段2: 检测人物区域...")
        # ⚡ 关键修复：如果 SAM2 失败，自动回退到 YOLO
        person_mask = None
        try:
            person_mask = self.detect_person_region(scene_image, method="auto")
        except Exception as e:
            logger.warning(f"人物区域检测失败: {e}，尝试使用 YOLO...")
            try:
                person_mask = self.detect_person_region(scene_image, method="yolo")
            except Exception as e2:
                logger.error(f"YOLO 检测也失败: {e2}")
        
        if person_mask is None:
            logger.warning("未检测到人物区域，将在场景中心区域注入身份")
            # 如果未检测到人物，创建一个中心区域的 mask（假设人物在中心）
            # 确保 scene_image 是 PIL Image
            if hasattr(scene_image, 'images') and isinstance(scene_image.images, list) and len(scene_image.images) > 0:
                scene_image = scene_image.images[0]
            w, h = scene_image.size
            center_x, center_y = w // 2, h // 2
            # 创建一个中心区域的 mask（约占图像的 30%）
            mask_size = int(min(w, h) * 0.3)
            person_mask = np.zeros((h, w), dtype=np.uint8)
            x1 = max(0, center_x - mask_size // 2)
            y1 = max(0, center_y - mask_size // 2)
            x2 = min(w, center_x + mask_size // 2)
            y2 = min(h, center_y + mask_size // 2)
            # ⚡ 关键修复：使用圆角矩形而不是硬矩形
            from PIL import ImageDraw
            mask_pil = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask_pil)
            corner_radius = int(mask_size * 0.1)
            draw.rounded_rectangle(
                [(x1, y1), (x2, y2)],
                radius=corner_radius,
                fill=255
            )
            person_mask = np.array(mask_pil)
            person_mask = self._expand_mask(person_mask, padding=30)
        
        # 阶段3: 身份注入
        logger.info("阶段3: 身份注入...")
        
        # 确保 scene_image 是 PIL Image（处理 Result 对象）
        scene_img = scene_image
        if hasattr(scene_image, 'images') and isinstance(scene_image.images, list) and len(scene_image.images) > 0:
            scene_img = scene_image.images[0]
        
        # 如果场景中已经有人物，先清理 mask 区域（用背景填充）
        # 这样可以避免"背景下面还有一张人像"的问题
        if person_mask is not None:
            scene_img = self._remove_existing_person(scene_img, person_mask)
        
        if identity_injector is not None:
            # ⚡ 关键修复：传递原始 prompt 给身份注入器，确保包含角色和服饰描述
            final_image = self._inject_identity(
                scene_image=scene_img,
                mask=person_mask,
                face_reference=face_reference,
                identity_injector=identity_injector,
                reference_strength=reference_strength,
                prompt=prompt,  # ⚡ 传递原始 prompt（包含角色和服饰描述）
                **kwargs
            )
        else:
            logger.warning("未提供身份注入器，返回场景图像")
            final_image = scene_img
        
        # 阶段4: 边缘融合
        logger.info("阶段4: 边缘融合...")
        # 确保 scene_image 是 PIL Image（处理 Result 对象）
        scene_img = scene_image
        if hasattr(scene_image, 'images') and isinstance(scene_image.images, list) and len(scene_image.images) > 0:
            scene_img = scene_image.images[0]
        final_image = self._blend_edges(scene_img, final_image, person_mask)
        
        logger.info("解耦生成完成")
        
        # 注意：人脸验证应该在最终图像上进行，而不是在场景生成后的中间图像
        # 验证逻辑由调用者（enhanced_image_generator）在最终图像上执行
        
        return final_image
    
    def _build_scene_prompt(self, original_prompt: str) -> str:
        """
        构建场景生成 prompt
        
        移除人物描述，只保留环境描述，确保生成纯场景（无人物）
        但保留镜头类型描述（如 wide shot, medium shot, close-up 等）
        """
        import re
        
        # 先提取镜头类型描述（需要保留）
        shot_patterns = [
            r'\([^)]*(?:wide|medium|close|shot|full body|upper body|head|shoulders|distant|view)[^)]*:\d+\.?\d*\)',
            r'(?:extreme\s+)?(?:wide|medium|close|full|american)\s+shot[^,)]*',
            r'(?:full\s+body|upper\s+body|head\s+and\s+shoulders|face\s+only|distant\s+view)[^,)]*',
        ]
        
        shot_descriptions = []
        for pattern in shot_patterns:
            matches = re.findall(pattern, original_prompt, flags=re.IGNORECASE)
            shot_descriptions.extend(matches)
        
        # 移除人物相关的关键词和描述
        # 常见的人物相关关键词
        person_keywords = [
            r'\b(person|people|human|character|man|woman|boy|girl|figure|individual)\b',
            r'\b(face|facial|head|hair|eyes|nose|mouth)\b',
            r'\b(wearing|clothing|clothes|robe|garment|outfit)\b',
            r'\b(standing|sitting|walking|running|posing)\b',
            r'\b(young|old|elderly|teenage)\s+(man|woman|person)',
        ]
        
        # 移除人物描述
        scene_prompt = original_prompt
        for pattern in person_keywords:
            scene_prompt = re.sub(pattern, '', scene_prompt, flags=re.IGNORECASE)
        
        # 清理多余的空格和标点
        scene_prompt = re.sub(r'\s+', ' ', scene_prompt).strip()
        scene_prompt = re.sub(r'[,\s]+,', ',', scene_prompt)  # 移除多余的逗号
        scene_prompt = re.sub(r'^,|,$', '', scene_prompt)  # 移除开头和结尾的逗号
        
        # 添加场景优先的前缀，强调环境
        # 同时保留镜头类型描述（放在最前面，确保构图正确）
        scene_prefix = ""
        if shot_descriptions:
            # 如果有镜头类型描述，放在最前面并加权
            shot_desc = " ".join(set(shot_descriptions))  # 去重
            scene_prefix = f"({shot_desc}:2.0), "  # 高权重确保构图
        
        scene_prefix += "detailed environment, rich atmosphere, cinematic scene, "
        
        # 不添加人物占位描述，确保生成纯场景
        # 人物会在后续步骤中通过身份注入添加
        
        return scene_prefix + scene_prompt
    
    def _inject_identity(
        self,
        scene_image: Image.Image,
        mask: np.ndarray,
        face_reference: Union[str, Image.Image],
        identity_injector: Any,
        reference_strength: int,
        **kwargs
    ) -> Image.Image:
        """
        在检测区域注入身份
        
        使用 Inpainting 方式将身份注入到人物区域
        """
        # 处理 Result 对象（来自 FluxWrapper）
        if hasattr(scene_image, 'images') and isinstance(scene_image.images, list) and len(scene_image.images) > 0:
            scene_image = scene_image.images[0]
        
        # 确保 scene_image 是 PIL Image
        if not isinstance(scene_image, Image.Image):
            logger.error(f"scene_image 必须是 PIL Image，当前类型: {type(scene_image)}")
            return scene_image
        
        # 加载参考图
        if isinstance(face_reference, str):
            face_reference = Image.open(face_reference).convert('RGB')
        
        # 转换 mask 为 PIL Image
        mask_image = Image.fromarray(mask)
        
        # 检查 identity_injector 是否支持 inpaint
        if hasattr(identity_injector, 'inpaint'):
            # 使用 inpaint 接口
            result = identity_injector.inpaint(
                image=scene_image,
                mask=mask_image,
                face_reference=face_reference,
                strength=reference_strength / 100.0,
                **kwargs
            )
        elif hasattr(identity_injector, 'generate_with_identity'):
            # 使用 PuLID 接口，生成新的人物图像
            # 然后合成到场景中
            logger.info("使用 PuLID 生成人物，然后合成到场景")
            
            # 获取mask区域的边界框
            mask_np = np.array(mask_image)
            coords = np.where(mask_np > 0)
            if len(coords[0]) == 0:
                return scene_image
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # ⚡ 关键修复：生成人物图像时，需要包含完整的角色描述（包括服饰和姿态）
            # 从 kwargs 中获取原始 prompt（包含角色描述）
            original_prompt = kwargs.get('prompt', '')
            
            # ⚡ 关键修复：从原始 prompt 中提取完整的角色描述
            # 包括：角色名称、气质、服饰、姿态、镜头类型
            person_prompt = original_prompt
            
            # 检查 prompt 中是否包含关键信息
            has_character = any(kw in person_prompt.lower() for kw in ['hanli', '韩立', 'character', 'person'])
            has_clothing = any(kw in person_prompt.lower() for kw in ['robe', 'clothing', 'clothes', 'garment', 'outfit', '服饰', '衣服', '道袍', '长袍', 'daoist', 'cultivator'])
            has_pose = any(kw in person_prompt.lower() for kw in ['pose', 'standing', 'sitting', 'lying', 'posture', 'upright'])
            has_shot = any(kw in person_prompt.lower() for kw in ['shot', 'close', 'medium', 'wide', 'full'])
            
            # ⚡ 关键修复：如果没有角色描述，添加基本描述
            if not has_character:
                person_prompt = f"a person, {person_prompt}"
            
            # ⚡ 关键修复：如果没有服饰描述，添加基本服饰描述（仙侠风格）
            if not has_clothing:
                # 检查是否是仙侠场景
                is_xianxia = any(kw in person_prompt.lower() for kw in ['xianxia', 'immortal', 'cultivator', '仙侠', '修仙', '修士', 'daoist'])
                if is_xianxia:
                    person_prompt = f"{person_prompt}, wearing traditional Chinese cultivator robe, deep cyan flowing fabric, not armor"
                else:
                    person_prompt = f"{person_prompt}, wearing appropriate clothing"
            
            # ⚡ 关键修复：确保包含姿态和镜头类型信息（如果原始prompt中有）
            # 这些信息对人物生成很重要
            if not has_pose:
                # 尝试从原始prompt中提取姿态信息
                pose_keywords = ['standing', 'sitting', 'lying', 'upright', 'posture']
                for kw in pose_keywords:
                    if kw in original_prompt.lower():
                        person_prompt = f"{person_prompt}, {kw}"
                        break
            
            # 添加镜头类型（如果原始prompt中有，确保人物生成时也包含）
            if has_shot:
                # 提取镜头类型描述
                import re
                shot_patterns = [
                    r'(?:extreme\s+)?(?:wide|medium|close|full|american)\s+shot[^,)]*',
                    r'(?:full\s+body|upper\s+body|head\s+and\s+shoulders|face\s+only|distant\s+view)[^,)]*',
                ]
                for pattern in shot_patterns:
                    matches = re.findall(pattern, original_prompt, flags=re.IGNORECASE)
                    if matches:
                        # 将镜头类型添加到人物prompt前面（高优先级）
                        shot_desc = " ".join(set(matches))
                        person_prompt = f"{shot_desc}, {person_prompt}"
                        break
            
            logger.info(f"人物生成 prompt: {person_prompt[:200]}...")
            
            # ⚡ 关键修复：从 kwargs 中移除 prompt，避免重复传递
            # 因为我们已经显式传递了 person_prompt
            kwargs_clean = {k: v for k, v in kwargs.items() if k != 'prompt'}
            # ⚡ 关键修复：确保传递 num_inference_steps 和 guidance_scale
            if 'num_inference_steps' not in kwargs_clean:
                kwargs_clean['num_inference_steps'] = 50  # 默认值
            if 'guidance_scale' not in kwargs_clean:
                kwargs_clean['guidance_scale'] = 7.5  # 默认值
            
            logger.info(f"身份注入参数: {x_max - x_min}x{y_max - y_min}, {kwargs_clean.get('num_inference_steps', 50)}步, guidance={kwargs_clean.get('guidance_scale', 7.5)}")
            person_image = identity_injector.generate_with_identity(
                prompt=person_prompt,
                face_reference=face_reference,
                reference_strength=reference_strength,
                width=x_max - x_min,
                height=y_max - y_min,
                **kwargs_clean
            )
            
            # 合成到场景
            result = self._composite_person(
                scene_image,
                person_image,
                mask_image,
                (x_min, y_min)
            )
        else:
            logger.warning("identity_injector 不支持 inpaint 或 generate_with_identity")
            result = scene_image
        
        return result
    
    def _composite_person(
        self,
        scene: Image.Image,
        person: Image.Image,
        mask: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """
        将人物图像合成到场景中（使用更好的融合方式，避免贴图感）
        """
        result = scene.copy()
        
        # 调整人物图像大小以匹配 mask
        person_resized = person.resize(
            (mask.width, mask.height),
            Image.Resampling.LANCZOS
        )
        
        # 确保 mask 和人物图像尺寸一致
        mask_resized = mask.resize(person_resized.size)
        
        # 转换为 numpy 数组进行更好的融合
        scene_array = np.array(result).astype(float)
        person_array = np.array(person_resized).astype(float)
        mask_array = np.array(mask_resized).astype(float) / 255.0
        
        # 如果 mask 是单通道，扩展为 3 通道
        if len(mask_array.shape) == 2:
            mask_array = mask_array[:, :, np.newaxis]
        
        # 获取合成区域
        x, y = position
        h, w = person_array.shape[:2]
        
        # 确保不越界
        scene_h, scene_w = scene_array.shape[:2]
        h = min(h, scene_h - y)
        w = min(w, scene_w - x)
        
        if h > 0 and w > 0:
            person_array = person_array[:h, :w]
            mask_array = mask_array[:h, :w]
            
            # 使用 alpha 混合进行更自然的融合
            # 在 mask 区域使用人物图像，在边缘区域进行渐变混合
            scene_region = scene_array[y:y+h, x:x+w]
            
            # Alpha 混合
            blended = scene_region * (1 - mask_array) + person_array * mask_array
            
            # 写回结果
            scene_array[y:y+h, x:x+w] = blended
        
        return Image.fromarray(scene_array.astype(np.uint8))
    
    def _remove_existing_person(
        self,
        scene: Image.Image,
        mask: np.ndarray
    ) -> Image.Image:
        """
        移除场景中已有的人物（用背景填充）
        
        这样可以避免"背景下面还有一张人像"的问题
        """
        try:
            from scipy import ndimage
        except ImportError:
            logger.warning("scipy 未安装，使用简单方法填充")
            # 简单方法：使用周围像素的平均值填充
            scene_array = np.array(scene)
            mask_bool = mask > 128  # 转换为布尔 mask
            
            # ⚡ 关键修复：使用更大的填充区域，并使用更智能的背景填充
            if mask_bool.any():
                # 获取 mask 区域的边界框
                coords = np.where(mask_bool)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    # ⚡ 关键修复：使用更大的填充区域（至少 50 像素）
                    padding = max(50, int(min(x_max - x_min, y_max - y_min) * 0.2))
                    y_min_ext = max(0, y_min - padding)
                    y_max_ext = min(scene_array.shape[0], y_max + padding)
                    x_min_ext = max(0, x_min - padding)
                    x_max_ext = min(scene_array.shape[1], x_max + padding)
                    
                    # 获取周围区域的背景
                    background_region = scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext].copy()
                    mask_region = mask_bool[y_min_ext:y_max_ext, x_min_ext:x_max_ext]
                    
                    # ⚡ 关键修复：使用更强的模糊（sigma=5），创建更自然的背景
                    # 对背景区域进行轻微模糊，然后填充到 mask 区域
                    background_blurred = ndimage.gaussian_filter(background_region, sigma=5)
                    
                    # ⚡ 关键修复：使用 Image.inpaint 风格的填充，而不是简单的替换
                    # 在 mask 区域使用模糊后的背景，但保持边缘的渐变
                    for c in range(3):  # RGB 三个通道
                        # 创建渐变 mask（边缘更透明）
                        mask_gradient = mask_region.astype(float)
                        # 对 mask 边缘进行羽化
                        mask_gradient = ndimage.gaussian_filter(mask_gradient, sigma=3)
                        # 使用渐变混合
                        background_region[:, :, c] = (
                            background_region[:, :, c] * (1 - mask_gradient) + 
                            background_blurred[:, :, c] * mask_gradient
                        )
                    
                    scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext] = background_region
            
            return Image.fromarray(scene_array.astype(np.uint8))
        
        # 使用 scipy 的方法
        scene_array = np.array(scene)
        mask_bool = mask > 128  # 转换为布尔 mask
        
        # ⚡ 关键修复：使用更大的填充区域，并使用更智能的背景填充
        if mask_bool.any():
            # 获取 mask 区域的边界框
            coords = np.where(mask_bool)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # ⚡ 关键修复：使用更大的填充区域（至少 50 像素）
                padding = max(50, int(min(x_max - x_min, y_max - y_min) * 0.2))
                y_min_ext = max(0, y_min - padding)
                y_max_ext = min(scene_array.shape[0], y_max + padding)
                x_min_ext = max(0, x_min - padding)
                x_max_ext = min(scene_array.shape[1], x_max + padding)
                
                # 获取周围区域的背景
                background_region = scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext].copy()
                mask_region = mask_bool[y_min_ext:y_max_ext, x_min_ext:x_max_ext]
                
                # ⚡ 关键修复：使用更强的模糊（sigma=5），创建更自然的背景
                # 对背景区域进行轻微模糊，然后填充到 mask 区域
                background_blurred = ndimage.gaussian_filter(background_region, sigma=5)
                
                # ⚡ 关键修复：使用 Image.inpaint 风格的填充，而不是简单的替换
                # 在 mask 区域使用模糊后的背景，但保持边缘的渐变
                for c in range(3):  # RGB 三个通道
                    # 创建渐变 mask（边缘更透明）
                    mask_gradient = mask_region.astype(float)
                    # 对 mask 边缘进行羽化
                    mask_gradient = ndimage.gaussian_filter(mask_gradient, sigma=3)
                    # 使用渐变混合
                    background_region[:, :, c] = (
                        background_region[:, :, c] * (1 - mask_gradient) + 
                        background_blurred[:, :, c] * mask_gradient
                    )
                
                scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext] = background_region
        
        return Image.fromarray(scene_array.astype(np.uint8))
    
    def _blend_edges(
        self,
        scene: Image.Image,
        final: Image.Image,
        mask: np.ndarray,
        feather_radius: int = 10
    ) -> Image.Image:
        """
        融合边缘，消除人物区域与背景的边界
        """
        # 创建羽化 mask
        mask_pil = Image.fromarray(mask)
        mask_feathered = mask_pil.filter(ImageFilter.GaussianBlur(feather_radius))
        
        # 转换为浮点数进行混合
        mask_array = np.array(mask_feathered).astype(float) / 255.0
        mask_array = mask_array[:, :, np.newaxis]  # 添加通道维度
        
        scene_array = np.array(scene).astype(float)
        final_array = np.array(final).astype(float)
        
        # 混合
        blended = scene_array * (1 - mask_array) + final_array * mask_array
        
        return Image.fromarray(blended.astype(np.uint8))
    
    # ==========================================
    # 质量评估模块
    # ==========================================
    
    def verify_face_similarity(
        self,
        generated_image: Image.Image,
        reference_image: Image.Image,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """
        验证生成图像与参考的人脸相似度
        
        Args:
            generated_image: 生成的图像
            reference_image: 参考图像
            threshold: 相似度阈值
            
        Returns:
            (是否通过, 相似度分数)
        """
        try:
            from insightface.app import FaceAnalysis
            
            # 加载人脸分析器
            if not hasattr(self, 'face_analyzer') or self.face_analyzer is None:
                # InsightFace 的 root 参数会在其下寻找 models/{name} 目录
                # 所以 root 应该是 gen_video 目录，这样它会找 gen_video/models/antelopev2
                insightface_root = os.path.dirname(self.model_base_path)
                self.face_analyzer = FaceAnalysis(
                    name='antelopev2',
                    root=insightface_root,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_analyzer.prepare(ctx_id=0)
            
            # 提取人脸特征
            gen_np = np.array(generated_image)
            ref_np = np.array(reference_image)
            
            gen_faces = self.face_analyzer.get(gen_np)
            ref_faces = self.face_analyzer.get(ref_np)
            
            if not gen_faces or not ref_faces:
                logger.warning("无法检测到人脸")
                return False, 0.0
            
            # 计算余弦相似度
            gen_emb = gen_faces[0].embedding
            ref_emb = ref_faces[0].embedding
            
            similarity = np.dot(gen_emb, ref_emb) / (
                np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb)
            )
            
            passed = similarity >= threshold
            
            logger.info(f"人脸相似度: {similarity:.3f} (阈值: {threshold})")
            
            return passed, float(similarity)
            
        except Exception as e:
            logger.error(f"人脸相似度验证失败: {e}")
            return False, 0.0
    
    def unload(self):
        """卸载所有模型"""
        if self.sam2_predictor is not None:
            del self.sam2_predictor
            self.sam2_predictor = None
        
        if hasattr(self, 'sam2_model'):
            del self.sam2_model
        
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None
        
        if hasattr(self, 'face_analyzer'):
            del self.face_analyzer
        
        self.sam2_loaded = False
        self.yolo_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("解耦融合引擎已卸载")


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "device": "cuda",
        "model_dir": "/vepfs-dev/shawn/vid/fanren/gen_video/models"
    }
    
    engine = DecoupledFusionEngine(config)
    
    # 测试 YOLO 加载
    print("\n测试 YOLO 加载...")
    engine.load_yolo()
    
    print("\n解耦融合引擎初始化成功!")
