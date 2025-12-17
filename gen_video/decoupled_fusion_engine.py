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
                
                # 查找配置文件
                config_file = os.path.join(self.sam2_path, "sam2_hiera_l.yaml")
                if not os.path.exists(config_file):
                    # 使用默认配置
                    config_file = "sam2_hiera_l"
                
                # 查找权重文件
                checkpoint = None
                for name in ["sam2_hiera_large.pt", "model.safetensors"]:
                    path = os.path.join(self.sam2_path, name)
                    if os.path.exists(path):
                        checkpoint = path
                        break
                
                if checkpoint is None:
                    raise FileNotFoundError(f"SAM2 权重文件未找到: {self.sam2_path}")
                
                # 构建模型
                sam2_model = build_sam2(config_file, checkpoint, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                
                self.sam2_loaded = True
                logger.info(f"SAM2 加载完成: {checkpoint}")
                
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
            
            # YOLO 会自动下载权重
            self.yolo_model = YOLO("yolov8n.pt")  # nano 版本，速度快
            
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
            method = "yolo"  # YOLO 更快更稳定
        
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
            mask[y1:y2, x1:x2] = 255
        
        # 扩展 mask 边缘 (用于自然融合)
        mask = self._expand_mask(mask, padding=20)
        
        return mask
    
    def _detect_with_sam2(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 SAM2 检测人物"""
        self.load_sam2()
        
        # 先用 YOLO 获取人物框作为 SAM2 的 prompt
        self.load_yolo()
        results = self.yolo_model(image, classes=[0], verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            logger.warning("未检测到人物框")
            return None
        
        # 获取最大的人物框
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_box = boxes[np.argmax(areas)]
        
        # 使用 SAM2 生成精确 mask
        if hasattr(self, 'sam2_predictor') and self.sam2_predictor is not None:
            self.sam2_predictor.set_image(image)
            
            # 使用框作为 prompt
            masks, scores, _ = self.sam2_predictor.predict(
                box=main_box,
                multimask_output=True
            )
            
            # 选择得分最高的 mask
            best_mask = masks[np.argmax(scores)]
            
            return (best_mask * 255).astype(np.uint8)
        else:
            # 使用 transformers 版本
            return self._detect_with_sam2_transformers(image, main_box)
    
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
        
        # 使用形态学膨胀扩展 mask
        kernel = np.ones((padding * 2, padding * 2), np.uint8)
        expanded = ndimage.binary_dilation(mask > 0, kernel)
        
        return (expanded * 255).astype(np.uint8)
    
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
            scene_image = scene_generator(
                prompt=scene_prompt,
                width=width,
                height=height,
                **kwargs
            ).images[0]
        else:
            logger.warning("未提供场景生成器，跳过场景生成")
            # 创建占位图像
            scene_image = Image.new('RGB', (width, height), color=(128, 128, 128))
        
        # 阶段2: 检测人物区域
        logger.info("阶段2: 检测人物区域...")
        person_mask = self.detect_person_region(scene_image)
        
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
            person_mask[y1:y2, x1:x2] = 255
            person_mask = self._expand_mask(person_mask, padding=20)
        
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
            final_image = self._inject_identity(
                scene_image=scene_img,
                mask=person_mask,
                face_reference=face_reference,
                identity_injector=identity_injector,
                reference_strength=reference_strength,
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
            
            # 生成人物图像
            # 使用原始 prompt 但添加人物描述，确保生成的人物符合场景
            original_prompt = kwargs.get('prompt', '')
            # 构建人物生成 prompt：保留环境描述，但强调人物
            person_prompt = original_prompt
            # 如果 prompt 中没有人物描述，添加基本的人物描述
            if 'person' not in person_prompt.lower() and 'character' not in person_prompt.lower():
                person_prompt = f"a person, {person_prompt}"
            
            logger.info(f"人物生成 prompt: {person_prompt[:100]}...")
            
            person_image = identity_injector.generate_with_identity(
                prompt=person_prompt,
                face_reference=face_reference,
                reference_strength=reference_strength,
                width=x_max - x_min,
                height=y_max - y_min,
                **kwargs
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
            
            # 对 mask 区域进行轻微的高斯模糊，模拟背景
            if mask_bool.any():
                # 获取 mask 区域的边界框
                coords = np.where(mask_bool)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    # 扩展边界框以获取周围区域
                    padding = 20
                    y_min_ext = max(0, y_min - padding)
                    y_max_ext = min(scene_array.shape[0], y_max + padding)
                    x_min_ext = max(0, x_min - padding)
                    x_max_ext = min(scene_array.shape[1], x_max + padding)
                    
                    # 获取周围区域的背景
                    background_region = scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext].copy()
                    mask_region = mask_bool[y_min_ext:y_max_ext, x_min_ext:x_max_ext]
                    
                    # 对背景区域进行轻微模糊，然后填充到 mask 区域
                    background_blurred = ndimage.gaussian_filter(background_region, sigma=3)
                    
                    # 在 mask 区域使用模糊后的背景
                    for c in range(3):  # RGB 三个通道
                        background_region[:, :, c][mask_region] = background_blurred[:, :, c][mask_region]
                    
                    scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext] = background_region
            
            return Image.fromarray(scene_array.astype(np.uint8))
        
        # 使用 scipy 的方法
        scene_array = np.array(scene)
        mask_bool = mask > 128  # 转换为布尔 mask
        
        # 对 mask 区域进行轻微的高斯模糊，模拟背景
        if mask_bool.any():
            # 获取 mask 区域的边界框
            coords = np.where(mask_bool)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # 扩展边界框以获取周围区域
                padding = 20
                y_min_ext = max(0, y_min - padding)
                y_max_ext = min(scene_array.shape[0], y_max + padding)
                x_min_ext = max(0, x_min - padding)
                x_max_ext = min(scene_array.shape[1], x_max + padding)
                
                # 获取周围区域的背景
                background_region = scene_array[y_min_ext:y_max_ext, x_min_ext:x_max_ext].copy()
                mask_region = mask_bool[y_min_ext:y_max_ext, x_min_ext:x_max_ext]
                
                # 对背景区域进行轻微模糊，然后填充到 mask 区域
                background_blurred = ndimage.gaussian_filter(background_region, sigma=3)
                
                # 在 mask 区域使用模糊后的背景
                for c in range(3):  # RGB 三个通道
                    background_region[:, :, c][mask_region] = background_blurred[:, :, c][mask_region]
                
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
