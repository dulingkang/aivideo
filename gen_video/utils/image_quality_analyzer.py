#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像质量分析器
用于自动评估生成的图像质量，包括：
- 人脸相似度（身份保持度）
- 构图质量（远景/中景/近景判断）
- 清晰度评估（拉普拉斯方差）
- 饱和度评估（色彩丰富度）
- 对比度评估
- 整体质量评分

Author: AI Video Team
Date: 2025-12-17
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ShotType(Enum):
    """镜头类型"""
    WIDE = "wide"           # 远景
    MEDIUM = "medium"       # 中景
    CLOSE = "close"         # 近景
    EXTREME_CLOSE = "extreme_close"  # 特写
    UNKNOWN = "unknown"     # 未知


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"  # 优秀 (90-100)
    GOOD = "good"           # 良好 (70-89)
    FAIR = "fair"           # 一般 (50-69)
    POOR = "poor"           # 较差 (30-49)
    BAD = "bad"             # 很差 (0-29)


@dataclass
class FaceSimilarityResult:
    """人脸相似度结果"""
    similarity: float = 0.0
    passed: bool = False
    threshold: float = 0.7
    face_detected_in_generated: bool = False
    face_detected_in_reference: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        # 确保所有值都是 Python 原生类型，以便 JSON 序列化
        return {
            "similarity": float(self.similarity) if self.similarity else 0.0,
            "passed": bool(self.passed),
            "threshold": float(self.threshold),
            "face_detected_in_generated": bool(self.face_detected_in_generated),
            "face_detected_in_reference": bool(self.face_detected_in_reference),
            "error": self.error
        }


@dataclass
class CompositionResult:
    """构图分析结果"""
    shot_type: ShotType = ShotType.UNKNOWN
    person_ratio: float = 0.0  # 人物占画面比例
    center_weight: float = 0.0  # 中心区域权重
    rule_of_thirds_score: float = 0.0  # 三分法评分
    face_position: Optional[Tuple[float, float]] = None  # 人脸在画面中的相对位置
    
    def to_dict(self) -> Dict[str, Any]:
        # 确保所有值都是 Python 原生类型
        face_pos = None
        if self.face_position:
            face_pos = (float(self.face_position[0]), float(self.face_position[1]))
        return {
            "shot_type": self.shot_type.value,
            "person_ratio": float(self.person_ratio),
            "center_weight": float(self.center_weight),
            "rule_of_thirds_score": float(self.rule_of_thirds_score),
            "face_position": face_pos
        }


@dataclass
class TechnicalQualityResult:
    """技术质量结果"""
    sharpness: float = 0.0          # 清晰度 (拉普拉斯方差)
    sharpness_level: QualityLevel = QualityLevel.FAIR
    saturation: float = 0.0         # 饱和度
    saturation_level: QualityLevel = QualityLevel.FAIR
    brightness: float = 0.0         # 亮度
    brightness_level: QualityLevel = QualityLevel.FAIR
    contrast: float = 0.0           # 对比度
    contrast_level: QualityLevel = QualityLevel.FAIR
    noise_level: float = 0.0        # 噪点水平 (越低越好)
    
    def to_dict(self) -> Dict[str, Any]:
        # 确保所有值都是 Python 原生类型
        return {
            "sharpness": float(self.sharpness),
            "sharpness_level": self.sharpness_level.value,
            "saturation": float(self.saturation),
            "saturation_level": self.saturation_level.value,
            "brightness": float(self.brightness),
            "brightness_level": self.brightness_level.value,
            "contrast": float(self.contrast),
            "contrast_level": self.contrast_level.value,
            "noise_level": float(self.noise_level)
        }


@dataclass
class ImageQualityReport:
    """图像质量报告"""
    image_path: Optional[str] = None
    image_size: Tuple[int, int] = (0, 0)
    timestamp: str = ""
    
    # 各项评估结果
    face_similarity: Optional[FaceSimilarityResult] = None
    composition: Optional[CompositionResult] = None
    technical: Optional[TechnicalQualityResult] = None
    
    # 综合评分
    overall_score: float = 0.0
    overall_level: QualityLevel = QualityLevel.FAIR
    
    # 问题和建议
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        # 确保所有值都是 Python 原生类型
        return {
            "image_path": self.image_path,
            "image_size": (int(self.image_size[0]), int(self.image_size[1])),
            "timestamp": self.timestamp,
            "face_similarity": self.face_similarity.to_dict() if self.face_similarity else None,
            "composition": self.composition.to_dict() if self.composition else None,
            "technical": self.technical.to_dict() if self.technical else None,
            "overall_score": float(self.overall_score),
            "overall_level": self.overall_level.value,
            "issues": list(self.issues),
            "suggestions": list(self.suggestions)
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class ImageQualityAnalyzer:
    """
    图像质量分析器
    
    提供全面的图像质量评估功能，用于验证生成图像的质量
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分析器
        
        Args:
            config: 配置字典，可选
        """
        self.config = config or {}
        self.face_analyzer = None
        self.device = self.config.get("device", "cuda")
        
        # 质量阈值
        self.thresholds = {
            "sharpness": {
                "excellent": 200,
                "good": 100,
                "fair": 50,
                "poor": 25
            },
            "saturation": {
                "excellent": 60,
                "good": 40,
                "fair": 25,
                "poor": 15
            },
            "brightness": {
                "min": 40,
                "max": 220,
                "optimal_min": 80,
                "optimal_max": 180
            },
            "contrast": {
                "excellent": 70,
                "good": 50,
                "fair": 35,
                "poor": 20
            }
        }
        
        # 镜头类型判断阈值 (人物占画面比例)
        self.shot_thresholds = {
            "extreme_close": 0.5,  # >50% 为特写
            "close": 0.25,         # 25-50% 为近景
            "medium": 0.1,         # 10-25% 为中景
            "wide": 0.0            # <10% 为远景
        }
    
    def _load_face_analyzer(self):
        """延迟加载人脸分析器"""
        if self.face_analyzer is not None:
            return
        
        try:
            from insightface.app import FaceAnalysis
            import os
            
            # 获取模型路径
            # InsightFace 会在 root/models/{name} 下查找模型
            # 所以 root 应该是包含 models 目录的父目录
            model_root = self.config.get("insightface_root", None)
            
            if model_root is None:
                # 默认路径：gen_video 目录（其下有 models/antelopev2）
                gen_video_dir = Path(__file__).parent.parent
                model_root = str(gen_video_dir)
            elif not os.path.isabs(model_root):
                # 相对路径时，基于 gen_video 目录
                gen_video_dir = Path(__file__).parent.parent
                model_root = str(gen_video_dir / model_root)
            
            logger.debug(f"InsightFace 模型根目录: {model_root}")
            
            # 检查模型是否存在
            model_path = os.path.join(model_root, "models", "antelopev2")
            if not os.path.exists(model_path):
                logger.warning(f"InsightFace 模型不存在: {model_path}")
                # 尝试备用路径
                alt_paths = [
                    os.path.join(gen_video_dir, "models", "antelopev2"),
                    os.path.expanduser("~/.insightface/models/antelopev2"),
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_root = os.path.dirname(os.path.dirname(alt_path))
                        logger.info(f"使用备用模型路径: {alt_path}")
                        break
            
            self.face_analyzer = FaceAnalysis(
                name='antelopev2',
                root=model_root,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0)
            logger.info("✅ 人脸分析器加载成功")
            
        except Exception as e:
            logger.warning(f"⚠️ 人脸分析器加载失败: {e}")
            self.face_analyzer = None
    
    def analyze(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        reference_image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        similarity_threshold: float = 0.7,
        expected_shot_type: Optional[str] = None
    ) -> ImageQualityReport:
        """
        分析图像质量
        
        Args:
            image: 要分析的图像 (路径、PIL Image 或 numpy array)
            reference_image: 参考图像 (用于人脸相似度比较)
            similarity_threshold: 相似度阈值
            expected_shot_type: 期望的镜头类型
            
        Returns:
            ImageQualityReport 分析报告
        """
        report = ImageQualityReport(timestamp=datetime.now().isoformat())
        
        # 加载图像
        pil_image, image_np, image_path = self._load_image(image)
        if pil_image is None:
            report.issues.append("无法加载图像")
            return report
        
        report.image_path = image_path
        report.image_size = pil_image.size
        
        # 1. 分析技术质量
        report.technical = self._analyze_technical_quality(image_np)
        
        # 2. 分析构图
        report.composition = self._analyze_composition(image_np, pil_image)
        
        # 3. 分析人脸相似度 (如果提供了参考图)
        if reference_image is not None:
            ref_pil, ref_np, _ = self._load_image(reference_image)
            if ref_pil is not None:
                report.face_similarity = self._analyze_face_similarity(
                    pil_image, ref_pil, similarity_threshold
                )
        
        # 4. 计算综合评分
        report.overall_score = self._calculate_overall_score(report)
        report.overall_level = self._score_to_level(report.overall_score)
        
        # 5. 生成问题和建议
        self._generate_issues_and_suggestions(report, expected_shot_type)
        
        return report
    
    def _load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray], Optional[str]]:
        """加载图像并转换为需要的格式"""
        image_path = None
        
        try:
            if isinstance(image, (str, Path)):
                image_path = str(image)
                pil_image = Image.open(image).convert('RGB')
                image_np = np.array(pil_image)
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
                image_np = np.array(pil_image)
            elif isinstance(image, np.ndarray):
                image_np = image
                if image_np.shape[-1] == 4:  # RGBA
                    image_np = image_np[:, :, :3]
                pil_image = Image.fromarray(image_np)
            else:
                return None, None, None
            
            return pil_image, image_np, image_path
            
        except Exception as e:
            logger.error(f"加载图像失败: {e}")
            return None, None, None
    
    def _analyze_technical_quality(self, image_np: np.ndarray) -> TechnicalQualityResult:
        """分析技术质量指标"""
        result = TechnicalQualityResult()
        
        # 转换为灰度图
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 1. 清晰度 (拉普拉斯方差)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result.sharpness = float(laplacian.var())
        result.sharpness_level = self._get_quality_level("sharpness", result.sharpness)
        
        # 2. 饱和度 (HSV 空间的 S 通道)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        result.saturation = float(hsv[:, :, 1].mean())
        result.saturation_level = self._get_quality_level("saturation", result.saturation)
        
        # 3. 亮度 (HSV 空间的 V 通道)
        result.brightness = float(hsv[:, :, 2].mean())
        result.brightness_level = self._get_brightness_level(result.brightness)
        
        # 4. 对比度 (灰度图的标准差)
        result.contrast = float(gray.std())
        result.contrast_level = self._get_quality_level("contrast", result.contrast)
        
        # 5. 噪点估计 (使用高频成分)
        # 使用高斯模糊后的差异来估计噪点
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.abs(gray.astype(float) - blurred.astype(float))
        result.noise_level = float(noise.mean())
        
        return result
    
    def _analyze_composition(
        self,
        image_np: np.ndarray,
        pil_image: Image.Image
    ) -> CompositionResult:
        """分析构图"""
        result = CompositionResult()
        h, w = image_np.shape[:2]
        
        # 尝试使用人脸检测来判断构图
        self._load_face_analyzer()
        
        face_bbox = None
        if self.face_analyzer is not None:
            try:
                faces = self.face_analyzer.get(image_np)
                if faces:
                    # 使用最大的人脸
                    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    face_bbox = largest_face.bbox  # [x1, y1, x2, y2]
                    
                    # 计算人脸占画面比例
                    face_w = face_bbox[2] - face_bbox[0]
                    face_h = face_bbox[3] - face_bbox[1]
                    face_area = face_w * face_h
                    image_area = w * h
                    result.person_ratio = face_area / image_area
                    
                    # 计算人脸中心位置 (归一化到 0-1)
                    face_center_x = (face_bbox[0] + face_bbox[2]) / 2 / w
                    face_center_y = (face_bbox[1] + face_bbox[3]) / 2 / h
                    result.face_position = (face_center_x, face_center_y)
                    
            except Exception as e:
                logger.debug(f"人脸检测失败: {e}")
        
        # 判断镜头类型
        if result.person_ratio > self.shot_thresholds["extreme_close"]:
            result.shot_type = ShotType.EXTREME_CLOSE
        elif result.person_ratio > self.shot_thresholds["close"]:
            result.shot_type = ShotType.CLOSE
        elif result.person_ratio > self.shot_thresholds["medium"]:
            result.shot_type = ShotType.MEDIUM
        elif result.person_ratio > 0:
            result.shot_type = ShotType.WIDE
        else:
            # 如果没检测到人脸，使用简单的亮度对比度判断
            result.shot_type = self._estimate_shot_type_by_contrast(image_np)
        
        # 计算中心权重 (用于判断人物是否在画面中心)
        center_region = image_np[h//4:3*h//4, w//4:3*w//4]
        result.center_weight = float(center_region.mean()) / 255.0
        
        # 计算三分法评分 (人脸位置是否在三分线上)
        if result.face_position:
            thirds_score = self._calculate_rule_of_thirds_score(result.face_position)
            result.rule_of_thirds_score = thirds_score
        
        return result
    
    def _estimate_shot_type_by_contrast(self, image_np: np.ndarray) -> ShotType:
        """通过对比度估计镜头类型"""
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 中心区域
        center = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = center.mean()
        
        # 边缘区域
        edge_regions = [
            gray[:h//8, :],      # 上
            gray[-h//8:, :],     # 下
            gray[:, :w//8],      # 左
            gray[:, -w//8:]      # 右
        ]
        edge_brightness = np.mean([r.mean() for r in edge_regions])
        
        contrast = abs(center_brightness - edge_brightness)
        
        if contrast > 60:
            return ShotType.CLOSE
        elif contrast > 30:
            return ShotType.MEDIUM
        else:
            return ShotType.WIDE
    
    def _calculate_rule_of_thirds_score(
        self,
        position: Tuple[float, float]
    ) -> float:
        """计算三分法评分"""
        x, y = position
        
        # 三分线位置: 1/3, 2/3
        thirds = [1/3, 2/3]
        
        # 计算到最近三分线的距离
        x_distance = min(abs(x - t) for t in thirds)
        y_distance = min(abs(y - t) for t in thirds)
        
        # 距离越小，分数越高
        x_score = max(0, 1 - x_distance * 3)
        y_score = max(0, 1 - y_distance * 3)
        
        return (x_score + y_score) / 2
    
    def _analyze_face_similarity(
        self,
        generated: Image.Image,
        reference: Image.Image,
        threshold: float
    ) -> FaceSimilarityResult:
        """分析人脸相似度"""
        result = FaceSimilarityResult(threshold=threshold)
        
        self._load_face_analyzer()
        
        if self.face_analyzer is None:
            result.error = "人脸分析器未加载"
            return result
        
        try:
            gen_np = np.array(generated)
            ref_np = np.array(reference)
            
            # 检测人脸
            gen_faces = self.face_analyzer.get(gen_np)
            ref_faces = self.face_analyzer.get(ref_np)
            
            result.face_detected_in_generated = len(gen_faces) > 0
            result.face_detected_in_reference = len(ref_faces) > 0
            
            if not gen_faces:
                result.error = "生成图像中未检测到人脸"
                return result
            
            if not ref_faces:
                result.error = "参考图像中未检测到人脸"
                return result
            
            # 计算余弦相似度
            gen_emb = gen_faces[0].embedding
            ref_emb = ref_faces[0].embedding
            
            similarity = np.dot(gen_emb, ref_emb) / (
                np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb)
            )
            
            result.similarity = float(similarity)
            result.passed = similarity >= threshold
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"人脸相似度分析失败: {e}")
        
        return result
    
    def _get_quality_level(self, metric: str, value: float) -> QualityLevel:
        """根据指标值获取质量等级"""
        thresholds = self.thresholds.get(metric, {})
        
        if value >= thresholds.get("excellent", float('inf')):
            return QualityLevel.EXCELLENT
        elif value >= thresholds.get("good", float('inf')):
            return QualityLevel.GOOD
        elif value >= thresholds.get("fair", float('inf')):
            return QualityLevel.FAIR
        elif value >= thresholds.get("poor", float('inf')):
            return QualityLevel.POOR
        else:
            return QualityLevel.BAD
    
    def _get_brightness_level(self, brightness: float) -> QualityLevel:
        """获取亮度等级"""
        thresholds = self.thresholds["brightness"]
        
        if brightness < thresholds["min"] or brightness > thresholds["max"]:
            return QualityLevel.POOR
        elif thresholds["optimal_min"] <= brightness <= thresholds["optimal_max"]:
            return QualityLevel.EXCELLENT
        else:
            return QualityLevel.GOOD
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """分数转换为等级"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 70:
            return QualityLevel.GOOD
        elif score >= 50:
            return QualityLevel.FAIR
        elif score >= 30:
            return QualityLevel.POOR
        else:
            return QualityLevel.BAD
    
    def _calculate_overall_score(self, report: ImageQualityReport) -> float:
        """计算综合评分"""
        scores = []
        weights = []
        
        # 技术质量权重: 40%
        if report.technical:
            tech = report.technical
            tech_score = 0
            
            # 清晰度 (占技术质量的 40%)
            sharpness_score = min(100, tech.sharpness / 2)
            tech_score += sharpness_score * 0.4
            
            # 饱和度 (占技术质量的 25%)
            saturation_score = min(100, tech.saturation * 1.5)
            tech_score += saturation_score * 0.25
            
            # 亮度 (占技术质量的 20%)
            # 最佳亮度在 80-180 范围内
            if 80 <= tech.brightness <= 180:
                brightness_score = 100
            else:
                distance = min(abs(tech.brightness - 80), abs(tech.brightness - 180))
                brightness_score = max(0, 100 - distance)
            tech_score += brightness_score * 0.2
            
            # 对比度 (占技术质量的 15%)
            contrast_score = min(100, tech.contrast * 1.5)
            tech_score += contrast_score * 0.15
            
            scores.append(tech_score)
            weights.append(0.4)
        
        # 构图权重: 20%
        if report.composition:
            comp = report.composition
            comp_score = 50  # 基础分
            
            # 三分法评分
            if comp.rule_of_thirds_score > 0:
                comp_score += comp.rule_of_thirds_score * 30
            
            # 人物比例合理性
            if comp.shot_type != ShotType.UNKNOWN:
                comp_score += 20
            
            scores.append(min(100, comp_score))
            weights.append(0.2)
        
        # 人脸相似度权重: 40%
        if report.face_similarity:
            face = report.face_similarity
            if face.similarity > 0:
                # 相似度 0.5-1.0 映射到 0-100
                face_score = max(0, (face.similarity - 0.5) * 200)
                scores.append(min(100, face_score))
                weights.append(0.4)
            elif not face.face_detected_in_generated:
                # 未检测到人脸，给予较低分数
                scores.append(30)
                weights.append(0.4)
        
        # 计算加权平均
        if not scores:
            return 50.0
        
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return round(weighted_score, 1)
    
    def _generate_issues_and_suggestions(
        self,
        report: ImageQualityReport,
        expected_shot_type: Optional[str] = None
    ):
        """生成问题和建议"""
        issues = []
        suggestions = []
        
        # 检查技术质量
        if report.technical:
            tech = report.technical
            
            if tech.sharpness_level in [QualityLevel.POOR, QualityLevel.BAD]:
                issues.append(f"图像模糊 (清晰度: {tech.sharpness:.1f})")
                suggestions.append("建议增加生成步数或降低引导强度")
            
            if tech.saturation_level in [QualityLevel.POOR, QualityLevel.BAD]:
                issues.append(f"色彩饱和度过低 (饱和度: {tech.saturation:.1f})")
                suggestions.append("建议在 prompt 中添加色彩描述词")
            
            if tech.brightness_level == QualityLevel.POOR:
                if tech.brightness < 50:
                    issues.append(f"图像过暗 (亮度: {tech.brightness:.1f})")
                    suggestions.append("建议调整场景光照描述")
                elif tech.brightness > 200:
                    issues.append(f"图像过亮 (亮度: {tech.brightness:.1f})")
                    suggestions.append("建议调整场景光照描述")
        
        # 检查人脸相似度
        if report.face_similarity:
            face = report.face_similarity
            
            if not face.face_detected_in_generated:
                issues.append("生成图像中未检测到人脸")
                suggestions.append("检查人物是否在画面中，或尝试使用近景镜头")
            elif not face.passed:
                issues.append(f"人脸相似度不足 ({face.similarity:.3f} < {face.threshold})")
                suggestions.append("建议增加参考强度或使用更清晰的参考图")
        
        # 检查构图
        if report.composition and expected_shot_type:
            actual_shot = report.composition.shot_type.value
            if actual_shot != expected_shot_type and report.composition.shot_type != ShotType.UNKNOWN:
                issues.append(f"镜头类型不匹配 (期望: {expected_shot_type}, 实际: {actual_shot})")
                suggestions.append("建议调整 prompt 中的镜头描述")
        
        report.issues = issues
        report.suggestions = suggestions
    
    def format_report(
        self,
        report: ImageQualityReport,
        verbose: bool = True
    ) -> str:
        """
        格式化报告为可读字符串
        
        Args:
            report: 质量报告
            verbose: 是否显示详细信息
            
        Returns:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("📊 图像质量分析报告")
        lines.append("=" * 60)
        
        # 基本信息
        if report.image_path:
            lines.append(f"📁 文件: {Path(report.image_path).name}")
        lines.append(f"📐 尺寸: {report.image_size[0]}x{report.image_size[1]}")
        lines.append(f"🕐 时间: {report.timestamp}")
        lines.append("")
        
        # 综合评分
        level_emoji = {
            QualityLevel.EXCELLENT: "🌟",
            QualityLevel.GOOD: "✅",
            QualityLevel.FAIR: "🟡",
            QualityLevel.POOR: "🟠",
            QualityLevel.BAD: "🔴"
        }
        emoji = level_emoji.get(report.overall_level, "❓")
        lines.append(f"🎯 综合评分: {report.overall_score:.1f}/100 {emoji} {report.overall_level.value.upper()}")
        lines.append("")
        
        # 人脸相似度
        if report.face_similarity:
            face = report.face_similarity
            lines.append("👤 人脸相似度:")
            if face.error:
                lines.append(f"   ⚠️ {face.error}")
            else:
                status = "✅ 通过" if face.passed else "❌ 未通过"
                lines.append(f"   相似度: {face.similarity:.3f} (阈值: {face.threshold}) {status}")
                
                # 相似度等级
                if face.similarity >= 0.8:
                    sim_level = "🟢 优秀"
                elif face.similarity >= 0.7:
                    sim_level = "🟡 良好"
                elif face.similarity >= 0.5:
                    sim_level = "🟠 一般"
                else:
                    sim_level = "🔴 较差"
                lines.append(f"   等级: {sim_level}")
            lines.append("")
        
        # 构图分析
        if report.composition and verbose:
            comp = report.composition
            lines.append("🎬 构图分析:")
            shot_emoji = {
                ShotType.EXTREME_CLOSE: "🔍",
                ShotType.CLOSE: "👁️",
                ShotType.MEDIUM: "📷",
                ShotType.WIDE: "🏞️",
                ShotType.UNKNOWN: "❓"
            }
            lines.append(f"   镜头类型: {shot_emoji.get(comp.shot_type, '')} {comp.shot_type.value}")
            if comp.person_ratio > 0:
                lines.append(f"   人物占比: {comp.person_ratio*100:.1f}%")
            if comp.face_position:
                lines.append(f"   人脸位置: ({comp.face_position[0]:.2f}, {comp.face_position[1]:.2f})")
            if comp.rule_of_thirds_score > 0:
                lines.append(f"   三分法评分: {comp.rule_of_thirds_score*100:.1f}%")
            lines.append("")
        
        # 技术质量
        if report.technical and verbose:
            tech = report.technical
            lines.append("📊 技术指标:")
            
            level_symbols = {
                QualityLevel.EXCELLENT: "🟢",
                QualityLevel.GOOD: "🟢",
                QualityLevel.FAIR: "🟡",
                QualityLevel.POOR: "🟠",
                QualityLevel.BAD: "🔴"
            }
            
            lines.append(f"   清晰度: {tech.sharpness:.1f} {level_symbols[tech.sharpness_level]}")
            lines.append(f"   饱和度: {tech.saturation:.1f} {level_symbols[tech.saturation_level]}")
            lines.append(f"   亮度: {tech.brightness:.1f} {level_symbols[tech.brightness_level]}")
            lines.append(f"   对比度: {tech.contrast:.1f} {level_symbols[tech.contrast_level]}")
            if tech.noise_level > 0:
                noise_level = "低" if tech.noise_level < 5 else ("中" if tech.noise_level < 10 else "高")
                lines.append(f"   噪点: {tech.noise_level:.1f} ({noise_level})")
            lines.append("")
        
        # 问题和建议
        if report.issues:
            lines.append("⚠️ 发现问题:")
            for issue in report.issues:
                lines.append(f"   • {issue}")
            lines.append("")
        
        if report.suggestions:
            lines.append("💡 优化建议:")
            for suggestion in report.suggestions:
                lines.append(f"   • {suggestion}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def log_report(
        self,
        report: ImageQualityReport,
        level: str = "info"
    ):
        """
        将报告输出到日志
        
        Args:
            report: 质量报告
            level: 日志级别 (debug, info, warning)
        """
        formatted = self.format_report(report, verbose=(level == "debug"))
        
        log_func = getattr(logger, level, logger.info)
        for line in formatted.split("\n"):
            log_func(line)
    
    def unload(self):
        """卸载模型释放资源"""
        if self.face_analyzer is not None:
            del self.face_analyzer
            self.face_analyzer = None
            
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug("图像质量分析器已卸载")


def analyze_image(
    image_path: str,
    reference_path: Optional[str] = None,
    threshold: float = 0.7
) -> ImageQualityReport:
    """
    分析图像质量的便捷函数
    
    Args:
        image_path: 图像文件路径
        reference_path: 参考图像路径（可选）
        threshold: 相似度阈值
        
    Returns:
        ImageQualityReport 分析报告
    """
    analyzer = ImageQualityAnalyzer()
    try:
        return analyzer.analyze(
            image_path,
            reference_image=reference_path,
            similarity_threshold=threshold
        )
    finally:
        analyzer.unload()


if __name__ == "__main__":
    """测试图像质量分析器"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python image_quality_analyzer.py <image_path> [reference_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    reference_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        analyzer = ImageQualityAnalyzer()
        report = analyzer.analyze(
            image_path,
            reference_image=reference_path,
            similarity_threshold=0.7
        )
        
        # 打印报告
        print(analyzer.format_report(report))
        
        # 保存 JSON
        json_path = Path(image_path).with_suffix('.quality.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n📁 JSON 报告已保存: {json_path}")
        
        analyzer.unload()
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
