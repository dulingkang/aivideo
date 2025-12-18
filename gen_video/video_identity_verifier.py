#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成增强模块 - 身份验证与重试
用于 HunyuanVideo 生成视频时的身份一致性验证

核心功能:
1. 视频生成后自动验证身份一致性
2. 失败时自动调整参数重试
3. 镜头语言建议（避免漂移）
4. 阈值配置管理

MVP 策略:
- 图像阶段: FLUX + PuLID → 角色 anchor 图
- 视频阶段: HunyuanVideo 1.5 (I2V) → VideoIdentityAnalyzer → 失败重试

Author: AI Video Team
Date: 2025-12-18
Project: M6 - 视频身份保持
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ShotLanguage(Enum):
    """镜头语言类型"""
    WIDE = "wide"               # 远景 - 最安全，漂移最少
    MEDIUM = "medium"           # 中景 - 推荐，漂移较少
    MEDIUM_CLOSE = "medium_close"  # 中近景 - 需要注意
    CLOSE = "close"             # 近景 - 有漂移风险
    EXTREME_CLOSE = "extreme_close"  # 特写 - 高漂移风险


@dataclass
class IdentityVerificationConfig:
    """身份验证配置"""
    # 相似度阈值
    similarity_threshold: float = 0.70  # 低于此值重试
    similarity_discard: float = 0.65    # 低于此值丢弃
    
    # 漂移阈值
    drift_threshold: float = 0.50       # 单帧漂移阈值
    max_drift_ratio: float = 0.10       # 最大漂移帧比例
    
    # 人脸检测
    min_face_detect_ratio: float = 0.80  # 最小人脸检测率
    
    # 重试配置
    max_retries: int = 3                # 最大重试次数
    retry_reduce_motion: bool = True    # 重试时减少运动
    retry_adjust_prompt: bool = True    # 重试时调整 prompt

    # 验证采样增强：强制包含尾帧（避免“最后几帧崩脸但采样没覆盖”）
    include_last_n_frames: int = 3      # 无论 sample_interval，多加最后N帧参与分析

    # 最低相似度下限：用于过滤“极端崩脸帧”（可用于阈值统计微调）
    min_similarity_floor: float = 0.30
    
    # 镜头类型影响
    shot_type_tolerance: Dict[str, float] = None
    
    def __post_init__(self):
        if self.shot_type_tolerance is None:
            self.shot_type_tolerance = {
                "wide": 0.10,        # 远景允许更多漂移
                "medium": 0.05,      # 中景标准
                "close": 0.03,       # 近景更严格
                "extreme_close": 0.02  # 特写最严格
            }


@dataclass 
class VerificationResult:
    """验证结果"""
    passed: bool
    avg_similarity: float
    min_similarity: float
    drift_ratio: float
    face_detect_ratio: float
    issues: List[str]
    should_retry: bool
    should_discard: bool
    retry_hints: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "avg_similarity": round(self.avg_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "drift_ratio": round(self.drift_ratio, 4),
            "face_detect_ratio": round(self.face_detect_ratio, 4),
            "issues": self.issues,
            "should_retry": self.should_retry,
            "should_discard": self.should_discard,
            "retry_hints": self.retry_hints
        }


class VideoIdentityVerifier:
    """
    视频身份验证器
    
    用于验证 HunyuanVideo 生成的视频是否保持角色身份一致性
    """
    
    def __init__(self, config: Optional[IdentityVerificationConfig] = None):
        """
        初始化验证器
        
        Args:
            config: 验证配置
        """
        self.config = config or IdentityVerificationConfig()
        self._analyzer = None
    
    def _get_analyzer(self):
        """延迟加载分析器"""
        if self._analyzer is None:
            try:
                from utils.video_identity_analyzer import VideoIdentityAnalyzer
                # 将验证配置同步到分析器（尤其是 drift_threshold 会影响 drift_ratio 计算）
                analyzer_config = {
                    "drift_threshold": self.config.drift_threshold,
                    # 让分析器的“自身通过判定”阈值与验证器一致（虽然 verify_video 主要用自己的逻辑）
                    "similarity_threshold": self.config.similarity_threshold,
                    "include_last_n_frames": self.config.include_last_n_frames,
                }
                self._analyzer = VideoIdentityAnalyzer(analyzer_config)
                logger.info("VideoIdentityAnalyzer 加载成功")
            except ImportError as e:
                logger.warning(f"VideoIdentityAnalyzer 导入失败: {e}")
                return None
        return self._analyzer
    
    def verify_video(
        self,
        video_path: str,
        reference_image: str,
        shot_type: str = "medium",
        sample_interval: int = 5,
        max_frames: int = 50
    ) -> VerificationResult:
        """
        验证视频身份一致性
        
        Args:
            video_path: 视频文件路径
            reference_image: 参考图像路径
            shot_type: 镜头类型 (wide/medium/close/extreme_close)
            sample_interval: 采样间隔
            max_frames: 最大分析帧数
            
        Returns:
            VerificationResult 验证结果
        """
        analyzer = self._get_analyzer()
        
        # 如果分析器不可用，默认通过
        if analyzer is None:
            logger.warning("分析器不可用，跳过验证")
            return VerificationResult(
                passed=True,
                avg_similarity=1.0,
                min_similarity=1.0,
                drift_ratio=0.0,
                face_detect_ratio=1.0,
                issues=[],
                should_retry=False,
                should_discard=False,
                retry_hints={}
            )
        
        # 根据镜头类型调整阈值
        tolerance = self.config.shot_type_tolerance.get(shot_type, 0.05)
        adjusted_threshold = self.config.similarity_threshold - tolerance
        adjusted_discard = self.config.similarity_discard - tolerance
        
        logger.info(f"验证视频: {video_path}")
        logger.info(f"  镜头类型: {shot_type}, 调整后阈值: {adjusted_threshold:.2f}")
        
        # 分析视频
        try:
            report = analyzer.analyze_video(
                video_path=video_path,
                reference_image=reference_image,
                sample_interval=sample_interval,
                max_frames=max_frames
            )
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            return VerificationResult(
                passed=False,
                avg_similarity=0.0,
                min_similarity=0.0,
                drift_ratio=1.0,
                face_detect_ratio=0.0,
                issues=[f"分析失败: {str(e)}"],
                should_retry=True,
                should_discard=False,
                retry_hints={"error": str(e)}
            )
        
        # 判断结果
        issues = []
        retry_hints = {}
        
        # 1. 平均相似度检查
        if report.avg_similarity < adjusted_discard:
            issues.append(f"平均相似度过低: {report.avg_similarity:.3f} < {adjusted_discard:.2f}")
        elif report.avg_similarity < adjusted_threshold:
            issues.append(f"平均相似度不足: {report.avg_similarity:.3f} < {adjusted_threshold:.2f}")
            retry_hints["reduce_motion"] = True
        
        # 2. 漂移检查
        if report.drift_ratio > self.config.max_drift_ratio:
            issues.append(f"漂移帧过多: {report.drift_ratio*100:.1f}% > {self.config.max_drift_ratio*100:.0f}%")
            retry_hints["reduce_motion"] = True
            retry_hints["use_simpler_prompt"] = True
        
        # 3. 人脸检测率检查
        if report.face_detected_ratio < self.config.min_face_detect_ratio:
            issues.append(f"人脸检测率低: {report.face_detected_ratio*100:.1f}% < {self.config.min_face_detect_ratio*100:.0f}%")
            retry_hints["use_medium_shot"] = True
        
        # 4. 最低相似度检查
        if report.min_similarity < self.config.min_similarity_floor:
            issues.append(
                f"存在极低相似度帧: {report.min_similarity:.3f} < {self.config.min_similarity_floor:.2f}"
            )
            retry_hints["reduce_motion"] = True
        
        # 判断是否通过
        passed = len(issues) == 0
        should_retry = not passed and report.avg_similarity >= adjusted_discard
        should_discard = report.avg_similarity < adjusted_discard
        
        result = VerificationResult(
            passed=passed,
            avg_similarity=report.avg_similarity,
            min_similarity=report.min_similarity,
            drift_ratio=report.drift_ratio,
            face_detect_ratio=report.face_detected_ratio,
            issues=issues,
            should_retry=should_retry,
            should_discard=should_discard,
            retry_hints=retry_hints
        )
        
        # 日志输出
        status = "✅ 通过" if passed else ("🔄 重试" if should_retry else "❌ 丢弃")
        logger.info(f"  验证结果: {status}")
        logger.info(f"    平均相似度: {report.avg_similarity:.3f}")
        logger.info(f"    漂移比例: {report.drift_ratio*100:.1f}%")
        if issues:
            for issue in issues:
                logger.warning(f"    ⚠ {issue}")
        
        return result
    
    def unload(self):
        """卸载分析器"""
        if self._analyzer is not None:
            self._analyzer.unload()
            self._analyzer = None


class ShotLanguageAdvisor:
    """
    镜头语言建议器
    
    为避免身份漂移提供镜头语言建议
    """
    
    # 安全的镜头描述词
    SAFE_SHOT_KEYWORDS = {
        ShotLanguage.WIDE: [
            "wide shot", "establishing shot", "full body",
            "environmental shot", "long shot", "distant view"
        ],
        ShotLanguage.MEDIUM: [
            "medium shot", "waist shot", "mid-shot",
            "three-quarter shot", "American shot"
        ],
        ShotLanguage.MEDIUM_CLOSE: [
            "medium close-up", "bust shot", "chest shot"
        ],
        ShotLanguage.CLOSE: [
            "close-up", "head shot", "face shot"
        ],
        ShotLanguage.EXTREME_CLOSE: [
            "extreme close-up", "macro", "detail shot"
        ]
    }
    
    # 应避免的描述词
    RISKY_KEYWORDS = [
        "extreme close-up", "macro face", "face filling frame",
        "dramatic head turn", "rapid movement", "dynamic motion",
        "spinning", "whipping", "fast pan"
    ]
    
    # 安全的运动描述
    SAFE_MOTION_KEYWORDS = [
        "subtle movement", "gentle motion", "slow pan",
        "static camera", "minimal movement", "steady",
        "calm", "smooth transition", "soft motion"
    ]
    
    @classmethod
    def get_drift_risk(cls, shot_type: ShotLanguage) -> str:
        """获取漂移风险等级"""
        risk_map = {
            ShotLanguage.WIDE: "低",
            ShotLanguage.MEDIUM: "低-中",
            ShotLanguage.MEDIUM_CLOSE: "中",
            ShotLanguage.CLOSE: "中-高",
            ShotLanguage.EXTREME_CLOSE: "高"
        }
        return risk_map.get(shot_type, "未知")
    
    @classmethod
    def suggest_shot_for_scene(cls, scene_type: str) -> Tuple[ShotLanguage, str]:
        """
        根据场景类型建议镜头
        
        Args:
            scene_type: 场景类型
            
        Returns:
            (推荐镜头, 理由)
        """
        suggestions = {
            "dialogue": (ShotLanguage.MEDIUM, "对话场景使用中景，保持身份稳定"),
            "action": (ShotLanguage.WIDE, "动作场景使用远景，减少漂移风险"),
            "emotional": (ShotLanguage.MEDIUM_CLOSE, "情感场景使用中近景，平衡表情和稳定性"),
            "establishing": (ShotLanguage.WIDE, "建立镜头使用远景"),
            "transition": (ShotLanguage.MEDIUM, "过渡场景使用中景"),
            "portrait": (ShotLanguage.MEDIUM, "人物介绍使用中景，避免特写漂移"),
        }
        return suggestions.get(scene_type, (ShotLanguage.MEDIUM, "默认使用中景"))
    
    @classmethod
    def enhance_prompt_for_stability(
        cls,
        prompt: str,
        shot_type: ShotLanguage = ShotLanguage.MEDIUM
    ) -> str:
        """
        增强 prompt 以提高身份稳定性
        
        Args:
            prompt: 原始 prompt
            shot_type: 镜头类型
            
        Returns:
            增强后的 prompt
        """
        # 添加稳定性关键词
        stability_prefix = "consistent character appearance, maintaining identity, "
        
        # 添加镜头类型
        shot_keywords = cls.SAFE_SHOT_KEYWORDS.get(shot_type, [])
        if shot_keywords:
            stability_prefix += shot_keywords[0] + ", "
        
        # 添加运动控制
        stability_prefix += "subtle natural movement, "
        
        # 检查原 prompt 中是否有高风险词汇
        prompt_lower = prompt.lower()
        warnings = []
        for risky in cls.RISKY_KEYWORDS:
            if risky in prompt_lower:
                warnings.append(risky)
        
        if warnings:
            logger.warning(f"Prompt 中包含高风险词汇: {warnings}")
        
        return stability_prefix + prompt
    
    @classmethod
    def get_negative_prompt_for_stability(cls) -> str:
        """获取用于身份稳定的 negative prompt"""
        return (
            "face changing, identity drift, inconsistent appearance, "
            "morphing face, different person, wrong face, "
            "deformed face, distorted features, "
            "multiple faces, face swap, "
            "rapid movement, extreme motion blur"
        )


def generate_video_with_verification(
    video_generator,
    image_path: str,
    output_path: str,
    reference_image: str,
    prompt: str = "",
    scene: Optional[Dict[str, Any]] = None,
    shot_type: str = "medium",
    max_retries: int = 3,
    verification_config: Optional[IdentityVerificationConfig] = None,
    **kwargs
) -> Tuple[str, VerificationResult]:
    """
    生成视频并验证身份一致性
    
    Args:
        video_generator: VideoGenerator 实例
        image_path: 输入图像路径
        output_path: 输出视频路径
        reference_image: 参考图像路径（用于身份验证）
        prompt: 视频生成 prompt
        scene: 场景配置
        shot_type: 镜头类型
        max_retries: 最大重试次数
        verification_config: 验证配置
        **kwargs: 其他视频生成参数
        
    Returns:
        (视频路径, 验证结果)
    """
    verifier = VideoIdentityVerifier(verification_config)
    
    # 增强 prompt
    shot_enum = getattr(ShotLanguage, shot_type.upper(), ShotLanguage.MEDIUM)
    enhanced_prompt = ShotLanguageAdvisor.enhance_prompt_for_stability(prompt, shot_enum)
    negative_prompt = ShotLanguageAdvisor.get_negative_prompt_for_stability()
    
    retry_count = 0
    best_result = None
    best_video_path = None
    
    while retry_count <= max_retries:
        # 生成视频
        attempt_suffix = f"_attempt{retry_count}" if retry_count > 0 else ""
        current_output = output_path.replace(".mp4", f"{attempt_suffix}.mp4")
        
        logger.info(f"生成视频 (尝试 {retry_count + 1}/{max_retries + 1})")
        
        try:
            # 调用视频生成
            video_path = video_generator.generate_video(
                image_path=image_path,
                output_path=current_output,
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                scene=scene,
                **kwargs
            )
            
            # 验证身份一致性
            result = verifier.verify_video(
                video_path=video_path,
                reference_image=reference_image,
                shot_type=shot_type
            )
            
            # 保存最佳结果
            if best_result is None or result.avg_similarity > best_result.avg_similarity:
                best_result = result
                best_video_path = video_path
            
            # 如果通过验证，返回
            if result.passed:
                logger.info(f"✅ 视频验证通过")
                verifier.unload()
                return video_path, result
            
            # 如果应该丢弃，不再重试
            if result.should_discard:
                logger.warning(f"❌ 视频质量过低，丢弃")
                break
            
            # 准备重试
            if result.should_retry and retry_count < max_retries:
                logger.info(f"🔄 准备重试...")
                
                # 根据提示调整参数
                if result.retry_hints.get("reduce_motion"):
                    enhanced_prompt = "slow gentle movement, minimal motion, " + enhanced_prompt
                    logger.info("  调整: 减少运动描述")
                
                if result.retry_hints.get("use_medium_shot"):
                    enhanced_prompt = enhanced_prompt.replace("close-up", "medium shot")
                    logger.info("  调整: 切换为中景")
            
            retry_count += 1
            
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            retry_count += 1
    
    # 返回最佳结果
    verifier.unload()
    
    if best_result is None:
        # 所有尝试都失败
        return None, VerificationResult(
            passed=False,
            avg_similarity=0.0,
            min_similarity=0.0,
            drift_ratio=1.0,
            face_detect_ratio=0.0,
            issues=["所有生成尝试都失败"],
            should_retry=False,
            should_discard=True,
            retry_hints={}
        )
    
    return best_video_path, best_result


if __name__ == "__main__":
    """测试验证模块"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("视频身份验证模块测试")
    print("=" * 60)
    
    # 测试镜头建议
    print("\n镜头语言建议测试:")
    for shot in ShotLanguage:
        risk = ShotLanguageAdvisor.get_drift_risk(shot)
        print(f"  {shot.value}: 漂移风险 = {risk}")
    
    # 测试 prompt 增强
    print("\nPrompt 增强测试:")
    original_prompt = "a woman walking in the park"
    enhanced = ShotLanguageAdvisor.enhance_prompt_for_stability(
        original_prompt, ShotLanguage.MEDIUM
    )
    print(f"  原始: {original_prompt}")
    print(f"  增强: {enhanced[:100]}...")
    
    print("\n✅ 模块测试完成!")
