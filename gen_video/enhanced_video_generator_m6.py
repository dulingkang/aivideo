#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成器增强模块
在 VideoGenerator 基础上添加身份验证功能

使用方法:
    from enhanced_video_generator_m6 import EnhancedVideoGeneratorM6
    
    generator = EnhancedVideoGeneratorM6("config.yaml")
    video_path, result = generator.generate_video_with_identity_check(
        image_path="input.png",
        output_path="output.mp4",
        reference_image="reference.jpg",
        scene=scene_config
    )

Author: AI Video Team
Date: 2025-12-18
Project: M6 - 视频身份保持
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from video_generator import VideoGenerator
from video_identity_verifier import (
    VideoIdentityVerifier,
    IdentityVerificationConfig,
    VerificationResult,
    ShotLanguageAdvisor,
    ShotLanguage
)

logger = logging.getLogger(__name__)


class EnhancedVideoGeneratorM6(VideoGenerator):
    """
    增强版视频生成器 - M6 身份保持
    
    在 VideoGenerator 基础上添加:
    1. 身份一致性验证
    2. 失败重试
    3. 镜头语言增强
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化增强版生成器"""
        super().__init__(config_path)
        
        # 身份验证配置
        identity_config = self.video_config.get("identity_verification", {})
        
        self.identity_config = IdentityVerificationConfig(
            similarity_threshold=identity_config.get("similarity_threshold", 0.70),
            similarity_discard=identity_config.get("similarity_discard", 0.65),
            drift_threshold=identity_config.get("drift_threshold", 0.50),
            max_drift_ratio=identity_config.get("max_drift_ratio", 0.10),
            min_face_detect_ratio=identity_config.get("min_face_detect_ratio", 0.80),
            min_similarity_floor=float(identity_config.get("min_similarity_floor", 0.30)),
            max_retries=identity_config.get("max_retries", 3),
            retry_reduce_motion=identity_config.get("retry_reduce_motion", True),
            retry_adjust_prompt=identity_config.get("retry_adjust_prompt", True),
            include_last_n_frames=int(identity_config.get("include_last_n_frames", 3)),
            shot_type_tolerance=identity_config.get("shot_type_tolerance"),
        )

        # 验证采样配置（避免每次硬编码）
        self.verification_sample_interval = int(identity_config.get("sample_interval", 5))
        self.verification_max_frames = int(identity_config.get("max_frames", 30))
        
        # 懒加载验证器
        self._verifier = None
        
        logger.info("EnhancedVideoGeneratorM6 初始化完成")
        logger.info(f"  身份验证阈值: {self.identity_config.similarity_threshold}")
    
    def _get_verifier(self) -> VideoIdentityVerifier:
        """获取验证器（懒加载）"""
        if self._verifier is None:
            self._verifier = VideoIdentityVerifier(self.identity_config)
        return self._verifier

    def _promote_video_to_output(self, src_video_path: str, dst_output_path: str) -> str:
        """
        将通过验证的尝试视频提升为最终输出（覆盖 dst_output_path）。
        这样调用方无论是否忽略返回值，都能拿到“通过验证”的最终视频。
        """
        if not src_video_path or not dst_output_path:
            return dst_output_path

        src = Path(src_video_path)
        dst = Path(dst_output_path)
        if not src.exists():
            logger.warning(f"无法提升视频：源文件不存在: {src}")
            return dst_output_path

        dst.parent.mkdir(parents=True, exist_ok=True)

        # 若路径不同则复制覆盖；保留 attempt 文件，避免丢失诊断素材
        if src.resolve() != dst.resolve():
            try:
                shutil.copy2(str(src), str(dst))
                logger.info(f"已将通过验证的视频复制为最终输出: {dst}")
            except Exception as e:
                logger.warning(f"复制最终输出失败（将返回源路径）: {e}")
                return src_video_path

        return str(dst)

    def _apply_layered_retry_tuning(
        self,
        scene: Dict[str, Any],
        shot_type: str,
        result: VerificationResult,
    ) -> str:
        """
        分层调参：根据失败类型/严重度调整 scene 与 hunyuanvideo 配置。
        返回（可能更新后的）shot_type。
        """
        identity_cfg = self.video_config.get("identity_verification", {}) or {}
        policy = identity_cfg.get("layered_tuning", {}) or {}

        # 阈值/策略参数（带默认值）
        catastrophic_min_sim = float(policy.get("catastrophic_min_similarity", 0.15))
        high_drift_ratio = float(policy.get("high_drift_ratio", 0.12))
        low_face_ratio = float(policy.get("low_face_detect_ratio", 0.80))
        inc_small = int(policy.get("steps_increase_small", 4))
        inc_large = int(policy.get("steps_increase_large", 8))
        inc_cat = int(policy.get("steps_increase_catastrophic", 12))
        steps_max = int(policy.get("steps_max", 45))
        downgrade_motion = bool(policy.get("downgrade_motion_on_retry", True))

        # 失败类型分层
        catastrophic = (result.min_similarity is not None and result.min_similarity < catastrophic_min_sim)
        drift_heavy = (result.drift_ratio is not None and result.drift_ratio > high_drift_ratio)
        face_low = (result.face_detect_ratio is not None and result.face_detect_ratio < low_face_ratio)

        # 1) prompt 层：更强的“锁脸/稳定”约束
        if catastrophic:
            scene["prompt"] = (
                "same face, no morphing, no face change, stable identity, "
                "avoid large head turn, avoid fast rotation, "
                "keep facial features consistent, "
                + scene.get("prompt", "")
            )
            logger.info("  分层调参: catastrophic(min_similarity低) → 强化锁脸 prompt")
        elif drift_heavy:
            scene["prompt"] = "stable face details, consistent facial features, " + scene.get("prompt", "")
            logger.info("  分层调参: drift_ratio高 → 强化面部一致性 prompt")
        elif face_low:
            scene["prompt"] = "face clearly visible, front-facing, avoid occlusion, " + scene.get("prompt", "")
            logger.info("  分层调参: 人脸检测率低 → 强化“可见脸” prompt")

        # 2) 运动层：漂移/崩脸通常先减运动
        scene["prompt"] = "minimal slow movement, static camera, " + scene.get("prompt", "")
        if downgrade_motion:
            mi = scene.get("motion_intensity")
            if mi == "dynamic":
                scene["motion_intensity"] = "moderate"
                logger.info("  分层调参: motion_intensity dynamic -> moderate")
            elif mi == "moderate" and (catastrophic or drift_heavy):
                scene["motion_intensity"] = "gentle"
                logger.info("  分层调参: motion_intensity moderate -> gentle")

        # 3) 镜头层：高漂移/崩脸/检出差时，优先回到更安全的 medium（避免 close / medium_close）
        if (catastrophic or drift_heavy or face_low) and shot_type in ("close", "extreme_close", "medium_close"):
            shot_type = "medium"
            logger.info("  分层调参: 高风险失败 → shot_type 回退到 medium")

        # 同步 prompt 中的镜头词（避免 prompt 仍然含 close-up 诱导）
        p = scene.get("prompt", "")
        for bad in ["extreme close-up", "extreme close up", "close-up", "close up", "medium close-up", "medium close up"]:
            p = p.replace(bad, "medium shot")
        scene["prompt"] = p

        # 移除高风险运动词（避免残留的 dynamic/fast 强运动描述影响后续重试）
        for bad in ["dynamic action", "fast movement", "energetic", "rapid movement", "strong motion"]:
            scene["prompt"] = scene["prompt"].replace(bad, "")

        # 4) 步数层：严重问题加大步数（更稳定）
        try:
            hv = self.video_config.get("hunyuanvideo", {}) or {}
            cur_steps = int(hv.get("num_inference_steps", 25))
            if catastrophic:
                inc = inc_cat
            elif drift_heavy:
                inc = inc_large
            else:
                inc = inc_small
            new_steps = min(cur_steps + inc, steps_max)
            self.video_config.setdefault("hunyuanvideo", {})
            self.video_config["hunyuanvideo"]["num_inference_steps"] = new_steps
            logger.info(f"  分层调参: num_inference_steps {cur_steps} -> {new_steps}")
        except Exception:
            pass

        # 5) 生成参数层：进一步降低运动噪声（通过 scene['_gen_kwargs'] 传给 VideoGenerator）
        try:
            gen_kwargs = scene.setdefault("_gen_kwargs", {})
            base_mb = float(gen_kwargs.get("motion_bucket_id", self.video_config.get("motion_bucket_id", 1.5)))
            base_noise = float(gen_kwargs.get("noise_aug_strength", self.video_config.get("noise_aug_strength", 0.00025)))
            if catastrophic or drift_heavy:
                gen_kwargs["motion_bucket_id"] = min(base_mb, 1.6)
                gen_kwargs["noise_aug_strength"] = min(base_noise, 0.00025)
                logger.info(
                    f"  分层调参: gen_kwargs motion_bucket_id->{gen_kwargs['motion_bucket_id']}, noise_aug_strength->{gen_kwargs['noise_aug_strength']}"
                )
        except Exception:
            pass

        # 6) guidance_scale 层：极端崩脸时略降 guidance（减少过度牵引导致的人脸畸变风险）
        try:
            hv = self.video_config.get("hunyuanvideo", {}) or {}
            cur_g = float(hv.get("guidance_scale", 7.5))
            if catastrophic:
                new_g = max(6.5, cur_g - 0.5)
            elif drift_heavy:
                new_g = max(7.0, cur_g - 0.3)
            else:
                new_g = cur_g
            if new_g != cur_g:
                self.video_config.setdefault("hunyuanvideo", {})
                self.video_config["hunyuanvideo"]["guidance_scale"] = new_g
                logger.info(f"  分层调参: guidance_scale {cur_g:.2f} -> {new_g:.2f}")
        except Exception:
            pass

        return shot_type
    
    def generate_video_with_identity_check(
        self,
        image_path: str,
        output_path: str,
        reference_image: Optional[str] = None,
        scene: Optional[Dict[str, Any]] = None,
        shot_type: str = "medium",
        enable_verification: bool = True,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Optional[VerificationResult]]:
        """
        生成视频并验证身份一致性
        
        Args:
            image_path: 输入图像路径（Anchor 图）
            output_path: 输出视频路径
            reference_image: 参考图像路径（用于身份验证，如果不传则使用 image_path）
            scene: 场景配置
            shot_type: 镜头类型 (wide/medium/close)
            enable_verification: 是否启用验证
            max_retries: 最大重试次数（覆盖配置）
            **kwargs: 其他传递给 generate_video 的参数
            
        Returns:
            (视频路径, 验证结果)
        """
        # 如果没有指定参考图，使用输入图
        if reference_image is None:
            reference_image = image_path
        
        # 重试次数（注意：max_retries=0 是合法值，表示不重试；不能用 `or`）
        if max_retries is None:
            max_retries = self.identity_config.max_retries
        
        # 从 scene 中获取或构建 prompt
        prompt = ""
        if scene:
            prompt = scene.get("prompt") or scene.get("description") or ""
        
        # 增强 prompt（添加身份稳定性关键词）
        shot_enum = getattr(ShotLanguage, shot_type.upper(), ShotLanguage.MEDIUM)
        enhanced_prompt = ShotLanguageAdvisor.enhance_prompt_for_stability(prompt, shot_enum)
        
        # 更新 scene 中的 prompt
        if scene is None:
            scene = {}
        scene["prompt"] = enhanced_prompt
        
        # 添加稳定性 negative prompt
        stability_negative = ShotLanguageAdvisor.get_negative_prompt_for_stability()
        existing_negative = scene.get("negative_prompt", "")
        if existing_negative:
            scene["negative_prompt"] = f"{existing_negative}, {stability_negative}"
        else:
            scene["negative_prompt"] = stability_negative
        
        retry_count = 0
        best_result = None
        best_video_path = None

        # 让重试真正产生“不同样本”：每次尝试设置不同 seed
        identity_cfg = self.video_config.get("identity_verification", {})
        base_seed = identity_cfg.get("seed_base", 42)
        seed_step = identity_cfg.get("seed_step", 1)
        try:
            base_seed = int(base_seed)
            seed_step = int(seed_step)
        except Exception:
            base_seed, seed_step = 42, 1

        # 记录原始 hunyuan 参数（用于重试时临时调参）
        hunyuan_cfg = self.video_config.get("hunyuanvideo", {})
        original_steps = hunyuan_cfg.get("num_inference_steps")
        original_guidance = hunyuan_cfg.get("guidance_scale")

        # 仅允许透传给 VideoGenerator.generate_video 的参数（避免 **kwargs 误传导致崩溃）
        allowed_kwargs = {"num_frames", "fps", "motion_bucket_id", "noise_aug_strength"}
        passthrough = {k: v for k, v in kwargs.items() if k in allowed_kwargs and v is not None}
        dropped = sorted([k for k in kwargs.keys() if k not in allowed_kwargs])
        if dropped:
            logger.warning(f"忽略不支持的参数: {dropped}")

        # 丢弃级别的兜底重试（hard case 可能靠 seed 才能救回）
        discard_retry_used = 0
        retry_on_discard = bool(identity_cfg.get("retry_on_discard", False))
        try:
            discard_retry_max = int(identity_cfg.get("discard_retry_max", 0) or 0)
        except Exception:
            discard_retry_max = 0
        
        while retry_count <= max_retries:
            # 构建输出路径
            if retry_count > 0:
                base, ext = os.path.splitext(output_path)
                current_output = f"{base}_attempt{retry_count}{ext}"
            else:
                current_output = output_path
            
            logger.info(f"生成视频 (尝试 {retry_count + 1}/{max_retries + 1})")
            logger.info(f"  镜头类型: {shot_type}")
            # 设置本次尝试的 seed（VideoGenerator 会读取 scene['seed']）
            scene["seed"] = base_seed + retry_count * seed_step
            
            try:
                # 合并分层调参写入的生成参数覆盖（例如 motion_bucket_id/noise_aug_strength）
                attempt_kwargs = dict(passthrough)
                extra = scene.get("_gen_kwargs") if isinstance(scene, dict) else None
                if isinstance(extra, dict) and extra:
                    for k, v in extra.items():
                        if k in allowed_kwargs and v is not None:
                            attempt_kwargs[k] = v

                # 调用父类方法生成视频
                video_path = self.generate_video(
                    image_path=image_path,
                    output_path=current_output,
                    scene=scene,
                    **attempt_kwargs
                )
                
                if video_path is None:
                    logger.error("视频生成失败，返回 None")
                    retry_count += 1
                    continue
                
                # 如果不启用验证，直接返回
                if not enable_verification:
                    logger.info("身份验证已禁用，直接返回视频")
                    return video_path, None
                
                # 验证身份一致性
                verifier = self._get_verifier()
                result = verifier.verify_video(
                    video_path=video_path,
                    reference_image=reference_image,
                    shot_type=shot_type,
                    sample_interval=self.verification_sample_interval,
                    max_frames=self.verification_max_frames
                )
                
                # 记录最佳结果
                if best_result is None or result.avg_similarity > best_result.avg_similarity:
                    best_result = result
                    best_video_path = video_path
                
                # 如果通过验证
                if result.passed:
                    logger.info(f"✅ 视频身份验证通过！相似度: {result.avg_similarity:.3f}")
                    final_path = self._promote_video_to_output(video_path, output_path)
                    return final_path, result
                
                # 如果应该丢弃
                if result.should_discard:
                    if retry_on_discard and discard_retry_used < discard_retry_max and retry_count < max_retries:
                        discard_retry_used += 1
                        logger.warning(
                            f"❌ 视频质量过低(丢弃级)，但启用 retry_on_discard：继续重试（{discard_retry_used}/{discard_retry_max}）"
                        )
                        shot_type = self._apply_layered_retry_tuning(scene, shot_type, result)
                        retry_count += 1
                        continue

                    logger.warning("❌ 视频质量过低，停止重试")
                    break
                
                # 准备重试
                if result.should_retry and retry_count < max_retries:
                    logger.info(f"🔄 准备重试...")

                    # 分层调参：按失败类型升级参数
                    shot_type = self._apply_layered_retry_tuning(scene, shot_type, result)

                    # 兼容旧 hint：如果提示要切换为中景，进一步做 prompt 替换
                    if result.retry_hints.get("use_medium_shot"):
                        scene["prompt"] = scene["prompt"].replace("close-up", "medium shot")
                        scene["prompt"] = scene["prompt"].replace("close up", "medium shot")
                        logger.info("  调整: prompt 替换 close-up -> medium shot")
                
                retry_count += 1
                
            except Exception as e:
                logger.error(f"视频生成异常: {e}")
                import traceback
                traceback.print_exc()
                retry_count += 1

        # 恢复原始步数（避免污染后续生成）
        try:
            if original_steps is not None:
                self.video_config.setdefault("hunyuanvideo", {})
                self.video_config["hunyuanvideo"]["num_inference_steps"] = original_steps
            if original_guidance is not None:
                self.video_config.setdefault("hunyuanvideo", {})
                self.video_config["hunyuanvideo"]["guidance_scale"] = original_guidance
        except Exception:
            pass
        
        # 返回最佳结果
        if best_video_path:
            logger.info(f"返回最佳尝试结果: 相似度 {best_result.avg_similarity:.3f}")
            final_path = self._promote_video_to_output(best_video_path, output_path)
            return final_path, best_result
        else:
            logger.error("所有尝试都失败")
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
    
    def unload_verifier(self):
        """卸载验证器"""
        if self._verifier is not None:
            self._verifier.unload()
            self._verifier = None
    
    def unload_all(self):
        """卸载所有模型"""
        self.unload_verifier()
        self.unload_model()


def quick_test():
    """快速测试"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("增强版视频生成器 M6 测试")
    print("=" * 60)
    
    # 检查是否有测试文件
    test_image = Path("reference_image/hanli_mid.jpg")
    
    if not test_image.exists():
        print(f"⚠ 测试图像不存在: {test_image}")
        print("跳过实际生成测试")
    else:
        print(f"✓ 找到测试图像: {test_image}")
    
    # 测试初始化
    print("\n初始化生成器...")
    try:
        generator = EnhancedVideoGeneratorM6("config.yaml")
        print(f"✓ 生成器初始化成功")
        print(f"  身份验证阈值: {generator.identity_config.similarity_threshold}")
        print(f"  最大重试次数: {generator.identity_config.max_retries}")
        
        # 测试镜头建议
        print("\n镜头漂移风险:")
        for shot in ShotLanguage:
            risk = ShotLanguageAdvisor.get_drift_risk(shot)
            print(f"  {shot.value}: {risk}")
        
        generator.unload_all()
        print("\n✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()
