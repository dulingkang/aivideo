#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试视频身份验证模块"""

import logging
logging.basicConfig(level=logging.INFO)

from video_identity_verifier import (
    VideoIdentityVerifier,
    IdentityVerificationConfig,
    ShotLanguageAdvisor,
    ShotLanguage
)

print("=" * 60)
print("视频身份验证模块测试")
print("=" * 60)

# 测试配置
config = IdentityVerificationConfig()
print(f"\n配置信息:")
print(f"  相似度阈值: {config.similarity_threshold}")
print(f"  丢弃阈值: {config.similarity_discard}")
print(f"  最大重试: {config.max_retries}")

# 测试镜头建议
print(f"\n镜头漂移风险:")
for shot in ShotLanguage:
    risk = ShotLanguageAdvisor.get_drift_risk(shot)
    print(f"  {shot.value}: {risk}")

# 测试 prompt 增强
print(f"\nPrompt 增强测试:")
original = "a woman walking in the park"
enhanced = ShotLanguageAdvisor.enhance_prompt_for_stability(original, ShotLanguage.MEDIUM)
print(f"  原始: {original}")
print(f"  增强: {enhanced[:100]}...")

# 测试 negative prompt
neg = ShotLanguageAdvisor.get_negative_prompt_for_stability()
print(f"\n稳定性 Negative Prompt:")
print(f"  {neg[:100]}...")

# 测试验证器创建
print(f"\n创建验证器...")
verifier = VideoIdentityVerifier(config)
print(f"  ✓ 验证器创建成功")

# 测试加载分析器
print(f"\n加载分析器...")
analyzer = verifier._get_analyzer()
if analyzer:
    print(f"  ✓ 分析器加载成功")
else:
    print(f"  ⚠ 分析器加载失败")

verifier.unload()
print(f"\n✅ 测试完成!")
