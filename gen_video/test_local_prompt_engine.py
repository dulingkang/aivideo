#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试本地Prompt Engine（无需LLM API）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.prompt_engine_v2 import PromptEngine, UserRequest


def test_local_mode():
    """测试完全本地模式（无LLM API）"""
    print("=" * 60)
    print("测试：完全本地模式（无需LLM API）")
    print("=" * 60)
    
    # 创建引擎（默认使用SimpleLLMClient，完全本地）
    engine = PromptEngine()
    
    print("\n✅ Prompt Engine已初始化（本地模式）")
    print("   - 使用SimpleLLMClient（规则引擎）")
    print("   - 无需外部LLM API")
    print("   - 完全本地运行")
    
    # 测试用例1：小说推文
    print("\n" + "-" * 60)
    print("测试用例1：小说推文")
    print("-" * 60)
    
    req1 = UserRequest(
        text="那夜他手握长剑，踏入断桥，月光如水洒在剑身上",
        scene_type="novel",
        style="xianxia_v2"
    )
    
    pkg1 = engine.run(req1)
    
    print(f"\n原始文本: {req1.text}")
    print(f"\n重写文本: {pkg1.rewritten_text}")
    print(f"\n最终Prompt: {pkg1.final_prompt}")
    print(f"\n镜头数: {len(pkg1.scene_struct.shots)}")
    for shot in pkg1.scene_struct.shots:
        print(f"  镜头{shot.shot_id}: {shot.description[:50]}... (时长: {shot.duration_secs:.1f}秒)")
    
    # 测试用例2：科学场景
    print("\n" + "-" * 60)
    print("测试用例2：科学场景")
    print("-" * 60)
    
    req2 = UserRequest(
        text="黑洞在太空中旋转，周围有吸积盘发出强烈的X射线",
        scene_type="scientific",
        target_model="hunyuanvideo"
    )
    
    pkg2 = engine.run(req2)
    
    print(f"\n原始文本: {req2.text}")
    print(f"\n最终Prompt: {pkg2.final_prompt}")
    print(f"\n目标模型: {pkg2.model_target}")
    
    # 测试用例3：风景场景
    print("\n" + "-" * 60)
    print("测试用例3：风景场景")
    print("-" * 60)
    
    req3 = UserRequest(
        text="一个美丽的山谷，有瀑布和彩虹，阳光透过云层洒下，远处有雪山，近处有绿树",
        scene_type="general"
    )
    
    pkg3 = engine.run(req3)
    
    print(f"\n原始文本: {req3.text}")
    print(f"\n重写文本: {pkg3.rewritten_text}")
    print(f"\n镜头数: {len(pkg3.scene_struct.shots)}")
    
    # 显示指标
    print("\n" + "=" * 60)
    print("指标统计")
    print("=" * 60)
    metrics = engine.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ 本地模式测试完成")
    print("=" * 60)
    print("\n说明：")
    print("  - 所有处理都在本地完成，无需外部API")
    print("  - 使用规则引擎进行文本重写和场景分解")
    print("  - 支持缓存，提升重复请求的性能")
    print("  - 可以完全离线运行")


if __name__ == "__main__":
    test_local_mode()

