#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Engine V2 测试脚本
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.prompt_engine_v2 import (
    PromptEngine,
    UserRequest,
    StyleStore,
    MemoryCache,
    SimpleLLMClient
)


def test_basic_usage():
    """测试基础使用"""
    print("=" * 60)
    print("测试1: 基础使用 - 小说推文")
    print("=" * 60)
    
    # 创建引擎
    engine = PromptEngine()
    
    # 创建用户请求
    req = UserRequest(
        text="那夜他手握长剑，踏入断桥，月光如水洒在剑身上",
        scene_type="novel",
        style="xianxia_v2",
        user_tier="professional",
        params={"duration": 5, "fps": 24}
    )
    
    # 执行处理
    pkg = engine.run(req)
    
    # 输出结果
    print(f"\n原始文本: {req.text}")
    print(f"\n重写文本: {pkg.rewritten_text}")
    print(f"\n最终Prompt: {pkg.final_prompt}")
    print(f"\nNegative Prompt: {pkg.negative[:150]}...")
    print(f"\n目标模型: {pkg.model_target}")
    print(f"\n镜头数: {len(pkg.scene_struct.shots)}")
    print(f"\nQA评分: {pkg.metadata.get('qa_score', 0)}/{pkg.metadata.get('qa_max_score', 0)}")
    print(f"\n处理时间: {pkg.metadata.get('processing_time', 0):.2f}秒")
    
    return pkg


def test_scientific_scene():
    """测试科学场景"""
    print("\n" + "=" * 60)
    print("测试2: 科学场景 - HunyuanVideo")
    print("=" * 60)
    
    engine = PromptEngine()
    
    req = UserRequest(
        text="黑洞在太空中旋转，周围有吸积盘发出强烈的X射线",
        scene_type="scientific",
        target_model="hunyuanvideo",
        params={"width": 1280, "height": 720}
    )
    
    pkg = engine.run(req)
    
    print(f"\n最终Prompt: {pkg.final_prompt}")
    print(f"\n目标模型: {pkg.model_target}")
    print(f"\n验证后的参数: {req.params}")
    
    return pkg


def test_cogvideox_adapter():
    """测试CogVideoX适配器"""
    print("\n" + "=" * 60)
    print("测试3: CogVideoX适配器")
    print("=" * 60)
    
    engine = PromptEngine()
    
    req = UserRequest(
        text="一个美丽的山谷，有瀑布和彩虹",
        scene_type="novel",
        target_model="cogvideox",
        params={"width": 1024, "height": 576}  # 会被适配器修正为720x480
    )
    
    pkg = engine.run(req)
    
    print(f"\n最终Prompt: {pkg.final_prompt}")
    print(f"\n目标模型: {pkg.model_target}")
    print(f"\n验证后的参数: {req.params}")
    print(f"  - 宽度: {req.params['width']} (已修正为720)")
    print(f"  - 高度: {req.params['height']} (已修正为480)")
    print(f"  - FPS: {req.params['fps']} (已修正为8)")
    
    return pkg


def test_style_store():
    """测试风格存储"""
    print("\n" + "=" * 60)
    print("测试4: 风格存储")
    print("=" * 60)
    
    # 从文件加载风格
    style_store = StyleStore(config_path="utils/style_templates.yaml")
    
    # 获取不同风格
    styles = ["xianxia_v2", "novel", "scientific", "general"]
    for style_id in styles:
        style = style_store.get(style_id)
        print(f"\n风格: {style_id}")
        print(f"  Pre-prompt: {style.pre_prompt}")
        print(f"  Post-prompt: {style.post_prompt}")
        print(f"  Negative: {', '.join(style.negative_list[:3])}...")


def test_cache():
    """测试缓存功能"""
    print("\n" + "=" * 60)
    print("测试5: 缓存功能")
    print("=" * 60)
    
    engine = PromptEngine()
    
    req = UserRequest(
        text="测试缓存功能的文本",
        scene_type="general"
    )
    
    # 第一次运行（无缓存）
    import time
    start1 = time.time()
    pkg1 = engine.run(req)
    time1 = time.time() - start1
    
    # 第二次运行（有缓存）
    start2 = time.time()
    pkg2 = engine.run(req)
    time2 = time.time() - start2
    
    print(f"\n第一次运行时间: {time1:.3f}秒")
    print(f"第二次运行时间: {time2:.3f}秒")
    print(f"加速比: {time1/time2:.2f}x")
    print(f"\n缓存命中数: {engine.metrics.get('cache_hits', 0)}")


def test_metrics():
    """测试指标收集"""
    print("\n" + "=" * 60)
    print("测试6: 指标收集")
    print("=" * 60)
    
    engine = PromptEngine()
    
    # 运行多个请求
    for i in range(3):
        req = UserRequest(
            text=f"测试请求 {i+1}",
            scene_type="general"
        )
        engine.run(req)
    
    # 显示指标
    metrics = engine.get_metrics()
    print("\n指标统计:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Prompt Engine V2 完整测试")
    print("=" * 60)
    
    try:
        # 测试1: 基础使用
        test_basic_usage()
        
        # 测试2: 科学场景
        test_scientific_scene()
        
        # 测试3: CogVideoX适配器
        test_cogvideox_adapter()
        
        # 测试4: 风格存储
        test_style_store()
        
        # 测试5: 缓存功能
        test_cache()
        
        # 测试6: 指标收集
        test_metrics()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

