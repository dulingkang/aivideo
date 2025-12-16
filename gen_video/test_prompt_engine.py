#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Engine测试脚本
测试专业级Prompt工程系统的各个模块
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.prompt_engine import (
    PromptEngine,
    PromptRewriter,
    SceneDecomposer,
    StyleController,
    CameraEngine,
    NegativePromptGenerator,
    PromptQA
)


def test_prompt_rewriter():
    """测试Prompt Rewriter"""
    print("=" * 60)
    print("测试1: Prompt Rewriter（Prompt重写器）")
    print("=" * 60)
    
    rewriter = PromptRewriter()
    
    test_cases = [
        ("一个男人在雪地里走路", "novel"),
        ("科学家在实验室工作", "scientific"),
        ("黑洞旋转", "scientific"),
    ]
    
    for user_input, scene_type in test_cases:
        rewritten = rewriter.rewrite(user_input, scene_type)
        print(f"\n输入: {user_input}")
        print(f"场景类型: {scene_type}")
        print(f"输出: {rewritten}")
        print("-" * 60)


def test_scene_decomposer():
    """测试Scene Decomposer"""
    print("\n" + "=" * 60)
    print("测试2: Scene Decomposer（场景语义解析器）")
    print("=" * 60)
    
    decomposer = SceneDecomposer()
    
    test_cases = [
        ("一个男人在雪地里走路", None),
        ("科学家在实验室工作", {
            "type": "scientific",
            "description": "a scientist working in a lab",
            "motion": {"type": "working"},
            "visual": {"lighting": "dramatic", "style": "scientific"}
        }),
    ]
    
    for user_input, scene in test_cases:
        components = decomposer.decompose(user_input, scene)
        prompt = decomposer.to_prompt(components)
        print(f"\n输入: {user_input}")
        print(f"场景: {scene}")
        print(f"解析结果:")
        print(f"  - 镜头: {components.shot}")
        print(f"  - 主体: {components.subject}")
        print(f"  - 动作: {components.action}")
        print(f"  - 环境: {components.environment}")
        print(f"  - 情绪: {components.emotion}")
        print(f"组合Prompt: {prompt}")
        print("-" * 60)


def test_style_controller():
    """测试Style Controller"""
    print("\n" + "=" * 60)
    print("测试3: Style Controller（风格控制器）")
    print("=" * 60)
    
    controller = StyleController()
    
    scene_types = ["novel", "scientific", "government", "enterprise", "chinese_modern"]
    
    for scene_type in scene_types:
        style = controller.get_style(scene_type)
        lighting = controller.get_lighting(scene_type)
        composition = controller.get_composition(scene_type)
        
        print(f"\n场景类型: {scene_type}")
        print(f"  风格描述: {style.get('description', 'N/A')}")
        print(f"  光线: {lighting}")
        print(f"  构图: {composition}")
        
        # 测试应用风格
        base_prompt = "a scene"
        styled = controller.apply_style(base_prompt, scene_type)
        print(f"  应用风格后: {styled}")
        print("-" * 60)


def test_camera_engine():
    """测试Camera Engine"""
    print("\n" + "=" * 60)
    print("测试4: Camera Engine（相机语言引擎）")
    print("=" * 60)
    
    engine = CameraEngine()
    
    test_configs = [
        {"shot_type": "wide", "movement": "static", "viewpoint": "third_person"},
        {"shot_type": "close", "movement": "push_in", "viewpoint": "third_person"},
        {"shot_type": "medium", "movement": "pan", "viewpoint": "aerial"},
    ]
    
    for config in test_configs:
        camera_prompt = engine.generate_camera_prompt(**config)
        print(f"\n配置: {config}")
        print(f"相机语言: {camera_prompt}")
        
        # 测试增强prompt
        base_prompt = "a beautiful scene"
        enhanced = engine.enhance_prompt_with_camera(base_prompt, config)
        print(f"增强后: {enhanced}")
        print("-" * 60)


def test_negative_prompt_generator():
    """测试Negative Prompt Generator"""
    print("\n" + "=" * 60)
    print("测试5: Negative Prompt Generator（反向提示词生成器）")
    print("=" * 60)
    
    generator = NegativePromptGenerator()
    
    model_types = ["hunyuanvideo", "cogvideox", "svd", "flux"]
    scene_types = [None, "scientific", "novel", "government"]
    
    for model_type in model_types:
        for scene_type in scene_types:
            negative = generator.generate(model_type=model_type, scene_type=scene_type)
            print(f"\n模型: {model_type}, 场景: {scene_type or 'general'}")
            print(f"Negative Prompt: {negative[:100]}...")
            print("-" * 60)


def test_prompt_qa():
    """测试Prompt QA"""
    print("\n" + "=" * 60)
    print("测试6: Prompt QA（质量评分器）")
    print("=" * 60)
    
    qa = PromptQA()
    
    test_prompts = [
        "a scene",  # 太短
        "a man walking in snow",  # 缺少细节
        "Cinematic scene, a young scientist walking slowly toward a quantum reactor in a high-tech futuristic lab with blue accent lights, calm but determined emotion, wide establishing shot, slow camera dolly forward, shallow depth of field, 35mm lens, cinematic lighting, high quality, detailed",  # 完整
    ]
    
    for prompt in test_prompts:
        result = qa.check(prompt)
        print(f"\nPrompt: {prompt[:80]}...")
        print(f"评分: {result['score']}/{result['max_score']}")
        print(f"缺失字段: {result['missing_fields']}")
        print(f"有质量关键词: {result['has_quality_keywords']}")
        print(f"词数: {result['word_count']}")
        if result['suggestions']:
            print(f"建议: {', '.join(result['suggestions'])}")
        print("-" * 60)


def test_full_prompt_engine():
    """测试完整的Prompt Engine流程"""
    print("\n" + "=" * 60)
    print("测试7: 完整Prompt Engine流程")
    print("=" * 60)
    
    engine = PromptEngine()
    
    test_cases = [
        {
            "user_input": "一个男人在雪地里走路",
            "scene": {
                "type": "novel",
                "description": "a man walking in snow",
                "motion_intensity": "gentle",
                "camera_motion": {"type": "pan"},
                "visual": {
                    "composition": "wide shot",
                    "lighting": "soft",
                    "style": "cinematic"
                }
            },
            "model_type": "cogvideox",
            "scene_type": "novel"
        },
        {
            "user_input": "科学家在实验室工作",
            "scene": {
                "type": "scientific",
                "description": "a scientist working in a lab",
                "motion_intensity": "moderate",
                "camera_motion": {"type": "static"},
                "visual": {
                    "composition": "medium shot",
                    "lighting": "dramatic",
                    "style": "scientific"
                }
            },
            "model_type": "hunyuanvideo",
            "scene_type": "scientific"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n【测试用例 {i}】")
        print(f"输入: {test_case['user_input']}")
        print(f"场景类型: {test_case['scene_type']}")
        print(f"模型类型: {test_case['model_type']}")
        
        result = engine.process(
            user_input=test_case['user_input'],
            scene=test_case['scene'],
            model_type=test_case['model_type'],
            scene_type=test_case['scene_type'],
            return_components=False
        )
        
        print(f"\n最终Prompt:")
        print(f"  {result['prompt']}")
        print(f"\nNegative Prompt:")
        print(f"  {result['negative_prompt'][:100]}...")
        print(f"\nQA评分: {result['qa_result']['score']}/{result['qa_result']['max_score']}")
        if result['qa_result'].get('suggestions'):
            print(f"建议: {', '.join(result['qa_result']['suggestions'][:2])}")
        print("=" * 60)


def test_quick_process():
    """测试快速处理"""
    print("\n" + "=" * 60)
    print("测试8: 快速处理（只返回prompt和negative_prompt）")
    print("=" * 60)
    
    engine = PromptEngine()
    
    prompt, negative = engine.quick_process(
        "黑洞旋转",
        scene_type="scientific",
        model_type="hunyuanvideo"
    )
    
    print(f"Prompt: {prompt}")
    print(f"Negative: {negative[:100]}...")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Prompt Engine 完整测试")
    print("=" * 60)
    
    # 运行所有测试
    test_prompt_rewriter()
    test_scene_decomposer()
    test_style_controller()
    test_camera_engine()
    test_negative_prompt_generator()
    test_prompt_qa()
    test_full_prompt_engine()
    test_quick_process()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

