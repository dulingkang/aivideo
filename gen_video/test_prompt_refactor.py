#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt模块重构验证测试

测试重构后的Prompt模块是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_module_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        from prompt import TokenEstimator, PromptParser, PromptOptimizer, PromptBuilder
        print("✓ 所有Prompt模块组件导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_token_estimator():
    """测试Token估算器"""
    print("\n" + "=" * 60)
    print("测试2: Token估算器")
    print("=" * 60)
    
    try:
        import torch
        from prompt import TokenEstimator
        
        device = torch.device("cpu")
        estimator = TokenEstimator(device=device, ascii_only_prompt=False)
        
        # 测试中文文本
        chinese_text = "仙侠风格，韩立，黑色长发，深绿道袍"
        tokens = estimator.estimate(chinese_text)
        print(f"✓ 中文文本Token估算: '{chinese_text}' -> {tokens} tokens")
        
        # 测试英文文本
        english_text = "xianxia fantasy, han li, long black hair, dark green robe"
        tokens = estimator.estimate(english_text)
        print(f"✓ 英文文本Token估算: '{english_text}' -> {tokens} tokens")
        
        return True
    except Exception as e:
        print(f"✗ Token估算器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_parser():
    """测试Prompt解析器"""
    print("\n" + "=" * 60)
    print("测试3: Prompt解析器")
    print("=" * 60)
    
    try:
        from prompt import PromptParser
        
        parser = PromptParser()
        
        # 测试提取第一个关键词
        text1 = "(韩立躺在沙地上:1.6)"
        first_keyword = parser.extract_first_keyword(text1)
        print(f"✓ 提取第一个关键词: '{text1}' -> '{first_keyword}'")
        
        # 测试提取核心关键词
        text2 = "仙侠风格，韩立，黑色长发，深绿道袍，躺在沙地上，感受灵气"
        core_keywords = parser.extract_core_keywords(text2, max_keywords=5)
        print(f"✓ 提取核心关键词: '{text2}' -> '{core_keywords}'")
        
        return True
    except Exception as e:
        print(f"✗ Prompt解析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_optimizer():
    """测试Prompt优化器"""
    print("\n" + "=" * 60)
    print("测试4: Prompt优化器")
    print("=" * 60)
    
    try:
        import torch
        from prompt import TokenEstimator, PromptOptimizer
        
        device = torch.device("cpu")
        estimator = TokenEstimator(device=device, ascii_only_prompt=False)
        optimizer = PromptOptimizer(estimator)
        
        # 测试优化
        parts = [
            "仙侠风格",
            "(韩立，黑色长发，深绿道袍:1.8)",
            "(躺在沙地上:1.6)",
            "(沙漠环境，青灰色天空:1.4)",
            "(中景，正面视角:1.3)"
        ]
        
        optimized = optimizer.smart_optimize_prompt(parts, max_tokens=70, ascii_only_prompt=False)
        print(f"✓ Prompt优化: {len(parts)} 个部分 -> {len(optimized)} 个部分")
        print(f"  优化后: {', '.join(optimized[:3])}...")
        
        return True
    except Exception as e:
        print(f"✗ Prompt优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_builder_basic():
    """测试Prompt构建器（基础功能）"""
    print("\n" + "=" * 60)
    print("测试5: Prompt构建器（基础功能）")
    print("=" * 60)
    
    try:
        import torch
        from prompt import TokenEstimator, PromptParser, PromptOptimizer, PromptBuilder
        
        # 创建模拟的SceneIntentAnalyzer
        class MockIntentAnalyzer:
            def analyze(self, scene):
                return {
                    'primary_entity': {
                        'type': 'character',
                        'keywords': ['韩立', 'han li'],
                        'weight': 1.8
                    },
                    'action_type': 'static',
                    'viewpoint': {
                        'type': 'front',
                        'weight': 1.8,
                        'explicit': False
                    },
                    'emphasis': ['正面视角', '面向镜头'],
                    'exclusions': [],
                    'weight_adjustments': {}
                }
        
        device = torch.device("cpu")
        estimator = TokenEstimator(device=device, ascii_only_prompt=False)
        parser = PromptParser()
        optimizer = PromptOptimizer(estimator)
        intent_analyzer = MockIntentAnalyzer()
        
        # 创建模拟的角色识别函数
        def identify_characters(scene):
            return ['hanli']
        
        def needs_character(scene):
            return True
        
        builder = PromptBuilder(
            token_estimator=estimator,
            parser=parser,
            optimizer=optimizer,
            intent_analyzer=intent_analyzer,
            character_profiles={},
            scene_profiles={},
            ascii_only_prompt=False,
            identify_characters_fn=identify_characters,
            needs_character_fn=needs_character
        )
        
        # 测试构建Prompt
        scene = {
            "title": "韩立躺在沙地上",
            "description": "韩立躺在沙地上，感受灵气",
            "camera": "中景",
            "visual": {
                "character_pose": "lying on sand"
            }
        }
        
        prompt = builder.build(scene=scene, include_character=True)
        print(f"✓ Prompt构建成功")
        print(f"  生成的Prompt: {prompt[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Prompt构建器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_generator_integration():
    """测试ImageGenerator集成"""
    print("\n" + "=" * 60)
    print("测试6: ImageGenerator集成")
    print("=" * 60)
    
    try:
        # 检查ImageGenerator是否可以正确导入
        from image_generator import ImageGenerator
        print("✓ ImageGenerator导入成功")
        
        # 检查build_prompt方法是否存在
        if hasattr(ImageGenerator, 'build_prompt'):
            print("✓ build_prompt方法存在")
        else:
            print("✗ build_prompt方法不存在")
            return False
        
        # 检查prompt_builder属性是否存在
        import inspect
        init_source = inspect.getsource(ImageGenerator.__init__)
        if 'self.prompt_builder' in init_source:
            print("✓ prompt_builder属性在__init__中初始化")
        else:
            print("✗ prompt_builder属性未在__init__中初始化")
            return False
        
        # 检查build_prompt是否委托给PromptBuilder
        build_prompt_source = inspect.getsource(ImageGenerator.build_prompt)
        if 'self.prompt_builder.build' in build_prompt_source:
            print("✓ build_prompt方法已正确委托给PromptBuilder")
        else:
            print("✗ build_prompt方法未委托给PromptBuilder")
            return False
        
        return True
    except Exception as e:
        print(f"✗ ImageGenerator集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Prompt模块重构验证测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_module_imports()))
    results.append(("Token估算器", test_token_estimator()))
    results.append(("Prompt解析器", test_prompt_parser()))
    results.append(("Prompt优化器", test_prompt_optimizer()))
    results.append(("Prompt构建器", test_prompt_builder_basic()))
    results.append(("ImageGenerator集成", test_image_generator_integration()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！重构成功！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，需要修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())








