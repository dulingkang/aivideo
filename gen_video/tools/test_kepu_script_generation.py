#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试科普脚本生成流程
验证脚本模板、知识库和快速生成工具的集成
"""

import sys
import json
from pathlib import Path

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.kepu_quick_generate import KepuQuickGenerator


def test_template_loading():
    """测试模板加载"""
    print("="*60)
    print("测试1: 模板加载")
    print("="*60)
    
    generator = KepuQuickGenerator()
    
    # 测试加载各个模板
    templates = [
        "universe_template.json",
        "quantum_template.json",
        "earth_template.json",
        "energy_template.json",
        "city_template.json",
        "biology_template.json",
        "ai_template.json"
    ]
    
    success_count = 0
    for template_name in templates:
        template = generator.load_template(template_name)
        if template:
            print(f"  ✅ {template_name} 加载成功")
            print(f"     标题: {template.get('title', 'N/A')}")
            print(f"     场景数: {len(template.get('scenes', []))}")
            success_count += 1
        else:
            print(f"  ❌ {template_name} 加载失败")
    
    print(f"\n模板加载结果: {success_count}/{len(templates)} 成功")
    return success_count == len(templates)


def test_topic_loading():
    """测试选题加载"""
    print("\n" + "="*60)
    print("测试2: 选题加载")
    print("="*60)
    
    generator = KepuQuickGenerator()
    
    # 列出所有选题
    topics = generator.list_topics()
    print(f"总选题数: {len(topics)}")
    
    # 按分类统计
    category_count = {}
    for topic in topics:
        category = topic.get('category_name', '未知')
        category_count[category] = category_count.get(category, 0) + 1
    
    print("\n分类统计:")
    for category, count in sorted(category_count.items()):
        print(f"  {category}: {count}个选题")
    
    # 测试查找选题
    test_topic = "什么是黑洞？"
    found_topic = generator.find_topic(test_topic)
    if found_topic:
        print(f"\n✅ 找到选题: {test_topic}")
        print(f"   分类: {found_topic.get('category_name')}")
        print(f"   难度: {found_topic.get('difficulty')}")
        print(f"   时长: {found_topic.get('duration')}秒")
    else:
        print(f"\n❌ 未找到选题: {test_topic}")
        return False
    
    return len(topics) >= 50  # 至少50个选题


def test_script_generation():
    """测试脚本生成"""
    print("\n" + "="*60)
    print("测试3: 脚本生成")
    print("="*60)
    
    generator = KepuQuickGenerator()
    
    # 测试生成脚本
    test_topics = [
        "什么是黑洞？",
        "量子纠缠是什么？",
        "地球内部结构"
    ]
    
    success_count = 0
    for topic_title in test_topics:
        topic = generator.find_topic(topic_title)
        if not topic:
            print(f"  ❌ 未找到选题: {topic_title}")
            continue
        
        try:
            script = generator.generate_script(topic, ip_character="kepu_gege")
            
            # 验证脚本结构
            required_fields = ['title', 'topic', 'category', 'duration', 'ip_character', 
                             'opening', 'scenes', 'ending', 'metadata']
            missing_fields = [f for f in required_fields if f not in script]
            
            if missing_fields:
                print(f"  ❌ {topic_title} 脚本缺少字段: {missing_fields}")
                continue
            
            # 验证场景数量
            num_scenes = len(script.get('scenes', []))
            expected_duration = script.get('duration', 60)
            content_duration = expected_duration - 12 - 12  # 减去开场和结尾
            expected_scenes = max(2, content_duration // 18)
            
            if num_scenes < 2:
                print(f"  ❌ {topic_title} 场景数量不足: {num_scenes}")
                continue
            
            print(f"  ✅ {topic_title} 脚本生成成功")
            print(f"     标题: {script.get('title')}")
            print(f"     场景数: {num_scenes}")
            print(f"     时长: {script.get('duration')}秒")
            print(f"     开场: {script.get('opening', {}).get('narration', '')[:30]}...")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ {topic_title} 脚本生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n脚本生成结果: {success_count}/{len(test_topics)} 成功")
    return success_count == len(test_topics)


def test_script_saving():
    """测试脚本保存"""
    print("\n" + "="*60)
    print("测试4: 脚本保存")
    print("="*60)
    
    generator = KepuQuickGenerator()
    
    topic = generator.find_topic("什么是黑洞？")
    if not topic:
        print("  ❌ 未找到测试选题")
        return False
    
    script = generator.generate_script(topic, ip_character="kepu_gege")
    
    # 保存脚本
    output_dir = Path(__file__).parent.parent / "outputs" / "kepu_test_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = output_dir / "test_script.json"
    with open(script_path, 'w', encoding='utf-8') as f:
        json.dump(script, f, ensure_ascii=False, indent=2)
    
    if script_path.exists():
        print(f"  ✅ 脚本已保存: {script_path}")
        print(f"     文件大小: {script_path.stat().st_size} 字节")
        return True
    else:
        print(f"  ❌ 脚本保存失败")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("科普脚本生成流程测试")
    print("="*60)
    
    results = []
    
    # 测试1: 模板加载
    results.append(("模板加载", test_template_loading()))
    
    # 测试2: 选题加载
    results.append(("选题加载", test_topic_loading()))
    
    # 测试3: 脚本生成
    results.append(("脚本生成", test_script_generation()))
    
    # 测试4: 脚本保存
    results.append(("脚本保存", test_script_saving()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查")
        return 1


if __name__ == '__main__':
    sys.exit(main())

