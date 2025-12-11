#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试科学主持人形象生成效果
验证LoRA和character_profiles配置是否正常工作
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import ModelManager

def test_host_person():
    """测试科学主持人形象生成"""
    print("=" * 70)
    print("🧪 测试科学主持人形象生成效果")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path("outputs/test_host_person")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化 ModelManager
        print("\n📦 初始化 ModelManager...")
        manager = ModelManager()
        
        # 检查 LoRA 配置
        lora_config = manager.lora_configs.get('host_face', {})
        lora_path = lora_config.get('lora_path')
        lora_alpha = lora_config.get('lora_alpha', 1.0)
        
        if not lora_path:
            print("❌ 未找到 LoRA 配置")
            return False
        
        print(f"\n✅ LoRA 配置:")
        print(f"   路径: {lora_path}")
        print(f"   Alpha: {lora_alpha}")
        
        # 检查文件是否存在
        if not Path(lora_path).exists():
            print(f"❌ LoRA 文件不存在: {lora_path}")
            return False
        
        print(f"✅ LoRA 文件存在")
        
        # 检查角色配置
        host_profile = manager.character_profiles.get('host_person', {})
        if not host_profile:
            print("⚠️  未找到 host_person 角色配置，将使用默认提示词")
        else:
            print(f"\n✅ 角色配置已加载:")
            print(f"   角色名: {host_profile.get('character_name', 'N/A')}")
            print(f"   身份: {host_profile.get('identity', 'N/A')}")
        
        # 测试场景列表
        test_scenes = [
            {
                "name": "演播室正面",
                "prompt": "科普主持人，专业形象，微笑，正式着装，演播室背景，正面",
                "description": "标准演播室场景，正面角度"
            },
            {
                "name": "演播室半身",
                "prompt": "科普主持人，专业形象，温和表情，商务正装，半身，演播室背景",
                "description": "演播室场景，半身像"
            },
            {
                "name": "纯色背景",
                "prompt": "科普主持人，专业形象，自信微笑，正式西装，正面，纯色背景",
                "description": "纯色背景，突出人物"
            },
            {
                "name": "户外自然光",
                "prompt": "科普主持人，专业形象，自然微笑，正式着装，户外场景，自然光",
                "description": "户外场景，自然光线"
            },
            {
                "name": "45度角",
                "prompt": "科普主持人，专业形象，温和，商务正装，45度角，专业摄影",
                "description": "45度角，专业摄影"
            }
        ]
        
        print(f"\n🎨 开始生成测试图像（共 {len(test_scenes)} 张）...")
        print("-" * 70)
        
        success_count = 0
        failed_scenes = []
        
        for i, scene in enumerate(test_scenes, 1):
            print(f"\n[{i}/{len(test_scenes)}] 生成: {scene['name']}")
            print(f"   描述: {scene['description']}")
            print(f"   提示词: {scene['prompt']}")
            
            try:
                image = manager.generate(
                    task="host_face",
                    prompt=scene['prompt'],
                    width=1024,
                    height=1024,
                    num_inference_steps=40,
                    seed=42 + i  # 使用不同的种子
                )
                
                if image:
                    # 保存图像
                    new_name = f"host_person_{i:02d}_{scene['name']}.png"
                    new_path = output_dir / new_name
                    image.save(new_path)
                    
                    print(f"   ✅ 生成成功: {new_path}")
                    success_count += 1
                else:
                    print(f"   ❌ 生成失败: 返回None")
                    failed_scenes.append(scene['name'])
                    
            except Exception as e:
                print(f"   ❌ 生成异常: {e}")
                failed_scenes.append(scene['name'])
        
        # 总结
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)
        print(f"✅ 成功: {success_count}/{len(test_scenes)}")
        print(f"❌ 失败: {len(failed_scenes)}/{len(test_scenes)}")
        
        if failed_scenes:
            print(f"\n失败的场景:")
            for scene_name in failed_scenes:
                print(f"  - {scene_name}")
        
        print(f"\n📁 输出目录: {output_dir.absolute()}")
        print(f"   所有测试图像已保存到该目录")
        
        # 生成测试报告
        report_path = output_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("科学主持人形象生成测试报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LoRA路径: {lora_path}\n")
            f.write(f"LoRA Alpha: {lora_alpha}\n")
            f.write(f"\n测试结果:\n")
            f.write(f"  成功: {success_count}/{len(test_scenes)}\n")
            f.write(f"  失败: {len(failed_scenes)}/{len(test_scenes)}\n")
            if failed_scenes:
                f.write(f"\n失败的场景:\n")
                for scene_name in failed_scenes:
                    f.write(f"  - {scene_name}\n")
        
        print(f"\n📄 测试报告已保存: {report_path}")
        
        return success_count == len(test_scenes)
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_host_person()
    sys.exit(0 if success else 1)

