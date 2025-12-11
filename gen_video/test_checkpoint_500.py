#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 checkpoint-500 LoRA 效果
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import ModelManager

def test_checkpoint_500():
    """测试 checkpoint-500 LoRA"""
    print("=" * 70)
    print("🧪 测试 checkpoint-500 LoRA")
    print("=" * 70)
    
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
        
        # 测试生成
        print("\n🎨 开始测试生成...")
        print("   提示词: 科普主持人，专业形象，微笑")
        print("   任务: host_face")
        print("   尺寸: 1024x1024")
        
        result = manager.generate(
            task="host_face",
            prompt="科普主持人，专业形象，微笑",
            width=1024,
            height=1024,
            num_inference_steps=40,
            seed=42
        )
        
        if result and 'image_path' in result:
            print(f"\n✅ 生成成功！")
            print(f"   图片路径: {result['image_path']}")
            print(f"   使用模型: {result.get('model_used', 'unknown')}")
            return True
        else:
            print(f"\n❌ 生成失败")
            print(f"   结果: {result}")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_checkpoint_500()
    sys.exit(0 if success else 1)


