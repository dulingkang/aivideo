#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 v2-1.json 文件，验证角色一致性是否正常工作
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from generate_novel_video import NovelVideoGenerator

def test_v2_1_json():
    """测试 v2-1.json 文件"""
    
    # 读取 JSON 文件
    json_path = project_root.parent / "lingjie" / "episode" / "1.v2-1.json"
    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        return False
    
    print(f"📖 读取 JSON 文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get('scenes', [])
    print(f"  ✓ 找到 {len(scenes)} 个场景")
    
    # 初始化生成器
    print("\n🔧 初始化小说推文生成器...")
    generator = NovelVideoGenerator()
    
    # 测试第一个包含韩立的场景
    hanli_scenes = [s for s in scenes if s.get('character', {}).get('id') == 'hanli']
    if not hanli_scenes:
        print("⚠️  未找到包含韩立的场景，测试第一个场景")
        test_scene = scenes[0] if scenes else None
    else:
        print(f"  ✓ 找到 {len(hanli_scenes)} 个包含韩立的场景")
        test_scene = hanli_scenes[0]
    
    if not test_scene:
        print("❌ 没有可测试的场景")
        return False
    
    # 提取场景信息
    character = test_scene.get('character', {})
    character_id = character.get('id')
    character_present = character.get('present', False)
    
    print(f"\n📝 测试场景信息:")
    print(f"  场景ID: {test_scene.get('scene_id')}")
    print(f"  角色ID: {character_id}")
    print(f"  角色出现: {character_present}")
    
    # 构建 prompt（从 visual_constraints 或其他字段）
    visual = test_scene.get('visual_constraints', {})
    environment = visual.get('environment', '')
    narration = test_scene.get('narration', {})
    narration_text = narration.get('text', '')
    
    # 构建 prompt
    prompt_parts = []
    if character_present and character_id == 'hanli':
        prompt_parts.append("韩立")
    if environment:
        prompt_parts.append(environment)
    if narration_text:
        # 提取关键描述
        prompt_parts.append(narration_text[:50])
    
    prompt = ", ".join(prompt_parts) if prompt_parts else "一个仙侠场景"
    print(f"  提示词: {prompt}")
    
    # 检查参考图是否存在
    ref_path = project_root / "reference_image" / "hanli_mid.jpg"
    if ref_path.exists():
        print(f"  ✓ 参考图存在: {ref_path}")
    else:
        print(f"  ⚠ 参考图不存在: {ref_path}")
        print(f"    将使用 ImageGenerator 的自动查找逻辑")
    
    # 测试生成（只生成图片，不生成视频，快速测试）
    print(f"\n🎨 开始生成图片（快速测试）...")
    try:
        result = generator.generate(
            prompt=prompt,
            output_dir=project_root / "outputs" / "test_novel_v2_1",
            width=768,
            height=1152,
            num_frames=24,  # 快速测试，只生成24帧
            fps=24,
            include_character=character_present,
            character_id=character_id,
            auto_character=True,
            enable_m6_identity=False,  # 快速测试，不启用 M6
            shot_type=test_scene.get('camera', {}).get('shot', 'medium'),
            motion_intensity=test_scene.get('quality_target', {}).get('motion_intensity', 'moderate'),
            m6_quick=True,  # 快速模式
        )
        
        print(f"\n✅ 生成成功!")
        print(f"  图片: {result.get('image')}")
        if 'video' in result:
            print(f"  视频: {result.get('video')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_v2_1_json()
    sys.exit(0 if success else 1)

