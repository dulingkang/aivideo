#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预览v2.1-exec转换结果（不依赖完整环境）

用于在运行实际生成前，验证JSON转换是否正确
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

def preview_conversion(json_path: str, start: int = 1, end: int = 3):
    """预览转换结果"""
    print("=" * 60)
    print("v2.1-exec转换预览")
    print("=" * 60)
    
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"✗ JSON文件不存在: {json_path}")
        return False
    
    # 加载JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get('scenes', [])
    print(f"\n✓ 加载了 {len(scenes)} 个场景")
    
    # 转换指定范围的场景
    scenes_to_convert = scenes[start:end]
    print(f"\n转换范围: {start} - {end-1} (共 {len(scenes_to_convert)} 个场景)")
    
    try:
        from json_v2_to_v21_converter import JSONV2ToV21Converter
        converter = JSONV2ToV21Converter()
        
        converted_scenes = []
        for i, scene in enumerate(scenes_to_convert):
            scene_id = scene.get('scene_id', start + i)
            scene_version = scene.get('version', '')
            
            print(f"\n场景 {scene_id}:")
            print(f"  原始版本: {scene_version}")
            
            if scene_version == 'v2':
                # 转换
                v21_scene = converter.convert_scene(scene)
                converted_scenes.append(v21_scene)
                
                print(f"  ✓ 转换成功")
                print(f"    Shot: {v21_scene['shot']['type']} (锁定: {v21_scene['shot']['locked']})")
                print(f"    Pose: {v21_scene['pose']['type']} (修正: {v21_scene['pose'].get('auto_corrected', False)})")
                print(f"    Model: {v21_scene['model_route']['base_model']} + {v21_scene['model_route']['identity_engine']}")
                print(f"    决策原因: {v21_scene['model_route'].get('decision_reason', 'N/A')}")
                
                # 检查角色锚
                character = v21_scene.get('character', {})
                if character.get('present'):
                    print(f"    角色: {character.get('id')} (性别: {character.get('gender')})")
            elif scene_version.startswith('v2.1'):
                print(f"  ✓ 已是v2.1-exec格式，无需转换")
                converted_scenes.append(scene)
            else:
                print(f"  ⚠ 未知版本: {scene_version}，跳过转换")
                converted_scenes.append(scene)
        
        # 保存转换结果（可选）
        output_file = Path(__file__).parent / "test_outputs" / f"preview_v21_{start}_{end}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        preview_data = {
            "episode": data.get("episode", 1),
            "title": data.get("title", ""),
            "scenes": converted_scenes,
            "conversion_info": {
                "original_file": str(json_path),
                "converted_count": len([s for s in converted_scenes if s.get('version', '').startswith('v2.1')]),
                "range": f"{start}-{end-1}"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(preview_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 预览结果已保存: {output_file}")
        print(f"\n总结:")
        print(f"  转换场景数: {len(converted_scenes)}")
        print(f"  v2.1-exec格式: {len([s for s in converted_scenes if s.get('version', '').startswith('v2.1')])}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 test_v21_batch_preview.py <json_path> [start] [end]")
        print("示例: python3 test_v21_batch_preview.py ../lingjie/episode/1.v2-1.json 1 3")
        sys.exit(1)
    
    json_path = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    success = preview_conversion(json_path, start, end)
    sys.exit(0 if success else 1)

