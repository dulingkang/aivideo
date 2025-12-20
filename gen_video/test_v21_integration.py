#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试v2.1系统集成

功能：
1. 测试JSON转换
2. 测试JSON校验
3. 测试Execution Executor
4. 测试适配器集成
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from gen_video.utils.json_v2_to_v21_converter import convert_json_file
from gen_video.utils.execution_validator import validate_json_file
from gen_video.utils.v21_executor_adapter import V21ExecutorAdapter, create_adapter_from_config
from gen_video.utils.execution_executor_v21 import ExecutionConfig, ExecutionMode


def test_full_workflow():
    """测试完整工作流"""
    print("=" * 60)
    print("v2.1系统完整工作流测试")
    print("=" * 60)
    
    # 1. 转换v2 → v2.1-exec
    print("\n步骤1: 转换v2 → v2.1-exec")
    input_json = "lingjie/episode/1.v2.json"
    output_json = "gen_video/test_outputs/1.v21_exec.json"
    
    if not Path(input_json).exists():
        print(f"  ⚠ 输入文件不存在: {input_json}")
        print("  跳过转换测试")
        return
    
    success = convert_json_file(input_json, output_json)
    if not success:
        print("  ✗ 转换失败")
        return
    
    print(f"  ✓ 转换成功: {output_json}")
    
    # 2. 校验JSON
    print("\n步骤2: 校验v2.1-exec JSON")
    is_valid, report = validate_json_file(output_json)
    print(f"  校验结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    if not is_valid:
        print("\n" + report)
        return
    
    # 3. 测试适配器
    print("\n步骤3: 测试适配器集成")
    try:
        adapter = create_adapter_from_config(
            "gen_video/config.yaml",
            execution_mode=ExecutionMode.STRICT
        )
        
        # 加载转换后的JSON
        with open(output_json, 'r', encoding='utf-8') as f:
            episode = json.load(f)
        
        # 测试第一个场景
        if episode.get("scenes"):
            scene = episode["scenes"][0]
            print(f"\n  测试场景 {scene.get('scene_id')}:")
            
            # 准备场景
            legacy_scene = adapter.prepare_scene_for_generation(scene)
            print(f"    ✓ 场景准备完成")
            print(f"    Model: {legacy_scene['generation_policy']['image_model']}")
            print(f"    Identity: {legacy_scene['generation_policy']['identity_engine']}")
            print(f"    Shot: {scene['shot']['type']} (锁定)")
            print(f"    Pose: {scene['pose']['type']} (锁定)")
            
            # 构建Prompt
            prompt = adapter.executor._build_prompt(scene)
            print(f"\n    生成的Prompt:")
            print(f"      {prompt[:100]}...")
            
    except Exception as e:
        print(f"  ✗ 适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ 集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_full_workflow()

