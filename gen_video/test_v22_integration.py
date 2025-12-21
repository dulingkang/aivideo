#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式集成测试

使用真实的v2.2-final格式JSON文件进行完整的集成测试
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def load_scene_json(json_path: str) -> dict:
    """加载场景JSON文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_integration_with_json(json_path: str):
    """使用JSON文件进行集成测试"""
    print("=" * 60)
    print("v2.2-final格式集成测试")
    print("=" * 60)
    print(f"\n使用JSON文件: {json_path}")
    
    # 加载JSON
    try:
        scene = load_scene_json(json_path)
        print(f"✓ JSON文件加载成功")
    except Exception as e:
        print(f"✗ JSON文件加载失败: {e}")
        return False
    
    # 创建输出目录
    json_file = Path(json_path)
    output_base = Path(__file__).parent / "outputs" / f"test_v22_integration_{json_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录: {output_base}")
    print(f"生成的图片将保存在: {output_base / 'scene_001' / 'novel_image.png'}")
    
    try:
        from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
        from execution_validator import ExecutionValidator
        
        # 1. 验证JSON
        print("\n" + "=" * 60)
        print("步骤1: JSON验证")
        print("=" * 60)
        validator = ExecutionValidator()
        validation_result = validator.validate_scene(scene)
        
        if not validation_result.is_valid:
            print(f"✗ JSON验证失败: {validation_result.errors_count} 个错误")
            for issue in validation_result.issues:
                if issue.level.value == "error":
                    print(f"  - [{issue.field}] {issue.message}")
                    if issue.suggestion:
                        print(f"    建议: {issue.suggestion}")
            return False
        
        print(f"✓ JSON验证通过")
        if validation_result.warnings_count > 0:
            print(f"  ⚠ 警告: {validation_result.warnings_count} 个")
            for issue in validation_result.issues:
                if issue.level.value == "warning":
                    print(f"    - [{issue.field}] {issue.message}")
        
        # 2. 格式规范化
        print("\n" + "=" * 60)
        print("步骤2: 格式规范化")
        print("=" * 60)
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        
        normalized = executor._normalize_scene_format(scene)
        print(f"✓ 格式规范化成功")
        print(f"  - scene_id: {normalized.get('scene_id')}")
        print(f"  - shot.type: {normalized.get('shot', {}).get('type')}")
        print(f"  - pose.type: {normalized.get('pose', {}).get('type')}")
        print(f"  - character.id: {normalized.get('character', {}).get('id')}")
        
        # 3. Prompt构建
        print("\n" + "=" * 60)
        print("步骤3: Prompt构建")
        print("=" * 60)
        prompt = executor._build_prompt(normalized)
        negative_prompt = executor._build_negative_prompt(normalized)
        
        print(f"✓ Prompt构建成功")
        print(f"  Prompt长度: {len(prompt)} 字符")
        print(f"  Prompt预览: {prompt[:150]}...")
        print(f"  负面词数量: {len(negative_prompt)}")
        
        # 保存Prompt
        prompt_path = output_base / "generated_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("生成的Prompt\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"Negative Prompt:\n{', '.join(negative_prompt)}\n")
        print(f"  ✓ Prompt已保存: {prompt_path}")
        
        # 4. 决策trace
        print("\n" + "=" * 60)
        print("步骤4: 决策trace")
        print("=" * 60)
        decision_trace = executor._build_decision_trace(normalized)
        
        print(f"✓ 决策trace构建成功")
        print(f"  Shot: {decision_trace['shot']['type']} (来源: {decision_trace['shot']['source']})")
        print(f"  Pose: {decision_trace['pose']['type']}")
        print(f"  Model: {decision_trace['model_route']['base_model']} + {decision_trace['model_route']['identity_engine']}")
        print(f"  Character: {decision_trace['character_anchor']['character_id']}")
        
        # 保存决策trace
        trace_path = output_base / "decision_trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(decision_trace, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 决策trace已保存: {trace_path}")
        
        # 5. 检查ImageGenerator
        print("\n" + "=" * 60)
        print("步骤5: ImageGenerator检查")
        print("=" * 60)
        
        image_generator = None
        try:
            from generate_novel_video import NovelVideoGenerator
            import yaml
            
            # 查找config文件
            config_path = Path(__file__).parent.parent / "config.yaml"
            if not config_path.exists():
                config_path = Path(__file__).parent / "config.yaml"
            
            if config_path.exists():
                print(f"  ℹ 找到配置文件: {config_path}")
                generator = NovelVideoGenerator(str(config_path))
                image_generator = generator.image_generator
                print("  ✓ ImageGenerator初始化成功")
            else:
                print(f"  ⚠ 未找到配置文件: {config_path}")
                print("  ℹ 将跳过实际图像生成")
        except Exception as e:
            print(f"  ⚠ ImageGenerator初始化失败: {e}")
            print("  ℹ 将跳过实际图像生成")
        
        # 6. 实际生成（如果ImageGenerator可用）
        if image_generator:
            print("\n" + "=" * 60)
            print("步骤6: 实际图像生成")
            print("=" * 60)
            
            try:
                # 更新执行器，传入ImageGenerator
                executor._image_generator = image_generator
                
                print("  开始生成图像...")
                result = executor.execute_scene(scene, str(output_base))
                
                if result.success:
                    print(f"  ✓ 图像生成成功")
                    print(f"    图像路径: {result.image_path}")
                    if result.image_path and Path(result.image_path).exists():
                        file_size = Path(result.image_path).stat().st_size / 1024
                        print(f"    ✓ 图像文件存在")
                        print(f"    文件大小: {file_size:.2f} KB")
                    else:
                        print(f"    ⚠ 图像文件不存在")
                else:
                    print(f"  ✗ 图像生成失败: {result.error_message}")
                    if result.decision_trace:
                        print(f"    决策trace: {result.decision_trace}")
            except Exception as e:
                print(f"  ✗ 图像生成异常: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n" + "=" * 60)
            print("步骤6: 跳过实际图像生成")
            print("=" * 60)
            print("  ℹ ImageGenerator未初始化，跳过实际生成")
            print(f"  ℹ 输出路径将位于: {output_base / 'scene_001' / 'novel_image.png'}")
        
        # 7. 保存测试结果
        print("\n" + "=" * 60)
        print("步骤7: 保存测试结果")
        print("=" * 60)
        
        # 复制原始JSON到输出目录
        json_output_path = output_base / json_file.name
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 测试JSON已保存: {json_output_path}")
        
        # 生成测试报告
        report_path = output_base / "test_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# v2.2-final格式集成测试报告\n\n")
            f.write(f"**测试时间**: {datetime.now().isoformat()}\n\n")
            f.write(f"**JSON文件**: {json_path}\n\n")
            f.write("## 测试结果\n\n")
            f.write(f"- ✅ JSON验证: 通过\n")
            f.write(f"- ✅ 格式规范化: 通过\n")
            f.write(f"- ✅ Prompt构建: 通过\n")
            f.write(f"- ✅ 决策trace: 通过\n")
            if image_generator:
                f.write(f"- {'✅' if result.success else '❌'} 图像生成: {'通过' if result.success else '失败'}\n")
            else:
                f.write(f"- ⏭️  图像生成: 跳过（ImageGenerator未初始化）\n")
            f.write("\n## Prompt\n\n")
            f.write(f"```\n{prompt}\n```\n\n")
            f.write("## 决策trace\n\n")
            f.write(f"```json\n{json.dumps(decision_trace, ensure_ascii=False, indent=2)}\n```\n")
        print(f"  ✓ 测试报告已保存: {report_path}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print(f"\n输出目录: {output_base}")
        print(f"  - 测试JSON: {json_output_path}")
        print(f"  - Prompt文件: {prompt_path}")
        print(f"  - 决策trace: {trace_path}")
        print(f"  - 测试报告: {report_path}")
        if image_generator and result.success:
            print(f"  - 图像文件: {result.image_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="v2.2-final格式集成测试")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="schemas/scene_v22_real_example.json",
        help="JSON文件路径（默认: schemas/scene_v22_real_example.json）"
    )
    
    args = parser.parse_args()
    
    json_path = Path(__file__).parent / args.json_path
    if not json_path.exists():
        print(f"✗ JSON文件不存在: {json_path}")
        print(f"\n可用的JSON文件:")
        json_files = list(Path(__file__).parent.glob("schemas/scene_v22*.json"))
        for f in json_files:
            print(f"  - {f}")
        sys.exit(1)
    
    success = test_integration_with_json(str(json_path))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

