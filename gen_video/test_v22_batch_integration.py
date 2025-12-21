#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final格式批量集成测试

批量测试多个v2.2-final格式JSON文件
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def test_single_json(json_path: str, output_base: Path) -> dict:
    """测试单个JSON文件"""
    json_file = Path(json_path)
    scene_name = json_file.stem
    
    print(f"\n{'='*60}")
    print(f"测试: {scene_name}")
    print(f"{'='*60}")
    
    # 加载JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
    except Exception as e:
        return {
            "scene_name": scene_name,
            "success": False,
            "error": f"JSON加载失败: {e}"
        }
    
    try:
        from execution_executor_v21 import ExecutionExecutorV21, ExecutionConfig, ExecutionMode
        from execution_validator import ExecutionValidator
        
        # 1. 验证JSON
        validator = ExecutionValidator()
        validation_result = validator.validate_scene(scene)
        
        if not validation_result.is_valid:
            return {
                "scene_name": scene_name,
                "success": False,
                "error": f"JSON验证失败: {validation_result.errors_count} 个错误",
                "errors": [issue.message for issue in validation_result.issues if issue.level.value == "error"]
            }
        
        # 2. 格式规范化
        config = ExecutionConfig(mode=ExecutionMode.STRICT)
        executor = ExecutionExecutorV21(config=config)
        normalized = executor._normalize_scene_format(scene)
        
        # 3. Prompt构建
        prompt = executor._build_prompt(normalized)
        negative_prompt = executor._build_negative_prompt(normalized)
        
        # 4. 决策trace
        decision_trace = executor._build_decision_trace(normalized)
        
        # 5. 保存结果
        scene_output_dir = output_base / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存Prompt
        prompt_path = scene_output_dir / "generated_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"Negative Prompt:\n{', '.join(negative_prompt)}\n")
        
        # 保存决策trace
        trace_path = scene_output_dir / "decision_trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(decision_trace, f, ensure_ascii=False, indent=2)
        
        # 复制原始JSON
        json_output_path = scene_output_dir / json_file.name
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ JSON验证: 通过")
        print(f"  ✓ Prompt构建: 成功 ({len(prompt)} 字符)")
        print(f"  ✓ 决策trace: 成功")
        print(f"  ✓ 结果已保存: {scene_output_dir}")
        
        return {
            "scene_name": scene_name,
            "success": True,
            "prompt_length": len(prompt),
            "negative_prompt_count": len(negative_prompt),
            "shot_type": decision_trace["shot"]["type"],
            "pose_type": decision_trace["pose"]["type"],
            "model": decision_trace["model_route"]["base_model"],
            "output_dir": str(scene_output_dir)
        }
        
    except Exception as e:
        import traceback
        return {
            "scene_name": scene_name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="v2.2-final格式批量集成测试")
    parser.add_argument(
        "json_files",
        nargs="+",
        help="JSON文件路径（可以多个）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认: outputs/test_v22_batch_YYYYMMDD_HHMMSS）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(__file__).parent / "outputs" / f"test_v22_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("v2.2-final格式批量集成测试")
    print("=" * 60)
    print(f"\n输出目录: {output_base}")
    print(f"测试文件数: {len(args.json_files)}")
    
    # 测试所有JSON文件
    results = []
    for json_path in args.json_files:
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"\n✗ JSON文件不存在: {json_path}")
            results.append({
                "scene_name": json_file.stem,
                "success": False,
                "error": "文件不存在"
            })
            continue
        
        result = test_single_json(str(json_file), output_base)
        results.append(result)
    
    # 生成汇总报告
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get("success", False))
    total_count = len(results)
    
    print(f"\n总计: {total_count} 个场景")
    print(f"成功: {success_count} 个 ({success_count/total_count*100:.1f}%)")
    print(f"失败: {total_count - success_count} 个")
    
    # 详细结果
    print("\n详细结果:")
    for result in results:
        if result.get("success", False):
            print(f"  ✓ {result['scene_name']}: {result['shot_type']} + {result['pose_type']} -> {result['model']}")
        else:
            print(f"  ✗ {result['scene_name']}: {result.get('error', '未知错误')}")
    
    # 保存汇总报告
    report_path = output_base / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total_count,
            "success": success_count,
            "failed": total_count - success_count,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    # 保存Markdown报告
    md_report_path = output_base / "batch_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# v2.2-final格式批量集成测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().isoformat()}\n\n")
        f.write(f"**测试文件数**: {total_count}\n\n")
        f.write("## 测试结果\n\n")
        f.write(f"- 总计: {total_count}\n")
        f.write(f"- 成功: {success_count} ({success_count/total_count*100:.1f}%)\n")
        f.write(f"- 失败: {total_count - success_count}\n\n")
        f.write("## 详细结果\n\n")
        for result in results:
            if result.get("success", False):
                f.write(f"- ✅ **{result['scene_name']}**: {result['shot_type']} + {result['pose_type']} -> {result['model']}\n")
            else:
                f.write(f"- ❌ **{result['scene_name']}**: {result.get('error', '未知错误')}\n")
    
    print(f"\n✓ 汇总报告已保存:")
    print(f"  - JSON: {report_path}")
    print(f"  - Markdown: {md_report_path}")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())

