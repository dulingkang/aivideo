#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 JSON 文件中的英文 prompt 字段替换为中文 description

策略：
- 如果 description 是中文，直接使用它作为 prompt
- 这样可以包含更多信息（中文用更少的 tokens 表达更多内容）

使用方法:
    # 预览模式（查看会修改什么）:
    python convert_prompts_to_chinese.py --input-dir lingjie/scenes/ --dry-run
    
    # 实际转换（会自动备份）:
    python convert_prompts_to_chinese.py --input-dir lingjie/scenes/
"""

import json
import argparse
from pathlib import Path
import re
import shutil


def has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def main():
    parser = argparse.ArgumentParser(description="批量将 JSON 文件中的 prompt 字段替换为中文 description")
    parser.add_argument("--input", type=str, help="输入 JSON 文件路径（单个文件）")
    parser.add_argument("--input-dir", type=str, help="输入目录（批量处理所有 JSON 文件）")
    parser.add_argument("--backup", action="store_true", default=True, help="转换前备份原文件（默认启用）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式：只显示会做的更改，不实际修改文件")
    
    args = parser.parse_args()
    
    # 确定要处理的文件列表
    files_to_process = []
    if args.input:
        files_to_process.append(Path(args.input))
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        files_to_process.extend(sorted(input_dir.glob("*.json")))
    else:
        print("错误: 请指定 --input 或 --input-dir")
        print("\n示例:")
        print("  # 预览模式（查看会修改什么）:")
        print("  python convert_prompts_to_chinese.py --input-dir lingjie/scenes/ --dry-run")
        print("\n  # 实际转换（会自动备份）:")
        print("  python convert_prompts_to_chinese.py --input-dir lingjie/scenes/")
        return
    
    print(f"找到 {len(files_to_process)} 个 JSON 文件")
    if args.dry_run:
        print("⚠ 预览模式：不会实际修改文件\n")
    
    total_scenes = 0
    converted_scenes = 0
    
    for file_path in files_to_process:
        print(f"{'='*70}")
        print(f"处理文件: {file_path.name}")
        print(f"{'='*70}")
        
        # 读取 JSON 文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ 读取文件失败: {e}")
            continue
        
        # 备份原文件
        if args.backup and not args.dry_run:
            backup_path = file_path.with_suffix('.json.bak')
            shutil.copy2(file_path, backup_path)
            print(f"  ✓ 已备份到: {backup_path.name}")
        
        # 转换场景数据
        file_converted = 0
        if "scenes" in data and isinstance(data["scenes"], list):
            for scene in data["scenes"]:
                total_scenes += 1
                scene_id = scene.get("id", "?")
                original_prompt = scene.get("prompt", "")
                description = scene.get("description", "")
                
                # 检查 prompt 是否需要转换
                if original_prompt:
                    # 如果 prompt 不是中文，且 description 是中文，则替换
                    if not has_chinese(original_prompt) and description and has_chinese(description):
                        print(f"\n  场景 {scene_id}:")
                        print(f"    原 prompt (英文): {original_prompt[:70]}...")
                        print(f"    新 prompt (中文): {description}")
                        
                        if not args.dry_run:
                            scene["prompt"] = description
                        
                        converted_scenes += 1
                        file_converted += 1
                    elif has_chinese(original_prompt):
                        # prompt 已经是中文，无需转换
                        pass
                    else:
                        # prompt 是英文但 description 不是中文或无 description
                        print(f"\n  场景 {scene_id}:")
                        print(f"    ⚠ prompt 是英文，但 description 不是中文或无 description")
                        print(f"      prompt: {original_prompt[:70]}...")
                        print(f"      description: {description[:70] if description else '(无)'}...")
        
        # 保存文件（如果不是预览模式且有修改）
        if not args.dry_run and file_converted > 0:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"\n  ✓ 文件已保存: {file_path.name} (转换了 {file_converted} 个场景的 prompt)")
            except Exception as e:
                print(f"  ✗ 保存文件失败: {e}")
        elif file_converted == 0:
            print(f"  ℹ 无需转换（所有 prompt 都已经是中文或没有可用的中文 description）")
        print()
    
    print(f"{'='*70}")
    print(f"处理完成：")
    print(f"  - 处理文件: {len(files_to_process)} 个")
    print(f"  - 总场景数: {total_scenes} 个")
    print(f"  - 转换场景: {converted_scenes} 个")
    print(f"{'='*70}")
    
    if args.dry_run:
        print("\n提示:")
        print("  - 使用 --backup 参数可以在修改前自动备份原文件（默认已启用）")
        print("  - 去掉 --dry-run 参数后才会实际修改文件")
    elif converted_scenes > 0:
        print("\n提示:")
        print("  - 原文件已自动备份为 .bak 文件")
        print("  - 所有 prompt 字段已替换为中文 description")


if __name__ == "__main__":
    main()