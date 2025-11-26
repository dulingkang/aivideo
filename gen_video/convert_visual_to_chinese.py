#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将 JSON 文件中的 visual 字段转换为中文

策略：
- 根据 description 智能提取不同部分，分别填入不同的 visual 字段
- composition: 构图描述（画面布局、主体与背景关系）
- environment: 环境描述（场景、天气、光线）
- character_pose: 角色姿势（动作、表情、姿态）
- fx: 特效（粒子、光效、扭曲等）

使用方法:
    # 预览模式（查看会修改什么）:
    python convert_visual_to_chinese.py --input-dir lingjie/scenes/ --dry-run
    
    # 实际转换（会自动备份）:
    python convert_visual_to_chinese.py --input-dir lingjie/scenes/
"""

import json
import argparse
from pathlib import Path
from typing import Dict
import re
import shutil


def has_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def extract_visual_fields_from_description(description: str) -> Dict[str, str]:
    """
    从中文 description 中智能提取信息，分别填入不同的 visual 字段
    
    Returns:
        dict: 包含 composition, environment, character_pose, fx 的字典
    """
    if not description or not has_chinese(description):
        return {}
    
    result = {}
    
    # 1. 提取 environment（环境描述）：场景、天气、光线
    # 关键词：沙漠、沙地、天空、太阳、月亮、宫殿、房间、森林、山谷、山峰、热、冷、暗、亮
    env_keywords = [
        "沙漠", "沙地", "沙砾", "沙粒", "天空", "太阳", "月亮", "星辰",
        "宫殿", "房间", "森林", "山谷", "山峰", "河流", "湖泊",
        "热", "燥热", "冷", "寒", "暗", "亮", "黯淡", "明亮",
        "夜晚", "白天", "黄昏", "黎明", "地面"
    ]
    
    env_parts = []
    for keyword in env_keywords:
        if keyword in description:
            # 找到关键词的位置，提取完整短语（从关键词向前找到句首或逗号，向后找到句末或逗号）
            idx = description.find(keyword)
            # 向前找句首
            start = idx
            while start > 0 and description[start-1] not in "，。、；：！？\n":
                start -= 1
            # 向后找句末
            end = idx + len(keyword)
            while end < len(description) and description[end] not in "，。、；：！？\n":
                end += 1
            phrase = description[start:end].strip()
            if phrase and phrase not in env_parts:
                env_parts.append(phrase)
    
    if env_parts:
        result["environment"] = ", ".join(env_parts[:2])  # 最多2个短语，避免过长
    
    # 2. 提取 character_pose（角色姿势）：动作、表情、姿态
    pose_keywords = [
        "躺着", "站着", "坐着", "蹲着", "跪着",
        "看着", "注视", "凝视", "仰望", "俯视",
        "回忆", "思考", "想", "脸色", "表情",
        "一动不动", "静止", "不动",
        "偏动", "转头", "回头", "睁大", "眯着", "皱眉",
        "韩立"  # 包含角色名的描述通常是角色姿势相关
    ]
    
    pose_parts = []
    for keyword in pose_keywords:
        if keyword in description:
            idx = description.find(keyword)
            # 提取包含关键词的完整短语
            start = idx
            while start > 0 and description[start-1] not in "，。、；：！？\n":
                start -= 1
            end = idx + len(keyword)
            while end < len(description) and description[end] not in "，。、；：！？\n":
                end += 1
            phrase = description[start:end].strip()
            if phrase and phrase not in pose_parts:
                pose_parts.append(phrase)
    
    if pose_parts:
        result["character_pose"] = ", ".join(pose_parts[:2])  # 最多2个短语
    
    # 3. 提取 fx（特效）：光效、粒子、扭曲
    fx_keywords = [
        "光芒", "灵光", "蓝芒", "红光", "绿光", "金光",
        "沙粒", "热浪", "火焰", "闪电", "迷雾", "波纹",
        "闪烁", "闪动", "发光", "流转", "闪动"
    ]
    
    fx_parts = []
    for keyword in fx_keywords:
        if keyword in description:
            idx = description.find(keyword)
            start = idx
            while start > 0 and description[start-1] not in "，。、；：！？\n":
                start -= 1
            end = idx + len(keyword)
            while end < len(description) and description[end] not in "，。、；：！？\n":
                end += 1
            phrase = description[start:end].strip()
            if phrase and phrase not in fx_parts:
                fx_parts.append(phrase)
    
    if fx_parts:
        result["fx"] = ", ".join(fx_parts[:2])  # 最多2个短语
    
    # 4. 提取 composition（构图）：整体画面布局、主体与背景关系
    # composition 应该包含整体画面描述，优先使用 description 的核心部分
    # 如果 description 不太长，直接使用；如果太长，提取前半部分（通常是主体描述）
    if len(description) <= 40:
        result["composition"] = description
    else:
        # 提取前半部分（通常是主体描述）
        # 尝试找到第一个逗号或句号，作为截断点
        truncate_pos = min(description.find("，"), description.find("。"))
        if truncate_pos > 0 and truncate_pos < 40:
            result["composition"] = description[:truncate_pos+1]
        else:
            result["composition"] = description[:35] + "..."
    
    return result


def main():
    parser = argparse.ArgumentParser(description="批量将 JSON 文件中的 visual 字段转换为中文")
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
        print("  python convert_visual_to_chinese.py --input-dir lingjie/scenes/ --dry-run")
        print("\n  # 实际转换（会自动备份）:")
        print("  python convert_visual_to_chinese.py --input-dir lingjie/scenes/")
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
                visual = scene.get("visual", {}) or {}
                
                if not isinstance(visual, dict):
                    continue
                
                description = scene.get("description", "")
                prompt = scene.get("prompt", "")
                
                # 只有当 description 或 prompt 是中文时，才转换 visual 字段
                has_chinese_desc = description and has_chinese(description)
                has_chinese_prompt = prompt and has_chinese(prompt)
                
                if has_chinese_desc or has_chinese_prompt:
                    # 检查 visual 字段是否需要转换：
                    # 1. 有英文内容需要转换
                    # 2. 或者虽然是中文但所有字段都相同（说明之前是简单复制，需要重新提取）
                    has_english_in_visual = False
                    all_fields_same = False
                    
                    # 检查是否有英文内容
                    for field_name in ["environment", "composition", "character_pose", "fx"]:
                        if field_name in visual and visual[field_name]:
                            if not has_chinese(str(visual[field_name])):
                                has_english_in_visual = True
                                break
                    
                    # 检查是否所有字段都相同（需要重新提取）
                    visual_values = []
                    for field_name in ["composition", "environment", "character_pose", "fx"]:
                        if field_name in visual and visual[field_name]:
                            visual_values.append(str(visual[field_name]).strip())
                    
                    # 如果所有非空字段都相同，需要重新提取
                    if len(visual_values) > 1 and len(set(visual_values)) == 1:
                        all_fields_same = True
                        print(f"\n  场景 {scene_id}:")
                        print(f"    ⚠ 检测到 visual 字段内容相同，将重新智能提取")
                    
                    if has_english_in_visual or all_fields_same:
                        if not all_fields_same:  # 如果是 all_fields_same，前面已经打印过了
                            print(f"\n  场景 {scene_id}:")
                        print(f"    description: {description[:60] if description else '(无)'}...")
                        
                        # 使用 description 智能提取 visual 字段
                        reference_text = description if has_chinese_desc else prompt
                        extracted = extract_visual_fields_from_description(reference_text)
                        
                        # 更新 visual 字段
                        if extracted:
                            for field_name in ["composition", "environment", "character_pose", "fx"]:
                                if field_name in extracted:
                                    old_value = visual.get(field_name, "")
                                    new_value = extracted[field_name]
                                    # 如果字段是英文，或者内容相同需要重新提取，就更新
                                    if not has_chinese(str(old_value)) or all_fields_same or str(old_value) != new_value:
                                        visual[field_name] = new_value
                                        if str(old_value) != new_value:
                                            print(f"    ✓ visual.{field_name}: {new_value[:50]}...")
                        
                        scene["visual"] = visual
                        converted_scenes += 1
                        file_converted += 1
        
        # 保存文件（如果不是预览模式且有修改）
        if not args.dry_run and file_converted > 0:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"\n  ✓ 文件已保存: {file_path.name} (转换了 {file_converted} 个场景的 visual 字段)")
            except Exception as e:
                print(f"  ✗ 保存文件失败: {e}")
        elif file_converted == 0:
            print(f"  ℹ 无需转换（所有 visual 字段都已经是中文或 description/prompt 不是中文）")
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
        print("  - visual 字段已智能提取为中文（根据 description 提取不同部分）")


if __name__ == "__main__":
    main()
