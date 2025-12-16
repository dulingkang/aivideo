#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 2.json 的 visual 字段，确保字段职责清晰且无重复
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any

def fix_visual_fields(file_path: Path):
    """修复 visual 字段"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scenes = data.get("scenes", [])
    
    # 手动修复每个场景的 visual 字段
    fixes = {
        1: {
            "composition": "青年将圆盘对准韩立，一道青濛濛光柱激射而出",
            "environment": "",
            "character_pose": "青年将圆盘对准韩立",
            "fx": "青濛濛光柱激射而出"
        },
        2: {
            "composition": "光柱照在韩立身上，韩立神色平静",
            "environment": "",
            "character_pose": "韩立神色平静，任凭探测",
            "fx": "光柱照在韩立身上"
        },
        3: {
            "composition": "圆盘灵光大放，青年满脸惊容",
            "environment": "",
            "character_pose": "青年满脸惊容，冲身后大叫",
            "fx": "圆盘灵光大放"
        },
        4: {
            "composition": "疤面大汉走过来，开口问韩立",
            "environment": "",
            "character_pose": "开口问韩立，但韩立听不懂",
            "fx": ""
        },
        5: {
            "composition": "大汉换了一种语言，韩立神色一动",
            "environment": "",
            "character_pose": "韩立神色一动，觉得耳熟",
            "fx": ""
        },
        6: {
            "composition": "大汉掏出淡黄色木盒，里面是一颗乳黄色圆珠",
            "environment": "",
            "character_pose": "大汉掏出淡黄色木盒",
            "fx": ""
        },
        7: {
            "composition": "大汉将圆珠按在指环晶石上，光芒大放",
            "environment": "",
            "character_pose": "大汉将圆珠按在指环晶石上",
            "fx": "光芒大放"
        },
        8: {
            "composition": "圆珠没入韩立身体，凉气窜入脑中",
            "environment": "",
            "character_pose": "",
            "fx": "圆珠没入，凉气窜入脑中"
        },
        9: {
            "composition": "韩立头颅刺痛，神识中浮现众多东西",
            "environment": "",
            "character_pose": "韩立头颅刺痛",
            "fx": "神识中浮现众多东西"
        },
        10: {
            "composition": "韩立与大汉对话，确认这里是青罗沙漠",
            "environment": "青罗沙漠，灵界",
            "character_pose": "韩立与大汉对话",
            "fx": ""
        },
        11: {
            "composition": "大汉提出签订血咒文书，韩立答应",
            "environment": "",
            "character_pose": "韩立答应",
            "fx": ""
        },
        12: {
            "composition": "骑士拿出血咒文书，咬破手指书写",
            "environment": "",
            "character_pose": "咬破手指书写",
            "fx": ""
        },
        13: {
            "composition": "韩立查看血咒文书，认出是符箓",
            "environment": "",
            "character_pose": "韩立查看血咒文书，分析其约束力",
            "fx": ""
        },
        14: {
            "composition": "骑士用匕首划韩立手腕，但无法划破",
            "environment": "",
            "character_pose": "",
            "fx": "无法划破"
        },
        15: {
            "composition": "大汉询问韩立是否修炼金刚诀，韩立确认第三层",
            "environment": "",
            "character_pose": "韩立确认第三层",
            "fx": ""
        },
        16: {
            "composition": "大汉拿出金莹剑，骑士戴上手套，剑身金芒四射",
            "environment": "",
            "character_pose": "大汉拿出金莹剑，骑士戴上手套",
            "fx": "金芒四射"
        },
        17: {
            "composition": "韩立发现灵具的奥秘，剑柄有灵石，手套也有灵石",
            "environment": "",
            "character_pose": "韩立发现灵具的奥秘",
            "fx": ""
        },
        18: {
            "composition": "金莹剑划破韩立手腕，鲜血流出，书写血咒符文",
            "environment": "",
            "character_pose": "",
            "fx": "划破，鲜血流出"
        },
        19: {
            "composition": "血咒文书血光大放，符文飞射而出，文书自燃",
            "environment": "",
            "character_pose": "",
            "fx": "血光大放，符文飞射而出，文书自燃"
        },
        20: {
            "composition": "远处灰尘大起，一只巨大的陆行龟出现，车厢门打开",
            "environment": "远处，灰尘大起",
            "character_pose": "",
            "fx": ""
        }
    }
    
    changes_count = 0
    for scene in scenes:
        scene_id = scene.get("id")
        if scene_id in fixes:
            visual = scene.get("visual", {}) or {}
            fix = fixes[scene_id]
            
            changed = False
            for key, value in fix.items():
                if visual.get(key) != value:
                    visual[key] = value
                    changed = True
            
            if changed:
                scene["visual"] = visual
                changes_count += 1
                print(f"场景 {scene_id}: 已修复 visual 字段")
    
    # 保存
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 修复完成！共修复 {changes_count} 个场景")

if __name__ == "__main__":
    file_path = Path("lingjie/scenes/2.json")
    fix_visual_fields(file_path)

