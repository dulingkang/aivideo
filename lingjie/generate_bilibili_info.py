#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成B站发布信息的脚本
根据scenes目录下的JSON文件，为每一集生成标题、描述等信息
"""

import json
import os
from pathlib import Path

# 标题翻译映射（英文标题 -> 中文标题）
# 如果映射中没有，将从章节对应记录中提取
TITLE_TRANSLATION = {
    "Strange Land.First Entry into the Spirit Realm": "陌生之地·沙漠苏醒",
    "Blood Curse Document.Tiandong Trading Company": "天东商号·血咒文书",
    "Nanqizi & Elder Fu.Sand Beasts": "陆行龟·沙虫兽袭",
    "False City Lord Revealed · Black Phoenix Standoff": "假城主现·黑凤对峙",
    "Beast Lair · Seven-Leaf Yin Blood Ganoderma": "妖兽巢穴·七叶阴血芝",
    "Blood Shadow Escape · Silver Firebird": "血影遁逃·银焰火鸟",
    # 可以根据需要添加更多映射
}

# 从章节对应记录中提取的标题映射（集数 -> 中文标题）
EPISODE_TITLE_MAP = {
    1: "陌生之地·沙漠苏醒",
    2: "天东商号·血咒文书",
    3: "陆行龟·沙虫兽袭",
    4: "战沙虫·符老金针",
    5: "草原戒备·安远初见",
    6: "如云客栈·金刚试锋",
    7: "兽潮将临·青狼初攻",
    8: "赤蟒围城·豹禽现身",
    9: "豹禽破城·营地再谋",
    10: "营地密谋·黑凤追踪",
    11: "假城主现·黑凤对峙",
    12: "妖兽巢穴·七叶阴血芝",
    13: "前往虞阳·路遇范胖",
    14: "虞阳生活·灵具炼制",
    15: "突破六层·九玄明玉潭",
    16: "青罗沙漠·重得宝物",
    # 17-21集：前往落日·初入落日之墓（需要具体拆分）
    # 22-30集：落日之墓冒险·灵族浮现（需要具体拆分）
    # 31-40集：器灵族控制·混沌谷大战（需要具体拆分）
    # 41-48集：炙光潭异变·巨人现世等（需要具体拆分）
}

def extract_scene_summary(scenes):
    """从场景中提取关键剧情摘要"""
    narrations = []
    for scene in scenes:
        if scene.get("id") != 0 and scene.get("id") != 999:  # 跳过开头和结尾
            narration = scene.get("narration", "")
            if narration:
                narrations.append(narration)
    
    # 取前3-5条关键旁白作为剧情摘要
    if len(narrations) > 5:
        key_points = narrations[0:1] + narrations[len(narrations)//3:len(narrations)//3+2] + narrations[-2:-1]
    elif len(narrations) > 2:
        key_points = [narrations[0], narrations[len(narrations)//2], narrations[-1]]
    else:
        key_points = narrations
    
    return key_points

def generate_description(episode, title_cn, scenes, chapter_info=None):
    """生成视频描述"""
    # 提取剧情要点
    summary = extract_scene_summary(scenes)
    
    # 构建描述
    description = f"""【凡人修仙传·灵界篇】第{episode}集：{title_cn}

📖 本集剧情：
"""
    
    # 添加关键剧情点
    for i, point in enumerate(summary[:4], 1):
        description += f"{i}. {point}\n"
    
    description += f"""
🎬 系列信息：
本视频为《凡人修仙传》灵界篇系列解说的第{episode}集
讲述韩立在灵界初期的冒险历程

🎭 主要角色：
- 韩立（主角）
- 云卷仙音（解说）

📚 原著：忘语《凡人修仙传》

💬 互动提示：
- 点赞、投币、收藏支持UP主
- 评论区分享你的看法
- 关注UP主，不错过更新

🎵 背景音乐：仙侠风格BGM

---
#凡人修仙传 #灵界篇 #韩立 #仙侠 #小说改编 #解说视频 #忘语
"""
    
    if chapter_info:
        description += f"\n📖 对应章节：{chapter_info}\n"
    
    return description

def translate_title(title_en, episode=None):
    """翻译标题"""
    # 优先使用集数映射表（最准确）
    if episode and episode in EPISODE_TITLE_MAP:
        return EPISODE_TITLE_MAP[episode]
    
    # 其次使用英文标题映射
    if title_en in TITLE_TRANSLATION:
        return TITLE_TRANSLATION[title_en]
    
    # 简单的翻译规则
    title_cn = title_en.replace("·", "·").replace(" & ", "·").replace(".", "·")
    title_cn = title_cn.replace(" ", "·")
    
    # 如果没有映射，返回处理后的标题
    return title_cn

def generate_bilibili_info():
    """生成所有集的B站发布信息"""
    scenes_dir = Path(__file__).parent / "scenes"
    
    if not scenes_dir.exists():
        print(f"错误：找不到scenes目录：{scenes_dir}")
        return
    
    # 读取章节对应记录（如果有的话）
    chapter_file = Path(__file__).parent / "章节对应记录.md"
    chapter_map = {}
    
    if chapter_file.exists():
        # 简单解析章节信息（可以根据需要改进）
        with open(chapter_file, "r", encoding="utf-8") as f:
            content = f.read()
            # 这里可以添加更复杂的解析逻辑
    
    all_info = []
    
    # 遍历所有JSON文件
    for json_file in sorted(scenes_dir.glob("*.json")):
        # 跳过备份文件
        if json_file.name.endswith(".bk") or json_file.name.endswith(".backup"):
            continue
            
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            episode = data.get("episode")
            title_en = data.get("title", "")
            scenes = data.get("scenes", [])
            
            if episode is None:
                print(f"警告：{json_file.name} 中没有episode字段，跳过")
                continue
            
            # 翻译标题
            title_cn = translate_title(title_en, episode)
            
            # 生成标题（B站格式）
            bilibili_title = f"【凡人修仙传·灵界篇】第{episode}集：{title_cn}"
            
            # 生成描述
            chapter_info = chapter_map.get(episode)
            description = generate_description(episode, title_cn, scenes, chapter_info)
            
            # 生成标签
            tags = [
                "凡人修仙传",
                "灵界篇",
                f"第{episode}集",
                "韩立",
                "仙侠",
                "小说改编",
                "解说视频",
                "忘语",
                "云卷仙音"
            ]
            
            info = {
                "episode": episode,
                "title_cn": title_cn,
                "title_en": title_en,
                "bilibili_title": bilibili_title,
                "description": description,
                "tags": tags
            }
            
            all_info.append(info)
            
        except Exception as e:
            print(f"错误：处理 {json_file.name} 时出错：{e}")
            continue
    
    # 按集数排序
    all_info.sort(key=lambda x: x["episode"])
    
    return all_info

def save_to_markdown(all_info, output_file):
    """保存为Markdown格式"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 凡人修仙传·灵界篇 - B站发布信息\n\n")
        f.write("本文件包含每一集的B站发布标题、描述和标签信息。\n\n")
        f.write("---\n\n")
        
        for info in all_info:
            f.write(f"## 第{info['episode']}集：{info['title_cn']}\n\n")
            f.write(f"**英文标题**：{info['title_en']}\n\n")
            f.write(f"**B站标题**：\n```\n{info['bilibili_title']}\n```\n\n")
            f.write(f"**视频描述**：\n```\n{info['description']}\n```\n\n")
            f.write(f"**标签**：\n")
            for tag in info['tags']:
                f.write(f"- {tag}\n")
            f.write("\n")
            f.write("---\n\n")

def save_to_json(all_info, output_file):
    """保存为JSON格式"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_info, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    print("开始生成B站发布信息...")
    
    all_info = generate_bilibili_info()
    
    if not all_info:
        print("错误：没有生成任何信息")
        return
    
    print(f"成功生成 {len(all_info)} 集的信息")
    
    # 保存为Markdown和JSON两种格式
    output_dir = Path(__file__).parent
    markdown_file = output_dir / "bilibili_release_info.md"
    json_file = output_dir / "bilibili_release_info.json"
    
    save_to_markdown(all_info, markdown_file)
    print(f"✓ 已保存Markdown格式：{markdown_file}")
    
    save_to_json(all_info, json_file)
    print(f"✓ 已保存JSON格式：{json_file}")
    
    # 打印前3集作为预览
    print("\n前3集预览：")
    print("=" * 80)
    for info in all_info[:3]:
        print(f"\n第{info['episode']}集：{info['title_cn']}")
        print(f"标题：{info['bilibili_title']}")
        print(f"描述（前200字）：{info['description'][:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()
