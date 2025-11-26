#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据分镜 JSON 生成标准 Markdown 剧本
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any


INTRO_TEMPLATE = (
    "仙友们好，我是云卷仙音——今天继续凡人修仙传灵界篇第{episode}集《{title}》，"
    "一起见证韩立在灵界的全新际遇。"
)

OUTRO_TEMPLATE = (
    "以上便是灵界篇第{episode}集《{title}》。原著：忘语。"
    "若仙友们喜欢，记得点赞、收藏与分享，我们下集再会。"
)


def load_scenes(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "scenes" in data and isinstance(data["scenes"], list):
            return data["scenes"]
        raise ValueError("JSON 顶层需要是数组，或包含 scenes 列表")
    if not isinstance(data, list):
        raise ValueError("JSON 内容必须是数组或包含 scenes 列表")
    return data


def sanitise(text: str) -> str:
    if not text:
        return ""
    return text.replace("\n", " ").replace("|", "｜").strip()


def quote(text: str) -> str:
    return sanitise(text).replace('"', "”")


def build_scene_table(scenes: Iterable[Dict[str, Any]]) -> str:
    rows = [
        "| 镜头 | 场景标题 | 画面设定 | 镜头语言 | 氛围风格 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for scene in scenes:
        scene_id = scene.get("scene_id") or scene.get("scene_number") or scene.get("id")
        if scene_id is None:
            continue
        title = sanitise(scene.get("title", f"场景{scene_id}"))
        environment = sanitise(scene.get("environment", ""))
        action = sanitise(scene.get("action", ""))
        camera = sanitise(scene.get("camera", ""))
        mood = sanitise(scene.get("mood", ""))
        rows.append(
            f"| {scene_id} | **{title}**：{environment} | {action} | {camera} | {mood} |"
        )
    return "\n".join(rows)


def build_narration_table(scenes: Iterable[Dict[str, Any]]) -> str:
    rows = [
        "| 镜头 | 旁白内容（云卷仙音） | 配音提示 |",
        "| --- | --- | --- |",
    ]
    for scene in scenes:
        scene_id = scene.get("scene_id") or scene.get("scene_number") or scene.get("id")
        if scene_id is None:
            continue
        narration = scene.get("narration", "")
        rows.append(f'| {scene_id} | "{quote(narration)}" | 平稳叙述 |')
    return "\n".join(rows)


def render_markdown(episode: str, title: str, scenes: List[Dict[str, Any]]) -> str:
    intro = INTRO_TEMPLATE.format(episode=episode, title=title)
    outro = OUTRO_TEMPLATE.format(episode=episode, title=title)
    scene_table = build_scene_table(scenes)
    narration_table = build_narration_table(scenes)
    lines = [
        f"# 凡人修仙传·灵界篇 第{episode}集《{title}》",
        "",
        "### 🎙️【开场解说稿】（云卷仙音旁白）",
        f"> {intro}",
        "",
        "### 🎬【分镜概览】",
        scene_table,
        "",
        "### 🎧【旁白台本】",
        narration_table,
        "",
        "### 🏷️【结束语】（云卷仙音旁白）",
        f"> {outro}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="根据分镜 JSON 生成 Markdown 剧本")
    parser.add_argument("--json", required=True, type=str, help="分镜 JSON 路径")
    parser.add_argument("--episode", required=True, type=str, help="集数编号")
    parser.add_argument("--title", required=True, type=str, help="本集标题")
    parser.add_argument(
        "--output", required=True, type=str, help="输出 Markdown 路径"
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    output_path = Path(args.output)
    scenes = load_scenes(json_path)

    markdown = render_markdown(args.episode, args.title, scenes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✓ 已生成 Markdown: {output_path}")


if __name__ == "__main__":
    main()

