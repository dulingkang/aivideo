#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本解析器
从 Markdown 格式的分镜脚本中提取场景和旁白信息
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional


class ScriptParser:
    """Markdown 脚本解析器"""
    
    def __init__(self, markdown_path: str, image_dir: str):
        """
        初始化解析器
        
        Args:
            markdown_path: Markdown 脚本路径
            image_dir: 图像目录路径
        """
        self.markdown_path = Path(markdown_path)
        self.image_dir = Path(image_dir)
        self.content = self._load_content()
    
    def _load_content(self) -> str:
        """加载 Markdown 内容"""
        with open(self.markdown_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_scenes(self) -> List[Dict]:
        """
        解析场景信息
        
        Returns:
            场景列表，每个场景包含：
            - scene_number: 场景编号
            - description: 场景描述
            - narration: 旁白文本
            - image_path: 图像路径（如果找到）
        """
        scenes = []
        
        # 解析分镜表格（场景描述）
        # 匹配格式: | 1️⃣ | **场景标题**：描述 | 动作 | 提示词 |
        scene_pattern = r'\|\s*([0-9️⃣]+)\s*\|\s*\*\*(.*?)\*\*[：:]?\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|'
        scene_matches = re.findall(scene_pattern, self.content, re.DOTALL)
        
        # 如果没找到，尝试更宽松的匹配
        if not scene_matches:
            # 尝试匹配不带**的格式
            scene_pattern = r'\|\s*([0-9️⃣]+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|'
            scene_matches = re.findall(scene_pattern, self.content, re.DOTALL)
            # 过滤掉表头
            scene_matches = [m for m in scene_matches if m[0].isdigit() and int(m[0]) <= 31]
        
        # 解析旁白表格
        # 匹配格式: | 1️⃣ | "旁白文本" | 语气 |
        narration_pattern = r'\|\s*([0-9️⃣]+)\s*\|\s*"([^"]+)"\s*\|\s*(.*?)\s*\|'
        narration_matches = re.findall(narration_pattern, self.content, re.DOTALL)
        
        # 创建旁白字典
        narration_dict = {}
        for match in narration_matches:
            try:
                scene_num = self._parse_scene_number(match[0])
                if scene_num is None:
                    continue
                narration_text = match[1].strip()
                narration_dict[scene_num] = narration_text
            except:
                continue
        
        # 获取所有图像文件
        image_files = self._get_image_files()
        print(f"找到 {len(image_files)} 张图像")
        
        # 构建场景列表
        for match in scene_matches:
            try:
                scene_num = self._parse_scene_number(match[0])
                if scene_num is None:
                    continue
                if len(match) >= 5:
                    scene_title = match[1].strip().strip('*')
                    scene_desc = match[2].strip()
                    action_desc = match[3].strip()
                    prompt = match[4].strip()
                else:
                    # 简化格式
                    scene_title = f"场景{scene_num}"
                    scene_desc = match[1].strip() if len(match) > 1 else ""
                    action_desc = match[2].strip() if len(match) > 2 else ""
                    prompt = match[3].strip() if len(match) > 3 else ""
                
                # 获取对应图像（按序号匹配）
                image_path = self._find_image_for_scene(scene_num, image_files)
                
                scene = {
                    'scene_number': scene_num,
                    'title': scene_title,
                    'description': scene_desc,
                    'action': action_desc,
                    'prompt': prompt,
                    'narration': narration_dict.get(scene_num, ''),
                    'image_path': image_path,
                }
                scenes.append(scene)
            except Exception as e:
                print(f"警告: 解析场景失败 {match}: {e}")
                continue
        
        # 如果表格解析失败或场景数不足，使用简单解析
        if not scenes or len(scenes) < 10:
            print("使用简单解析模式...")
            scenes = self._parse_simple_format(image_files, narration_dict)
        
        # 按场景编号排序
        scenes.sort(key=lambda x: x['scene_number'])
        
        return scenes

    @staticmethod
    def _parse_scene_number(token: str) -> Optional[int]:
        digits = re.findall(r'\d+', token)
        if not digits:
            return None
        return int(''.join(digits))
    
    def _get_image_files(self) -> List[Path]:
        """获取所有图像文件"""
        image_files = []
        
        # 查找 jpg 和 png 文件
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(self.image_dir.glob(ext))
        
        # 排序：优先使用 jpgsrc 目录，然后按文件名排序
        image_files.sort(key=lambda x: (
            0 if 'jpgsrc' in str(x) else 1,  # jpgsrc 目录优先
            self._extract_number(str(x)),  # 按数字排序
            str(x)
        ))
        
        return image_files
    
    def _extract_number(self, filename: str) -> int:
        """从文件名中提取数字"""
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])  # 使用最后一个数字
        return 999999
    
    def _find_image_for_scene(self, scene_num: int, image_files: List[Path]) -> Optional[str]:
        """为场景找到对应的图像"""
        if not image_files:
            return None
        
        # 优先查找 jpgsrc 目录
        jpgsrc_files = [f for f in image_files if 'jpgsrc' in str(f)]
        if jpgsrc_files:
            image_files = jpgsrc_files
        
        # 尝试直接匹配场景编号
        # 场景编号从1开始，图像可能从8开始（根据实际文件）
        # 使用 scene_num - 1 作为索引（如果图像数量足够）
        if scene_num <= len(image_files):
            return str(image_files[scene_num - 1])
        
        # 如果场景数量超过图像数量，循环使用
        return str(image_files[(scene_num - 1) % len(image_files)])
    
    def _parse_simple_format(self, image_files: List[Path], narration_dict: Dict = None) -> List[Dict]:
        """简单格式解析（如果表格解析失败）"""
        if narration_dict is None:
            narration_dict = {}
        
        scenes = []
        
        # 提取所有"镜头"或数字开头的场景
        lines = self.content.split('\n')
        scene_num = 0
        
        # 查找所有包含场景编号的行
        for i, line in enumerate(lines):
            # 匹配 "### 镜头X" 或 "| X |" 格式
            match = re.search(r'(?:镜头|场景)\s*([0-9️⃣]+)', line)
            if match:
                parsed = self._parse_scene_number(match.group(1))
                scene_num = parsed or scene_num
            elif re.search(r'^\|\s*([0-9️⃣]+)\s*\|', line):
                # 表格行
                match = re.search(r'^\|\s*([0-9️⃣]+)\s*\|', line)
                if match:
                    parsed = self._parse_scene_number(match.group(1))
                    scene_num = parsed or scene_num
            
            if scene_num > 0 and scene_num <= 31:
                # 查找对应的图像
                if scene_num <= len(image_files):
                    image_path = str(image_files[scene_num - 1])
                else:
                    image_path = str(image_files[(scene_num - 1) % len(image_files)]) if image_files else None
                
                # 提取描述（下一行或当前行）
                description = line.strip()
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('|') and not next_line.startswith('#'):
                        description = next_line
                
                scenes.append({
                    'scene_number': scene_num,
                    'title': f'场景{scene_num}',
                    'description': description,
                    'action': '',
                    'prompt': '',
                    'narration': narration_dict.get(scene_num, ''),
                    'image_path': image_path,
                })
                
                # 避免重复
                if scene_num >= 31:
                    break
        
        # 如果还是没找到，直接使用图像文件创建场景
        if not scenes and image_files:
            for i, image_file in enumerate(image_files[:31]):
                scenes.append({
                    'scene_number': i + 1,
                    'title': f'场景{i + 1}',
                    'description': f'场景 {i + 1}',
                    'action': '',
                    'prompt': '',
                    'narration': narration_dict.get(i + 1, ''),
                    'image_path': str(image_file),
                })
        
        return scenes
    
    def extract_opening_narration(self) -> str:
        """提取开场白"""
        lines = self.content.split('\n')
        opening_lines = []
        capture = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('### 🎙️【开场解说稿】'):
                capture = True
                continue
            if capture:
                if stripped.startswith('### '):
                    break
                if stripped.startswith('>'):
                    opening_lines.append(stripped.lstrip('> ').strip())
                elif stripped:
                    opening_lines.append(stripped)
        if opening_lines:
            return ' '.join(opening_lines).strip()

        # 兼容旧格式（引号包裹）
        pattern = r'### 🎙️【开场解说稿】.*?"(.*?)"'
        match = re.search(pattern, self.content, re.DOTALL)
        if match:
            return match.group(1).strip().replace('\n', ' ')
        return ''
    
    def extract_ending_narration(self) -> str:
        """提取结束语"""
        lines = self.content.split('\n')
        ending_lines = []
        capture = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('### 🏷️【结束语】'):
                capture = True
                continue
            if capture:
                if stripped.startswith('### '):
                    break
                if stripped.startswith('>'):
                    ending_lines.append(stripped.lstrip('> ').strip())
                elif stripped:
                    ending_lines.append(stripped)
        if ending_lines:
            return ' '.join(ending_lines).strip()

        pattern = r'### 🏷️【结束语】.*?"(.*?)"'
        match = re.search(pattern, self.content, re.DOTALL)
        if match:
            return match.group(1).strip().replace('\n', ' ')
        return ''
    
    def get_full_narration(
        self,
        scenes: Optional[List[Dict]] = None,
        include_opening: bool = True,
        include_ending: bool = True,
    ) -> str:
        """获取完整旁白（包括开场、各场景旁白、结束语）"""
        narration_parts = []
        
        # 开场白
        if include_opening:
            opening = self.extract_opening_narration()
            if opening:
                narration_parts.append(opening)
        
        # 各场景旁白
        scenes = scenes if scenes is not None else self.parse_scenes()
        for scene in scenes:
            if scene.get('narration'):
                narration_parts.append(scene['narration'])
        
        # 结束语
        if include_ending:
            ending = self.extract_ending_narration()
            if ending:
                narration_parts.append(ending)
        
        return ' '.join(narration_parts)
    
    def to_json(
        self,
        output_path: str,
        scenes: Optional[List[Dict]] = None,
        total_scene_count: Optional[int] = None,
    ):
        """导出为 JSON 格式"""
        from pathlib import Path
        
        # 创建输出目录
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        scenes = scenes if scenes is not None else self.parse_scenes()
        if total_scene_count is None:
            total_scene_count = len(self.parse_scenes())
        opening = self.extract_opening_narration()
        ending = self.extract_ending_narration()
        include_ending = len(scenes) >= total_scene_count
        full_narration = self.get_full_narration(
            scenes,
            include_opening=True,
            include_ending=include_ending,
        )
        
        script = {
            'title': '凡人修仙传·灵界篇②：青罗沙漠',
            'opening_narration': opening,
            'ending_narration': ending if include_ending else '',
            'full_narration': full_narration,
            'scenes': scenes,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 脚本已导出: {output_path}")
        return script


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="解析 Markdown 脚本")
    parser.add_argument("--markdown", type=str, required=True, help="Markdown 脚本路径")
    parser.add_argument("--image-dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--output", type=str, help="输出 JSON 路径")
    
    args = parser.parse_args()
    
    # 解析脚本
    parser = ScriptParser(args.markdown, args.image_dir)
    scenes = parser.parse_scenes()
    
    print(f"解析到 {len(scenes)} 个场景")
    for scene in scenes[:5]:  # 显示前5个场景
        print(f"场景 {scene['scene_number']}: {scene.get('title', '')}")
        print(f"  图像: {scene.get('image_path', 'N/A')}")
        print(f"  旁白: {scene.get('narration', '')[:50]}...")
    
    # 导出 JSON
    if args.output:
        parser.to_json(args.output)


if __name__ == "__main__":
    main()

