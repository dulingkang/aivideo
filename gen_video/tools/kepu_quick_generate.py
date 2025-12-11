#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科普视频快速生成工具
输入：选题 → 输出：完整视频

功能：
1. 从知识库读取选题信息
2. 根据选题生成脚本JSON
3. 调用主流程生成完整视频
"""

import os
import sys
import yaml
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import AIVideoPipeline


class KepuQuickGenerator:
    """科普视频快速生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化生成器"""
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path(__file__).parent.parent / self.config_path).resolve()
        
        # 加载知识库
        topics_yaml_path = Path(__file__).parent.parent / "knowledge_base" / "kepu_topics.yaml"
        with open(topics_yaml_path, 'r', encoding='utf-8') as f:
            self.topics_config = yaml.safe_load(f)
        
        # 加载场景提示词库
        scenes_yaml_path = Path(__file__).parent.parent / "prompt" / "kepu_scenes.yaml"
        with open(scenes_yaml_path, 'r', encoding='utf-8') as f:
            self.scenes_config = yaml.safe_load(f)
        
        self.kepu_topics = self.topics_config.get('kepu_topics', [])
        self.kepu_scenes = self.scenes_config.get('kepu_scenes', {})
        
        # 初始化视频生成流水线（延迟加载，不占用启动显存）
        print("初始化视频生成流水线（延迟加载，不占用启动显存）...")
        # 只创建实例，不预加载模型（延迟加载）
        self.pipeline = AIVideoPipeline(
            str(self.config_path),
            load_image=True,  # 延迟加载
            load_video=True,  # 延迟加载
            load_tts=True,  # 延迟加载
            load_subtitle=True,  # 延迟加载
            load_composer=True  # 延迟加载
        )
        
        # 创建输出目录
        self.output_dir = Path(__file__).parent.parent / "outputs" / "kepu_videos"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_topic(self, topic_title: str) -> Optional[Dict]:
        """从知识库中查找选题"""
        for category_data in self.kepu_topics:
            for topic in category_data.get('topics', []):
                if topic.get('title') == topic_title:
                    # 添加分类信息
                    topic['category_name'] = category_data.get('category', '')
                    return topic
        return None
    
    def list_topics(self) -> List[Dict]:
        """列出所有可用选题"""
        all_topics = []
        for category_data in self.kepu_topics:
            category_name = category_data.get('category', '')
            for topic in category_data.get('topics', []):
                topic_copy = topic.copy()
                topic_copy['category_name'] = category_name
                all_topics.append(topic_copy)
        return all_topics
    
    def load_template(self, template_name: str) -> Optional[Dict]:
        """加载脚本模板"""
        template_path = Path(__file__).parent.parent / "templates" / template_name
        if not template_path.exists():
            print(f"  ⚠️  模板文件不存在: {template_name}，使用默认模板")
            return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️  加载模板失败: {e}，使用默认模板")
            return None
    
    def generate_script(self, topic: Dict, ip_character: str = "kepu_gege") -> Dict:
        """根据选题生成脚本JSON"""
        title = topic.get('title', '科普视频')
        scene_type = topic.get('scene_type', 'universe')
        duration = topic.get('duration', 60)
        difficulty = topic.get('difficulty', '初级')
        category_name = topic.get('category_name', '')
        template_name = topic.get('script_template', f'{scene_type}_template.json')
        description = topic.get('description', '')
        
        # 尝试加载模板
        template = self.load_template(template_name)
        
        # 获取场景配置
        scene_config = self.kepu_scenes.get(scene_type, {})
        scene_examples = scene_config.get('examples', [])
        base_prompt = scene_config.get('base_prompt', '')
        
        # 选择场景提示词
        def select_scene_prompt():
            if scene_examples:
                return random.choice(scene_examples)
            return base_prompt
        
        # 生成开场
        character_name = "科普哥哥" if ip_character == "kepu_gege" else "未来姐姐"
        opening_narration = f"大家好，我是{character_name}。今天我们来聊聊{title}。"
        
        # 如果使用模板，从模板中获取开场和结尾
        if template:
            opening_narration = template.get('opening', {}).get('narration', opening_narration)
            opening_narration = opening_narration.replace('[主题]', title)
            ending_narration = template.get('ending', {}).get('narration', "今天的科普就到这里，我们下期再见！")
        else:
            ending_narration = "今天的科普就到这里，我们下期再见！"
        
        # 生成内容场景（根据时长决定场景数量）
        # 每个场景15-20秒，开场12秒，结尾12秒
        content_duration = duration - 12 - 12  # 减去开场和结尾
        num_scenes = max(2, content_duration // 18)  # 至少2个场景
        
        # 如果使用模板，从模板中获取场景结构
        if template and template.get('scenes'):
            template_scenes = template.get('scenes', [])
            # 根据需要的场景数量调整
            if len(template_scenes) >= num_scenes:
                scenes = template_scenes[:num_scenes]
            else:
                # 如果模板场景不够，补充场景
                scenes = template_scenes.copy()
                for i in range(len(template_scenes), num_scenes):
                    scenes.append({
                        "id": i + 1,
                        "narration": f"{description}这是关于{title}的重要知识点。",
                        "scene": {
                            "prompt": select_scene_prompt(),
                            "duration": content_duration // num_scenes,
                            "type": "content"
                        },
                        # 添加主持人信息，确保生成时包含主持人形象
                        "characters": [ip_character],
                        "description": f"{character_name}在讲解{title}的相关知识"
                    })
            
            # 更新场景的时长和ID，并确保包含主持人信息
            for i, scene in enumerate(scenes):
                scene['id'] = i + 1
                scene['scene']['duration'] = content_duration // num_scenes
                # 确保每个场景都包含主持人信息（如果模板中没有）
                if 'characters' not in scene or not scene.get('characters'):
                    scene['characters'] = [ip_character]
                if 'description' not in scene or not scene.get('description'):
                    scene['description'] = f"{character_name}在讲解{title}的相关知识"
        else:
            # 生成默认场景
            scenes = []
            for i in range(num_scenes):
                scene_id = i + 1
                # 根据描述生成更智能的旁白
                if description:
                    narration = f"{description}让我们深入了解{title}的相关知识。"
                else:
                    narration = f"这是关于{title}的第{scene_id}个知识点讲解。"
                scenes.append({
                    "id": scene_id,
                    "narration": narration,
                    "scene": {
                        "prompt": select_scene_prompt(),
                        "duration": content_duration // num_scenes,
                        "type": "content"
                    },
                    # 添加主持人信息，确保生成时包含主持人形象
                    "characters": [ip_character],
                    "description": f"{character_name}在讲解{title}的相关知识"
                })
        
        # 构建完整脚本
        script = {
            "title": title,
            "topic": category_name,
            "category": scene_type,
            "duration": duration,
            "ip_character": ip_character,
            "opening": {
                "narration": opening_narration,
                "scene": {
                    "prompt": select_scene_prompt() if not template else template.get('opening', {}).get('scene', {}).get('prompt', select_scene_prompt()),
                    "duration": 12,
                    "type": "opening"
                },
                # 开场场景也需要包含主持人
                "characters": [ip_character],
                "description": f"{character_name}向大家介绍{title}"
            },
            "scenes": scenes,
            "ending": {
                "narration": ending_narration,
                "scene": {
                    "prompt": select_scene_prompt() if not template else template.get('ending', {}).get('scene', {}).get('prompt', select_scene_prompt()),
                    "duration": 12,
                    "type": "ending"
                },
                # 结尾场景也需要包含主持人
                "characters": [ip_character],
                "description": f"{character_name}向大家告别"
            },
            "metadata": {
                "target_platform": "douyin",
                "tags": ["科普", "科学", "教育"] + topic.get('keywords', []),
                "difficulty": difficulty
            }
        }
        
        return script
    
    def generate_video(self, topic_title: str, ip_character: str = "kepu_gege", 
                      custom_script: Optional[Dict] = None) -> Path:
        """生成完整视频"""
        print(f"\n{'='*60}")
        print(f"开始生成科普视频: {topic_title}")
        print(f"{'='*60}")
        
        # 查找选题
        topic = self.find_topic(topic_title)
        if not topic:
            raise ValueError(f"未找到选题: {topic_title}")
        
        print(f"选题信息:")
        print(f"  标题: {topic.get('title')}")
        print(f"  分类: {topic.get('category_name')}")
        print(f"  难度: {topic.get('difficulty')}")
        print(f"  时长: {topic.get('duration')}秒")
        
        # 生成或使用自定义脚本
        if custom_script:
            script = custom_script
            print("使用自定义脚本")
        else:
            print("生成脚本...")
            script = self.generate_script(topic, ip_character)
        
        # 保存脚本JSON
        script_filename = f"{topic_title.replace(' ', '_').replace('？', '')}_script.json"
        script_path = self.output_dir / script_filename
        with open(script_path, 'w', encoding='utf-8') as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        
        print(f"脚本已保存: {script_path}")
        
        # 生成输出名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{topic_title.replace(' ', '_').replace('？', '')}_{timestamp}"
        
        # 调用主流程生成视频
        print(f"\n开始生成视频...")
        print(f"输出名称: {output_name}")
        
        try:
            self.pipeline.process_script(
                script_path=str(script_path),
                output_name=output_name
            )
            
            # 查找生成的视频文件
            output_video = self.output_dir / f"{output_name}.mp4"
            if not output_video.exists():
                # 尝试在其他可能的位置查找
                possible_paths = [
                    self.pipeline.paths['output_dir'] / f"{output_name}.mp4",
                    Path(self.pipeline.paths['output_dir']) / f"{output_name}.mp4",
                ]
                for path in possible_paths:
                    if path.exists():
                        output_video = path
                        break
            
            if output_video.exists():
                print(f"\n✅ 视频生成成功: {output_video}")
                return output_video
            else:
                print(f"\n⚠️  视频生成完成，但未找到输出文件")
                print(f"   请检查输出目录: {self.pipeline.paths['output_dir']}")
                return output_video  # 返回预期路径
                
        except Exception as e:
            print(f"\n❌ 视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description='科普视频快速生成工具')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--topic', type=str, required=True,
                       help='选题标题（从知识库中选择）')
    parser.add_argument('--ip-character', type=str, default='kepu_gege',
                       choices=['kepu_gege', 'weilai_jiejie'],
                       help='IP角色（默认: kepu_gege）')
    parser.add_argument('--script', type=str, default=None,
                       help='自定义脚本JSON文件路径（可选）')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用选题')
    
    args = parser.parse_args()
    
    generator = KepuQuickGenerator(config_path=args.config)
    
    if args.list:
        # 列出所有选题
        print("\n可用选题列表:")
        print("="*60)
        topics = generator.list_topics()
        for idx, topic in enumerate(topics, 1):
            print(f"{idx}. {topic.get('title')}")
            print(f"   分类: {topic.get('category_name')}")
            print(f"   难度: {topic.get('difficulty')}")
            print(f"   时长: {topic.get('duration')}秒")
            print()
    else:
        # 生成视频
        custom_script = None
        if args.script:
            with open(args.script, 'r', encoding='utf-8') as f:
                custom_script = json.load(f)
        
        try:
            output_path = generator.generate_video(
                topic_title=args.topic,
                ip_character=args.ip_character,
                custom_script=custom_script
            )
            print(f"\n✅ 完成！视频已生成: {output_path}")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

