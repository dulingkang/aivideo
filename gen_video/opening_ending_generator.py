#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开头和结尾视频生成器
生成固定的开头和结尾视频，可在所有集数中复用
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont

# 添加 InstantID 路径
INSTANTID_PATH = os.environ.get("INSTANTID_PATH", "../InstantID")
if os.path.exists(INSTANTID_PATH):
    sys.path.insert(0, INSTANTID_PATH)

from image_generator import ImageGenerator


class OpeningEndingGenerator:
    """开头和结尾视频生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化生成器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.image_config = self.config.get('image', {})
        self.video_config = self.config.get('video', {})
        self.paths = self.config.get('paths', {})
        
        # 输出目录
        self.output_dir = Path(self.paths.get('output_dir', 'outputs')) / "opening_ending"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化图像生成器（用于生成开头和结尾的图像）
        self.image_generator = ImageGenerator(config_path)
        if self.image_generator.pipeline is None:
            self.image_generator.load_pipeline()
    
    def generate_opening_image(self, episode: Optional[int] = None, title: Optional[str] = None) -> Path:
        """
        生成开头图像
        
        Args:
            episode: 集数（可选，不用于图像生成，文字通过字幕添加）
            title: 标题（可选，不用于图像生成，文字通过字幕添加）
        
        Returns:
            生成的图像路径
        """
        print("\n=== 生成开头图像 ===")
        
        # 构建开头提示词（去掉文字，只生成画面）
        # 画面：云雾缭绕的仙域上空，金色卷轴在空中展开，光芒闪动后卷轴飞入远方云层
        prompt = """cinematic opening, xianxia fantasy,
        ethereal clouds, misty immortal realm sky,
        golden scroll unfurling in the air, ancient Chinese scroll,
        golden light radiating from scroll, scroll flying into distant clouds,
        soft mystical atmosphere, gentle peaceful,
        wide shot, aerial view, slow natural camera movement,
        photorealistic, cinematic lighting, 4k, masterpiece"""
        
        # 强调不要文字，文字会在视频合成时通过字幕添加
        negative_prompt = """text, letters, words, Chinese characters, 
        noisy, chaotic, complex details, dark atmosphere,
        violent, aggressive, modern elements, text overlay,
        fast movement, shaky camera, camera shake"""
        
        output_path = self.output_dir / "opening.png"
        
        try:
            # 使用图像生成器生成开头图像
            generated_path = self.image_generator.generate_image(
                prompt,
                output_path,
                negative_prompt=negative_prompt,
                use_lora=False,  # 开头不使用角色LoRA
            )
            print(f"✓ 开头图像已生成: {generated_path}")
            return generated_path
        except Exception as e:
            print(f"✗ 开头图像生成失败: {e}")
            # 如果生成失败，创建一个简单的占位图像
            return self._create_placeholder_image(output_path, "opening")
    
    def generate_ending_image(self) -> Path:
        """
        生成结尾图像
        
        Returns:
            生成的图像路径
        """
        print("\n=== 生成结尾图像 ===")
        
        # 构建结尾提示词（去掉文字，只生成画面）
        # 画面：夜色下的仙城或古塔剪影，天空洒落灵光，画面淡出到黑色
        prompt = """cinematic ending, xianxia fantasy,
        night scene, dark sky with stars,
        silhouette immortal city ancient pagoda in the distance,
        spiritual light particles falling from sky, gentle light rays,
        mysterious calm atmosphere, peaceful,
        fading to black, gradual darkening,
        wide shot, cinematic, static composition,
        photorealistic, cinematic lighting, 4k, masterpiece"""
        
        # 强调不要文字和快速运动，文字会在视频合成时添加
        negative_prompt = """text, letters, words, Chinese characters,
        bright, daytime, noisy, chaotic, complex details,
        violent, aggressive, modern elements,
        fast movement, shaking, vibrating particles, excessive motion"""
        
        output_path = self.output_dir / "ending.png"
        
        try:
            # 使用图像生成器生成结尾图像
            generated_path = self.image_generator.generate_image(
                prompt,
                output_path,
                negative_prompt=negative_prompt,
                use_lora=False,  # 结尾不使用角色LoRA
            )
            print(f"✓ 结尾图像已生成: {generated_path}")
            return generated_path
        except Exception as e:
            print(f"✗ 结尾图像生成失败: {e}")
            # 如果生成失败，创建一个简单的占位图像
            return self._create_placeholder_image(output_path, "ending")
    
    def _create_placeholder_image(self, output_path: Path, image_type: str) -> Path:
        """创建占位图像（如果生成失败）"""
        width = self.image_config.get('width', 1024)
        height = self.image_config.get('height', 1024)
        
        # 创建渐变背景
        img = Image.new('RGB', (width, height), color=(20, 20, 40) if image_type == "ending" else (240, 240, 255))
        draw = ImageDraw.Draw(img)
        
        # 添加文字
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        text = "Opening" if image_type == "opening" else "Ending"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"✓ 已创建占位图像: {output_path}")
        return output_path
    
    def generate_opening_video(self, episode: Optional[int] = None, title: Optional[str] = None, duration: float = 6.0) -> Path:
        """
        生成开头视频
        
        Args:
            episode: 集数
            title: 标题
            duration: 视频时长（秒）
        
        Returns:
            生成的视频路径
        """
        print(f"\n=== 生成开头视频（{duration}秒）===")
        
        # 生成或获取开头图像
        opening_image = self.output_dir / "opening.png"
        if not opening_image.exists():
            self.generate_opening_image(episode, title)
        
        # 使用视频生成器生成视频
        from video_generator import VideoGenerator
        video_gen = VideoGenerator("config.yaml")
        
        output_path = self.output_dir / "opening.mp4"
        
        # 计算帧数（使用四舍五入，小于0.5向下，大于等于0.5向上，减少累积误差）
        import math
        fps = self.video_config.get('fps', 12)
        num_frames = round(duration * fps)  # 使用标准四舍五入
        
        # 限制最大帧数
        max_frames = self.video_config.get('max_frames', 180)
        if num_frames > max_frames:
            num_frames = max_frames
            duration = num_frames / fps
        
        # 创建场景数据，指定为静态缓慢运动
        scene_data = {
            "duration": duration,
            "visual": {
                "motion": "gentle, subtle, slow, static"
            }
        }
        
        try:
            video_gen.generate_video(
                str(opening_image),
                str(output_path),
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=3,  # 进一步降低，减少闪烁，保持轻微稳定运动（完全静态可能不稳定）
                noise_aug_strength=0.0005,  # 进一步降低噪声，减少闪烁
                scene=scene_data,  # 传递场景数据，用于自动降低运动参数
            )
            print(f"✓ 开头视频已生成: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ 开头视频生成失败: {e}")
            raise
    
    def generate_ending_video(self, duration: float = 5.0) -> Path:
        """
        生成结尾视频
        
        Args:
            duration: 视频时长（秒）
        
        Returns:
            生成的视频路径
        """
        print(f"\n=== 生成结尾视频（{duration}秒）===")
        
        # 生成或获取结尾图像
        ending_image = self.output_dir / "ending.png"
        if not ending_image.exists():
            self.generate_ending_image()
        
        if not ending_image.exists():
            raise FileNotFoundError(f"结尾图像生成失败: {ending_image}")
        
        # 使用视频生成器生成视频
        from video_generator import VideoGenerator
        video_gen = VideoGenerator("config.yaml")
        
        output_path = self.output_dir / "ending.mp4"
        
        # 计算帧数（使用四舍五入，小于0.5向下，大于等于0.5向上，减少累积误差）
        import math
        fps = self.video_config.get('fps', 12)
        num_frames = round(duration * fps)  # 使用标准四舍五入
        
        # 限制最大帧数
        max_frames = self.video_config.get('max_frames', 180)
        if num_frames > max_frames:
            num_frames = max_frames
            duration = num_frames / fps
        
        # 创建场景数据，指定为静态缓慢运动，减少粒子晃动
        scene_data = {
            "duration": duration,
            "visual": {
                "motion": "gentle, subtle, slow, static, calm"
            }
        }
        
        try:
            video_gen.generate_video(
                str(ending_image),
                str(output_path),
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=3,  # 进一步降低，减少闪烁和粒子晃动，保持轻微稳定运动
                noise_aug_strength=0.0005,  # 进一步降低噪声，减少闪烁
                scene=scene_data,  # 传递场景数据，用于自动降低运动参数
            )
            print(f"✓ 结尾视频已生成: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ 结尾视频生成失败: {e}")
            raise
    
    def ensure_opening_ending_videos(
        self, 
        episode: Optional[int] = None, 
        title: Optional[str] = None,
        opening_duration: Optional[float] = None,
        ending_duration: Optional[float] = None,
    ) -> tuple[Path, Path]:
        """
        确保开头和结尾视频存在，如果不存在则生成
        
        Args:
            episode: 集数
            title: 标题
            opening_duration: 开头视频时长（秒），如果为None则使用默认6秒
            ending_duration: 结尾视频时长（秒），如果为None则使用默认5秒
        
        Returns:
            (opening_video_path, ending_video_path)
        """
        opening_path = self.output_dir / "opening.mp4"
        ending_path = self.output_dir / "ending.mp4"
        
        # 检查开头视频
        if not opening_path.exists():
            print("\n[开头视频] 不存在，开始生成...")
            try:
                duration = opening_duration or 6.0
                self.generate_opening_video(episode, title, duration)
            except Exception as e:
                print(f"⚠ 开头视频生成失败: {e}")
                print("  将跳过开头视频")
        else:
            print(f"\n[开头视频] ✓ 使用已有: {opening_path}")
        
        # 检查结尾视频
        if not ending_path.exists():
            print("\n[结尾视频] 不存在，开始生成...")
            try:
                duration = ending_duration or 5.0
                self.generate_ending_video(duration)
            except Exception as e:
                print(f"⚠ 结尾视频生成失败: {e}")
                print("  将跳过结尾视频")
        else:
            print(f"\n[结尾视频] ✓ 使用已有: {ending_path}")
        
        return opening_path, ending_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成开头和结尾视频")
    parser.add_argument("--episode", type=int, help="集数")
    parser.add_argument("--title", type=str, help="标题")
    parser.add_argument("--opening-duration", type=float, default=6.0, help="开头视频时长（秒）")
    parser.add_argument("--ending-duration", type=float, default=5.0, help="结尾视频时长（秒）")
    parser.add_argument("--opening-only", action="store_true", help="只生成开头")
    parser.add_argument("--ending-only", action="store_true", help="只生成结尾")
    
    args = parser.parse_args()
    
    generator = OpeningEndingGenerator()
    
    if args.opening_only:
        generator.generate_opening_video(args.episode, args.title, args.opening_duration)
    elif args.ending_only:
        generator.generate_ending_video(args.ending_duration)
    else:
        generator.generate_opening_video(args.episode, args.title, args.opening_duration)
        generator.generate_ending_video(args.ending_duration)
    
    print("\n✓ 完成！")

