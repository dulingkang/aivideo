#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt增强工具
用于根据图像内容自动增强prompt，使其更符合HunyuanVideo的要求
"""

from typing import Dict, Any, Optional
from PIL import Image
import os


class PromptEnhancer:
    """Prompt增强器"""
    
    def __init__(self):
        """初始化增强器"""
        self.quality_keywords = [
            "high quality", "cinematic", "detailed", "professional",
            "photorealistic", "excellent composition", "visual appeal"
        ]
        
        self.motion_keywords = {
            "gentle": ["gentle motion", "slow movement", "peaceful atmosphere", "smooth transition"],
            "moderate": ["smooth motion", "natural movement", "fluid animation"],
            "dynamic": ["dynamic motion", "fast movement", "energetic action", "vibrant movement"]
        }
        
        self.camera_keywords = {
            "static": ["stable camera", "fixed position"],
            "pan": ["smooth camera pan", "horizontal movement"],
            "zoom": ["gentle camera zoom", "gradual zoom"],
            "dolly": ["smooth camera dolly movement", "forward/backward movement"]
        }
    
    def enhance_prompt(
        self,
        base_prompt: str,
        scene: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None
    ) -> str:
        """
        增强prompt，使其更详细和符合HunyuanVideo要求
        
        Args:
            base_prompt: 基础prompt
            scene: 场景配置
            image_path: 图像路径（可选，用于分析图像内容）
        
        Returns:
            增强后的prompt
        """
        enhanced_parts = []
        
        # 1. 添加基础描述
        if base_prompt:
            enhanced_parts.append(base_prompt)
        elif scene:
            description = scene.get('description', '') or scene.get('prompt', '')
            if description:
                enhanced_parts.append(description)
        
        # 2. 添加视觉细节
        if scene:
            visual = scene.get('visual', {})
            if isinstance(visual, dict):
                composition = visual.get('composition', '')
                if composition:
                    enhanced_parts.append(composition)
                
                lighting = visual.get('lighting', '')
                if lighting:
                    enhanced_parts.append(f"lighting: {lighting}")
                
                mood = visual.get('mood', '')
                if mood:
                    enhanced_parts.append(f"mood: {mood}")
        
        # 3. 添加运动描述
        motion_intensity = scene.get('motion_intensity', 'moderate') if scene else 'moderate'
        if motion_intensity in self.motion_keywords:
            enhanced_parts.extend(self.motion_keywords[motion_intensity][:2])  # 使用前2个关键词
        
        # 4. 添加镜头运动
        if scene:
            camera_motion = scene.get('camera_motion', {})
            if isinstance(camera_motion, dict):
                camera_type = camera_motion.get('type', 'static')
                if camera_type in self.camera_keywords:
                    enhanced_parts.append(self.camera_keywords[camera_type][0])
        
        # 5. 添加质量关键词
        enhanced_parts.extend(self.quality_keywords[:3])  # 使用前3个质量关键词
        
        # 6. 组合所有部分
        enhanced_prompt = ", ".join(enhanced_parts)
        
        # 7. 确保prompt足够详细
        word_count = len(enhanced_prompt.split())
        if word_count < 20:
            # 如果prompt太短，添加通用描述
            enhanced_prompt += ". The scene is rich in detail with excellent composition and visual appeal"
        
        return enhanced_prompt
    
    def analyze_image_for_prompt(self, image_path: str) -> Dict[str, Any]:
        """
        分析图像内容，提取可用于prompt的信息
        
        Args:
            image_path: 图像路径
        
        Returns:
            分析结果字典
        """
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # 简单的图像分析
            analysis = {
                "resolution": f"{width}x{height}",
                "aspect_ratio": width / height if height > 0 else 1.0,
                "is_landscape": width > height,
                "is_portrait": height > width,
                "is_square": abs(width - height) < 10
            }
            
            # 根据文件名提取信息
            filename = os.path.basename(image_path)
            filename_lower = filename.lower()
            
            if "scene" in filename_lower:
                analysis["type"] = "scene"
            elif "face" in filename_lower or "portrait" in filename_lower:
                analysis["type"] = "portrait"
            else:
                analysis["type"] = "general"
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def suggest_prompt_improvements(self, current_prompt: str) -> list:
        """
        建议prompt改进
        
        Args:
            current_prompt: 当前prompt
        
        Returns:
            改进建议列表
        """
        suggestions = []
        
        word_count = len(current_prompt.split())
        
        if word_count < 15:
            suggestions.append("Prompt太短，建议添加更多细节描述（至少15-20个词）")
        
        if "style" not in current_prompt.lower():
            suggestions.append("建议添加风格描述（如：realistic, cinematic, scientific）")
        
        if "motion" not in current_prompt.lower() and "movement" not in current_prompt.lower():
            suggestions.append("建议添加运动描述（如：gentle motion, smooth movement）")
        
        if "quality" not in current_prompt.lower() and "high quality" not in current_prompt.lower():
            suggestions.append("建议添加质量关键词（如：high quality, cinematic, detailed）")
        
        if word_count > 100:
            suggestions.append("Prompt可能过长，建议精简到50-80个词")
        
        return suggestions


if __name__ == "__main__":
    """测试Prompt增强器"""
    enhancer = PromptEnhancer()
    
    # 测试场景
    scene = {
        "description": "a black hole in space",
        "motion_intensity": "gentle",
        "camera_motion": {"type": "slow_zoom"},
        "visual": {
            "style": "scientific",
            "composition": "wide shot"
        }
    }
    
    # 增强prompt
    enhanced = enhancer.enhance_prompt("", scene=scene)
    print(f"增强后的prompt: {enhanced}")
    
    # 获取改进建议
    suggestions = enhancer.suggest_prompt_improvements(enhanced)
    if suggestions:
        print(f"\n改进建议:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

