#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分析工具
对比生成的prompt和实际图片，找出可以优化的地方
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import torch


class ImageAnalyzer:
    """图像分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.clip_model = None
        self.clip_processor = None
        self.clip_tokenizer = None
        self._load_clip_model()
    
    def _load_clip_model(self):
        """加载CLIP模型用于图像分析"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            print("  加载CLIP模型用于图像分析...")
            model_name = "openai/clip-vit-large-patch14"
            self.clip_model = CLIPModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            print("  ✓ CLIP模型加载成功")
        except Exception as e:
            print(f"  ⚠ CLIP模型加载失败: {e}")
            print("  ℹ 将使用基础图像分析（不依赖CLIP）")
            self.clip_model = None
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        scene: Optional[Dict[str, Any]] = None,
        actual_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析图像，对比prompt和实际内容
        
        Args:
            image_path: 图像路径
            prompt: 生成的prompt
            scene: 场景JSON数据（可选）
            
        Returns:
            {
                "prompt_analysis": Dict,  # Prompt分析结果
                "image_analysis": Dict,  # 图像分析结果
                "comparison": Dict,  # 对比分析结果
                "suggestions": List[str],  # 优化建议
            }
        """
        print(f"\n分析图像: {os.path.basename(image_path)}")
        
        # 使用实际prompt（如果提供）
        prompt_to_analyze = actual_prompt if actual_prompt else prompt
        print(f"Prompt: {prompt_to_analyze[:100]}...")
        
        # 1. 分析Prompt
        prompt_analysis = self._analyze_prompt(prompt_to_analyze, scene)
        
        # 2. 分析图像
        image_analysis = self._analyze_image_content(image_path, prompt_to_analyze)
        
        # 3. 对比分析
        comparison = self._compare_prompt_and_image(prompt_analysis, image_analysis)
        
        # 4. 生成优化建议
        suggestions = self._generate_suggestions(prompt_analysis, image_analysis, comparison, scene)
        
        return {
            "prompt_analysis": prompt_analysis,
            "image_analysis": image_analysis,
            "comparison": comparison,
            "suggestions": suggestions,
        }
    
    def _analyze_prompt(self, prompt: str, scene: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """分析Prompt内容"""
        prompt_lower = prompt.lower()
        
        # 提取关键元素
        # 注意：如果prompt中有"no person"或"no character"，则不应该认为has_character=True
        has_exclusion = any(kw in prompt_lower for kw in ["no person", "no character", "no human", "no people", "无人物", "无角色", "无人"])
        elements = {
            "has_character": not has_exclusion and any(kw in prompt_lower for kw in ["character", "person", "han li", "韩立", "人物", "角色"]),
            "has_object": any(kw in prompt_lower for kw in ["object", "scroll", "item", "物体", "物品", "卷轴"]),
            "has_environment": any(kw in prompt_lower for kw in ["sky", "desert", "ground", "gravel", "sand", "天空", "沙漠", "地面", "砂砾", "沙砾"]),
            "viewpoint": self._extract_viewpoint(prompt_lower),
            "shot_type": self._extract_shot_type(prompt_lower),
            "action_type": self._extract_action_type(prompt_lower),
            "facing_direction": self._extract_facing_direction(prompt_lower),
        }
        
        # 从scene中提取期望内容
        expected = {}
        if scene:
            visual = scene.get("visual", {})
            if isinstance(visual, dict):
                expected["character_pose"] = visual.get("character_pose", "")
                expected["composition"] = visual.get("composition", "")
                expected["camera"] = scene.get("camera", "")
                expected["action"] = scene.get("action", "")
        
        return {
            "elements": elements,
            "expected": expected,
            "prompt_text": prompt,
        }
    
    def _analyze_image_content(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """分析图像内容"""
        if not os.path.exists(image_path):
            return {"error": "图像文件不存在"}
        
        image = Image.open(image_path)
        width, height = image.size
        
        # 基础图像分析
        analysis = {
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": width / height,
        }
        
        # 使用CLIP分析图像内容（如果可用）
        if self.clip_model is not None:
            clip_analysis = self._analyze_with_clip(image, prompt)
            analysis.update(clip_analysis)
        else:
            # 基础分析（不依赖CLIP）
            analysis["detected_elements"] = self._basic_image_analysis(image)
        
        return analysis
    
    def _analyze_with_clip(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """使用CLIP模型分析图像"""
        try:
            # 定义检测项
            check_items = [
                "character facing camera",
                "character from behind",
                "character side view",
                "close-up shot",
                "wide shot",
                "medium shot",
                "golden scroll",
                "desert background",
                "sky background",
                "character with action",
                "static character",
            ]
            
            # 使用CLIP计算相似度
            inputs = self.clip_processor(
                text=check_items,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.clip_model.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # 提取高相似度的项
            detected = {}
            for i, item in enumerate(check_items):
                score = probs[0][i].item()
                if score > 0.1:  # 阈值
                    detected[item] = score
            
            return {
                "detected_elements": detected,
                "primary_element": max(detected.items(), key=lambda x: x[1])[0] if detected else None,
            }
        except Exception as e:
            print(f"  ⚠ CLIP分析失败: {e}")
            return {"detected_elements": {}}
    
    def _basic_image_analysis(self, image: Image.Image) -> Dict[str, float]:
        """基础图像分析（不依赖CLIP）"""
        # 简单的图像分析
        # 这里可以添加基于像素的分析，如检测主要颜色、亮度等
        return {
            "analysis_method": "basic",
            "note": "需要CLIP模型进行详细分析"
        }
    
    def _extract_viewpoint(self, prompt_lower: str) -> str:
        """从prompt中提取视角"""
        if any(kw in prompt_lower for kw in ["facing camera", "front view", "正面", "面向镜头"]):
            return "front"
        elif any(kw in prompt_lower for kw in ["back view", "from behind", "背影", "背后"]):
            return "back"
        elif any(kw in prompt_lower for kw in ["side view", "profile", "侧面", "侧身"]):
            return "side"
        elif any(kw in prompt_lower for kw in ["top-down", "aerial", "俯视", "鸟瞰"]):
            return "top"
        else:
            return "unknown"
    
    def _extract_shot_type(self, prompt_lower: str) -> str:
        """从prompt中提取镜头类型"""
        if any(kw in prompt_lower for kw in ["close-up", "extreme close-up", "特写", "近景"]):
            return "close-up"
        elif any(kw in prompt_lower for kw in ["wide shot", "distant", "远景", "远距离"]):
            return "wide"
        elif any(kw in prompt_lower for kw in ["medium shot", "中景", "半身"]):
            return "medium"
        else:
            return "unknown"
    
    def _extract_action_type(self, prompt_lower: str) -> str:
        """从prompt中提取动作类型"""
        if any(kw in prompt_lower for kw in ["attack", "fight", "run", "jump", "攻击", "战斗", "奔跑", "跳跃"]):
            return "dynamic"
        elif any(kw in prompt_lower for kw in ["tilt", "turn", "move", "侧", "转", "移动"]):
            return "moderate"
        elif any(kw in prompt_lower for kw in ["still", "motionless", "静止", "不动"]):
            return "static"
        else:
            return "unknown"
    
    def _extract_facing_direction(self, prompt_lower: str) -> str:
        """从prompt中提取朝向"""
        if any(kw in prompt_lower for kw in ["facing camera", "front view", "正面", "面向镜头"]):
            return "front"
        elif any(kw in prompt_lower for kw in ["back view", "from behind", "背影", "背后"]):
            return "back"
        else:
            return "unknown"
    
    def _compare_prompt_and_image(
        self,
        prompt_analysis: Dict[str, Any],
        image_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """对比Prompt和图像"""
        comparison = {
            "matches": [],
            "mismatches": [],
            "missing": [],
            "extra": [],
        }
        
        prompt_elements = prompt_analysis.get("elements", {})
        image_elements = image_analysis.get("detected_elements", {})
        
        # 检查视角匹配
        expected_viewpoint = prompt_elements.get("viewpoint")
        if expected_viewpoint != "unknown":
            if "character facing camera" in image_elements and expected_viewpoint == "front":
                comparison["matches"].append("视角：正面")
            elif "character from behind" in image_elements and expected_viewpoint == "back":
                comparison["matches"].append("视角：背面")
            elif expected_viewpoint == "front" and "character from behind" in image_elements:
                comparison["mismatches"].append("视角不匹配：期望正面，实际背面")
            elif expected_viewpoint == "back" and "character facing camera" in image_elements:
                comparison["mismatches"].append("视角不匹配：期望背面，实际正面")
        
        # 检查镜头类型匹配
        expected_shot = prompt_elements.get("shot_type")
        if expected_shot != "unknown":
            if "close-up shot" in image_elements and expected_shot == "close-up":
                comparison["matches"].append("镜头类型：特写")
            elif "wide shot" in image_elements and expected_shot == "wide":
                comparison["matches"].append("镜头类型：远景")
            elif "medium shot" in image_elements and expected_shot == "medium":
                comparison["matches"].append("镜头类型：中景")
        
        # 检查角色存在
        if prompt_elements.get("has_character"):
            # 检查各种角色视角：正面、背面、侧面、动作、静态
            character_detected = (
                "character facing camera" in image_elements or 
                "character from behind" in image_elements or
                "character side view" in image_elements or
                "character with action" in image_elements or
                "static character" in image_elements
            )
            if character_detected:
                comparison["matches"].append("角色存在")
            else:
                comparison["missing"].append("角色未检测到")
        
        # 检查物体存在
        if prompt_elements.get("has_object"):
            if "golden scroll" in image_elements:
                comparison["matches"].append("物体存在：卷轴")
            else:
                comparison["missing"].append("物体未检测到：卷轴")
        
        return comparison
    
    def _generate_suggestions(
        self,
        prompt_analysis: Dict[str, Any],
        image_analysis: Dict[str, Any],
        comparison: Dict[str, Any],
        scene: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于不匹配项生成建议
        for mismatch in comparison.get("mismatches", []):
            if "视角不匹配" in mismatch:
                if "期望正面，实际背面" in mismatch:
                    suggestions.append("⚠ 视角问题：期望正面但生成了背面。建议在prompt中增加 '(facing camera, front view:1.8)' 并添加负面提示 'back view, from behind'")
                elif "期望背面，实际正面" in mismatch:
                    suggestions.append("ℹ 视角问题：期望背面但生成了正面。如果确实需要背面，建议在prompt中明确添加 'back view, from behind'")
        
        # 基于缺失项生成建议
        for missing in comparison.get("missing", []):
            if "角色未检测到" in missing:
                suggestions.append("⚠ 角色缺失：prompt中描述了角色但图像中未检测到。建议检查prompt中角色描述的权重和位置")
            elif "物体未检测到" in missing:
                suggestions.append("⚠ 物体缺失：prompt中描述了物体但图像中未检测到。建议在prompt最前面添加物体描述，使用高权重（如2.0）")
        
        # 基于镜头类型生成建议
        prompt_elements = prompt_analysis.get("elements", {})
        expected_shot = prompt_elements.get("shot_type")
        if expected_shot == "close-up" and "close-up shot" not in image_analysis.get("detected_elements", {}):
            suggestions.append("⚠ 镜头距离问题：期望特写但实际可能是中景或远景。建议在prompt中明确添加 'extreme close-up' 或 'close-up' 并提高权重")
        elif expected_shot == "wide" and "wide shot" not in image_analysis.get("detected_elements", {}):
            suggestions.append("⚠ 镜头距离问题：期望远景但实际可能是中景或特写。建议在prompt中明确添加 'wide shot' 或 'distant view' 并提高权重")
        
        # 基于动作类型生成建议
        action_type = prompt_elements.get("action_type")
        if action_type == "dynamic" and "character with action" not in image_analysis.get("detected_elements", {}):
            suggestions.append("ℹ 动作问题：期望动态动作但可能不够明显。建议在prompt中明确描述动作细节，如 'attacking', 'running' 等")
        
        # 基于场景数据生成建议
        if scene:
            visual = scene.get("visual", {})
            if isinstance(visual, dict):
                character_pose = visual.get("character_pose", "")
                if character_pose and "facing camera" in character_pose.lower():
                    # 如果期望正面但实际是背面
                    if "character from behind" in image_analysis.get("detected_elements", {}):
                        suggestions.append("🔴 关键问题：character_pose中指定了'facing camera'但生成了背面。建议：1) 增加正面朝向权重至1.8；2) 添加负面提示'back view, from behind'；3) 检查prompt中是否有冲突的描述")
        
        # 基于镜头距离生成建议
        image_detected = image_analysis.get("detected_elements", {})
        if "wide shot" in image_detected and prompt_elements.get("shot_type") == "close-up":
            suggestions.append("🔴 关键问题：期望特写但生成了远景。建议：1) 在prompt最前面添加 '(extreme close-up:2.0)'；2) 添加负面提示 'wide shot, distant view'；3) 检查camera字段是否正确")
        elif "close-up shot" in image_detected and prompt_elements.get("shot_type") == "wide":
            suggestions.append("🔴 关键问题：期望远景但生成了特写。建议：1) 在prompt中添加 '(wide shot, distant view:1.8)'；2) 添加负面提示 'close-up, extreme close-up'")
        
        return suggestions
    
    def analyze_batch(
        self,
        scenes: List[Dict[str, Any]],
        image_dir: str,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量分析图像
        
        Args:
            scenes: 场景列表（包含prompt和image_path）
            image_dir: 图像目录
            output_file: 输出文件路径（可选）
        """
        results = []
        
        for i, scene in enumerate(scenes):
            image_path = scene.get("image_path")
            if not image_path:
                continue
            
            # 构建完整路径
            if not os.path.isabs(image_path):
                full_path = os.path.join(image_dir, image_path)
            else:
                full_path = image_path
            
            if not os.path.exists(full_path):
                print(f"  ⚠ 场景 {i+1}: 图像不存在: {full_path}")
                continue
            
            # 获取prompt（从scene或需要重新生成）
            prompt = scene.get("prompt") or scene.get("description") or ""
            
            # 分析图像
            try:
                result = self.analyze_image(full_path, prompt, scene)
                result["scene_id"] = scene.get("id", i)
                result["image_path"] = full_path
                results.append(result)
                print(f"  ✓ 场景 {i+1} 分析完成")
            except Exception as e:
                print(f"  ✗ 场景 {i+1} 分析失败: {e}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 分析结果已保存到: {output_file}")
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """生成分析报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("图像分析报告")
        report_lines.append("=" * 80)
        
        for result in results:
            scene_id = result.get("scene_id", "?")
            image_path = result.get("image_path", "?")
            
            report_lines.append(f"\n场景 {scene_id}: {os.path.basename(image_path)}")
            report_lines.append("-" * 80)
            
            # 对比结果
            comparison = result.get("comparison", {})
            if comparison.get("matches"):
                report_lines.append("✓ 匹配项:")
                for match in comparison["matches"]:
                    report_lines.append(f"  - {match}")
            
            if comparison.get("mismatches"):
                report_lines.append("✗ 不匹配项:")
                for mismatch in comparison["mismatches"]:
                    report_lines.append(f"  - {mismatch}")
            
            if comparison.get("missing"):
                report_lines.append("⚠ 缺失项:")
                for missing in comparison["missing"]:
                    report_lines.append(f"  - {missing}")
            
            # 优化建议
            suggestions = result.get("suggestions", [])
            if suggestions:
                report_lines.append("\n💡 优化建议:")
                for suggestion in suggestions:
                    report_lines.append(f"  {suggestion}")
        
        return "\n".join(report_lines)

