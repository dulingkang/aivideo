#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2.2-final JSON 验证和标准化工具

确保所有场景使用一致的配置，避免因JSON格式不一致导致的问题
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class JSONValidatorV22:
    """v2.2-final JSON 验证和标准化器"""
    
    # 标准配置模板（所有场景应该使用这些默认值）
    STANDARD_CONFIG = {
        "generation_params": {
            "width": 1536,
            "height": 1536,
            "num_inference_steps": 50,  # 标准推理步数
            "guidance_scale": 7.5,
            "seed": -1
        },
        "character": {
            "lora_config": {
                "lora_path": "models/lora/hanli/pytorch_lora_weights.safetensors",
                "weight": 0.9,
                "trigger": "hanli"
            },
            "reference_image": "character_references/hanli_reference.jpg"
        },
        "model_route": {
            "base_model": "flux",
            "identity_engine": "pulid"
        }
    }
    
    def __init__(self):
        self.issues = []
        self.fixed_count = 0
    
    def validate_and_fix(self, scene_path: Path, fix: bool = True) -> Tuple[bool, List[str]]:
        """
        验证并修复 JSON 文件
        
        Args:
            scene_path: JSON 文件路径
            fix: 是否自动修复
            
        Returns:
            (is_valid, issues)
        """
        self.issues = []
        
        try:
            with open(scene_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return False, [f"JSON解析失败: {e}"]
        
        # 检查版本
        if data.get("version") != "v2.2-final":
            self.issues.append(f"版本不正确: {data.get('version')}，应该是 v2.2-final")
        
        if "scene" not in data:
            return False, ["缺少 'scene' 字段"]
        
        scene = data["scene"]
        
        # 1. 检查 generation_params
        gen_params = scene.get("generation_params", {})
        standard_gen = self.STANDARD_CONFIG["generation_params"]
        
        if gen_params.get("num_inference_steps") != standard_gen["num_inference_steps"]:
            issue = f"推理步数不一致: {gen_params.get('num_inference_steps')} (应该是 {standard_gen['num_inference_steps']})"
            self.issues.append(issue)
            if fix:
                gen_params["num_inference_steps"] = standard_gen["num_inference_steps"]
                self.fixed_count += 1
                logger.info(f"  ✓ 修复推理步数: {standard_gen['num_inference_steps']}")
        
        if gen_params.get("width") != standard_gen["width"]:
            issue = f"宽度不一致: {gen_params.get('width')} (应该是 {standard_gen['width']})"
            self.issues.append(issue)
            if fix:
                gen_params["width"] = standard_gen["width"]
                self.fixed_count += 1
        
        if gen_params.get("height") != standard_gen["height"]:
            issue = f"高度不一致: {gen_params.get('height')} (应该是 {standard_gen['height']})"
            self.issues.append(issue)
            if fix:
                gen_params["height"] = standard_gen["height"]
                self.fixed_count += 1
        
        # 2. 检查 character.lora_config
        character = scene.get("character", {})
        if character.get("present", True):
            lora_config = character.get("lora_config", {})
            standard_lora = self.STANDARD_CONFIG["character"]["lora_config"]
            
            if lora_config.get("lora_path") != standard_lora["lora_path"]:
                issue = f"LoRA路径不一致: {lora_config.get('lora_path')} (应该是 {standard_lora['lora_path']})"
                self.issues.append(issue)
                if fix:
                    lora_config["lora_path"] = standard_lora["lora_path"]
                    self.fixed_count += 1
                    logger.info(f"  ✓ 修复LoRA路径: {standard_lora['lora_path']}")
            
            if character.get("reference_image") != self.STANDARD_CONFIG["character"]["reference_image"]:
                issue = f"参考图路径不一致: {character.get('reference_image')}"
                self.issues.append(issue)
                if fix:
                    character["reference_image"] = self.STANDARD_CONFIG["character"]["reference_image"]
                    self.fixed_count += 1
                    logger.info(f"  ✓ 修复参考图路径")
        
        # 3. 检查 model_route
        model_route = scene.get("model_route", {})
        standard_route = self.STANDARD_CONFIG["model_route"]
        
        if model_route.get("base_model") != standard_route["base_model"]:
            issue = f"基础模型不一致: {model_route.get('base_model')} (应该是 {standard_route['base_model']})"
            self.issues.append(issue)
            if fix:
                model_route["base_model"] = standard_route["base_model"]
                self.fixed_count += 1
        
        if model_route.get("identity_engine") != standard_route["identity_engine"]:
            issue = f"身份引擎不一致: {model_route.get('identity_engine')} (应该是 {standard_route['identity_engine']})"
            self.issues.append(issue)
            if fix:
                model_route["identity_engine"] = standard_route["identity_engine"]
                self.fixed_count += 1
        
        # 4. 检查必需字段
        required_fields = [
            "id", "scene_id", "shot", "pose", "character", 
            "prompt", "generation_params", "model_route"
        ]
        for field in required_fields:
            if field not in scene:
                self.issues.append(f"缺少必需字段: {field}")
        
        # 保存修复后的文件
        if fix and self.fixed_count > 0:
            try:
                with open(scene_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"  ✓ 已保存修复后的文件: {scene_path}")
            except Exception as e:
                self.issues.append(f"保存修复文件失败: {e}")
        
        return len(self.issues) == 0, self.issues
    
    def validate_batch(self, scenes_dir: Path, fix: bool = True) -> Dict[str, Any]:
        """
        批量验证和修复场景文件
        
        Args:
            scenes_dir: 场景目录
            fix: 是否自动修复
            
        Returns:
            验证结果统计
        """
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "fixed": 0,
            "issues": []
        }
        
        scene_files = sorted(scenes_dir.glob("scene_*.json"))
        
        for scene_file in scene_files:
            results["total"] += 1
            logger.info(f"验证: {scene_file.name}")
            
            is_valid, issues = self.validate_and_fix(scene_file, fix=fix)
            
            if is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["issues"].append({
                    "file": scene_file.name,
                    "issues": issues
                })
            
            if self.fixed_count > 0:
                results["fixed"] += 1
                self.fixed_count = 0  # 重置计数器
        
        return results


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证和修复 v2.2-final JSON 文件")
    parser.add_argument("scenes_dir", type=Path, help="场景目录路径")
    parser.add_argument("--fix", action="store_true", help="自动修复不一致的配置")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    validator = JSONValidatorV22()
    results = validator.validate_batch(args.scenes_dir, fix=args.fix)
    
    print("=" * 60)
    print("JSON 验证结果")
    print("=" * 60)
    print(f"总计: {results['total']}")
    print(f"有效: {results['valid']}")
    print(f"无效: {results['invalid']}")
    if args.fix:
        print(f"已修复: {results['fixed']}")
    
    if results["issues"]:
        print("\n问题详情:")
        for item in results["issues"]:
            print(f"  {item['file']}:")
            for issue in item["issues"]:
                print(f"    - {issue}")
    
    return 0 if results["invalid"] == 0 else 1


if __name__ == "__main__":
    exit(main())

