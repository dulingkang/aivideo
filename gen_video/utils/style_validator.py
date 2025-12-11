#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风格配置验证工具
用于验证风格配置是否正确，并提供风格列表
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml


class StyleValidator:
    """风格配置验证器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化验证器"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = None
        self.style_templates = {}
        self.default_style = "realistic"
        
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        hunyuan_config = self.config.get('video', {}).get('hunyuanvideo', {})
        self.style_templates = hunyuan_config.get('style_templates', {})
        self.default_style = hunyuan_config.get('default_style', 'realistic')
    
    def get_available_styles(self) -> List[str]:
        """获取所有可用的风格列表"""
        return list(self.style_templates.keys())
    
    def validate_style(self, style_name: str) -> bool:
        """验证风格名称是否有效"""
        if not style_name:
            return False
        return style_name in self.style_templates
    
    def get_style_template(self, style_name: str) -> Optional[Dict[str, Any]]:
        """获取风格模板"""
        return self.style_templates.get(style_name)
    
    def get_default_style(self) -> str:
        """获取默认风格"""
        return self.default_style
    
    def validate_scene_style(self, scene: Dict[str, Any]) -> tuple[bool, str, Optional[str]]:
        """
        验证场景中的风格配置
        
        Returns:
            (is_valid, style_name, error_message)
        """
        # 优先从visual.style获取
        visual = scene.get('visual', {})
        if isinstance(visual, dict):
            style_name = visual.get('style', '')
            if style_name:
                if self.validate_style(style_name):
                    return True, style_name, None
                else:
                    return False, style_name, f"无效的风格名称: {style_name}，可用风格: {', '.join(self.get_available_styles())}"
        
        # 从scene.style获取
        style_name = scene.get('style', '')
        if style_name:
            if self.validate_style(style_name):
                return True, style_name, None
            else:
                return False, style_name, f"无效的风格名称: {style_name}，可用风格: {', '.join(self.get_available_styles())}"
        
        # 使用默认风格
        return True, self.default_style, None
    
    def get_style_info(self, style_name: Optional[str] = None) -> Dict[str, Any]:
        """获取风格信息"""
        if style_name is None:
            style_name = self.default_style
        
        template = self.style_templates.get(style_name)
        if template:
            return {
                "name": style_name,
                "keywords": template.get('keywords', []),
                "description": template.get('description', ''),
                "negative_keywords": template.get('negative_keywords', [])
            }
        else:
            return {
                "name": style_name,
                "error": f"风格不存在: {style_name}"
            }


def validate_style_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    验证风格配置文件的完整性
    
    Returns:
        验证结果字典
    """
    try:
        validator = StyleValidator(config_path)
        
        result = {
            "valid": True,
            "available_styles": validator.get_available_styles(),
            "default_style": validator.get_default_style(),
            "style_count": len(validator.style_templates),
            "errors": []
        }
        
        # 验证每个风格模板
        for style_name, template in validator.style_templates.items():
            if not isinstance(template, dict):
                result["errors"].append(f"风格 '{style_name}' 的模板格式错误：必须是字典")
                result["valid"] = False
                continue
            
            if 'keywords' not in template and 'description' not in template:
                result["errors"].append(f"风格 '{style_name}' 缺少 keywords 或 description")
                result["valid"] = False
        
        # 验证默认风格是否存在
        if validator.default_style not in validator.style_templates:
            result["errors"].append(f"默认风格 '{validator.default_style}' 不存在于 style_templates 中")
            result["valid"] = False
        
        return result
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "available_styles": [],
            "default_style": "realistic",
            "style_count": 0,
            "errors": [str(e)]
        }


if __name__ == "__main__":
    """测试风格验证器"""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "gen_video" / "config.yaml"
    
    print("=" * 60)
    print("风格配置验证")
    print("=" * 60)
    
    # 验证配置
    result = validate_style_config(str(config_path))
    
    if result["valid"]:
        print(f"\n✓ 风格配置有效")
        print(f"  可用风格数量: {result['style_count']}")
        print(f"  默认风格: {result['default_style']}")
        print(f"\n可用风格列表:")
        for style in result["available_styles"]:
            print(f"  - {style}")
    else:
        print(f"\n✗ 风格配置无效")
        print(f"  错误:")
        for error in result["errors"]:
            print(f"    - {error}")
        sys.exit(1)
    
    # 测试场景验证
    print("\n" + "=" * 60)
    print("场景风格验证测试")
    print("=" * 60)
    
    validator = StyleValidator(str(config_path))
    
    test_scenes = [
        {"visual": {"style": "scientific"}},
        {"style": "commercial"},
        {"visual": {"style": "invalid_style"}},
        {},
    ]
    
    for i, scene in enumerate(test_scenes, 1):
        is_valid, style_name, error = validator.validate_scene_style(scene)
        print(f"\n测试场景 {i}:")
        print(f"  场景数据: {scene}")
        if is_valid:
            print(f"  ✓ 有效，使用风格: {style_name}")
        else:
            print(f"  ✗ 无效: {error}")

