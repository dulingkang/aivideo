#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片生成辅助功能：
1. 为开始和结束场景设置特殊处理，避免重复生成
2. 可以在图片上叠加标题文字
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

def should_skip_scene_generation(scene: Dict[str, Any]) -> bool:
    """
    判断是否应该跳过场景图片生成
    
    Args:
        scene: 场景数据
    
    Returns:
        True 如果应该跳过（使用已有图片），False 如果需要生成
    """
    scene_id = scene.get("id")
    
    # 开始场景（id=0）和结束场景（id=999）如果已有图片路径，就跳过生成
    if scene_id in [0, 999]:
        image_path = scene.get("image_path")
        if image_path and Path(image_path).exists():
            return True
    
    return False

def get_shared_image_path(scene_id: int, episode: Optional[int] = None) -> Optional[str]:
    """
    获取共享图片路径
    
    Args:
        scene_id: 场景ID（0=开始，999=结束）
        episode: 集数（可选，用于开始场景）
    
    Returns:
        共享图片路径，如果不存在则返回None
    """
    shared_dir = Path("gen_video/shared_images")
    
    if scene_id == 0:
        # 开始场景：优先使用带集数的图片，否则使用模板
        if episode:
            ep_image = shared_dir / f"opening_ep{episode}.png"
            if ep_image.exists():
                return str(ep_image)
        template_image = shared_dir / "opening_template.png"
        if template_image.exists():
            return str(template_image)
    
    elif scene_id == 999:
        # 结束场景：使用模板
        template_image = shared_dir / "ending_template.png"
        if template_image.exists():
            return str(template_image)
    
    return None

# 使用说明：
# 在 image_generator.py 的 generate_from_script 方法中：
# 
# for idx, scene in enumerate(scenes, start=1):
#     scene_id = scene.get("id")
#     
#     # 检查是否应该跳过生成（开始/结束场景且有图片）
#     if should_skip_scene_generation(scene):
#         print(f"跳过场景 {scene_id}（使用已有图片）")
#         continue
#     
#     # 如果没有图片路径，尝试使用共享图片
#     if not scene.get("image_path"):
#         shared_path = get_shared_image_path(scene_id, episode)
#         if shared_path:
#             scene["image_path"] = shared_path
#             print(f"使用共享图片: {shared_path}")
#             continue
#     
#     # ... 正常生成图片的代码 ...

