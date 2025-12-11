#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deforum风格的相机运动模块
实现参数化的相机运动（zoom、pan、rotate），用于静态图像动画化
参考：ComfyUI Deforum
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
import math


class DeforumCameraMotion:
    """Deforum风格的相机运动生成器"""
    
    def __init__(self):
        """初始化相机运动生成器"""
        pass
    
    def apply_camera_motion(
        self,
        image: Image.Image,
        num_frames: int,
        motion_params: Dict[str, Any],
        curve: str = "ease_in_out",
    ) -> List[np.ndarray]:
        """
        应用Deforum风格的相机运动
        
        Args:
            image: 输入图像
            num_frames: 帧数
            motion_params: 运动参数
                {
                    "zoom": {"start": 1.0, "end": 1.1},
                    "pan_x": {"start": 0.0, "end": 0.05},
                    "pan_y": {"start": 0.0, "end": 0.0},
                    "rotate": {"start": 0.0, "end": 0.0},
                }
            curve: 运动曲线类型（"linear", "ease_in", "ease_out", "ease_in_out"）
        
        Returns:
            视频帧列表（numpy数组）
        """
        width, height = image.size
        frames = []
        
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0.0  # 0.0 to 1.0
            
            # 应用运动曲线
            t_eased = self._apply_curve(t, curve)
            
            # 插值参数
            zoom = self._interpolate(
                motion_params.get("zoom", {}).get("start", 1.0),
                motion_params.get("zoom", {}).get("end", 1.0),
                t_eased
            )
            pan_x = self._interpolate(
                motion_params.get("pan_x", {}).get("start", 0.0),
                motion_params.get("pan_x", {}).get("end", 0.0),
                t_eased
            )
            pan_y = self._interpolate(
                motion_params.get("pan_y", {}).get("start", 0.0),
                motion_params.get("pan_y", {}).get("end", 0.0),
                t_eased
            )
            rotate = self._interpolate(
                motion_params.get("rotate", {}).get("start", 0.0),
                motion_params.get("rotate", {}).get("end", 0.0),
                t_eased
            )
            
            # 应用变换
            frame = self._apply_transform(image, zoom, pan_x, pan_y, rotate, width, height)
            frames.append(np.array(frame))
        
        return frames
    
    def _apply_curve(self, t: float, curve: str) -> float:
        """应用运动曲线"""
        if curve == "linear":
            return t
        elif curve == "ease_in":
            return t * t
        elif curve == "ease_out":
            return 1.0 - (1.0 - t) * (1.0 - t)
        elif curve == "ease_in_out":
            # smoothstep函数
            return t * t * (3.0 - 2.0 * t)
        else:
            return t
    
    def _interpolate(self, start: float, end: float, t: float) -> float:
        """线性插值"""
        return start + (end - start) * t
    
    def _apply_transform(
        self,
        image: Image.Image,
        zoom: float,
        pan_x: float,
        pan_y: float,
        rotate: float,
        target_width: int,
        target_height: int,
    ) -> Image.Image:
        """
        应用相机变换（zoom、pan、rotate）
        
        Args:
            image: 输入图像
            zoom: 缩放比例（1.0=无缩放，>1.0=放大，<1.0=缩小）
            pan_x: 水平平移（-1.0到1.0，相对于图像宽度）
            pan_y: 垂直平移（-1.0到1.0，相对于图像高度）
            rotate: 旋转角度（度）
            target_width: 目标宽度
            target_height: 目标高度
        
        Returns:
            变换后的图像
        """
        # 计算缩放后的尺寸
        zoomed_width = int(image.width * zoom)
        zoomed_height = int(image.height * zoom)
        
        # 缩放图像
        if zoom != 1.0:
            zoomed_image = image.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
        else:
            zoomed_image = image.copy()
        
        # 计算平移偏移（相对于原始图像尺寸）
        pan_offset_x = int(pan_x * image.width)
        pan_offset_y = int(pan_y * image.height)
        
        # 计算裁剪区域（中心对齐，考虑平移）
        crop_left = (zoomed_width - target_width) // 2 - pan_offset_x
        crop_top = (zoomed_height - target_height) // 2 - pan_offset_y
        crop_right = crop_left + target_width
        crop_bottom = crop_top + target_height
        
        # 确保裁剪区域在图像范围内
        crop_left = max(0, min(zoomed_width, crop_left))
        crop_top = max(0, min(zoomed_height, crop_top))
        crop_right = max(crop_left, min(zoomed_width, crop_right))
        crop_bottom = max(crop_top, min(zoomed_height, crop_bottom))
        
        # 裁剪
        if crop_right > crop_left and crop_bottom > crop_top:
            cropped = zoomed_image.crop((crop_left, crop_top, crop_right, crop_bottom))
        else:
            cropped = zoomed_image
        
        # 如果裁剪后尺寸不对，调整大小
        if cropped.size != (target_width, target_height):
            cropped = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 应用旋转
        if abs(rotate) > 0.1:
            rotated = cropped.rotate(rotate, expand=False, resample=Image.Resampling.BILINEAR)
            # 如果旋转后尺寸变化，裁剪回目标尺寸
            if rotated.size != (target_width, target_height):
                left = (rotated.width - target_width) // 2
                top = (rotated.height - target_height) // 2
                rotated = rotated.crop((left, top, left + target_width, top + target_height))
            return rotated
        
        return cropped
    
    def generate_motion_params_from_scene(
        self,
        scene: Dict[str, Any],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        根据场景描述自动生成相机运动参数
        
        Args:
            scene: 场景JSON数据
            default_params: 默认参数（可选）
        
        Returns:
            相机运动参数
        """
        if default_params is None:
            default_params = {
                "zoom": {"start": 1.0, "end": 1.1},
                "pan_x": {"start": 0.0, "end": 0.0},
                "pan_y": {"start": 0.0, "end": 0.0},
                "rotate": {"start": 0.0, "end": 0.0},
            }
        
        # 从场景中提取信息
        motion = scene.get("visual", {}).get("motion", {})
        camera = scene.get("camera", "").lower()
        composition = scene.get("visual", {}).get("composition", "").lower()
        description = scene.get("description", "").lower()
        
        # 如果motion字段明确指定了类型
        if isinstance(motion, dict):
            motion_type = motion.get("type", "").lower()
            direction = motion.get("direction", "").lower()
            speed = motion.get("speed", "slow").lower()
            
            # 根据motion类型设置参数
            if motion_type == "zoom" or "zoom" in camera:
                if "in" in direction or "推" in description or "近" in description:
                    default_params["zoom"] = {"start": 1.0, "end": 1.15}  # zoom in
                elif "out" in direction or "拉" in description or "远" in description:
                    default_params["zoom"] = {"start": 1.1, "end": 1.0}  # zoom out
                else:
                    default_params["zoom"] = {"start": 1.0, "end": 1.1}  # 默认轻微放大
            
            elif motion_type == "pan" or "pan" in camera or "横移" in description:
                if "left" in direction or "左" in description:
                    default_params["pan_x"] = {"start": 0.05, "end": -0.05}  # 左移
                elif "right" in direction or "右" in description:
                    default_params["pan_x"] = {"start": -0.05, "end": 0.05}  # 右移
                else:
                    default_params["pan_x"] = {"start": 0.0, "end": 0.05}  # 默认右移
            
            elif motion_type == "tilt" or "tilt" in camera:
                if "up" in direction or "上" in description or "仰" in description:
                    default_params["pan_y"] = {"start": 0.05, "end": -0.05}  # 上移
                elif "down" in direction or "下" in description or "俯" in description:
                    default_params["pan_y"] = {"start": -0.05, "end": 0.05}  # 下移
            
            elif motion_type == "orbit" or "orbit" in camera or "环绕" in description:
                default_params["rotate"] = {"start": 0.0, "end": 5.0}  # 轻微旋转
        
        # 根据composition推断（如果没有明确指定）
        elif not isinstance(motion, dict) or not motion.get("type"):
            if "wide" in composition or "distant" in composition or "远景" in composition:
                # 远景场景 → 缓慢zoom in
                default_params["zoom"] = {"start": 1.0, "end": 1.2}
            elif "medium" in composition or "mid" in composition or "中景" in composition:
                # 中景场景 → 轻微pan
                default_params["pan_x"] = {"start": 0.0, "end": 0.05}
            elif "close" in composition or "close-up" in composition or "特写" in composition:
                # 特写场景 → 静态或轻微zoom
                default_params["zoom"] = {"start": 1.0, "end": 1.05}
        
        return default_params

