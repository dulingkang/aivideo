#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试场景生成问题

用于排查场景生成失败的原因，特别是视频缺失问题
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

def find_scene_files(output_dir: str, scene_id: str) -> Dict[str, Any]:
    """查找场景相关的所有文件"""
    output_path = Path(output_dir)
    scene_info = {
        "scene_id": scene_id,
        "image": None,
        "video": None,
        "m6_report": None,
        "log_files": [],
        "partial_files": []
    }
    
    # 查找图片
    for pattern in [f"scene_{scene_id:03d}/*.png", f"scene_{scene_id:03d}/*.jpg"]:
        for img_file in output_path.glob(pattern):
            if "novel_image" in img_file.name or "scene_image" in img_file.name:
                scene_info["image"] = {
                    "path": str(img_file),
                    "size": img_file.stat().st_size,
                    "exists": True
                }
                break
    
    # 查找视频
    for pattern in [f"scene_{scene_id:03d}/*.mp4", f"scene_{scene_id:03d}/*.avi"]:
        for vid_file in output_path.glob(pattern):
            if "novel_video" in vid_file.name or "scene_video" in vid_file.name:
                scene_info["video"] = {
                    "path": str(vid_file),
                    "size": vid_file.stat().st_size,
                    "exists": True
                }
                break
    
    # 查找M6报告
    for pattern in [f"**/*scene_{scene_id:03d}*m6*.json", f"**/*scene_{scene_id:03d}*identity*.json"]:
        for m6_file in output_path.glob(pattern):
            scene_info["m6_report"] = {
                "path": str(m6_file),
                "exists": True
            }
            try:
                with open(m6_file, 'r') as f:
                    scene_info["m6_report"]["content"] = json.load(f)
            except:
                pass
            break
    
    # 查找日志文件
    for log_file in output_path.rglob("*.log"):
        if scene_id in log_file.name or f"scene_{scene_id:03d}" in log_file.name:
            scene_info["log_files"].append(str(log_file))
    
    # 查找部分生成的文件（可能是中断的）
    for partial_file in output_path.glob(f"**/*scene_{scene_id:03d}*.tmp"):
        scene_info["partial_files"].append({
            "path": str(partial_file),
            "size": partial_file.stat().st_size
        })
    
    return scene_info

def analyze_scene_status(scene_info: Dict[str, Any]) -> Dict[str, Any]:
    """分析场景状态"""
    status = {
        "image_status": "missing",
        "video_status": "missing",
        "m6_status": "unknown",
        "issues": [],
        "recommendations": []
    }
    
    # 图片状态
    if scene_info["image"]:
        status["image_status"] = "exists"
        if scene_info["image"]["size"] < 1000:  # 小于1KB可能是损坏的
            status["issues"].append("图片文件异常小，可能损坏")
    else:
        status["issues"].append("图片文件缺失")
        status["recommendations"].append("需要重新生成图片")
    
    # 视频状态
    if scene_info["video"]:
        status["video_status"] = "exists"
        if scene_info["video"]["size"] < 10000:  # 小于10KB可能是损坏的
            status["issues"].append("视频文件异常小，可能损坏")
    else:
        status["issues"].append("视频文件缺失")
        status["recommendations"].append("需要重新生成视频")
    
    # M6状态
    if scene_info["m6_report"]:
        status["m6_status"] = "exists"
        m6_content = scene_info["m6_report"].get("content", {})
        if isinstance(m6_content, dict):
            # 检查M6验证结果
            if "verification" in m6_content:
                verification = m6_content["verification"]
                if verification.get("passed", False):
                    status["m6_status"] = "passed"
                else:
                    status["m6_status"] = "failed"
                    status["issues"].append(f"M6验证失败: {verification.get('reason', '未知原因')}")
            elif "similarity" in m6_content:
                similarity = m6_content.get("similarity", 0)
                if similarity < 0.65:
                    status["m6_status"] = "low_similarity"
                    status["issues"].append(f"M6相似度过低: {similarity:.3f} < 0.65")
    else:
        status["m6_status"] = "missing"
        if scene_info["video"]:
            status["recommendations"].append("视频已生成但缺少M6报告，可能需要手动验证")
    
    # 部分文件
    if scene_info["partial_files"]:
        status["issues"].append(f"发现 {len(scene_info['partial_files'])} 个部分生成的文件，可能生成被中断")
        status["recommendations"].append("清理部分文件后重新生成")
    
    return status

def print_report(scene_info: Dict[str, Any], status: Dict[str, Any]):
    """打印报告"""
    print("=" * 60)
    print(f"场景 {scene_info['scene_id']} 调试报告")
    print("=" * 60)
    print()
    
    print("📁 文件状态:")
    print(f"  图片: {status['image_status']}")
    if scene_info["image"]:
        print(f"    - 路径: {scene_info['image']['path']}")
        print(f"    - 大小: {scene_info['image']['size'] / 1024:.2f} KB")
    
    print(f"  视频: {status['video_status']}")
    if scene_info["video"]:
        print(f"    - 路径: {scene_info['video']['path']}")
        print(f"    - 大小: {scene_info['video']['size'] / 1024 / 1024:.2f} MB")
    
    print(f"  M6报告: {status['m6_status']}")
    if scene_info["m6_report"]:
        print(f"    - 路径: {scene_info['m6_report']['path']}")
        if "content" in scene_info["m6_report"]:
            content = scene_info["m6_report"]["content"]
            if isinstance(content, dict):
                if "similarity" in content:
                    print(f"    - 相似度: {content['similarity']:.3f}")
                if "verification" in content:
                    verification = content["verification"]
                    print(f"    - 验证结果: {'通过' if verification.get('passed') else '失败'}")
    
    if scene_info["log_files"]:
        print(f"  日志文件: {len(scene_info['log_files'])} 个")
        for log_file in scene_info["log_files"][:5]:  # 只显示前5个
            print(f"    - {log_file}")
    
    if scene_info["partial_files"]:
        print(f"  部分文件: {len(scene_info['partial_files'])} 个")
        for partial_file in scene_info["partial_files"]:
            print(f"    - {partial_file['path']} ({partial_file['size'] / 1024:.2f} KB)")
    
    print()
    print("⚠️  问题:")
    if status["issues"]:
        for issue in status["issues"]:
            print(f"  - {issue}")
    else:
        print("  ✅ 未发现问题")
    
    print()
    print("💡 建议:")
    if status["recommendations"]:
        for rec in status["recommendations"]:
            print(f"  - {rec}")
    else:
        print("  ✅ 无需操作")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="调试场景生成问题")
    parser.add_argument("--output-dir", type=str, default="gen_video/outputs/lingjie_ep1_v2",
                        help="输出目录")
    parser.add_argument("--scene-id", type=str, required=True,
                        help="场景ID（如：001 或 1）")
    
    args = parser.parse_args()
    
    # 标准化场景ID（确保是3位数字字符串）
    try:
        scene_id_int = int(args.scene_id)
        scene_id = f"{scene_id_int:03d}"
    except ValueError:
        # 如果已经是字符串格式（如 "001"），直接使用
        scene_id = args.scene_id.zfill(3)
    
    # 查找文件
    scene_info = find_scene_files(args.output_dir, scene_id)
    
    # 分析状态
    status = analyze_scene_status(scene_info)
    
    # 打印报告
    print_report(scene_info, status)

if __name__ == "__main__":
    main()

