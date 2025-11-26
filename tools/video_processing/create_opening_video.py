#!/usr/bin/env python3
"""
创建开头视频
组合多个场景片段：165_scene_165 + 170_scene_021(前3秒) + 165_scene_245
"""

import argparse
import subprocess
from pathlib import Path
import tempfile
import shutil

def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def trim_video(input_path: Path, output_path: Path, start_time: float = 0.0, duration: float = None, mute: bool = True):
    """裁剪视频（重新编码确保兼容性，可选静音）"""
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-ss', str(start_time),
        '-y'
    ]
    
    if duration:
        cmd.extend(['-t', str(duration)])
    
    # 重新编码以确保兼容性
    cmd.extend([
        '-c:v', 'libx264',  # 重新编码视频
        '-preset', 'fast',  # 快速编码
        '-crf', '23',       # 质量控制
    ])
    
    if mute:
        cmd.append('-an')  # 去掉音频（静音）
    else:
        cmd.extend(['-c:a', 'aac'])  # 重新编码音频
    
    cmd.append(str(output_path))
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)

def concatenate_videos(video_paths: list[Path], output_path: Path, mute: bool = True):
    """拼接多个视频文件（重新编码确保兼容性，可选静音）"""
    if len(video_paths) == 1:
        # 只有一个视频时，也需要重新编码以确保格式一致
        cmd = [
            'ffmpeg', '-i', str(video_paths[0]),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-y'
        ]
        if mute:
            cmd.append('-an')  # 去掉音频（静音）
        else:
            cmd.extend(['-c:a', 'aac'])
        cmd.append(str(output_path))
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return
    
    # 创建临时文件列表
    concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
    try:
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.absolute()}'\n")
        
        # 重新编码以确保所有视频参数一致
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file),
            '-c:v', 'libx264',  # 重新编码视频
            '-preset', 'fast',  # 快速编码
            '-crf', '23',       # 质量控制
            '-y'
        ]
        if mute:
            cmd.append('-an')  # 去掉音频（静音）
        else:
            cmd.extend(['-c:a', 'aac'])  # 重新编码音频
        
        cmd.append(str(output_path))
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        if concat_file.exists():
            concat_file.unlink()

def create_opening_video(
    scene_165_path: Path,
    scene_021_path: Path,
    scene_245_path: Path,
    output_path: Path,
    scene_021_duration: float = 3.0
):
    """
    创建开头视频
    
    Args:
        scene_165_path: 165_scene_165 视频路径（完整使用）
        scene_021_path: 170_scene_021 视频路径（使用前3秒）
        scene_245_path: 165_scene_245 视频路径（完整使用）
        output_path: 输出视频路径
        scene_021_duration: scene_021使用时长（默认3秒）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件
    for name, path in [
        ("165_scene_165", scene_165_path),
        ("170_scene_021", scene_021_path),
        ("165_scene_245", scene_245_path)
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} 文件不存在: {path}")
    
    print("=" * 60)
    print("创建开头视频")
    print("=" * 60)
    print()
    
    # 显示视频信息
    video_info = {}
    for name, path in [
        ("165_scene_165", scene_165_path),
        ("170_scene_021", scene_021_path),
        ("165_scene_245", scene_245_path)
    ]:
        duration = get_video_duration(Path(path))
        video_info[name] = duration
        print(f"✅ {name}: {duration:.2f}秒")
    
    # 检查170_scene_021是否有足够的时长
    if video_info["170_scene_021"] < scene_021_duration:
        print(f"⚠️  警告: 170_scene_021只有{video_info['170_scene_021']:.2f}秒，小于请求的{scene_021_duration:.2f}秒")
        scene_021_duration = video_info["170_scene_021"]
    
    print()
    print("组合方案：")
    print(f"1. 165_scene_165（完整: {video_info['165_scene_165']:.2f}秒）")
    print(f"2. 170_scene_021（前{scene_021_duration:.2f}秒）")
    print(f"3. 165_scene_245（完整: {video_info['165_scene_245']:.2f}秒）")
    
    total_duration = video_info["165_scene_165"] + scene_021_duration + video_info["165_scene_245"]
    print(f"\n预估总时长: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    print()
    
    # 创建临时目录
    temp_dir = output_path.parent / "temp_opening"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 处理165_scene_165（完整使用，静音）
        part1 = temp_dir / "part1_scene_165.mp4"
        print(f"步骤1: 处理 165_scene_165（完整，静音）...")
        trim_video(scene_165_path, part1, start_time=0.0, duration=None, mute=True)
        
        # 2. 处理170_scene_021（前3秒，静音）
        part2 = temp_dir / "part2_scene_021_3s.mp4"
        print(f"步骤2: 裁剪 170_scene_021（前{scene_021_duration:.2f}秒，静音）...")
        trim_video(scene_021_path, part2, start_time=0.0, duration=scene_021_duration, mute=True)
        
        # 3. 处理165_scene_245（完整使用，静音）
        part3 = temp_dir / "part3_scene_245.mp4"
        print(f"步骤3: 处理 165_scene_245（完整，静音）...")
        trim_video(scene_245_path, part3, start_time=0.0, duration=None, mute=True)
        
        # 4. 拼接所有片段（静音）
        print(f"步骤4: 拼接所有片段（静音，后续统一添加BGM和旁白）...")
        concatenate_videos([part1, part2, part3], output_path, mute=True)
        
        # 验证输出
        actual_duration = get_video_duration(output_path)
        print()
        print("=" * 60)
        print(f"✓ 开头视频创建完成！")
        print("=" * 60)
        print(f"输出文件: {output_path}")
        print(f"实际时长: {actual_duration:.2f}秒 ({actual_duration/60:.2f}分钟)")
        print()
        print("💡 提示：")
        print(f"  在实际使用时，可以根据开头文本的时长（如3-5秒）")
        print(f"  使用 ffmpeg 裁剪此视频到合适长度：")
        print(f'  ffmpeg -i "{output_path}" -t 5.0 -c copy output_trimmed.mp4')
        print()
        
    finally:
        # 清理临时文件
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"✓ 清理临时文件: {temp_dir}")

def main():
    parser = argparse.ArgumentParser(description='创建开头视频')
    parser.add_argument('--scene-165', required=True,
                       help='165_scene_165 视频路径')
    parser.add_argument('--scene-021', required=True,
                       help='170_scene_021 视频路径')
    parser.add_argument('--scene-245', required=True,
                       help='165_scene_245 视频路径')
    parser.add_argument('--output', '-o', required=True,
                       help='输出视频路径')
    parser.add_argument('--scene-021-duration', type=float, default=3.0,
                       help='scene_021使用时长（默认: 3.0秒）')
    
    args = parser.parse_args()
    
    create_opening_video(
        Path(args.scene_165),
        Path(args.scene_021),
        Path(args.scene_245),
        Path(args.output),
        scene_021_duration=args.scene_021_duration
    )

if __name__ == '__main__':
    main()

