#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说推文批量生成工具

功能：
1. 批量处理 JSON 场景文件
2. 支持多场景并行/串行生成
3. 自动错误重试
4. 生成详细报告
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import traceback

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generate_novel_video import NovelVideoGenerator


class BatchNovelGenerator:
    """批量小说推文生成器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化批量生成器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self.generator = NovelVideoGenerator()
        self.results = []
        self.errors = []
        
    def load_scenes_from_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        从 JSON 文件加载场景列表
        
        Args:
            json_path: JSON 文件路径
        
        Returns:
            场景列表
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON 文件不存在: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scenes = data.get('scenes', [])
        print(f"  ✓ 从 {json_path} 加载了 {len(scenes)} 个场景")
        return scenes
    
    def extract_prompt_from_scene(self, scene: Dict[str, Any]) -> str:
        """
        从场景字典中提取 prompt
        
        Args:
            scene: 场景字典
        
        Returns:
            提示词字符串
        """
        # 尝试多种方式提取 prompt
        prompt_parts = []
        
        # 1. 从 visual_constraints 提取
        visual = scene.get('visual_constraints', {})
        if isinstance(visual, dict):
            environment = visual.get('environment', '')
            if environment:
                prompt_parts.append(environment)
        
        # 2. 从 narration 提取
        narration = scene.get('narration', {})
        if isinstance(narration, dict):
            narration_text = narration.get('text', '')
            if narration_text:
                # 提取关键描述（前100字）
                prompt_parts.append(narration_text[:100])
        
        # 3. 从 character 提取
        character = scene.get('character', {})
        if character.get('present', False):
            character_id = character.get('id', '')
            if character_id == 'hanli':
                prompt_parts.insert(0, "韩立")
        
        # 4. 从其他字段提取
        if not prompt_parts:
            # 尝试从其他字段提取
            description = scene.get('description', '')
            if description:
                prompt_parts.append(description)
            else:
                prompt_parts.append("一个仙侠场景")
        
        return ", ".join(prompt_parts) if prompt_parts else "一个仙侠场景"
    
    def generate_scene(
        self,
        scene: Dict[str, Any],
        output_base_dir: Path,
        scene_index: int,
        total_scenes: int,
        enable_m6: bool = True,
        quick_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        生成单个场景
        
        Args:
            scene: 场景字典
            output_base_dir: 输出基础目录
            scene_index: 场景索引
            total_scenes: 总场景数
            enable_m6: 是否启用 M6 身份验证
            quick_mode: 快速模式（减少帧数）
        
        Returns:
            生成结果字典
        """
        scene_id = scene.get('scene_id', scene_index)
        print(f"\n{'='*60}")
        print(f"生成场景 {scene_index + 1}/{total_scenes} (ID: {scene_id})")
        print(f"{'='*60}")
        
        # 提取 prompt
        prompt = self.extract_prompt_from_scene(scene)
        print(f"  提示词: {prompt[:100]}...")
        
        # 提取场景参数
        character = scene.get('character', {})
        character_present = character.get('present', False)
        character_id = character.get('id') if character_present else None
        
        camera = scene.get('camera', {})
        shot_type = camera.get('shot', 'medium')
        
        quality_target = scene.get('quality_target', {})
        motion_intensity = quality_target.get('motion_intensity', 'moderate')
        
        # 构建输出目录
        scene_output_dir = output_base_dir / f"scene_{scene_id:03d}"
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成参数
        width = scene.get('width', 768)
        height = scene.get('height', 1152)
        num_frames = 24 if quick_mode else scene.get('num_frames', 120)
        fps = scene.get('target_fps', 24) or 24
        
        print(f"  参数: {width}x{height}, {num_frames}帧, {fps}fps")
        print(f"  镜头: {shot_type}, 运动强度: {motion_intensity}")
        if character_present:
            print(f"  角色: {character_id} (M6: {'启用' if enable_m6 else '禁用'})")
        
        try:
            # 生成视频
            result = self.generator.generate(
                prompt=prompt,
                output_dir=scene_output_dir,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                scene=scene,
                include_character=character_present,
                character_id=character_id,
                auto_character=True,
                enable_m6_identity=enable_m6 if character_present else False,
                auto_m6_identity=enable_m6,
                shot_type=shot_type,
                motion_intensity=motion_intensity,
                m6_quick=quick_mode,
            )
            
            print(f"  ✅ 生成成功!")
            print(f"     图片: {result.get('image')}")
            if 'video' in result:
                print(f"     视频: {result.get('video')}")
            
            return {
                'scene_id': scene_id,
                'scene_index': scene_index,
                'status': 'success',
                'prompt': prompt,
                'result': result,
                'error': None,
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ 生成失败: {error_msg}")
            traceback.print_exc()
            
            return {
                'scene_id': scene_id,
                'scene_index': scene_index,
                'status': 'error',
                'prompt': prompt,
                'result': None,
                'error': error_msg,
            }
    
    def generate_batch(
        self,
        json_path: Path,
        output_dir: Path,
        enable_m6: bool = True,
        quick_mode: bool = False,
        max_retries: int = 2,
        start_index: int = 0,
        end_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        批量生成场景
        
        Args:
            json_path: JSON 场景文件路径
            output_dir: 输出目录
            enable_m6: 是否启用 M6 身份验证
            quick_mode: 快速模式
            max_retries: 最大重试次数
            start_index: 开始索引（用于断点续传）
            end_index: 结束索引（用于分批处理）
        
        Returns:
            批量生成结果
        """
        print("="*60)
        print("小说推文批量生成")
        print("="*60)
        
        # 加载场景
        scenes = self.load_scenes_from_json(json_path)
        
        # 过滤场景范围
        if end_index is None:
            end_index = len(scenes)
        scenes = scenes[start_index:end_index]
        
        print(f"\n生成范围: {start_index} - {end_index-1} (共 {len(scenes)} 个场景)")
        print(f"输出目录: {output_dir}")
        print(f"M6 身份验证: {'启用' if enable_m6 else '禁用'}")
        print(f"快速模式: {'是' if quick_mode else '否'}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 批量生成
        results = []
        for i, scene in enumerate(scenes):
            scene_index = start_index + i
            
            # 生成场景（带重试）
            result = None
            for retry in range(max_retries + 1):
                if retry > 0:
                    print(f"  🔄 重试 {retry}/{max_retries}...")
                
                result = self.generate_scene(
                    scene=scene,
                    output_base_dir=output_dir,
                    scene_index=scene_index,
                    total_scenes=len(scenes),
                    enable_m6=enable_m6,
                    quick_mode=quick_mode,
                )
                
                if result['status'] == 'success':
                    break
            
            results.append(result)
            
            # 保存中间结果
            if (i + 1) % 5 == 0:
                self._save_progress(output_dir, results, scenes)
        
        # 保存最终结果
        self._save_progress(output_dir, results, scenes)
        
        # 生成报告
        report = self._generate_report(results, output_dir)
        
        return {
            'results': results,
            'report': report,
        }
    
    def _save_progress(self, output_dir: Path, results: List[Dict], scenes: List[Dict]):
        """保存进度"""
        progress_file = output_dir / "progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_scenes': len(scenes),
                'completed': len(results),
                'results': results,
            }, f, ensure_ascii=False, indent=2)
    
    def _generate_report(self, results: List[Dict], output_dir: Path) -> Dict[str, Any]:
        """生成报告"""
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        errors = sum(1 for r in results if r['status'] == 'error')
        
        success_rate = (success / total * 100) if total > 0 else 0
        
        # 统计错误
        error_details = []
        for r in results:
            if r['status'] == 'error':
                error_details.append({
                    'scene_id': r['scene_id'],
                    'prompt': r['prompt'][:50] + '...',
                    'error': r['error'],
                })
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'success': success,
                'errors': errors,
                'success_rate': f"{success_rate:.1f}%",
            },
            'errors': error_details,
        }
        
        # 保存报告
        report_file = output_dir / "batch_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成 Markdown 报告
        md_report = self._generate_markdown_report(report, results)
        md_file = output_dir / "batch_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"\n{'='*60}")
        print("批量生成完成")
        print(f"{'='*60}")
        print(f"总计: {total}")
        print(f"成功: {success} ({success_rate:.1f}%)")
        print(f"失败: {errors}")
        print(f"\n报告已保存:")
        print(f"  JSON: {report_file}")
        print(f"  Markdown: {md_file}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict, results: List[Dict]) -> str:
        """生成 Markdown 格式报告"""
        md = f"""# 小说推文批量生成报告

生成时间: {report['timestamp']}

## 摘要

- **总计**: {report['summary']['total']} 个场景
- **成功**: {report['summary']['success']} 个
- **失败**: {report['summary']['errors']} 个
- **成功率**: {report['summary']['success_rate']}

## 失败场景详情

"""
        if report['errors']:
            for error in report['errors']:
                md += f"### 场景 {error['scene_id']}\n\n"
                md += f"- **提示词**: {error['prompt']}\n"
                md += f"- **错误**: {error['error']}\n\n"
        else:
            md += "无失败场景 ✅\n"
        
        md += "\n## 成功场景列表\n\n"
        for r in results:
            if r['status'] == 'success':
                md += f"- 场景 {r['scene_id']}: {r['prompt'][:50]}...\n"
        
        return md


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="小说推文批量生成工具")
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='JSON 场景文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认: outputs/batch_novel_<timestamp>）'
    )
    parser.add_argument(
        '--enable-m6',
        action='store_true',
        default=True,
        help='启用 M6 身份验证（默认: 启用）'
    )
    parser.add_argument(
        '--disable-m6',
        action='store_true',
        help='禁用 M6 身份验证'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速模式（减少帧数，用于测试）'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='开始索引（用于断点续传）'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='结束索引（用于分批处理）'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='最大重试次数（默认: 2）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    # 如果是在 gen_video 目录下执行，相对路径应该相对于 gen_video 目录
    json_path_str = args.json
    json_path = Path(json_path_str)
    
    if not json_path.is_absolute():
        # 处理相对路径
        # 如果路径以 ../ 开头，从 gen_video 目录向上查找
        # 否则，相对于 gen_video 目录
        if json_path_str.startswith('../'):
            # 去掉 ../ 前缀，然后从 fanren 目录开始
            relative_path = json_path_str[3:]  # 去掉 '../'
            json_path = project_root.parent / relative_path
        else:
            # 相对于 gen_video 目录
            json_path = project_root / json_path
        
        # 规范化路径（处理 .. 和 .）
        json_path = json_path.resolve()
    
    # 解析输出目录路径
    if args.output_dir:
        output_dir_str = args.output_dir
        output_dir = Path(output_dir_str)
        if not output_dir.is_absolute():
            # 处理相对路径
            if output_dir_str.startswith('../'):
                # 去掉 ../ 前缀，然后从 fanren 目录开始
                relative_path = output_dir_str[3:]  # 去掉 '../'
                output_dir = project_root.parent / relative_path
            else:
                # 相对于 gen_video 目录
                output_dir = project_root / output_dir
            output_dir = output_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / f"batch_novel_{timestamp}"
    
    # M6 设置
    enable_m6 = args.enable_m6 and not args.disable_m6
    
    # 创建生成器
    generator = BatchNovelGenerator()
    
    # 批量生成
    result = generator.generate_batch(
        json_path=json_path,
        output_dir=output_dir,
        enable_m6=enable_m6,
        quick_mode=args.quick,
        max_retries=args.max_retries,
        start_index=args.start,
        end_index=args.end,
    )
    
    # 返回状态码
    success_count = result['report']['summary']['success']
    total_count = result['report']['summary']['total']
    
    if success_count == total_count:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

