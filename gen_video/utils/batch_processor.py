#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理工具
支持批量生成视频、批量分析质量等
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, max_workers: int = 2):
        """
        初始化批量处理器
        
        Args:
            max_workers: 最大并发数（视频生成需要大量显存，建议1-2）
        """
        self.max_workers = max_workers
    
    def process_videos(
        self,
        prompts: List[str],
        output_dir: Path,
        processor_func: Callable[[str, Path], Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量处理视频生成
        
        Args:
            prompts: 提示词列表
            output_dir: 输出目录
            processor_func: 处理函数，接受(prompt, output_dir)参数
            **kwargs: 传递给处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        results = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        log_file = output_dir / f"batch_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        print(f"开始批量处理 {len(prompts)} 个视频")
        print(f"输出目录: {output_dir}")
        print(f"日志文件: {log_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 由于视频生成需要大量显存，使用串行处理
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] 处理: {prompt[:50]}...")
            
            try:
                # 创建子目录
                sub_dir = output_dir / f"video_{i:03d}"
                sub_dir.mkdir(exist_ok=True)
                
                # 处理
                result = processor_func(prompt, sub_dir, **kwargs)
                result["prompt"] = prompt
                result["index"] = i
                result["status"] = "success"
                results.append(result)
                
                print(f"  ✅ 成功: {result.get('video_path', 'N/A')}")
                
                # 记录日志
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{datetime.now()}] [{i}/{len(prompts)}] SUCCESS: {prompt}\n")
                    f.write(f"  Video: {result.get('video_path', 'N/A')}\n")
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ❌ 失败: {error_msg}")
                
                result = {
                    "prompt": prompt,
                    "index": i,
                    "status": "error",
                    "error": error_msg
                }
                results.append(result)
                
                # 记录错误日志
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{datetime.now()}] [{i}/{len(prompts)}] ERROR: {prompt}\n")
                    f.write(f"  Error: {error_msg}\n")
        
        elapsed_time = time.time() - start_time
        
        # 生成总结报告
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        
        print("\n" + "=" * 60)
        print("批量处理完成")
        print("=" * 60)
        print(f"总数量: {len(prompts)}")
        print(f"成功: {success_count}")
        print(f"失败: {error_count}")
        print(f"总耗时: {elapsed_time/60:.1f} 分钟")
        print(f"平均耗时: {elapsed_time/len(prompts):.1f} 秒/个")
        print(f"日志文件: {log_file}")
        
        # 保存结果到JSON
        results_file = output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total": len(prompts),
                    "success": success_count,
                    "error": error_count,
                    "elapsed_time": elapsed_time
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"结果文件: {results_file}")
        
        return results
    
    def analyze_videos(
        self,
        video_paths: List[str],
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        批量分析视频质量
        
        Args:
            video_paths: 视频文件路径列表
            output_dir: 输出目录（可选）
            
        Returns:
            分析结果列表
        """
        from utils.video_quality_analyzer import VideoQualityAnalyzer
        
        analyzer = VideoQualityAnalyzer()
        results = []
        
        print(f"开始批量分析 {len(video_paths)} 个视频")
        print("=" * 60)
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] 分析: {Path(video_path).name}")
            
            try:
                result = analyzer.analyze(video_path)
                results.append(result)
                
                print(f"  ✅ 总体评分: {result['overall_score']:.1f}/100")
                print(f"     色彩: {result['color_analysis']['color_score']*100:.1f}/100")
                print(f"     一致性: {result['consistency_analysis']['flicker_score']*100:.1f}/100")
                print(f"     清晰度: {result['sharpness_analysis']['score']*100:.1f}/100")
                
            except Exception as e:
                print(f"  ❌ 分析失败: {e}")
                results.append({
                    "video_path": video_path,
                    "error": str(e)
                })
        
        # 保存结果
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / f"quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 分析结果已保存到: {results_file}")
        
        return results


if __name__ == "__main__":
    """测试批量处理器"""
    from generate_novel_video import NovelVideoGenerator
    
    # 测试批量生成
    prompts = [
        "一个美丽的山谷，有瀑布和彩虹",
        "那夜他手握长剑，踏入断桥",
        "黑洞在太空中旋转"
    ]
    
    output_dir = Path("outputs/batch_test")
    generator = NovelVideoGenerator()
    
    def process_prompt(prompt: str, output_dir: Path):
        result = generator.generate(
            prompt=prompt,
            output_dir=output_dir
        )
        return result
    
    processor = BatchProcessor(max_workers=1)
    results = processor.process_videos(prompts, output_dir, process_prompt)
    
    print("\n批量处理完成！")

