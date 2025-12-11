#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控工具
用于监控视频生成的性能指标
"""

import time
import torch
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenerationMetrics:
    """生成指标"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    memory_before: float = 0.0
    memory_after: float = 0.0
    memory_peak: float = 0.0
    num_frames: int = 0
    resolution: tuple = (0, 0)
    num_inference_steps: int = 0
    retry_count: int = 0
    success: bool = False
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """生成耗时（秒）"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def memory_used(self) -> float:
        """使用的显存（GB）"""
        return self.memory_after - self.memory_before
    
    @property
    def fps_generation(self) -> float:
        """生成速度（帧/秒）"""
        if self.duration > 0 and self.num_frames > 0:
            return self.num_frames / self.duration
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "duration": self.duration,
            "memory_used_gb": self.memory_used,
            "memory_peak_gb": self.memory_peak,
            "num_frames": self.num_frames,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "num_inference_steps": self.num_inference_steps,
            "retry_count": self.retry_count,
            "fps_generation": self.fps_generation,
            "success": self.success,
            "error": self.error,
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_file: Optional[str] = None):
        """初始化监控器"""
        self.log_file = log_file
        self.metrics_history: list[GenerationMetrics] = []
    
    def start_generation(self, num_frames: int, resolution: tuple, num_inference_steps: int) -> GenerationMetrics:
        """开始监控一次生成"""
        metrics = GenerationMetrics(
            num_frames=num_frames,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
        )
        
        # 记录生成前显存
        if torch.cuda.is_available():
            metrics.memory_before = torch.cuda.memory_allocated() / 1024**3
        
        return metrics
    
    def update_memory_peak(self, metrics: GenerationMetrics):
        """更新峰值显存"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / 1024**3
            if current > metrics.memory_peak:
                metrics.memory_peak = current
    
    def end_generation(self, metrics: GenerationMetrics, success: bool = True, error: Optional[str] = None):
        """结束监控一次生成"""
        metrics.end_time = time.time()
        metrics.success = success
        metrics.error = error
        
        # 记录生成后显存
        if torch.cuda.is_available():
            metrics.memory_after = torch.cuda.memory_allocated() / 1024**3
        
        # 保存到历史
        self.metrics_history.append(metrics)
        
        # 打印摘要
        self._print_summary(metrics)
        
        # 保存到日志文件
        if self.log_file:
            self._save_to_log(metrics)
    
    def _print_summary(self, metrics: GenerationMetrics):
        """打印性能摘要"""
        print(f"\n  📊 性能摘要:")
        print(f"     - 耗时: {metrics.duration:.1f}秒")
        print(f"     - 生成速度: {metrics.fps_generation:.2f} 帧/秒")
        print(f"     - 显存使用: {metrics.memory_used:.2f}GB")
        if metrics.memory_peak > 0:
            print(f"     - 峰值显存: {metrics.memory_peak:.2f}GB")
        if metrics.retry_count > 0:
            print(f"     - 重试次数: {metrics.retry_count}")
        if not metrics.success:
            print(f"     - 状态: 失败 ({metrics.error})")
    
    def _save_to_log(self, metrics: GenerationMetrics):
        """保存到日志文件"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **metrics.to_dict()
        }
        
        # 追加到日志文件
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.metrics_history:
            return {}
        
        successful = [m for m in self.metrics_history if m.success]
        if not successful:
            return {"error": "没有成功的生成记录"}
        
        durations = [m.duration for m in successful]
        memory_usages = [m.memory_used for m in successful]
        fps_rates = [m.fps_generation for m in successful if m.fps_generation > 0]
        
        return {
            "total_generations": len(self.metrics_history),
            "successful_generations": len(successful),
            "failed_generations": len(self.metrics_history) - len(successful),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_usage": sum(memory_usages) / len(memory_usages),
            "max_memory_usage": max(memory_usages),
            "avg_fps": sum(fps_rates) / len(fps_rates) if fps_rates else 0,
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        if "error" in stats:
            print(f"  ⚠ {stats['error']}")
            return
        
        print(f"\n  📈 性能统计:")
        print(f"     - 总生成次数: {stats['total_generations']}")
        print(f"     - 成功: {stats['successful_generations']}, 失败: {stats['failed_generations']}")
        print(f"     - 平均耗时: {stats['avg_duration']:.1f}秒")
        print(f"     - 耗时范围: {stats['min_duration']:.1f}秒 - {stats['max_duration']:.1f}秒")
        print(f"     - 平均显存使用: {stats['avg_memory_usage']:.2f}GB")
        print(f"     - 最大显存使用: {stats['max_memory_usage']:.2f}GB")
        if stats['avg_fps'] > 0:
            print(f"     - 平均生成速度: {stats['avg_fps']:.2f} 帧/秒")


# 全局监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor(log_file: Optional[str] = None) -> PerformanceMonitor:
    """获取全局监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(log_file)
    return _global_monitor


if __name__ == "__main__":
    """测试性能监控器"""
    monitor = PerformanceMonitor("test_performance.log")
    
    # 模拟一次生成
    metrics = monitor.start_generation(
        num_frames=24,
        resolution=(640, 480),
        num_inference_steps=30
    )
    
    # 模拟生成过程
    import time
    time.sleep(1)
    
    # 更新峰值显存
    if torch.cuda.is_available():
        monitor.update_memory_peak(metrics)
    
    # 结束生成
    monitor.end_generation(metrics, success=True)
    
    # 打印统计
    monitor.print_statistics()

