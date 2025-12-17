#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显存管理器
用于优化批量生成时的显存管理和模型加载策略

功能特性:
1. 显存监控 - 实时监控显存使用情况
2. 智能卸载 - 根据显存压力自动卸载不活跃的模型
3. 模型缓存 - 缓存常用模型，避免重复加载
4. 延迟加载 - 按需加载模型
5. 批量优化 - 优化批量生成时的显存分配

Author: AI Video Team
Date: 2025-12-17
"""

import gc
import time
import torch
import weakref
import threading
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryPriority(Enum):
    """模型显存优先级"""
    CRITICAL = 1    # 核心模型，尽量不卸载
    HIGH = 2        # 高优先级，较少使用时可卸载
    MEDIUM = 3      # 中等优先级
    LOW = 4         # 低优先级，优先卸载


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    loader: Callable[[], Any]  # 模型加载函数
    unloader: Optional[Callable[[Any], None]] = None  # 模型卸载函数
    priority: MemoryPriority = MemoryPriority.MEDIUM
    estimated_size_gb: float = 0.0  # 估计的显存占用
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    loaded: bool = False
    instance: Any = None


@dataclass
class MemoryStats:
    """显存统计"""
    total_gb: float = 0.0
    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    free_gb: float = 0.0
    cached_gb: float = 0.0
    
    @classmethod
    def current(cls) -> 'MemoryStats':
        """获取当前显存状态"""
        if not torch.cuda.is_available():
            return cls()
        
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        cached = reserved - allocated
        free = total - reserved
        
        return cls(
            total_gb=total,
            allocated_gb=allocated,
            reserved_gb=reserved,
            free_gb=free,
            cached_gb=cached
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total_gb": round(self.total_gb, 2),
            "allocated_gb": round(self.allocated_gb, 2),
            "reserved_gb": round(self.reserved_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "cached_gb": round(self.cached_gb, 2),
            "usage_percent": round((self.reserved_gb / self.total_gb) * 100, 1) if self.total_gb > 0 else 0
        }


class MemoryManager:
    """
    显存管理器
    
    提供统一的显存管理接口，支持：
    - 模型注册和延迟加载
    - 显存监控和预警
    - 智能卸载策略
    - 批量生成优化
    """
    
    def __init__(
        self,
        max_memory_gb: Optional[float] = None,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        auto_cleanup: bool = True
    ):
        """
        初始化显存管理器
        
        Args:
            max_memory_gb: 最大可用显存（GB），None 表示自动检测
            warning_threshold: 显存警告阈值（0-1）
            critical_threshold: 显存危险阈值（0-1）
            auto_cleanup: 是否自动清理缓存
        """
        self.models: Dict[str, ModelInfo] = {}
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.auto_cleanup = auto_cleanup
        
        # 检测可用显存
        if torch.cuda.is_available():
            if self.max_memory_gb is None:
                self.max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"显存管理器初始化: 最大显存 {self.max_memory_gb:.1f}GB")
        else:
            self.max_memory_gb = 0
            logger.warning("CUDA 不可用，显存管理器将以 CPU 模式运行")
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 显存历史记录
        self._memory_history: List[Dict[str, Any]] = []
    
    def register_model(
        self,
        name: str,
        loader: Callable[[], Any],
        unloader: Optional[Callable[[Any], None]] = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        estimated_size_gb: float = 0.0
    ):
        """
        注册模型
        
        Args:
            name: 模型名称
            loader: 模型加载函数
            unloader: 模型卸载函数
            priority: 优先级
            estimated_size_gb: 估计的显存占用
        """
        with self._lock:
            self.models[name] = ModelInfo(
                name=name,
                loader=loader,
                unloader=unloader,
                priority=priority,
                estimated_size_gb=estimated_size_gb
            )
            logger.debug(f"注册模型: {name} (优先级: {priority.name}, 预估: {estimated_size_gb:.1f}GB)")
    
    def get_model(self, name: str, ensure_memory: bool = True) -> Any:
        """
        获取模型实例（延迟加载）
        
        Args:
            name: 模型名称
            ensure_memory: 是否确保有足够显存
            
        Returns:
            模型实例
        """
        with self._lock:
            if name not in self.models:
                raise KeyError(f"模型未注册: {name}")
            
            info = self.models[name]
            
            # 如果模型未加载
            if not info.loaded or info.instance is None:
                # 检查显存
                if ensure_memory:
                    self._ensure_memory(info.estimated_size_gb)
                
                # 加载模型
                logger.info(f"加载模型: {name}")
                start_time = time.time()
                
                try:
                    info.instance = info.loader()
                    info.loaded = True
                    
                    load_time = time.time() - start_time
                    logger.info(f"模型 {name} 加载完成 ({load_time:.1f}秒)")
                    
                except Exception as e:
                    logger.error(f"模型 {name} 加载失败: {e}")
                    raise
            
            # 更新使用信息
            info.last_used = time.time()
            info.use_count += 1
            
            return info.instance
    
    def unload_model(self, name: str, force: bool = False):
        """
        卸载模型
        
        Args:
            name: 模型名称
            force: 是否强制卸载
        """
        with self._lock:
            if name not in self.models:
                return
            
            info = self.models[name]
            
            if not info.loaded:
                return
            
            # 检查优先级
            if not force and info.priority == MemoryPriority.CRITICAL:
                logger.warning(f"模型 {name} 是关键模型，跳过卸载")
                return
            
            logger.info(f"卸载模型: {name}")
            
            # 调用自定义卸载函数
            if info.unloader and info.instance:
                try:
                    info.unloader(info.instance)
                except Exception as e:
                    logger.warning(f"模型 {name} 自定义卸载失败: {e}")
            
            # 删除实例引用
            info.instance = None
            info.loaded = False
            
            # 清理显存
            self._cleanup_memory()
    
    def unload_all(self, include_critical: bool = False):
        """
        卸载所有模型
        
        Args:
            include_critical: 是否包括关键模型
        """
        with self._lock:
            for name, info in list(self.models.items()):
                if info.loaded:
                    if include_critical or info.priority != MemoryPriority.CRITICAL:
                        self.unload_model(name, force=include_critical)
    
    def _ensure_memory(self, required_gb: float):
        """
        确保有足够的显存
        
        Args:
            required_gb: 需要的显存（GB）
        """
        if not torch.cuda.is_available():
            return
        
        stats = MemoryStats.current()
        
        # 检查是否需要释放显存
        if stats.free_gb < required_gb:
            logger.warning(f"显存不足: 需要 {required_gb:.1f}GB, 可用 {stats.free_gb:.1f}GB")
            
            # 尝试智能卸载
            self._smart_unload(required_gb - stats.free_gb)
            
            # 再次检查
            stats = MemoryStats.current()
            if stats.free_gb < required_gb:
                logger.error(f"无法释放足够显存: 需要 {required_gb:.1f}GB, 可用 {stats.free_gb:.1f}GB")
    
    def _smart_unload(self, required_gb: float):
        """
        智能卸载模型以释放显存
        
        优先卸载：
        1. 优先级低的模型
        2. 最久未使用的模型
        3. 使用次数少的模型
        
        Args:
            required_gb: 需要释放的显存（GB）
        """
        with self._lock:
            # 获取已加载的模型，按优先级和使用时间排序
            loaded_models = [
                (name, info) for name, info in self.models.items()
                if info.loaded and info.priority != MemoryPriority.CRITICAL
            ]
            
            # 排序：优先级低的在前，同优先级按最后使用时间排序
            loaded_models.sort(key=lambda x: (
                -x[1].priority.value,  # 优先级值越大越靠前（越先卸载）
                x[1].last_used         # 最后使用时间越早越靠前
            ))
            
            freed_gb = 0.0
            for name, info in loaded_models:
                if freed_gb >= required_gb:
                    break
                
                logger.info(f"智能卸载模型: {name} (预计释放 {info.estimated_size_gb:.1f}GB)")
                self.unload_model(name)
                freed_gb += info.estimated_size_gb
    
    def _cleanup_memory(self):
        """清理显存缓存"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_stats(self) -> MemoryStats:
        """获取当前显存状态"""
        return MemoryStats.current()
    
    def log_stats(self, prefix: str = ""):
        """记录显存状态到日志"""
        stats = self.get_stats()
        usage_pct = (stats.reserved_gb / stats.total_gb) * 100 if stats.total_gb > 0 else 0
        
        status_emoji = "🟢" if usage_pct < self.warning_threshold * 100 else (
            "🟡" if usage_pct < self.critical_threshold * 100 else "🔴"
        )
        
        logger.info(
            f"{prefix}显存 {status_emoji}: "
            f"已分配={stats.allocated_gb:.2f}GB, "
            f"已保留={stats.reserved_gb:.2f}GB, "
            f"可用={stats.free_gb:.2f}GB, "
            f"使用率={usage_pct:.1f}%"
        )
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        with self._lock:
            return [name for name, info in self.models.items() if info.loaded]
    
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        with self._lock:
            if name not in self.models:
                return None
            
            info = self.models[name]
            return {
                "name": info.name,
                "priority": info.priority.name,
                "estimated_size_gb": info.estimated_size_gb,
                "loaded": info.loaded,
                "last_used": info.last_used,
                "use_count": info.use_count
            }
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """
        显存管理上下文
        
        记录操作前后的显存变化
        
        Args:
            operation_name: 操作名称
            
        Example:
            with memory_manager.memory_context("generate_image"):
                image = generator.generate(...)
        """
        before = MemoryStats.current()
        start_time = time.time()
        
        try:
            yield
        finally:
            after = MemoryStats.current()
            duration = time.time() - start_time
            
            delta = after.allocated_gb - before.allocated_gb
            delta_sign = "+" if delta >= 0 else ""
            
            logger.debug(
                f"{operation_name} 完成: "
                f"耗时={duration:.1f}s, "
                f"显存变化={delta_sign}{delta:.2f}GB "
                f"({before.allocated_gb:.2f}GB → {after.allocated_gb:.2f}GB)"
            )
            
            # 记录历史
            self._memory_history.append({
                "operation": operation_name,
                "timestamp": time.time(),
                "duration": duration,
                "before": before.to_dict(),
                "after": after.to_dict(),
                "delta_gb": delta
            })
            
            # 自动清理
            if self.auto_cleanup:
                usage_pct = after.reserved_gb / after.total_gb if after.total_gb > 0 else 0
                if usage_pct > self.critical_threshold:
                    logger.warning(f"显存使用率 {usage_pct*100:.1f}% 超过危险阈值，执行清理")
                    self._cleanup_memory()
    
    @contextmanager
    def batch_context(self, batch_name: str = "batch"):
        """
        批量生成上下文
        
        在批量开始时记录状态，结束时清理
        
        Args:
            batch_name: 批量名称
        """
        logger.info(f"开始批量 '{batch_name}'")
        self.log_stats(f"[{batch_name}] 开始前 - ")
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"完成批量 '{batch_name}' (耗时: {duration:.1f}s)")
            self.log_stats(f"[{batch_name}] 完成后 - ")
            
            # 批量结束后清理缓存
            if self.auto_cleanup:
                self._cleanup_memory()


# 全局实例
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(**kwargs) -> MemoryManager:
    """获取全局显存管理器"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(**kwargs)
    return _global_memory_manager


def log_memory_status(prefix: str = ""):
    """快捷函数：记录当前显存状态"""
    stats = MemoryStats.current()
    if stats.total_gb > 0:
        usage_pct = (stats.reserved_gb / stats.total_gb) * 100
        logger.info(
            f"{prefix}显存: 已分配={stats.allocated_gb:.2f}GB, "
            f"已保留={stats.reserved_gb:.2f}GB, "
            f"可用={stats.free_gb:.2f}GB ({usage_pct:.1f}%)"
        )


def cleanup_memory():
    """快捷函数：清理显存缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    """测试显存管理器"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("显存管理器测试")
    print("=" * 60)
    
    # 创建管理器
    manager = MemoryManager()
    
    # 显示当前状态
    stats = manager.get_stats()
    print(f"\n当前显存状态:")
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")
    
    # 注册模型示例
    def dummy_loader():
        return "dummy_model"
    
    def dummy_unloader(model):
        pass
    
    manager.register_model(
        name="test_model",
        loader=dummy_loader,
        unloader=dummy_unloader,
        priority=MemoryPriority.MEDIUM,
        estimated_size_gb=2.0
    )
    
    # 获取模型
    with manager.memory_context("load_test_model"):
        model = manager.get_model("test_model")
        print(f"\n加载的模型: {model}")
    
    # 显示加载的模型
    print(f"\n已加载模型: {manager.get_loaded_models()}")
    
    # 卸载模型
    manager.unload_all()
    print(f"\n卸载后已加载模型: {manager.get_loaded_models()}")
    
    print("\n✅ 测试完成!")
