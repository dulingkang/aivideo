#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils 工具模块

提供各种辅助工具和实用函数：
- image_quality_analyzer: 图像质量分析
- video_quality_analyzer: 视频质量分析
- video_identity_analyzer: 视频身份分析（M6）
- memory_manager: 显存管理器
- performance_monitor: 性能监控
- prompt_engine: Prompt 生成引擎
- prompt_engine_v2: Prompt 生成引擎 V2
- batch_processor: 批量处理工具
- model_router: 模型路由
"""

from .image_quality_analyzer import (
    ImageQualityAnalyzer,
    ImageQualityReport,
    FaceSimilarityResult,
    CompositionResult,
    TechnicalQualityResult,
    ShotType,
    QualityLevel,
    analyze_image
)

from .video_quality_analyzer import (
    VideoQualityAnalyzer,
    analyze_video
)

from .video_identity_analyzer import (
    VideoIdentityAnalyzer,
    VideoIdentityReport,
    FrameIdentityResult,
    analyze_video as analyze_video_identity
)

from .memory_manager import (
    MemoryManager,
    MemoryStats,
    MemoryPriority,
    ModelInfo,
    get_memory_manager,
    log_memory_status,
    cleanup_memory
)

from .performance_monitor import (
    PerformanceMonitor,
    GenerationMetrics,
    get_monitor
)

__all__ = [
    # Image Quality
    'ImageQualityAnalyzer',
    'ImageQualityReport',
    'FaceSimilarityResult',
    'CompositionResult',
    'TechnicalQualityResult',
    'ShotType',
    'QualityLevel',
    'analyze_image',
    # Video Quality
    'VideoQualityAnalyzer',
    'analyze_video',
    # Video Identity (M6)
    'VideoIdentityAnalyzer',
    'VideoIdentityReport',
    'FrameIdentityResult',
    'analyze_video_identity',
    # Memory Manager
    'MemoryManager',
    'MemoryStats',
    'MemoryPriority',
    'ModelInfo',
    'get_memory_manager',
    'log_memory_status',
    'cleanup_memory',
    # Performance Monitor
    'PerformanceMonitor',
    'GenerationMetrics',
    'get_monitor',
]

