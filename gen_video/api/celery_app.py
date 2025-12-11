"""
Celery应用配置
用于异步任务处理（图像生成、视频生成等）
"""
from celery import Celery
import os
from pathlib import Path

# Redis配置（可以从环境变量读取）
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 创建Celery应用
celery_app = Celery(
    "video_gen_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]  # 包含任务模块
)

# Celery配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 任务超时时间：1小时
    task_soft_time_limit=3300,  # 软超时：55分钟
    worker_prefetch_multiplier=1,  # 每个worker一次只处理一个任务
    worker_max_tasks_per_child=50,  # 每个worker处理50个任务后重启（释放内存）
)

