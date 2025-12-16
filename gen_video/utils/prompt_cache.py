#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt缓存管理器
支持持久化缓存，避免重复计算
"""

from typing import Optional, Dict, Any
import json
import hashlib
from pathlib import Path
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PromptCache:
    """Prompt缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/prompts", ttl_days: int = 7):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            ttl_days: 缓存有效期（天数）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 24 * 3600
    
    def _get_cache_key(self, text: str, scene_type: str, style: Optional[str] = None) -> str:
        """生成缓存键"""
        key_str = f"{text}|{scene_type}|{style or ''}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, text: str, scene_type: str, style: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取缓存
        
        Args:
            text: 原始文本
            scene_type: 场景类型
            style: 风格（可选）
            
        Returns:
            缓存的结果，如果不存在或已过期则返回None
        """
        cache_key = self._get_cache_key(text, scene_type, style)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否过期
            cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time > timedelta(seconds=self.ttl_seconds):
                # 缓存已过期，删除文件
                cache_path.unlink()
                return None
            
            return data.get('result')
            
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return None
    
    def set(
        self,
        text: str,
        scene_type: str,
        result: Dict[str, Any],
        style: Optional[str] = None
    ) -> None:
        """
        设置缓存
        
        Args:
            text: 原始文本
            scene_type: 场景类型
            result: 缓存的结果
            style: 风格（可选）
        """
        cache_key = self._get_cache_key(text, scene_type, style)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = {
                'text': text,
                'scene_type': scene_type,
                'style': style,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def clear_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的文件数
        """
        count = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                if now - cache_time > timedelta(seconds=self.ttl_seconds):
                    cache_file.unlink()
                    count += 1
            except:
                # 如果文件损坏，也删除
                cache_file.unlink()
                count += 1
        
        return count
    
    def clear_all(self) -> int:
        """
        清理所有缓存
        
        Returns:
            清理的文件数
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count


# 全局缓存实例
_global_cache: Optional[PromptCache] = None


def get_cache(cache_dir: str = "cache/prompts") -> PromptCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = PromptCache(cache_dir)
    return _global_cache

