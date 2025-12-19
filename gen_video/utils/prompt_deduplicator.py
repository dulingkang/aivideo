#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt 去重工具
统一管理 prompt 构建过程中的去重逻辑，避免重复描述
"""

import re
from typing import List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class PromptDeduplicator:
    """Prompt 去重工具类"""
    
    def __init__(self):
        # 无意义词（中文和英文）
        self.skip_words = {
            '的', '了', '在', '是', '有', '和', '与', '及', '或', '也', '就', '都', '还', '更', '最',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        # 同义词映射（用于检测重复）
        self.synonyms = {
            # 地面相关
            '地面': ['ground', 'floor', '地面可见', 'ground visible', 'floor visible', '地面清晰可见'],
            '地面可见': ['ground visible', 'floor visible', '地面', '地面清晰可见'],
            'ground visible': ['floor visible', '地面可见', '地面', '地面清晰可见'],
            
            # 全身相关
            '全身': ['full body', 'whole body', 'entire body', '全身可见', 'full body visible'],
            '全身可见': ['full body visible', 'full body', '全身'],
            
            # 沙漠相关
            '沙漠': ['desert', '沙漠景观', 'desert landscape', '沙漠地面'],
            '沙漠景观': ['desert landscape', '沙漠', 'desert'],
            
            # 躺相关
            '躺': ['lying', 'lie', '躺在地上', 'lying on ground', 'prone'],
            '躺在地上': ['lying on ground', 'lying', '躺', 'prone'],
            
            # 可见性相关
            '可见': ['visible', 'clearly visible', '可见', '清晰可见'],
            '清晰可见': ['clearly visible', 'visible', '可见'],
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        提取文本中的关键词
        
        Args:
            text: 待提取的文本
        
        Returns:
            关键词列表
        """
        if not text:
            return []
        
        # 使用中文逗号和英文逗号分割
        parts = re.split(r'[，,]\s*', text)
        keywords = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # 移除权重标记 (xxx:1.5)
            part = re.sub(r'\(([^:]+):[\d.]+\)', r'\1', part)
            part = re.sub(r'\(([^)]+)\)', r'\1', part)
            
            # 分割成单词（支持中英文）
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+(?:\s+[a-zA-Z]+)*', part)
            
            for word in words:
                word = word.strip()
                if word and word not in self.skip_words and len(word) > 1:
                    keywords.append(word.lower())
        
        return keywords
    
    def is_duplicate(self, new_desc: str, existing_texts: List[str], threshold: float = 0.6) -> bool:
        """
        检查新描述是否与已有文本重复
        
        Args:
            new_desc: 新描述
            existing_texts: 已有文本列表（可以是 prompt 的各个部分）
            threshold: 重复阈值（0-1），超过此值认为是重复
        
        Returns:
            是否重复
        """
        if not new_desc or not existing_texts:
            return False
        
        new_keywords = set(self.extract_keywords(new_desc))
        if not new_keywords:
            return False
        
        # 合并所有已有文本
        combined_existing = " ".join(existing_texts)
        existing_keywords = set(self.extract_keywords(combined_existing))
        
        if not existing_keywords:
            return False
        
        # 计算关键词重叠率
        overlap = new_keywords & existing_keywords
        
        # 检查同义词
        for new_kw in new_keywords:
            for existing_kw in existing_keywords:
                # 直接匹配
                if new_kw == existing_kw:
                    overlap.add(new_kw)
                # 检查同义词
                elif self._are_synonyms(new_kw, existing_kw):
                    overlap.add(new_kw)
                # 检查包含关系（如"地面可见"包含"地面"）
                elif new_kw in existing_kw or existing_kw in new_kw:
                    overlap.add(new_kw)
        
        # 计算重叠率
        overlap_ratio = len(overlap) / len(new_keywords) if new_keywords else 0
        
        is_dup = overlap_ratio >= threshold
        
        if is_dup:
            logger.debug(f"  检测到重复: {new_desc[:50]}... (重叠率: {overlap_ratio:.2f}, 阈值: {threshold})")
        
        return is_dup
    
    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """检查两个词是否是同义词"""
        word1_lower = word1.lower()
        word2_lower = word2.lower()
        
        # 直接匹配
        if word1_lower == word2_lower:
            return True
        
        # 检查同义词映射
        for key, synonyms in self.synonyms.items():
            if word1_lower == key.lower() and word2_lower in [s.lower() for s in synonyms]:
                return True
            if word2_lower == key.lower() and word1_lower in [s.lower() for s in synonyms]:
                return True
        
        return False
    
    def filter_duplicates(
        self,
        new_descriptions: List[str],
        existing_texts: List[str],
        threshold: float = 0.6
    ) -> List[str]:
        """
        过滤掉与已有文本重复的描述
        
        Args:
            new_descriptions: 新描述列表
            existing_texts: 已有文本列表
            threshold: 重复阈值
        
        Returns:
            过滤后的描述列表
        """
        filtered = []
        
        for desc in new_descriptions:
            if not self.is_duplicate(desc, existing_texts, threshold):
                filtered.append(desc)
            else:
                logger.debug(f"  跳过重复描述: {desc[:50]}...")
        
        return filtered
    
    def merge_prompt_parts(self, parts: List[str], separator: str = ", ") -> str:
        """
        合并 prompt 部分，并去重
        
        Args:
            parts: prompt 部分列表
            separator: 分隔符
        
        Returns:
            合并后的 prompt
        """
        if not parts:
            return ""
        
        # 过滤空部分
        parts = [p.strip() for p in parts if p and p.strip()]
        
        if not parts:
            return ""
        
        # 去重：检查每个部分是否与前面的部分重复
        filtered_parts = []
        existing_keywords = set()
        
        for part in parts:
            part_keywords = set(self.extract_keywords(part))
            
            # 检查是否与已有部分重复
            if part_keywords and existing_keywords:
                overlap = part_keywords & existing_keywords
                overlap_ratio = len(overlap) / len(part_keywords) if part_keywords else 0
                
                # 如果重叠率低于阈值，添加该部分
                if overlap_ratio < 0.6:
                    filtered_parts.append(part)
                    existing_keywords.update(part_keywords)
                else:
                    logger.debug(f"  跳过重复部分: {part[:50]}... (重叠率: {overlap_ratio:.2f})")
            else:
                # 如果没有关键词或没有已有部分，直接添加
                filtered_parts.append(part)
                if part_keywords:
                    existing_keywords.update(part_keywords)
        
        return separator.join(filtered_parts)


# 全局实例
_deduplicator = None


def get_deduplicator() -> PromptDeduplicator:
    """获取全局去重工具实例"""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = PromptDeduplicator()
    return _deduplicator


def is_duplicate(new_desc: str, existing_texts: List[str], threshold: float = 0.6) -> bool:
    """便捷函数：检查是否重复"""
    return get_deduplicator().is_duplicate(new_desc, existing_texts, threshold)


def filter_duplicates(
    new_descriptions: List[str],
    existing_texts: List[str],
    threshold: float = 0.6
) -> List[str]:
    """便捷函数：过滤重复描述"""
    return get_deduplicator().filter_duplicates(new_descriptions, existing_texts, threshold)


def merge_prompt_parts(parts: List[str], separator: str = ", ") -> str:
    """便捷函数：合并并去重 prompt 部分"""
    return get_deduplicator().merge_prompt_parts(parts, separator)

