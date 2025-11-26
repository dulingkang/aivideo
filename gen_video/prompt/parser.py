"""
Prompt解析器

负责从文本中提取核心关键词，去除重复和冗余描述。
"""

import re
from typing import List


class PromptParser:
    """Prompt解析器"""
    
    def extract_first_keyword(self, text: str) -> str:
        """
        从文本中提取第一个关键词（用于紧急精简）
        
        区分近义词和不同特征：
        - 近义词（如"仙侠风格，古风，修仙"）-> 只保留第一个
        - 不同特征（如"黑色长发，深绿道袍"）-> 保留完整，不拆分
        
        Args:
            text: 原始文本，如"仙侠风格，古风，修仙" 或 "(黑色长发，深绿道袍:1.7)"
        
        Returns:
            精简后的关键词，如"仙侠风格" 或 "(黑色长发，深绿道袍:1.7)"
        """
        if not text:
            return ""
        
        # 移除权重标记，提取内容
        content = text
        weight = None
        has_brackets = False
        
        # 检查是否有权重标记
        if text.startswith("(") and ":" in text:
            has_brackets = True
            # 提取内容和权重
            match = re.match(r'\(([^:]+):([^)]+)\)', text)
            if match:
                content = match.group(1).strip()
                weight = match.group(2).strip()
            else:
                # 如果没有匹配到，尝试简单分割
                if ":" in text:
                    parts = text.split(":", 1)
                    content = parts[0].strip("()")
                    weight = parts[1].strip("()")
        
        # 按逗号、顿号分割关键词
        keywords = re.split(r'[，,、]', content)
        keywords = [kw.strip() for kw in keywords if kw.strip()]
        
        if not keywords:
            return text
        
        # 判断是否是不同特征的组合（不应该拆分）
        # 特征关键词：头发、衣服、眼睛、身体等不同方面的描述
        feature_keywords = [
            "头发", "发", "hair", "robe", "道袍", "袍", "衣服", "服装", "服饰",
            "眼睛", "眼神", "eye", "身体", "身材", "body", "肩", "shoulder",
            "黑色", "深绿", "green", "black", "long", "dark"
        ]
        
        # 检查是否包含特征关键词
        has_features = any(any(fk in kw for fk in feature_keywords) for kw in keywords)
        
        # 如果包含特征关键词，说明是不同的特征描述，应该保留完整
        if has_features and len(keywords) > 1:
            # 保留完整内容，不拆分
            if has_brackets and weight:
                return f"({content}:{weight})"
            elif has_brackets:
                return f"({content})"
            else:
                return content
        
        # 否则是近义词组合，只保留第一个
        first_keyword = keywords[0]
        
        # 如果有权重标记，恢复格式
        if has_brackets and weight:
            return f"({first_keyword}:{weight})"
        elif has_brackets:
            return f"({first_keyword})"
        else:
            return first_keyword
    
    def extract_core_keywords(self, text: str, max_keywords: int = 8) -> str:
        """
        提取核心关键词，去除重复词和冗余描述
        
        Args:
            text: 原始文本描述
            max_keywords: 最大关键词数量（默认8个，确保不超过77 tokens）
        
        Returns:
            精简后的核心关键词字符串
        """
        if not text:
            return ""
        
        # 去除权重标记和括号，只保留核心内容
        text = re.sub(r'\([^)]*:\d+\.?\d*\)', '', text)  # 移除权重标记
        text = re.sub(r'[():]', ',', text)  # 将括号和冒号替换为逗号
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = text.strip()
        
        # 按逗号、句号、分号分割成关键词
        keywords = re.split(r'[，,。;；\s]+', text)
        
        # 清理每个关键词
        cleaned_keywords = []
        seen_words = set()  # 用于去重
        
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword:
                continue
            
            # 去除常见无意义词（中文）
            skip_words = {'的', '了', '在', '是', '有', '和', '与', '及', '或', '及', '也', '就', '都', '还', '更', '最'}
            # 去除常见无意义词（英文）
            skip_words_en = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # 检查是否是单字的无意义词
            if len(keyword) == 1 and keyword in skip_words:
                continue
            if keyword.lower() in skip_words_en:
                continue
            
            # 提取核心词汇（去除重复）
            # 检查关键词是否包含已见过的核心词
            keyword_lower = keyword.lower()
            is_duplicate = False
            
            # 检查是否与已有关键词重复或高度相似
            for seen in seen_words:
                if keyword_lower in seen.lower() or seen.lower() in keyword_lower:
                    is_duplicate = True
                    break
                # 检查是否有大量重叠的字符
                if len(keyword) > 2 and len(seen) > 2:
                    overlap = len(set(keyword_lower) & set(seen.lower()))
                    if overlap > min(len(keyword), len(seen)) * 0.7:  # 70%重叠认为是重复
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                cleaned_keywords.append(keyword)
                # 记录核心词用于去重（提取2-4字的核心词）
                words = re.findall(r'[\u4e00-\u9fff]{2,4}|[a-zA-Z]{3,}', keyword)
                for word in words:
                    seen_words.add(word.lower())
                
                # 如果达到最大数量，停止
                if len(cleaned_keywords) >= max_keywords:
                    break
        
        # 合并为字符串
        result = ', '.join(cleaned_keywords)
        
        # 如果结果过长，进一步精简
        if len(result) > 60:  # 大约40个字符，留出空间给权重标记
            # 只保留前6个关键词
            result = ', '.join(cleaned_keywords[:6])
        
        return result









