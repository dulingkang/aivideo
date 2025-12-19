"""
Token估算器

负责准确计算CLIP token数量，确保Prompt不超过77 tokens限制。
"""

from typing import Optional
import re


class TokenEstimator:
    """CLIP Token估算器"""
    
    def __init__(self, ascii_only_prompt: bool = False):
        """
        初始化Token估算器
        
        Args:
            ascii_only_prompt: 是否只使用ASCII字符（英文）
        """
        self.ascii_only_prompt = ascii_only_prompt
        self._clip_tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self) -> None:
        """加载CLIPTokenizer（如果可用）"""
        try:
            from transformers import CLIPTokenizer
            # 优先使用SDXL实际使用的CLIP-L/14 tokenizer
            # SDXL使用两个tokenizer：CLIP-L/14（主要）和CLIP-G（辅助）
            # 我们使用CLIP-L/14，这是实际限制77 tokens的tokenizer
            try:
                # ⚡ 关键修复：优先使用本地SDXL模型，避免网络下载
                import os
                local_sdxl_path = "/vepfs-dev/shawn/vid/fanren/gen_video/models/sdxl-base"
                if os.path.exists(os.path.join(local_sdxl_path, "tokenizer")):
                    # 使用本地路径
                    self._clip_tokenizer = CLIPTokenizer.from_pretrained(
                        local_sdxl_path,
                        subfolder="tokenizer",
                        local_files_only=True
                    )
                    print(f"  ✓ CLIPTokenizer 从本地加载成功: {local_sdxl_path}")
                else:
                    # 尝试从HuggingFace加载（但设置 local_files_only=True 优先使用缓存）
                    self._clip_tokenizer = CLIPTokenizer.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        subfolder="tokenizer",
                        local_files_only=True  # 只使用本地缓存，不联网下载
                    )
                    print(f"  ✓ CLIPTokenizer 从缓存加载成功")
            except Exception:
                # 如果本地加载失败，尝试使用CLIP-L/14（SDXL实际使用的）
                try:
                    self._clip_tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        local_files_only=True  # 只使用本地缓存，不联网下载
                    )
                    print(f"  ✓ CLIPTokenizer 从CLIP-L/14缓存加载成功")
                except Exception:
                    # 如果都失败，不加载tokenizer，使用估算方法
                    print(f"  ⚠ 无法从本地或缓存加载 CLIPTokenizer，将使用保守估算")
                    self._clip_tokenizer = None
        except Exception as e:
            print(f"  ⚠ 无法加载 CLIPTokenizer，使用保守估算: {e}")
            self._clip_tokenizer = None
    
    def estimate(self, text: str) -> int:
        """
        准确计算CLIP token数量
        
        CLIP tokenizer通常会将文本tokenize，每个单词可能对应1-2个tokens
        使用实际的 CLIPTokenizer 进行准确计算，确保不超过 77 tokens
        
        对于中文和英文混合的文本，tokenizer 会正确处理：
        - 中文：通常一个字一个 token（或更多，取决于编码）
        - 英文：一个词可能 1-3 个 tokens（取决于长度和复杂度）
        
        Args:
            text: 要估算的文本
            
        Returns:
            估算的token数量
        """
        if not text:
            return 0
        
        # 优先使用真实的tokenizer
        if self._clip_tokenizer is not None:
            try:
                tokens = self._clip_tokenizer(text, truncation=False, return_tensors="pt")
                actual_tokens = tokens.input_ids.shape[1]
                return actual_tokens
            except Exception as e:
                print(f"  ⚠ Tokenizer 计算失败，使用保守估算: {e}")
                self._clip_tokenizer = None  # 标记为失败，下次尝试重新加载
        
        # 如果无法使用 tokenizer，使用更准确的估算方法
        return self._estimate_fallback(text)
    
    def _estimate_fallback(self, text: str) -> int:
        """
        回退估算方法（当tokenizer不可用时）
        
        Args:
            text: 要估算的文本
            
        Returns:
            估算的token数量
        """
        # 分离中文字符和英文单词
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_words = re.findall(r'[a-zA-Z]+', text)
        
        # 判断是否主要是中文prompt（根据配置）
        is_chinese_prompt = (
            not self.ascii_only_prompt 
            if hasattr(self, 'ascii_only_prompt') 
            else len(chinese_chars) > len(' '.join(english_words))
        )
        
        # 中文：CLIP tokenizer中，中文字符通常1个token（大多数常见字符）
        # 如果主要是中文prompt，使用更准确的估算（1.05倍，考虑少数复杂字符和标点）
        # 如果是混合prompt，使用保守估算（1.2倍）
        chinese_multiplier = 1.05 if is_chinese_prompt and len(chinese_chars) > 0 else 1.2
        chinese_tokens = len(chinese_chars) * chinese_multiplier
        
        # 英文：每个词约 1.5-2.5 tokens（取决于长度）
        english_tokens = sum(len(word) * 0.25 + 1.0 for word in english_words)
        
        # 其他字符（标点、括号、权重标记等）
        special_chars = text.count(":") + text.count("(") + text.count(")") + text.count(",") + text.count(".")
        special_tokens = special_chars * 1.5
        
        # 括号内的内容（权重标记）会显著增加 token 数
        weight_markers = text.count(":")  # 权重标记
        weight_tokens = weight_markers * 2.5  # 每个权重标记如 ":1.3" 约 2-3 tokens
        
        # 括号本身也会增加 token 数
        bracket_tokens = (text.count("(") + text.count(")")) * 1.0
        
        # 总估算
        estimated = chinese_tokens + english_tokens + special_tokens + weight_tokens + bracket_tokens
        
        # 安全边界：中文prompt更稳定，使用较小的安全边界（15%）；混合prompt使用较大边界（25%）
        if is_chinese_prompt and len(chinese_chars) > len(' '.join(english_words)) * 2:
            # 主要是中文，使用较小的安全边界
            safety_multiplier = 1.15
        else:
            # 混合或主要是英文，使用较大的安全边界
            safety_multiplier = 1.25
        
        return int(estimated * safety_multiplier)

















