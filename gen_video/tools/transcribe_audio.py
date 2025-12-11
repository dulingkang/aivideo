#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别音频内容
使用 WhisperX 识别参考音频的实际文本内容，确保 prompt_text 与音频匹配
"""

import argparse
import sys
from pathlib import Path

# 添加gen_video路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from subtitle_generator import SubtitleGenerator
    SUBTITLE_GENERATOR_AVAILABLE = True
except ImportError:
    SUBTITLE_GENERATOR_AVAILABLE = False
    print("⚠️  无法导入 SubtitleGenerator，将使用备用方法")


def transcribe_with_whisperx(audio_path: Path, config_path: str = "config.yaml") -> str:
    """使用 WhisperX 识别音频内容"""
    if not SUBTITLE_GENERATOR_AVAILABLE:
        return None
    
    try:
        subtitle_gen = SubtitleGenerator(config_path)
        
        # 使用 WhisperX 识别音频
        print(f"使用 WhisperX 识别音频: {audio_path.name}")
        result = subtitle_gen.transcribe(str(audio_path))
        
        if result and 'text' in result:
            return result['text']
        elif result and 'segments' in result:
            # 合并所有片段
            text = ' '.join([seg.get('text', '') for seg in result['segments']])
            return text
        else:
            return None
    except Exception as e:
        print(f"⚠️  WhisperX 识别失败: {e}")
        return None


def transcribe_with_whisper_simple(audio_path: Path) -> str:
    """使用简单的 Whisper 识别（备用方法）"""
    try:
        import whisper
        
        print(f"使用 Whisper 识别音频: {audio_path.name}")
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path), language="zh")
        
        return result.get("text", "").strip()
    except ImportError:
        print("⚠️  Whisper 未安装，跳过")
        return None
    except Exception as e:
        print(f"⚠️  Whisper 识别失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='识别音频内容，生成匹配的 prompt_text')
    parser.add_argument('--audio', type=str, required=True,
                       help='参考音频文件路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出识别的文本到文件（可选）')
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ 音频文件不存在: {audio_path}")
        return 1
    
    print("="*60)
    print("识别音频内容")
    print("="*60)
    print(f"\n音频文件: {audio_path.name}")
    
    # 尝试使用 WhisperX
    text = transcribe_with_whisperx(audio_path, args.config)
    
    # 如果失败，尝试使用简单的 Whisper
    if not text:
        print("\n尝试使用 Whisper（备用方法）...")
        text = transcribe_with_whisper_simple(audio_path)
    
    if text:
        print(f"\n✅ 识别成功:")
        print(f"   {text}")
        print(f"\n   长度: {len(text)} 字符")
        
        # 保存到文件
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\n✅ 识别的文本已保存到: {output_path}")
        
        print("\n请将识别的文本复制到 config.yaml 的 prompt_text 字段")
        return 0
    else:
        print("\n❌ 无法识别音频内容")
        print("\n建议:")
        print("1. 手动确认音频中的实际内容")
        print("2. 使用音频前5-7秒对应的文本作为 prompt_text")
        print("3. 确保 prompt_text 与音频内容完全匹配")
        return 1


if __name__ == '__main__':
    sys.exit(main())

