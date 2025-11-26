#!/usr/bin/env python3
"""
镜头语义标注脚本：使用BLIP-2/CLIP生成描述和embedding
"""

import os
import sys
import json
import argparse
import hashlib
import random
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from sentence_transformers import SentenceTransformer
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("警告: transformers 或 sentence-transformers 未安装")

def load_models(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """加载BLIP-2和CLIP模型"""
    if not BLIP_AVAILABLE:
        raise ImportError("需要安装: pip install transformers sentence-transformers")
    
    print(f"加载模型到设备: {device}")
    
    # BLIP-2 for captioning
    print("加载 BLIP-2...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    
    # CLIP for embeddings
    print("加载 CLIP (sentence-transformers)...")
    clip_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    return processor, blip_model, clip_model, device

def describe_image(image_path, processor, model, device, language='english'):
    """
    使用BLIP-2生成图像描述
    
    Args:
        image_path: 图像路径
        processor: BLIP处理器
        model: BLIP模型
        device: 设备
        language: 语言，'chinese' 或 'english'（BLIP-2主要支持英文）
    
    注意：BLIP-2不支持中文生成，使用中文prompt会导致乱码
    """
    image = Image.open(image_path).convert('RGB')
    
    # BLIP-2主要支持英文，使用英文prompt
    # 如果有字幕，会优先使用字幕（中文），视觉描述只作为补充
    prompt = "a picture of"
    
    inputs = processor(image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=50)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

def get_embedding(text, clip_model):
    """使用CLIP获取文本embedding"""
    embedding = clip_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def load_aligned_subtitles(subtitle_json):
    """加载已对齐的字幕数据"""
    if subtitle_json and Path(subtitle_json).exists():
        with open(subtitle_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式：
        # 1. {"scenes": {"scene_001": {...}}} (align_subtitles.py生成)
        # 2. {"scene_001": {...}} (直接字典格式)
        if "scenes" in data:
            return data["scenes"]
        elif isinstance(data, dict):
            # 检查是否是场景字典格式
            if any(k.startswith("scene_") for k in data.keys()):
                return data
        return {}
    return {}

def infer_chinese_description_from_context(visual_caption_en, subtitle_text, use_llm=False, use_translation=True, prefer_visual=True):
    """
    根据字幕和英文视觉描述，推断出准确的中文描述
    
    策略：
    1. 如果有视觉描述：优先使用视觉描述（描述画面内容）
    2. 如果有字幕：作为补充信息（对话和剧情）
    3. 如果只有字幕没有视觉描述：使用字幕（向后兼容）
    4. 如果只有视觉描述没有字幕：翻译为中文
    
    Args:
        visual_caption_en: 英文视觉描述（如 "a man standing in a cave"）
        subtitle_text: 中文字幕（如 "这是我宗的身份令牌"）
        use_llm: 是否使用LLM进行智能整合（需要API）
        use_translation: 是否使用百度翻译API（默认True）
        prefer_visual: 是否优先使用视觉描述（默认True，用于开头/结尾场景查找）
    
    Returns:
        准确的中文描述（优先视觉描述，字幕作为补充）
    """
    # 情况1：有视觉描述（优先）
    if visual_caption_en and visual_caption_en.strip():
        # 翻译视觉描述为中文
        visual_desc_cn = translate_visual_to_chinese(visual_caption_en, use_translation_api=use_translation)
        
        # 如果有字幕，可以作为补充（但视觉描述优先）
        if subtitle_text and subtitle_text.strip():
            # 组合：视觉描述 + [字幕补充]
            # 例如："洞府内的场景 [对话: 这是我宗的身份令牌]"
            # 但为了简洁，通常只使用视觉描述即可
            # 如果 prefer_visual=True，只返回视觉描述
            if prefer_visual:
                return visual_desc_cn
            else:
                # 向后兼容：如果 prefer_visual=False，返回字幕
                return subtitle_text
        else:
            # 只有视觉描述，返回翻译后的中文
            return visual_desc_cn
    
    # 情况2：没有视觉描述，只有字幕（向后兼容）
    if subtitle_text and subtitle_text.strip():
        return subtitle_text
    
    # 情况3：都没有
    return ""

def translate_with_baidu(english_text, app_id=None, secret_key=None):
    """
    使用百度翻译API将英文翻译为中文
    
    Args:
        english_text: 英文文本
        app_id: 百度翻译APP ID（从环境变量BAIDU_TRANSLATE_APP_ID获取）
        secret_key: 百度翻译密钥（从环境变量BAIDU_TRANSLATE_SECRET_KEY获取）
    
    Returns:
        中文翻译结果，失败时返回原文本
    """
    if not english_text or not english_text.strip():
        return ""
    
    # 从环境变量获取API密钥
    if app_id is None:
        app_id = os.environ.get('BAIDU_TRANSLATE_APP_ID', '20251125002505258')
    if secret_key is None:
        secret_key = os.environ.get('BAIDU_TRANSLATE_SECRET_KEY', 'T3iF1iFXvPh6XNuYWrMu')
    
    if not app_id or not secret_key:
        print("  警告: 未配置百度翻译API，返回英文描述")
        return english_text
    
    try:
        # 百度翻译API参数
        url = "https://api.fanyi.baidu.com/api/trans/vip/translate"
        salt = str(random.randint(32768, 65536))
        sign_str = app_id + english_text + salt + secret_key
        sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
        
        params = {
            'q': english_text,
            'from': 'en',
            'to': 'zh',
            'appid': app_id,
            'salt': salt,
            'sign': sign
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        result = response.json()
        
        if 'trans_result' in result and len(result['trans_result']) > 0:
            translated = result['trans_result'][0]['dst']
            print(f"  ✓ 翻译: {english_text[:50]}... -> {translated[:50]}...")
            return translated
        elif 'error_code' in result:
            print(f"  警告: 百度翻译API错误 {result.get('error_code')}: {result.get('error_msg', '未知错误')}")
            return english_text
        else:
            print(f"  警告: 百度翻译返回异常: {result}")
            return english_text
            
    except Exception as e:
        print(f"  警告: 百度翻译失败: {e}，返回英文描述")
        return english_text

def translate_visual_to_chinese(english_text, use_translation_api=True, app_id=None, secret_key=None):
    """
    将英文视觉描述翻译为中文
    
    Args:
        english_text: 英文描述（如 "a man standing in a cave"）
        use_translation_api: 是否使用翻译API（默认True，使用百度翻译）
        app_id: 百度翻译APP ID（可选，优先使用参数，否则从环境变量读取）
        secret_key: 百度翻译密钥（可选，优先使用参数，否则从环境变量读取）
    
    Returns:
        中文描述（如 "一个男子站在洞府中"）
    """
    if not english_text:
        return ""
    
    if use_translation_api:
        # 使用百度翻译API
        return translate_with_baidu(english_text, app_id=app_id, secret_key=secret_key)
    
    # 不使用API时返回英文（embedding模型支持多语言）
    return english_text

def combine_descriptions(visual_caption, subtitle_text, prefer_chinese=True, enhance_with_visual=True, 
                         use_llm=False, use_translation=True, prefer_visual=False):
    """
    智能合并视觉描述和字幕文本，生成准确的中文描述
    
    策略：
    1. 优先使用视觉描述（描述画面内容，适合开头/结尾场景查找）
    2. 如果有字幕：作为补充信息（对话和剧情）
    3. 向后兼容：如果 prefer_visual=False，仍使用原逻辑（字幕优先）
    
    Args:
        visual_caption: BLIP-2生成的图像描述（英文）
        subtitle_text: 字幕文本（中文）
        prefer_chinese: 是否优先使用中文描述（默认True）
        enhance_with_visual: 是否用视觉描述增强字幕描述（默认True）
        use_llm: 是否使用LLM进行智能整合和翻译（默认False）
        use_translation: 是否使用百度翻译API（默认True）
        prefer_visual: 是否优先使用视觉描述（默认False，用于开头/结尾场景查找时设为True）
    
    Returns:
        合并后的描述文本（优先视觉描述，字幕作为补充）
    """
    # 使用智能推断函数
    return infer_chinese_description_from_context(
        visual_caption, 
        subtitle_text, 
        use_llm=use_llm,
        use_translation=use_translation,
        prefer_visual=prefer_visual
    )

def process_scene_keyframes(keyframe_dir, output_json, models=None, subtitle_json=None, 
                            prefer_chinese=True, skip_no_subtitle=False, use_llm_translation=False,
                            use_translation=True):
    """
    处理所有关键帧，生成描述和embedding（整合字幕信息）
    
    Args:
        keyframe_dir: 关键帧目录
        output_json: 输出JSON路径
        models: (processor, blip_model, clip_model, device) 元组
        subtitle_json: 已对齐的字幕JSON路径（可选）
        prefer_chinese: 是否优先使用中文描述（默认True）
        skip_no_subtitle: 是否跳过没有字幕的场景（默认False，仍会生成视觉描述）
    """
    keyframe_dir = Path(keyframe_dir)
    output_json = Path(output_json)
    
    if models is None:
        processor, blip_model, clip_model, device = load_models()
    else:
        processor, blip_model, clip_model, device = models
    
    # 加载字幕数据
    aligned_subtitles = load_aligned_subtitles(subtitle_json)
    if aligned_subtitles:
        print(f"✓ 加载字幕数据: {len(aligned_subtitles)} 个场景有字幕")
    
    keyframes = sorted(keyframe_dir.glob('*.jpg'))
    print(f"处理 {len(keyframes)} 个关键帧...")
    
    scene_metadata = {}
    
    for i, keyframe_path in enumerate(keyframes, 1):
        # 从文件名提取场景ID
        # 格式可能是：episode_171_clean-Scene-001_start.jpg 或 scene_001_start.jpg
        import re
        scene_match = re.search(r'Scene-(\d+)', keyframe_path.stem)
        if scene_match:
            scene_num = int(scene_match.group(1))
            scene_name = f"scene_{scene_num:03d}"  # scene_001
        else:
            # 备用方案：从文件名开头提取
            parts = keyframe_path.stem.split('_')
            if len(parts) >= 2 and parts[0] == 'scene':
                scene_name = f"{parts[0]}_{parts[1]}"
            else:
                # 最后备用：使用文件名前缀
                scene_name = keyframe_path.stem.split('_')[0]
        
        if scene_name not in scene_metadata:
            # 获取该场景的字幕信息
            scene_subtitle_data = aligned_subtitles.get(scene_name, {})
            subtitle_text = scene_subtitle_data.get("combined_text", "")
            subtitle_segments = scene_subtitle_data.get("subtitles", [])
            
            scene_metadata[scene_name] = {
                "text": "",
                "visual_caption": "",
                "subtitle_text": subtitle_text,
                "subtitles": subtitle_segments,
                "embedding": None,
                "keyframes": []
            }
        
        print(f"[{i}/{len(keyframes)}] 处理 {keyframe_path.name}...")
        
        try:
            # 获取该场景的字幕信息
            scene_subtitle_data = aligned_subtitles.get(scene_name, {})
            subtitle_text = scene_subtitle_data.get("combined_text", "")
            
            # 如果跳过没有字幕的场景，且当前场景没有字幕
            if skip_no_subtitle and not subtitle_text:
                print(f"  跳过（无字幕）")
                continue
            
            # 生成视觉描述（BLIP-2只支持英文）
            # 重要：即使有字幕，也生成视觉描述（用于场景理解，特别是开头/结尾场景查找）
            # 视觉描述描述画面内容，字幕描述对话，两者互补
            visual_caption = ""
            # 总是生成视觉描述（不再跳过）
            # 理由：视觉描述描述画面内容，字幕描述对话，对于场景检索（特别是开头/结尾）需要画面描述
            try:
                visual_caption = describe_image(keyframe_path, processor, blip_model, device, language='english')
                if visual_caption:
                    print(f"  ✓ 视觉描述: {visual_caption[:50]}...")
            except Exception as e:
                print(f"  ⚠ 视觉描述生成失败: {e}")
                visual_caption = ""
            
            # 获取embedding（使用第一个关键帧）
            if scene_metadata[scene_name]["embedding"] is None:
                # 智能合并视觉描述和字幕，生成准确的中文描述
                # 优先使用视觉描述（描述画面内容），字幕作为补充
                # 如果没有字幕，会使用百度翻译将英文描述翻译为中文
                combined_text = combine_descriptions(
                    visual_caption, 
                    subtitle_text, 
                    prefer_chinese=prefer_chinese,
                    use_translation=use_translation,
                    prefer_visual=True  # 优先使用视觉描述（用于场景检索）
                )
                
                # 生成embedding（使用合并后的文本，主要是中文）
                embedding = get_embedding(combined_text, clip_model)
                scene_metadata[scene_name]["embedding"] = embedding
                scene_metadata[scene_name]["text"] = combined_text
                # 保存原始视觉描述（英文）和字幕（中文）供参考
                scene_metadata[scene_name]["visual_caption"] = visual_caption
            
            scene_metadata[scene_name]["keyframes"].append({
                "path": str(keyframe_path),
                "caption": visual_caption
            })
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            continue
    
    # 保存结果
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(scene_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 标注完成: {output_json}")
    print(f"  共处理 {len(scene_metadata)} 个场景")
    
    # 统计信息
    scenes_with_subtitles = sum(1 for s in scene_metadata.values() if s.get("subtitle_text"))
    scenes_chinese_only = sum(1 for s in scene_metadata.values() 
                              if s.get("text") and s.get("subtitle_text") and 
                              not s.get("visual_caption"))
    print(f"  其中 {scenes_with_subtitles} 个场景整合了字幕")
    if prefer_chinese:
        print(f"  其中 {scenes_chinese_only} 个场景仅使用中文字幕（无英文视觉描述）")
    
    return scene_metadata

def main():
    parser = argparse.ArgumentParser(description='生成场景描述和embedding')
    parser.add_argument('--input', '-i', required=True, help='关键帧目录')
    parser.add_argument('--output', '-o', required=True, help='输出JSON路径')
    parser.add_argument('--subtitles', '-s', help='已对齐的字幕JSON路径（可选）')
    parser.add_argument('--device', '-d', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--prefer-chinese', action='store_true', default=True,
                       help='优先使用中文描述（默认True，有字幕时直接使用字幕）')
    parser.add_argument('--skip-no-subtitle', action='store_true',
                       help='跳过没有字幕的场景（默认False，仍会生成视觉描述）')
    parser.add_argument('--use-llm-translation', action='store_true',
                       help='使用LLM进行智能翻译和整合（需要API配置，默认False）')
    parser.add_argument('--baidu-app-id', help='百度翻译APP ID（也可通过环境变量BAIDU_TRANSLATE_APP_ID设置）')
    parser.add_argument('--baidu-secret-key', help='百度翻译密钥（也可通过环境变量BAIDU_TRANSLATE_SECRET_KEY设置）')
    parser.add_argument('--no-translate', action='store_true',
                       help='禁用翻译，直接使用英文描述（embedding模型支持多语言）')
    
    args = parser.parse_args()
    
    # 设置百度翻译API密钥（如果通过命令行参数提供）
    if args.baidu_app_id:
        os.environ['BAIDU_TRANSLATE_APP_ID'] = args.baidu_app_id
    if args.baidu_secret_key:
        os.environ['BAIDU_TRANSLATE_SECRET_KEY'] = args.baidu_secret_key
    
    keyframe_dir = Path(args.input)
    if not keyframe_dir.exists():
        print(f"错误: 输入目录不存在: {keyframe_dir}")
        return 1
    
    try:
        models = load_models(args.device)
        process_scene_keyframes(
            keyframe_dir, args.output, models, args.subtitles,
            prefer_chinese=args.prefer_chinese,
            skip_no_subtitle=args.skip_no_subtitle,
            use_llm_translation=args.use_llm_translation,
            use_translation=not args.no_translate
        )
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

