#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M6 端到端测试脚本
MVP 策略验证：HunyuanVideo + VideoIdentityAnalyzer + 失败重试

流程:
1. 加载 Anchor 图 (Hanli)
2. 调用 EnhancedVideoGeneratorM6 生成视频
3. 自动执行身份验证
4. 输出最终报告

Author: AI Video Team
Date: 2025-12-18
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('m6_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# 添加路径
sys.path.insert(0, os.path.abspath("."))

from enhanced_video_generator_m6 import EnhancedVideoGeneratorM6
from video_identity_verifier import ShotLanguage

def run_test(args: argparse.Namespace):
    print("=" * 60)
    print("M6 MVP 端到端测试")
    print("=" * 60)
    
    # 1. 设置路径
    input_image = args.input_image
    reference_image = args.reference_image or args.input_image  # 默认使用同一张作为参考
    output_dir = args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = os.path.join(output_dir, f"{args.output_prefix}_{timestamp}.mp4")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入
    if not os.path.exists(input_image):
        logger.error(f"❌ 输入图像不存在: {input_image}")
        return
    
    logger.info(f"输入图像: {input_image}")
    logger.info(f"输出路径: {output_video}")
    
    # 2. 初始化生成器
    logger.info("初始化增强版生成器...")
    try:
        generator = EnhancedVideoGeneratorM6("config.yaml")
        
        # 覆盖 HunyuanVideo 参数（会在首次生成时被 load_model 读取）
        if args.model_path:
            generator.video_config.setdefault("hunyuanvideo", {})
            generator.video_config["hunyuanvideo"]["model_path"] = args.model_path
        if args.num_frames is not None:
            generator.video_config.setdefault("hunyuanvideo", {})
            generator.video_config["hunyuanvideo"]["num_frames"] = int(args.num_frames)
        if args.num_inference_steps is not None:
            generator.video_config.setdefault("hunyuanvideo", {})
            generator.video_config["hunyuanvideo"]["num_inference_steps"] = int(args.num_inference_steps)
        if args.width is not None:
            generator.video_config.setdefault("hunyuanvideo", {})
            generator.video_config["hunyuanvideo"]["width"] = int(args.width)
        if args.height is not None:
            generator.video_config.setdefault("hunyuanvideo", {})
            generator.video_config["hunyuanvideo"]["height"] = int(args.height)

        logger.info(
            "  配置覆盖: model_path=%s, num_frames=%s, steps=%s, size=%sx%s",
            args.model_path or "(config.yaml)",
            str(args.num_frames) if args.num_frames is not None else "(config.yaml)",
            str(args.num_inference_steps) if args.num_inference_steps is not None else "(config.yaml)",
            str(args.width) if args.width is not None else "(config.yaml)",
            str(args.height) if args.height is not None else "(config.yaml)",
        )
        
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return
    
    # 3. 生成视频
    logger.info("开始生成视频...")
    
    scene_config = {
        "prompt": args.prompt,
        "description": "Han Li portrait shot",
        "motion_intensity": args.motion_intensity,
        "negative_prompt": args.negative_prompt or ""
    }
    
    try:
        video_path, result = generator.generate_video_with_identity_check(
            image_path=input_image,
            output_path=output_video,
            reference_image=reference_image,
            scene=scene_config,
            shot_type=args.shot_type,
            enable_verification=(not args.no_verify),
            max_retries=args.max_retries
        )
        
        # 4. 输出结果
        print("\n" + "=" * 60)
        print("测试结果")
        print("=" * 60)
        
        if video_path:
            print(f"📁 视频路径: {video_path}")
            
            if result:
                status = "✅ 验证通过" if result.passed else "❌ 验证失败"
                print(f"🎯 最终状态: {status}")
                print(f"   平均相似度: {result.avg_similarity:.3f}")
                print(f"   漂移比例: {result.drift_ratio*100:.1f}%")
                print(f"   人脸检测率: {result.face_detect_ratio*100:.1f}%")
                
                if result.issues:
                    print("⚠️ 发现问题:")
                    for issue in result.issues:
                        print(f"   • {issue}")
                
                # 保存结果 JSON
                result_json = os.path.join(output_dir, f"result_{timestamp}.json")
                with open(result_json, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                print(f"📄 详细报告已保存: {result_json}")
            else:
                print("⚠ 无验证结果 (可能已禁用验证)")
        else:
            print("❌ 视频生成完全失败 (所有重试均未成功)")
            if result and result.issues:
                print("最后一轮失败原因:")
                for issue in result.issues:
                    print(f"   • {issue}")
    
    except Exception as e:
        logger.error(f"❌ 测试过程异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        generator.unload_all()
        logger.info("测试结束，资源已释放")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M6 端到端测试（HunyuanVideo + 身份验证 + 重试）")
    parser.add_argument("--input-image", default="reference_image/hanli_mid.jpg", help="输入 Anchor 图路径")
    parser.add_argument("--reference-image", default=None, help="参考图路径（用于身份验证，默认同 input-image）")
    parser.add_argument("--output-dir", default="outputs/m6_test", help="输出目录")
    parser.add_argument("--output-prefix", default="hanli_test", help="输出文件名前缀")

    parser.add_argument("--shot-type", default="medium", choices=["wide", "medium", "medium_close", "close", "extreme_close"], help="镜头类型")
    parser.add_argument("--max-retries", type=int, default=2, help="最大重试次数（覆盖 config）")
    parser.add_argument("--no-verify", action="store_true", help="禁用身份验证（仅生成视频）")

    parser.add_argument("--prompt", default="Han Li standing in a mystical garden, gentle breeze moving his hair, subtle movement, high quality, cinematic lighting", help="视频 prompt（会被稳定性增强器自动增强）")
    parser.add_argument("--negative-prompt", default="", help="额外 negative prompt（会叠加稳定性 negative prompt）")
    parser.add_argument("--motion-intensity", default="gentle", choices=["gentle", "moderate", "dynamic"], help="运动强度")

    # HunyuanVideo 覆盖参数（可选）
    parser.add_argument("--model-path", default=None, help="HunyuanVideo 模型路径（可选，覆盖 config）")
    parser.add_argument("--num-frames", type=int, default=None, help="帧数（可选，覆盖 config）")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="推理步数（可选，覆盖 config）")
    parser.add_argument("--width", type=int, default=None, help="宽度（可选，覆盖 config）")
    parser.add_argument("--height", type=int, default=None, help="高度（可选，覆盖 config）")

    run_test(parser.parse_args())
