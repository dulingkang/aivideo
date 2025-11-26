#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为所有角色生成参考图像

读取 character_profiles.yaml 中的所有角色，使用文生图为每个角色生成一张参考图像，
并保存到参考图像目录中。这样以后就可以使用这些生成的参考图像，而不需要手动提供参考照片。
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from image_generator import ImageGenerator


def build_character_prompt(profile: Dict[str, Any], use_chinese: bool = False) -> str:
    """根据角色模板构建角色描述 prompt（用于生成参考图像）
    
    使用英文描述，精简内容，确保不超过 77 tokens（CLIP 限制）
    强调角色独特特征，避免生成类似韩立的特征
    """
    if not profile:
        return ""
    
    parts = []
    
    # 1. 风格（简洁）
    parts.append("xianxia fantasy")
    
    # 2. 角色身份（简洁，只保留关键信息，强调独特性）
    identity = profile.get("identity", "")
    if identity:
        # 简化身份描述，只保留核心信息
        identity_short = identity.split(",")[0] if "," in identity else identity
        # 移除权重标记（如果有）
        identity_short = identity_short.split(":")[0].strip()
        parts.append(f"{identity_short}")
    
    # 2.5. 强调角色独特性（避免生成类似韩立的特征）
    # 添加角色特有的面部特征描述
    character_name = profile.get("character_name", "")
    if character_name and "hanli" not in character_name.lower() and "han li" not in character_name.lower():
        # 对于非韩立角色，强调独特的面部特征
        # 通过强调不同的年龄、气质、面部特征来区分
        if profile.get("face_keywords"):
            # 如果有关键词，使用关键词（通常包含独特特征）
            face_desc = profile["face_keywords"]
            if len(face_desc) > 25:
                face_desc = face_desc[:25]
            parts.append(face_desc)
    
    # 3. 发型描述（精简）
    hair = profile.get("hair", {})
    if hair.get("prompt_keywords"):
        # 如果有关键词，使用关键词（通常更简洁）
        hair_desc = hair["prompt_keywords"]
        # 限制长度
        if len(hair_desc) > 30:
            hair_desc = hair_desc[:30]
        parts.append(hair_desc)
    elif hair.get("style") and hair.get("color"):
        # 简化：只保留颜色和样式
        parts.append(f"{hair['color']} {hair['style']}")
    
    # 4. 服饰描述（精简）
    clothes = profile.get("clothes", {})
    if clothes.get("prompt_keywords"):
        clothes_desc = clothes["prompt_keywords"]
        # 限制长度
        if len(clothes_desc) > 30:
            clothes_desc = clothes_desc[:30]
        parts.append(clothes_desc)
    elif clothes.get("style") and clothes.get("color"):
        parts.append(f"{clothes['color']} {clothes['style']}")
    
    # 5. 面部特征（精简，但已在 2.5 中处理，这里避免重复）
    # 注意：如果已经在 2.5 中添加了 face_keywords，这里不再重复添加
    if not (character_name and "hanli" not in character_name.lower() and profile.get("face_keywords")):
        if profile.get("face_keywords"):
            face_desc = profile["face_keywords"]
            # 限制长度
            if len(face_desc) > 25:
                face_desc = face_desc[:25]
            parts.append(face_desc)
    
    # 6. 身体特征（精简，可选）
    body = profile.get("body", {})
    if body.get("build"):
        build_desc = body["build"]
        if len(build_desc) > 20:
            build_desc = build_desc[:20]
        parts.append(build_desc)
    
    # 7. 镜头描述（InstantID 必需，强调正面半身照）
    # InstantID 需要：正面、半身照（不要太远）、清晰面部
    # 强调正面视角，避免侧面或斜面
    parts.append("medium shot")  # 半身照，不要太远
    parts.append("upper body")  # 上半身
    parts.append("front view")  # 正面视角（必需）
    parts.append("facing camera")  # 面向镜头（强调正面）
    parts.append("straight on")  # 正对镜头（强调正面）
    parts.append("clear face")  # 清晰面部（必需）
    parts.append("face visible")  # 面部可见（强调）
    parts.append("good lighting")  # 光线充足
    
    # 8. 质量标签（精简）
    parts.append("high quality, 4k, detailed face")
    
    # 组合 prompt
    prompt = ", ".join(parts)
    
    # 估算 token 数（简单估算：每个单词约 1 token，标点符号约 0.5 token）
    # 如果超过 70 tokens，进一步精简
    word_count = len(prompt.split())
    estimated_tokens = word_count + prompt.count(",") * 0.5
    
    if estimated_tokens > 70:
        # 进一步精简：只保留最关键的信息
        parts_simplified = []
        # 1. 风格（必需）
        parts_simplified.append("xianxia fantasy")
        # 2. 身份（精简，只保留第一个关键词）
        if identity:
            identity_short = identity.split(",")[0].split(":")[0].strip()
            if len(identity_short) > 20:
                identity_short = identity_short[:20]
            parts_simplified.append(identity_short)
        # 3. 发型（精简）
        if hair.get("color") and hair.get("style"):
            parts_simplified.append(f"{hair['color']} {hair['style']}")
        # 4. 服饰（精简）
        if clothes.get("color") and clothes.get("style"):
            parts_simplified.append(f"{clothes['color']} {clothes['style']}")
        # 5. 镜头描述（必需，强调正面半身照）
        parts_simplified.append("medium shot, upper body, front view, facing camera, clear face")
        # 6. 质量（精简）
        parts_simplified.append("high quality, 4k")
        
        prompt = ", ".join(parts_simplified)
        # 再次检查
        word_count = len(prompt.split())
        estimated_tokens = word_count + prompt.count(",") * 0.5
        
        # 如果还是太长，进一步精简质量标签
        if estimated_tokens > 70:
            parts_final = []
            parts_final.append("xianxia fantasy")
            if identity:
                identity_short = identity.split(",")[0].split(":")[0].strip()[:15]
                parts_final.append(identity_short)
            if hair.get("color") and hair.get("style"):
                parts_final.append(f"{hair['color']} {hair['style']}")
            if clothes.get("color") and clothes.get("style"):
                parts_final.append(f"{clothes['color']} {clothes['style']}")
            parts_final.append("medium shot, upper body, front view, facing camera, clear face")
            parts_final.append("high quality")
            prompt = ", ".join(parts_final)
    
    return prompt


def generate_character_reference(
    generator: ImageGenerator,
    character_id: str,
    profile: Dict[str, Any],
    output_dir: Path,
    use_chinese: bool = True,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """为单个角色生成参考图像"""
    print(f"\n{'='*60}")
    print(f"生成角色参考图像: {character_id}")
    print(f"角色名称: {profile.get('character_name', character_id)}")
    print(f"{'='*60}")
    
    # 构建 prompt
    prompt = build_character_prompt(profile, use_chinese)
    print(f"Prompt: {prompt[:200]}...")
    
    # 输出路径
    output_path = output_dir / f"{character_id}_reference.png"
    
    # 如果已存在，询问是否覆盖
    if output_path.exists():
        print(f"  ⚠ 参考图像已存在: {output_path}")
        print(f"  ℹ 跳过生成（如需重新生成，请先删除该文件）")
        return output_path
    
    try:
        # 生成图像（使用文生图，不使用参考照片）
        # 创建一个虚拟场景，标识为非韩立角色，强制使用文生图
        # 注意：即使是韩立，生成参考图像时也使用文生图
        virtual_scene = {
            "id": 0,
            "description": f"{profile.get('character_name', character_id)} reference image",
            "prompt": prompt,
            "characters": [{"name": "dummy_character"}] if character_id == "hanli" else [{"name": character_id}],
        }
        
        # 对于生成参考图像，强制使用 SDXL 文生图（不使用 InstantID）
        # 直接调用 _generate_image_sdxl，绕过角色识别逻辑
        print(f"  ℹ 使用 SDXL 文生图生成参考图像（不使用参考照片）")
        
        # 保存原始引擎和 pipeline
        original_engine = generator.engine
        original_pipeline = generator.pipeline
        
        # 确保 SDXL pipeline 已加载
        if generator.engine == "instantid":
            print(f"  ℹ 当前使用 InstantID 引擎，临时切换到 SDXL...")
            generator.engine = "sdxl"
            # 加载 SDXL pipeline（如果还没有加载）
            if not hasattr(generator, '_sdxl_pipeline_backup'):
                print(f"  ℹ 加载 SDXL pipeline...")
                generator._load_sdxl_pipeline()
                generator._sdxl_pipeline_backup = generator.pipeline
            else:
                generator.pipeline = generator._sdxl_pipeline_backup
        elif generator.pipeline is None:
            print(f"  ℹ 加载 SDXL pipeline...")
            generator._load_sdxl_pipeline()
        
        try:
            # 临时禁用所有参考图像相关配置，确保完全使用纯文生图
            original_face_ref_dir = generator.face_reference_dir
            original_face_ref_images = generator.face_reference_images
            original_ref_images = generator.reference_images
            original_use_ip_adapter = generator.use_ip_adapter
            original_use_img2img = generator.use_img2img
            
            # 临时清空所有参考图像配置
            generator.face_reference_dir = None
            generator.face_reference_images = []
            generator.reference_images = []
            generator.use_ip_adapter = False
            generator.use_img2img = False
            # 也禁用 LoRA（避免 LoRA 中包含韩立的面部特征）
            original_use_lora = generator.use_lora
            generator.use_lora = False
            
            print(f"  ℹ 已临时禁用所有参考图像、IP-Adapter 和 LoRA，使用纯文生图")
            
            # 构建负面提示词，避免生成类似韩立的特征和侧面/斜面
            # 对于非韩立角色，添加负面提示词来避免生成韩立的特征
            negative_prompt = generator.negative_prompt
            
            # 添加负面提示词，避免生成侧面或斜面的脸（InstantID 需要正面）
            face_angle_negative = "side view, side profile, side face, angled face, three-quarter view, profile view, looking away, turned head"
            if negative_prompt:
                negative_prompt = f"{negative_prompt}, {face_angle_negative}"
            else:
                negative_prompt = face_angle_negative
            
            if character_id != "hanli":
                # 添加负面提示词，避免生成韩立的特征
                hanli_negative = "Han Li, calm cultivator, lean physique, light-colored robe, golden trim, calm focused eyes, similar face to Han Li"
                negative_prompt = f"{negative_prompt}, {hanli_negative}"
                print(f"  ℹ 添加负面提示词，避免生成类似韩立的特征和侧面/斜面")
            else:
                print(f"  ℹ 添加负面提示词，避免生成侧面/斜面的脸（InstantID 需要正面）")
            
            # 直接调用 SDXL 生成方法（禁用 IP-Adapter）
            result_path = generator._generate_image_sdxl(
                prompt=prompt,
                output_path=output_path,
                negative_prompt=negative_prompt,
                seed=seed,
                reference_image_path=None,  # 不使用参考图像
                face_reference_image_path=None,  # 不使用面部参考
                use_lora=None,
                scene=None,  # 不使用场景（避免角色识别）
                use_ip_adapter_override=False,  # 强制禁用 IP-Adapter
            )
            
            # 恢复原始配置
            generator.face_reference_dir = original_face_ref_dir
            generator.face_reference_images = original_face_ref_images
            generator.reference_images = original_ref_images
            generator.use_ip_adapter = original_use_ip_adapter
            generator.use_img2img = original_use_img2img
            generator.use_lora = original_use_lora
        finally:
            # 恢复原始引擎和 pipeline
            generator.engine = original_engine
            generator.pipeline = original_pipeline
        
        print(f"  ✓ 参考图像已生成: {result_path}")
        return result_path
        
    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="为所有角色生成参考图像")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, help="参考图像输出目录（默认：face_reference_dir 或 reference_image_dir）")
    parser.add_argument("--characters", type=str, nargs="+", help="指定要生成的角色ID（默认：所有角色）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的参考图像")
    parser.add_argument("--seed", type=int, help="随机种子（用于可重复生成）")
    parser.add_argument("--use-chinese", action="store_true", default=False, help="使用中文 prompt（默认：False，使用英文）")
    parser.add_argument("--no-chinese", dest="use_chinese", action="store_false", help="不使用中文 prompt（默认）")
    
    args = parser.parse_args()
    
    # 初始化图像生成器
    print("初始化图像生成器...")
    generator = ImageGenerator(args.config)
    
    # 加载模型
    print("加载图像生成模型...")
    generator.load_pipeline()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 使用配置中的 face_reference_dir 或 reference_image_dir
        face_ref_dir = generator.image_config.get("face_reference_dir")
        ref_dir = generator.image_config.get("reference_image_dir")
        if face_ref_dir:
            output_dir = Path(face_ref_dir)
        elif ref_dir:
            output_dir = Path(ref_dir) / "character_references"
        else:
            output_dir = Path("gen_video/character_references")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"参考图像输出目录: {output_dir}")
    
    # 加载角色配置
    character_profiles = generator.character_profiles
    if not character_profiles:
        print("⚠ 未找到角色配置，请检查 character_profiles.yaml 文件")
        return
    
    print(f"\n找到 {len(character_profiles)} 个角色配置")
    
    # 确定要生成的角色
    if args.characters:
        characters_to_generate = {char_id: character_profiles.get(char_id) 
                                 for char_id in args.characters 
                                 if char_id in character_profiles}
        if len(characters_to_generate) < len(args.characters):
            missing = set(args.characters) - set(characters_to_generate.keys())
            print(f"⚠ 以下角色未找到: {missing}")
    else:
        characters_to_generate = character_profiles
    
    print(f"将生成 {len(characters_to_generate)} 个角色的参考图像\n")
    
    # 生成每个角色的参考图像
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for char_id, profile in characters_to_generate.items():
        output_path = output_dir / f"{char_id}_reference.png"
        
        # 检查是否已存在
        if args.skip_existing and output_path.exists():
            print(f"  ⏭ 跳过 {char_id}（已存在）")
            skip_count += 1
            continue
        
        # 生成参考图像
        result = generate_character_reference(
            generator=generator,
            character_id=char_id,
            profile=profile,
            output_dir=output_dir,
            use_chinese=args.use_chinese,
            seed=args.seed,
        )
        
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print(f"生成完成！")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(characters_to_generate)}")
    print(f"{'='*60}")
    print(f"\n参考图像保存在: {output_dir}")
    print(f"\n使用说明：")
    print(f"  1. 生成的参考图像可用于后续的场景生成")
    print(f"  2. 在 config.yaml 中设置 face_reference_dir 或 reference_image_dir 指向此目录")
    print(f"  3. 系统会自动使用这些参考图像进行角色生成")


if __name__ == "__main__":
    main()

