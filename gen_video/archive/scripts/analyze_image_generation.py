#!/usr/bin/env python3
"""
分析图像生成问题：为什么scene_002和scene_004生成的图像不像韩立

检查点：
1. 角色识别是否正确
2. LoRA是否正确加载和应用
3. InstantID参考图像是否正确使用
4. 权重参数是否足够高
"""

import json
import yaml
from pathlib import Path

def analyze_generation_process():
    """分析生成过程中的可能问题"""
    
    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载场景数据
    script_json = Path(__file__).parent.parent / "lingjie" / "episode" / "1.json"
    with open(script_json, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    scenes = script_data.get("scenes", [])
    
    print("="*70)
    print("🔍 分析scene_002和scene_004图像生成问题")
    print("="*70)
    
    # 检查scene_002 (id=1)
    scene_002 = next((s for s in scenes if s.get("id") == 1), None)
    # 检查scene_004 (id=3)  
    scene_004 = next((s for s in scenes if s.get("id") == 3), None)
    
    for scene, name in [(scene_002, "scene_002"), (scene_004, "scene_004")]:
        if not scene:
            continue
            
        print(f"\n{'='*70}")
        print(f"📸 {name} (id={scene.get('id')})")
        print(f"{'='*70}")
        
        # 1. 检查角色识别
        print("\n1️⃣ 角色识别检查:")
        keywords = ["han li", "hanli", "韩立"]
        combined_text = " ".join([
            scene.get("description", ""),
            scene.get("prompt", ""),
            scene.get("narration", ""),
        ]).lower()
        
        found = [kw for kw in keywords if kw in combined_text]
        if found:
            print(f"  ✅ 应该能识别为'hanli' (找到关键词: {found})")
        else:
            print(f"  ❌ 可能无法识别为'hanli' (未找到关键词)")
        
        # 2. 检查prompt
        print(f"\n2️⃣ Prompt分析:")
        prompt = scene.get("prompt", "")
        print(f"  Prompt: {prompt}")
        if "han li" in prompt.lower() or "hanli" in prompt.lower():
            print(f"  ✅ Prompt中包含韩立")
        else:
            print(f"  ⚠ Prompt中未明确包含'han li'或'hanli'")
        
        # 3. 检查camera类型（影响权重）
        camera = scene.get("camera", "")
        print(f"\n3️⃣ 镜头类型:")
        print(f"  Camera: {camera}")
        
        is_wide = any(kw in camera.lower() for kw in ["wide", "top-down", "long", "establish"])
        is_close = any(kw in camera.lower() for kw in ["close", "close-up"])
        is_medium = any(kw in camera.lower() for kw in ["medium"])
        
        if is_wide:
            print(f"  📷 远景/全身 -> ip_adapter_scale = 0.95 * 0.85 = 0.8075")
        elif is_close:
            print(f"  📷 近景/特写 -> ip_adapter_scale = 0.95 * 1.3 = 1.235 (最高)")
        elif is_medium:
            print(f"  📷 中景/半身 -> ip_adapter_scale = 0.95 * 1.35 = 1.2825 (最高)")
        else:
            print(f"  📷 其他 -> 默认权重")
    
    # 4. 配置检查
    print(f"\n{'='*70}")
    print("⚙️ 配置参数检查")
    print(f"{'='*70}")
    
    lora_config = config['image']['lora']
    instantid_config = config['image']['instantid']
    
    print(f"\n📦 LoRA配置:")
    print(f"  enabled: {lora_config.get('enabled')}")
    print(f"  adapter_name: {lora_config.get('adapter_name')}")
    print(f"  alpha: {lora_config.get('alpha')} {'⚠️ 可能不够高' if lora_config.get('alpha', 0) < 0.7 else '✅ 足够高'}")
    
    print(f"\n🎭 InstantID配置:")
    print(f"  face_emb_scale: {instantid_config.get('face_emb_scale')} {'✅ 非常高' if instantid_config.get('face_emb_scale', 0) >= 0.9 else '⚠️ 可能不够高'}")
    print(f"  face_kps_scale: {instantid_config.get('face_kps_scale')}")
    face_image_path = Path(instantid_config.get('face_image_path', ''))
    if face_image_path.exists():
        print(f"  face_image_path: ✅ {face_image_path}")
    else:
        print(f"  face_image_path: ❌ {face_image_path} (不存在!)")
    
    print(f"\n🎨 风格LoRA配置:")
    style_lora = lora_config.get('style_lora', {})
    if isinstance(style_lora, dict):
        print(f"  enabled: {style_lora.get('enabled')}")
        print(f"  adapter_name: {style_lora.get('adapter_name')}")
        print(f"  alpha: {style_lora.get('alpha')}")
        if style_lora.get('alpha', 1.0) >= 1.0:
            print(f"  ⚠️ 风格LoRA权重很高(1.0)，可能会覆盖角色特征!")
    
    # 5. 可能的问题
    print(f"\n{'='*70}")
    print("💡 可能的问题和解决方案")
    print(f"{'='*70}")
    
    issues = []
    solutions = []
    
    if style_lora.get('alpha', 1.0) >= 1.0:
        issues.append("风格LoRA权重过高(1.0)，可能覆盖角色LoRA的特征")
        solutions.append("考虑降低style_lora的alpha到0.7-0.8")
    
    if lora_config.get('alpha', 0.7) < 0.75:
        issues.append("角色LoRA权重可能不够高")
        solutions.append("考虑提高lora.alpha到0.75-0.80")
    
    if instantid_config.get('face_emb_scale', 0.95) < 0.95:
        issues.append("InstantID人脸权重可能不够高")
        solutions.append("确保face_emb_scale >= 0.95")
    
    if issues:
        print("\n⚠️ 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
            print(f"     解决方案: {solutions[i-1]}")
    else:
        print("\n✅ 配置参数看起来都正常")
    
    print(f"\n{'='*70}")
    print("🔍 建议检查生成的日志:")
    print("  1. 是否有 '检测到角色: hanli（韩立），自动加载LoRA: hanli'")
    print("  2. 是否有 '✅ LoRA 已加载: hanli'")
    print("  3. 是否有 '✓ 使用用户指定的角色LoRA: hanli (alpha=0.70)'")
    print("  4. 是否有 '✓ 已应用LoRA适配器: [hanli, anime_style]'")
    print("  5. ip_adapter_scale的实际值是多少")
    print("  6. 是否使用了正确的参考图像 (hanli_mid.png)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    analyze_generation_process()

