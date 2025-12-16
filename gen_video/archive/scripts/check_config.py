#!/usr/bin/env python3
"""检查配置是否正确：验证韩立的LoRA和参考图像配置"""

import yaml
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("="*60)
print("检查韩立角色配置")
print("="*60)

# 检查LoRA配置
lora_config = config['image']['lora']
print(f"\n📦 LoRA配置:")
print(f"  enabled: {lora_config.get('enabled')}")
print(f"  weights_path: {lora_config.get('weights_path')}")
print(f"  adapter_name: {lora_config.get('adapter_name')}")
print(f"  alpha: {lora_config.get('alpha')}")

lora_path = Path(lora_config.get('weights_path', ''))
if lora_path.exists():
    print(f"  ✅ LoRA文件存在: {lora_path}")
    print(f"     文件大小: {lora_path.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"  ❌ LoRA文件不存在: {lora_path}")

# 检查InstantID配置
instantid_config = config['image']['instantid']
print(f"\n🎭 InstantID配置:")
print(f"  face_image_path: {instantid_config.get('face_image_path')}")
print(f"  face_emb_scale: {instantid_config.get('face_emb_scale')}")
print(f"  face_kps_scale: {instantid_config.get('face_kps_scale')}")

face_image_path = Path(instantid_config.get('face_image_path', ''))
if face_image_path.exists():
    print(f"  ✅ 参考图像存在: {face_image_path}")
    print(f"     文件大小: {face_image_path.stat().st_size / 1024:.1f} KB")
else:
    print(f"  ❌ 参考图像不存在: {face_image_path}")

# 检查风格LoRA
style_lora = lora_config.get('style_lora', {})
if isinstance(style_lora, dict):
    print(f"\n🎨 风格LoRA配置:")
    print(f"  enabled: {style_lora.get('enabled')}")
    print(f"  adapter_name: {style_lora.get('adapter_name')}")
    print(f"  alpha: {style_lora.get('alpha')}")

print(f"\n{'='*60}")
print("💡 当前配置总结:")
print(f"  1. LoRA权重 (alpha): {lora_config.get('alpha')}")
print(f"  2. InstantID人脸权重 (face_emb_scale): {instantid_config.get('face_emb_scale')}")
print(f"  3. 这两个值都应该足够高以确保韩立的相似度")
print(f"{'='*60}\n")

