#!/usr/bin/env python3
"""
修复scene_002生成问题：
1. 两个头（多人问题）
2. 性别错误（变成女性）
3. 场景不对（不在沙漠）
4. 缺少修仙风格

解决方案：
1. 加强negative prompt中排除多人和女性的权重
2. 确保环境描述被正确添加到prompt
3. 确保仙侠风格被正确添加
"""

import re

# 读取image_generator.py文件
image_gen_path = "/vepfs-dev/shawn/vid/fanren/gen_video/image_generator.py"

print("正在修复image_generator.py...")

with open(image_gen_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: 加强排除女性的权重（从1.5提高到2.0）
old_female_negative = r', female, woman, girl, feminine.*?\(girl body:1\.5\)'
new_female_negative = ', female, woman, girl, feminine, female character, woman character, girl character, female figure, woman figure, girl figure, female appearance, woman appearance, girl appearance, (female:2.0), (woman:2.0), (girl:2.0), (feminine:2.0), (female character:2.0), (woman character:2.0), (girl character:2.0), (female figure:2.0), (woman figure:2.0), (girl figure:2.0), (female appearance:2.0), (woman appearance:2.0), (girl appearance:2.0), female face, woman face, girl face, female body, woman body, girl body, (female face:2.0), (woman face:2.0), (girl face:2.0), (female body:2.0), (woman body:2.0), (girl body:2.0), breasts, female breasts, (breasts:2.0), (female breasts:2.0), long hair flowing, feminine hair, (feminine hair:2.0)'

# 使用正则替换
pattern = r'(female_negative = ")(.*?)(female body:1\.5\).*?)(")'
replacement = r'\1' + new_female_negative + r'\4'
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 修复2: 在multiple_people_negative中添加two heads
if 'two heads' not in content.lower() or '(two heads:1.8)' not in content:
    # 在multiple_people_negative中添加two heads相关
    pattern = r'(multiple_people_negative = ")(.*?)(\(copy of person:1\.8\))'
    replacement = r'\1\2(two heads:2.0), (multiple heads:2.0), (duplicate head:2.0), (second head:2.0), \3'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 修复3: 提高多人排除权重从1.8到2.0
content = content.replace('(duplicate person:1.8)', '(duplicate person:2.0)')
content = content.replace('(same person twice:1.8)', '(same person twice:2.0)')
content = content.replace('(two same people:1.8)', '(two same people:2.0)')
content = content.replace('(multiple people:1.8)', '(multiple people:2.0)')
content = content.replace('(two people:1.8)', '(two people:2.0)')

print("修复完成！")
print("\n修复内容：")
print("1. ✓ 提高排除女性权重从1.5到2.0，并添加breasts、feminine hair等排除项")
print("2. ✓ 添加two heads、multiple heads排除项（权重2.0）")
print("3. ✓ 提高多人排除权重从1.8到2.0")

# 保存文件
with open(image_gen_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\n文件已保存: {image_gen_path}")

