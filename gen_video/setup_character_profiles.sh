#!/bin/bash
# 创建角色档案目录结构
# 用于存放多角度参考图和表情库

echo "==========================================="
echo "创建角色档案目录结构"
echo "==========================================="

# 配置
BASE_DIR="${1:-/vepfs-dev/shawn/vid/fanren/gen_video/character_profiles}"

echo "基础目录: $BASE_DIR"

# 创建目录结构
echo ""
echo "1. 创建目录结构..."

# 韩立角色目录
mkdir -p "$BASE_DIR/hanli/expressions"
mkdir -p "$BASE_DIR/hanli/poses"

# 其他角色模板
mkdir -p "$BASE_DIR/_template/expressions"
mkdir -p "$BASE_DIR/_template/poses"

echo "   ✓ 目录结构已创建"

# 创建 README
echo ""
echo "2. 创建 README 文件..."

cat > "$BASE_DIR/README.md" << 'EOF'
# 角色档案系统 (Character Profiles)

这个目录用于存放角色的多角度参考图和表情库，
参考可灵 Element Library 的多参考图方案。

## 目录结构

```
character_profiles/
├── hanli/                      # 韩立角色
│   ├── front.png              # 正面参考图
│   ├── three_quarter.png      # 3/4 侧面
│   ├── side.png               # 全侧面
│   ├── back.png               # 背面（可选）
│   ├── expressions/           # 表情库
│   │   ├── neutral.png        # 中性表情
│   │   ├── happy.png          # 开心
│   │   ├── sad.png            # 悲伤
│   │   ├── angry.png          # 愤怒
│   │   └── surprised.png      # 惊讶
│   ├── poses/                 # 姿势库（可选）
│   │   ├── standing.png       # 站立
│   │   ├── sitting.png        # 坐着
│   │   └── fighting.png       # 战斗姿势
│   └── metadata.yaml          # 角色元数据
└── _template/                  # 新角色模板
    ├── expressions/
    └── poses/
```

## 参考图要求

### 角度参考图
- **分辨率**: 建议 512x512 或更高
- **背景**: 纯色背景或透明背景
- **光照**: 柔和均匀光照
- **表情**: 中性表情
- **文件格式**: PNG (推荐) 或 JPG

### 表情库
- **尺寸**: 保持与角度参考图一致
- **角度**: 建议使用正面或 3/4 侧面
- **表情清晰**: 确保表情特征明显

## 元数据格式 (metadata.yaml)

```yaml
id: hanli
name: 韩立
gender: male
age_range: "20-25"
style_tags:
  - ancient_chinese
  - cultivator
  - handsome
appearance:
  hair: "black, long, tied with ribbon"
  eyes: "deep brown, determined"
  build: "lean, tall"
  clothing_default: "blue cultivator robe"
notes: |
  主角，凡人修仙传
  性格：谨慎、聪明、坚韧
```

## 使用方式

角色档案会被 Execution Planner V3 自动：

1. **根据相机角度选择参考图**
   - 侧面镜头 → 使用 side.png
   - 正面镜头 → 使用 front.png

2. **根据情绪选择表情参考**
   - sad 情绪 → 使用 expressions/sad.png
   - angry 情绪 → 使用 expressions/angry.png

3. **混合使用**
   - 主参考图 + 表情参考图
   - 提高人物一致性

## 添加新角色

1. 复制 `_template` 目录
2. 重命名为角色 ID
3. 添加参考图像
4. 创建 metadata.yaml
5. 在 config.yaml 中注册角色

```yaml
character_profiles:
  characters:
    new_character:
      id: new_character
      name: 新角色
      profile_dir: new_character
```
EOF

echo "   ✓ README.md 已创建"

# 创建韩立的元数据
echo ""
echo "3. 创建韩立角色元数据..."

cat > "$BASE_DIR/hanli/metadata.yaml" << 'EOF'
# 韩立角色档案
id: hanli
name: 韩立
gender: male
age_range: "20-25"
style_tags:
  - ancient_chinese
  - cultivator
  - xianxia
  - handsome
  - protagonist

appearance:
  hair: "black, long, tied with simple ribbon"
  eyes: "deep brown, sharp and observant"
  face: "handsome, calm expression, slight stubble"
  build: "lean, athletic, tall"
  clothing_default: "blue cultivator robe with silver trim"

personality:
  - cautious
  - intelligent
  - resourceful
  - determined
  - pragmatic

notes: |
  《凡人修仙传》主角
  从凡人一步步修炼成仙的传奇故事
  
  关键特征：
  - 眼神锐利但平静
  - 表情通常沉稳
  - 动作干练
  
  参考图来源：
  - hanli_mid.jpg: 主要参考
  
  Prompt 模板：
  "Han Li, young Chinese cultivator, black long hair tied with ribbon,
   blue cultivation robe, sharp observant eyes, calm determined expression,
   lean athletic build, Chinese xianxia style"
EOF

echo "   ✓ hanli/metadata.yaml 已创建"

# 提示添加参考图
echo ""
echo "==========================================="
echo "目录结构创建完成！"
echo "==========================================="
echo ""
echo "下一步："
echo "1. 将韩立的多角度参考图放入 $BASE_DIR/hanli/"
echo "   - front.png (正面)"
echo "   - three_quarter.png (3/4侧面)"
echo "   - side.png (侧面)"
echo ""
echo "2. 将表情参考图放入 $BASE_DIR/hanli/expressions/"
echo "   - neutral.png"
echo "   - happy.png"
echo "   - sad.png"
echo "   - angry.png"
echo ""
echo "3. 确保 config.yaml 中的 character_profiles.profiles_dir 指向正确路径"
echo ""
