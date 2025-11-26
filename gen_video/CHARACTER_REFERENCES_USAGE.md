# 角色参考图像使用说明

## 概述

系统现在支持为所有角色生成参考图像，并在生成场景图像时自动使用对应的参考图像。这样就不需要手动为每个角色准备参考照片了。

## 使用步骤

### 1. 生成角色参考图像

首先，运行脚本为所有角色生成参考图像：

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python generate_character_references.py --config config.yaml
```

**参数说明：**
- `--config`: 配置文件路径（默认：config.yaml）
- `--output-dir`: 参考图像输出目录（默认：自动检测或使用 `gen_video/character_references`）
- `--characters`: 指定要生成的角色ID（例如：`--characters huangliang_lingjun huan_cangqi`）
- `--skip-existing`: 跳过已存在的参考图像
- `--seed`: 使用固定随机种子（用于可重复生成）

**示例：**
```bash
# 生成所有角色的参考图像
python generate_character_references.py --config config.yaml

# 只生成特定角色
python generate_character_references.py --characters huangliang_lingjun huan_cangqi dumu_juren

# 重新生成所有角色（覆盖已存在的）
python generate_character_references.py --config config.yaml
```

### 2. 配置参考图像目录

生成的参考图像默认保存在 `gen_video/character_references/` 目录。

**方式一：在 config.yaml 中配置（推荐）**

在 `config.yaml` 中添加或修改 `character_reference_dir` 配置：

```yaml
image:
  character_reference_dir: /vepfs-dev/shawn/vid/fanren/gen_video/gen_video/character_references
```

**方式二：自动检测（无需配置）**

如果未配置 `character_reference_dir`，系统会自动检测：
1. 检查 `face_reference_dir` 的父目录下是否有 `character_references` 子目录
2. 检查 `face_reference_dir` 本身是否包含 `*_reference.png` 文件
3. 如果都不存在，使用默认路径 `gen_video/character_references`

### 3. 自动使用参考图像

配置完成后，系统会在生成场景图像时**自动**：

1. **识别场景中的角色**：自动识别场景描述中的角色（如黄粱灵君、寰姓少年等）
2. **选择对应的参考图像**：根据角色ID自动选择对应的参考图像
   - 例如：识别到 `huangliang_lingjun` → 使用 `huangliang_lingjun_reference.png`
   - 例如：识别到 `huan_cangqi` → 使用 `huan_cangqi_reference.png`
3. **生成场景图像**：使用选中的参考图像生成场景图像

**示例输出：**
```
生成场景图像 5/20: ...
  ✓ 识别到角色: huangliang_lingjun，使用对应参考图像: huangliang_lingjun_reference.png
  ✓ 使用角色参考图像: huangliang_lingjun -> huangliang_lingjun_reference.png
```

## 参考图像命名规则

参考图像必须按照以下命名规则：

```
{character_id}_reference.png
```

例如：
- `huangliang_lingjun_reference.png`（黄粱灵君）
- `huan_cangqi_reference.png`（寰姓少年）
- `dumu_juren_reference.png`（独目巨人）

角色ID必须与 `character_profiles.yaml` 中的角色ID一致。

## 工作流程

### 完整流程

```
1. 生成参考图像
   python generate_character_references.py
   ↓
2. 配置参考图像目录（可选，系统会自动检测）
   config.yaml: character_reference_dir
   ↓
3. 生成场景图像（自动使用参考图像）
   python generate_from_script.py --script scenes/43.json
   ↓
4. 系统自动识别角色并选择对应的参考图像
   - 识别场景中的角色
   - 查找对应的参考图像
   - 使用参考图像生成场景图像
```

### 角色识别逻辑

系统会根据以下信息识别场景中的角色：

1. **场景描述**（`description` 字段）
2. **提示词**（`prompt` 字段）
3. **角色姿态**（`visual.character_pose` 字段）
4. **旁白**（`narration` 字段）

支持的角色关键词：
- `huangliang_lingjun`: "黄粱灵君", "huangliang", "Huangliang Spirit Lord"
- `huan_cangqi`: "寰姓少年", "寰天奇", "huan cangqi", "huan tianqi", "Huan youth"
- `dumu_juren`: "独目巨人", "one-eyed giant", "giant", "巨人"
- 其他角色：根据 `character_profiles.yaml` 中的配置

## 注意事项

1. **韩立角色**：韩立仍然使用 InstantID（需要参考照片），不会使用生成的参考图像
2. **参考图像质量**：生成的参考图像质量取决于 prompt 和模型，如果质量不满意，可以：
   - 调整 `character_profiles.yaml` 中的角色描述
   - 使用不同的随机种子重新生成
   - 手动优化参考图像
3. **参考图像更新**：如果修改了角色描述，需要重新生成参考图像
4. **目录结构**：确保参考图像目录结构正确，系统才能自动找到

## 故障排除

### 问题1：系统没有使用参考图像

**检查：**
1. 参考图像是否存在：`ls gen_video/character_references/*_reference.png`
2. 角色ID是否匹配：参考图像文件名中的角色ID必须与 `character_profiles.yaml` 中的ID一致
3. 配置是否正确：检查 `config.yaml` 中的 `character_reference_dir` 配置

**解决：**
- 确保参考图像命名正确：`{character_id}_reference.png`
- 检查系统启动时的日志，看是否加载了参考图像

### 问题2：参考图像还是参考了韩立的面部

**原因：** 生成参考图像时可能使用了韩立的参考图像

**解决：**
- 重新生成参考图像，确保使用纯文生图（脚本已自动处理）
- 检查生成日志，确认已禁用 IP-Adapter 和 LoRA

### 问题3：找不到参考图像

**检查：**
1. 参考图像目录是否存在
2. 文件命名是否正确
3. 配置文件中的路径是否正确

**解决：**
- 运行 `python generate_character_references.py` 重新生成
- 检查 `config.yaml` 中的 `character_reference_dir` 配置

## 高级用法

### 为特定角色生成参考图像

```bash
# 只生成黄粱灵君和寰姓少年的参考图像
python generate_character_references.py --characters huangliang_lingjun huan_cangqi
```

### 使用固定种子生成（可重复）

```bash
# 使用固定种子，确保生成结果可重复
python generate_character_references.py --seed 42
```

### 重新生成所有参考图像

```bash
# 删除已存在的参考图像
rm gen_video/character_references/*_reference.png

# 重新生成
python generate_character_references.py
```

## 参考图像目录结构

```
gen_video/
├── character_references/          # 角色参考图像目录
│   ├── huangliang_lingjun_reference.png
│   ├── huan_cangqi_reference.png
│   ├── dumu_juren_reference.png
│   └── ...
└── ...
```

## 总结

使用角色参考图像后，系统可以：
- ✅ 自动识别场景中的角色
- ✅ 自动选择对应的参考图像
- ✅ 根据角色描述生成图像（不需要手动提供参考照片）
- ✅ 保持角色一致性（使用参考图像确保角色外观一致）

现在你可以直接运行场景生成，系统会自动使用这些参考图像！

