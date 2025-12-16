# 角色一致性修复方案

## 问题分析

### 1. 性别变成女性
**根本原因**：
- InstantID + LoRA冲突，导致性别特征被覆盖
- 性别标记权重（1.8）不够高
- LoRA可能学习到了女性特征

### 2. 镜头太近
**根本原因**：
- 竖屏模式下，镜头距离检测可能失效
- 特写场景没有被正确识别和转换

### 3. SDXL + InstantID + LoRA 不能保持一致性
**根本原因**：
- LoRA权重（0.75）过高，与InstantID的face_emb_scale（0.95）冲突
- 风格LoRA（0.65）进一步干扰
- 三个模型相互竞争，导致特征混乱

## 解决方案

### 方案A：优化当前配置（推荐先试）

1. **降低LoRA权重，优先InstantID**
   - LoRA alpha: 0.75 → 0.5-0.6（降低，让InstantID主导）
   - 风格LoRA: 0.65 → 0.4-0.5（进一步降低或禁用）

2. **提高InstantID权重**
   - face_emb_scale: 0.95 → 0.98-1.0（最大化人脸相似度）

3. **增强性别约束**
   - 性别标记权重: 1.8 → 2.5（更高权重）
   - 女性排除权重: 2.0 → 2.5（更强排除）

4. **检查参考图片**
   - 确保参考图片是正脸、清晰的韩立形象
   - 如果参考图片有问题，建议替换

### 方案B：使用原生韩立形象（如果方案A不行）

如果InstantID + LoRA仍然冲突，建议：
1. **直接使用凡人修仙传动画/漫画中的韩立截图**
2. **使用IP-Adapter替代LoRA**（更稳定，不会覆盖InstantID）
3. **或者只使用InstantID + 参考图片**（完全依赖InstantID，不用LoRA）

### 方案C：分场景策略

- **特写/近景**：只用InstantID（禁用LoRA），face_emb_scale: 1.0
- **中景**：InstantID + LoRA（低权重0.5），face_emb_scale: 0.95
- **远景**：InstantID + LoRA（正常权重0.6），face_emb_scale: 0.90

## 实施步骤

### 第一步：检查参考图片

```bash
# 检查参考图片是否存在且清晰
ls -lh /vepfs-dev/shawn/vid/fanren/gen_video/reference_image/hanli_mid.png
```

### 第二步：优化配置

见下面的配置更改

### 第三步：增强性别约束

在prompt builder中提高性别标记权重

### 第四步：修复镜头距离

确保所有场景都正确识别镜头类型

