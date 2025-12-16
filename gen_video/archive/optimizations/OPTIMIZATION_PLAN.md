# 优化方案实施计划

## 方案A：简化架构（推荐，立即实施）

### 原则
以InstantID为主，减少LoRA干扰，通过prompt控制风格

### 配置修改

```yaml
instantid:
  face_emb_scale: 0.85  # 从0.98降到0.85，给其他系统留空间，避免过度僵硬

lora:
  enabled: true
  alpha: 0.3  # 从0.5降到0.3，仅辅助，不覆盖InstantID

style_lora:
  enabled: false  # 禁用风格LoRA，通过prompt控制风格
```

### Prompt简化

**修改前**：
```
(Chinese xianxia anime style:1.8), (3D rendered anime:1.6), (detailed character:1.4), (cinematic lighting:1.4), 4k, sharp focus, traditional Chinese fantasy aesthetic, immortal cultivator style
```

**修改后**：
```
Chinese xianxia anime style, 3D rendered anime, detailed character, cinematic lighting, 4k, sharp focus, traditional Chinese fantasy aesthetic
```

### 优点
- 简化系统，减少冲突
- InstantID负责人脸，prompt负责风格
- 更容易调试和优化

### 预期效果
- 人脸相似度：保持（InstantID权重0.85仍然有效）
- 风格一致性：通过prompt和基础模型保证
- 角色特征：角色LoRA（0.3）辅助保持
- 整体质量：应该更稳定，减少冲突

---

## 方案B：分离式架构（中期实施）

### 原则
根据不同场景使用不同的组合策略

### 配置修改

```yaml
instantid:
  face_emb_scale: 0.90  # 适中权重

lora:
  enabled: true
  alpha: 0.4  # 中等权重

style_lora:
  enabled: true
  alpha: 0.5  # 中等权重
```

### 场景选择逻辑

```python
# 人物特写/近景：InstantID主导
if is_close_up or is_medium_shot:
    use_instantid = True
    instantid_face_emb_scale = 0.90
    lora_alpha = 0.3  # 低权重辅助
    style_lora_enabled = False  # 禁用风格LoRA
    # 通过prompt控制风格

# 人物远景：角色LoRA主导
elif is_wide_shot and has_character:
    use_instantid = False  # 禁用InstantID（远景人脸不重要）
    lora_alpha = 0.6  # 高权重
    style_lora_alpha = 0.5  # 中等权重
    # 使用LoRA控制角色和风格

# 纯场景：风格LoRA主导
else:
    use_instantid = False
    lora_enabled = False
    style_lora_alpha = 0.7  # 高权重
    # 主要通过风格LoRA和prompt
```

### 优点
- 不同场景使用最适合的组合
- 避免不必要的系统冲突
- 更精细的控制

### 缺点
- 实现复杂，需要修改代码逻辑
- 需要测试不同场景的效果

---

## 方案C：更换基础模型（长期考虑）

### 推荐模型

1. **Animagine XL**
   - 专门训练的动漫风格SDXL模型
   - 基础风格就是动漫，不需要风格LoRA
   - InstantID可能表现更好

2. **Anything系列**
   - 专门针对动漫训练的模型
   - 可能需要调整InstantID适配

### 实施步骤

1. 测试Animagine XL + InstantID的效果
2. 如果效果好，迁移到新模型
3. 重新训练或调整LoRA（如果需要）

### 优点
- 基础模型匹配，效果应该更好
- 减少风格LoRA的需求
- 整体架构更简单

### 缺点
- 需要重新测试和调整
- 可能需要重新训练LoRA

---

## 立即实施的优化（方案A简化版）

### 步骤1：修改配置

修改 `config.yaml`：

```yaml
instantid:
  face_emb_scale: 0.85  # 从0.98降到0.85

lora:
  alpha: 0.3  # 从0.5降到0.3

style_lora:
  enabled: false  # 禁用
```

### 步骤2：简化Prompt

修改 `prompt/builder.py`：

```python
# 移除权重标记，简化风格描述
hanli_style = "Chinese xianxia anime style, 3D rendered anime, detailed character, cinematic lighting, 4k, sharp focus, traditional Chinese fantasy aesthetic"
```

### 步骤3：测试效果

重新生成场景2，检查：
- [ ] 人脸相似度是否保持
- [ ] 风格是否为动漫风格
- [ ] 角色特征是否保持
- [ ] 整体质量是否提升

---

## 如果效果仍不好

### 进一步优化

1. **继续降低InstantID权重**：0.85 → 0.80
2. **完全移除角色LoRA**：只用InstantID + prompt
3. **测试不同的参考图**：确保参考图质量高
4. **考虑方案B（分离式架构）**：根据场景选择不同组合

### 最终方案

如果以上都不行，考虑：
- 使用方案C（更换基础模型）
- 或者重新思考整个架构，可能需要更激进的改变

---

## 关键指标

### 评估标准

1. **人脸相似度**：是否像韩立（参考图）
2. **风格一致性**：是否为动漫风格
3. **角色特征**：发型、服装等是否一致
4. **整体质量**：画面是否自然、不僵硬
5. **稳定性**：不同场景是否一致

### 预期改进

- **稳定性**：↑↑（减少系统冲突）
- **自然度**：↑↑（降低InstantID权重，避免僵硬）
- **风格一致性**：↑（通过prompt控制，但可能不如LoRA稳定）
- **角色特征**：→（可能略有下降，但应该可接受）

