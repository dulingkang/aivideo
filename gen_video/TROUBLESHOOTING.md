# 图片问题排查指南

## 当前修改总结

### ✅ 已完成的修改
1. **禁用LoRA**：角色LoRA和风格LoRA都已禁用
2. **InstantID权重**：从0.98降低到0.82
3. **HanLi.prompt模板**：已创建并增强（添加了性别、发型、服装等具体特征）
4. **风格分离**：风格只在Scene层注入
5. **移除冲突代码**：移除了secondary_parts中的风格标签

### 📝 当前Prompt结构
```
约束 -> HanLi.prompt模板 -> 场景描述 -> 环境 -> Scene层风格 -> 其他
```

## 如果图片还是不对，请检查：

### 1. 具体问题类型
请明确说明：
- **形象问题**：人脸不像韩立？还是整体气质不对？
- **风格问题**：是写实风格？还是其他风格？
- **细节问题**：发型、服装、表情等是否正确？

### 2. 可能的原因和解决方案

#### 原因A：HanLi.prompt模板不够具体
**症状**：形象模糊，不够像韩立

**解决方案**：
- 进一步细化模板描述
- 添加更多具体的视觉特征（发型、服装颜色、面部特征等）

#### 原因B：风格描述不够强
**症状**：风格不是动漫，或风格不一致

**解决方案**：
- 增强Scene层风格描述（已增强：添加了"3D rendered anime"）
- 检查negative prompt是否正确排除写实风格

#### 原因C：InstantID权重不合适
**症状**：
- 如果僵硬：权重可能还是太高，降低到0.75-0.80
- 如果不像：权重可能太低，提高到0.85-0.88

#### 原因D：参考图问题
**症状**：人脸相似度不够

**解决方案**：
- 检查参考图质量
- 确保参考图风格正确
- 检查参考图路径是否正确

### 3. 诊断步骤

#### 步骤1：查看生成的Prompt
运行生成时，查看日志中的prompt输出，确认：
- HanLi.prompt模板是否被使用
- 风格是否正确注入
- Prompt结构是否正确

#### 步骤2：检查具体场景
分析问题场景的具体表现：
- 哪些场景有问题？
- 问题是否一致？
- 是否某些场景更好？

#### 步骤3：逐步调整
根据问题类型，逐步调整参数：
1. 如果是形象问题：增强HanLi.prompt模板
2. 如果是风格问题：增强风格描述或调整negative prompt
3. 如果是僵硬问题：降低InstantID权重
4. 如果是不像问题：提高InstantID权重或检查参考图

### 4. 进一步优化建议

如果以上都不行，可以考虑：

#### 选项1：进一步增强HanLi.prompt模板
添加更多具体的视觉特征，例如：
```
a young male cultivator from Chinese xianxia novel,
male, man, masculine,
slim but resilient build, narrow shoulders,
long black hair flowing down, dark green simple cultivator robe, practical and plain,
sharp calm eyes, slightly cold expression, angular face,
low-key temperament, restrained emotions,
standing upright, calm posture,
no arrogance, no flamboyance,
quiet, cautious, intelligent, survival-oriented
```

#### 选项2：调整InstantID权重
根据实际情况微调：
- 如果僵硬：0.82 → 0.78-0.80
- 如果不像：0.82 → 0.85-0.88

#### 选项3：增强风格描述
如果需要更强调动漫风格，可以添加权重：
```
(Chinese xianxia anime illustration:1.3), (3D rendered anime:1.2), anime cinematic style, cinematic lighting
```

#### 选项4：检查参考图
确保参考图：
- 质量高、清晰
- 符合期望的风格（动漫风格）
- 角度合适（正脸或15°侧脸）

