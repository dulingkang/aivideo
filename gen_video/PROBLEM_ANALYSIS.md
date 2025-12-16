# 图片问题分析

## 当前状态检查

### ✅ 已确认
1. HanLi.prompt模板文件存在且内容正确
2. 代码逻辑正确（加载模板、分离风格）
3. 配置已修改（LoRA禁用、InstantID权重0.82）

### ❓ 可能的问题

#### 问题1：Prompt模板可能不够强
当前HanLi.prompt内容：
```
a young male cultivator from Chinese xianxia novel,
slim but resilient build,
sharp calm eyes, slightly cold expression,
low-key temperament, restrained emotions,
dark simple cultivator robe, practical and plain,
no arrogance, no flamboyance,
quiet, cautious, intelligent, survival-oriented
```

**可能不足**：
- 缺少具体的视觉特征描述（发型、服装细节等）
- 描述较为抽象（temperament, emotions），可能不够具体

#### 问题2：风格注入可能不够强
当前Scene层风格：
```
Chinese xianxia illustration, anime cinematic, cinematic lighting
```

**可能不足**：
- 没有权重标记，可能不够强调
- 可能需要更明确的动漫风格关键词

#### 问题3：InstantID权重可能仍然不合适
当前：`face_emb_scale: 0.82`

**可能问题**：
- 0.82可能还是太高（如果还是僵硬）
- 或者太低（如果人脸相似度不够）

#### 问题4：参考图可能有问题
- InstantID的参考图质量可能不够好
- 参考图可能不符合期望的风格

## 诊断步骤

### 1. 检查实际生成的Prompt
需要查看实际生成的prompt内容，确认：
- HanLi.prompt模板是否被正确使用
- 风格是否正确注入
- Prompt结构是否正确

### 2. 检查生成的图片
需要分析：
- 形象问题：是人脸不像，还是整体气质不对？
- 风格问题：是写实风格，还是其他风格？
- 细节问题：服装、发型、表情等是否正确？

### 3. 检查InstantID配置
需要确认：
- 参考图路径是否正确
- InstantID是否真的在工作
- 权重是否合适

## 建议的改进方向

### 改进1：增强HanLi.prompt模板
添加更具体的视觉特征：
- 发型描述（黑色长发等）
- 服装细节（深绿色道袍等）
- 面部特征（眼神、表情等）

### 改进2：增强风格描述
如果风格不对，可能需要：
- 添加明确的动漫风格关键词
- 使用权重标记（如果必要）
- 排除写实风格（negative prompt）

### 改进3：调整InstantID权重
根据实际情况调整：
- 如果僵硬：降低到0.75-0.80
- 如果不像：提高到0.85-0.88

### 改进4：检查参考图
确保参考图：
- 质量高、清晰
- 符合期望的风格
- 角度合适（正脸或15°侧脸）

