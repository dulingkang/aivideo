# PuLID 注入流程说明

## 生成流程

PuLID 的生成流程是**同时进行**的，不是分步骤的：

```
1. Prompt 编码（T5/CLIP）
   ↓
2. 身份嵌入提取（PuLID Encoder）
   ↓
3. 去噪过程（同时使用 Prompt 和 ID Embedding）
   ├─ Prompt 控制：场景、姿态、服饰、环境
   └─ ID Embedding 控制：人脸特征、身份一致性
   ↓
4. 解码图像
```

## 关键点

### 1. Prompt 的作用
- **场景描述**：沙漠、环境、背景
- **姿态描述**：lying（躺）、sitting（坐）、standing（站）
- **服饰描述**：deep cyan cultivator robe
- **镜头类型**：full body shot, wide shot

### 2. ID Embedding 的作用
- **人脸特征**：眼睛、鼻子、嘴巴、脸型
- **身份一致性**：确保是同一个人

### 3. 为什么会出现"站着"而不是"躺着"？

**原因**：
1. Prompt 中"lying"描述的权重不够高
2. 或者"lying"描述的位置太靠后，被其他描述覆盖
3. 模型默认倾向于生成"standing"（站姿）

**解决方案**：
1. 在 prompt **最前面**添加高权重的"lying"描述（权重 2.5-3.0）
2. 添加负面提示词："NOT standing, NOT upright"
3. 确保"lying"描述在角色描述之前或紧跟在镜头类型之后

## Prompt 优先级顺序（推荐）

```
1. 镜头类型（最高优先级）
   (full body shot, ground visible, feet visible:2.5)

2. 姿态描述（高优先级，确保不被覆盖）
   (lying on ground, lying on sand, horizontal position, prone position, NOT standing:3.0)

3. 场景描述
   (vast desert landscape, sand dunes, desert floor visible:2.0)

4. 角色描述（性别、服饰、发型）
   (Male:1.8), (deep cyan cultivator robe:1.8), ...

5. 原始 prompt
   韩立, Gray-green desert floor, 韩立保持不动，静静体会脚下炽热的沙地。
```

## 当前实现

在 `execution_planner_v3.py` 中：
1. 场景分析器识别"保持不动"+"脚下"为"lying"
2. 自动添加高权重的"lying"描述到 prompt 前面
3. 确保姿态描述在角色描述之前

