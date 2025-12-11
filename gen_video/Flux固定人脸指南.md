# Flux 固定科学主持人形象指南

## 📋 问题说明

**InstantID 的 IP-Adapter 权重不兼容 Flux**。InstantID 是为 SDXL 设计的，其 IP-Adapter 权重无法直接用于 Flux 模型。

## ✅ 解决方案

### 方案1: 使用 Flux 专用的 IP-Adapter（推荐）

Flux 需要使用 Flux 专用的 IP-Adapter 权重。推荐使用以下选项：

#### 选项 A: Flux 标准 IP-Adapter
- **模型**: `XLabs-AI/flux-ip-adapter`
- **用途**: 通用图像参考，可以用于固定风格和部分人脸特征
- **下载方式**:
```bash
# 使用 huggingface-cli 下载
huggingface-cli download XLabs-AI/flux-ip-adapter --local-dir models/instantid/ip-adapter-flux
```

#### 选项 B: IP-Adapter FaceID Plus for Flux（最佳选择，如果有）
- **用途**: 专门用于固定人脸身份
- **注意**: 需要确认是否有 Flux 版本的 FaceID Plus
- **如果存在，下载方式类似**:
```bash
huggingface-cli download <repo-id>/flux-ip-adapter-faceid-plus --local-dir models/instantid/ip-adapter-faceid-flux
```

### 方案2: 使用 LoRA + IP-Adapter 组合（当前可用）

如果暂时没有 Flux 专用的 FaceID IP-Adapter，可以使用以下组合：

1. **训练科学主持人的 LoRA**
   - 使用科学主持人的多张照片训练 LoRA
   - LoRA 可以固定角色的面部特征、发型、服装风格等
   - 训练工具: Kohya SS 或其他 LoRA 训练工具

2. **配合标准 IP-Adapter**
   - 使用 Flux 标准 IP-Adapter 作为风格参考
   - LoRA 负责固定人脸特征
   - IP-Adapter 负责场景和风格

### 方案3: 使用 SDXL + InstantID（备选）

如果 Flux 的 IP-Adapter 方案不可用，可以考虑：
- 使用 SDXL 模型 + InstantID（完全兼容）
- 生成图像后再进行风格迁移或后处理

## 🔧 当前代码状态

当前代码已经：
- ✅ 修复了 InsightFace 初始化问题（SCRFD 模型识别）
- ✅ 成功提取人脸特征（embedding shape: (512,)）
- ⚠️ 但 InstantID 的 IP-Adapter 权重不兼容 Flux，无法加载

## 📝 实施步骤

### 步骤1: 下载 Flux IP-Adapter 权重

```bash
# 创建目录
mkdir -p /vepfs-dev/shawn/vid/fanren/gen_video/models/instantid/ip-adapter-flux

# 下载 Flux IP-Adapter
cd /vepfs-dev/shawn/vid/fanren/gen_video/models/instantid/ip-adapter-flux
huggingface-cli download XLabs-AI/flux-ip-adapter --local-dir .
```

### 步骤2: 更新配置

在 `model_manager.py` 中更新 IP-Adapter 路径：

```python
self.instantid_paths = {
    "instantid": str(models_root / "instantid"),
    "controlnet": str(models_root / "instantid" / "ControlNet"),
    "ip_adapter": str(models_root / "instantid" / "ip-adapter-flux"),  # 使用 Flux 版本
}
```

### 步骤3: 重新运行

代码会自动加载 Flux IP-Adapter 权重，并使用人脸 embedding 生成图像。

## 🎯 最佳实践：固定科学主持人形象

### 方法1: LoRA + IP-Adapter（推荐）

1. **准备科学主持人照片**
   - 收集 20-50 张不同角度、不同场景的科学主持人照片
   - 确保照片清晰，人脸完整可见

2. **训练 LoRA**
   - 使用 Kohya SS 训练科学主持人 LoRA
   - 训练参数建议：
     - Learning rate: 1e-4
     - Steps: 1000-2000
     - Batch size: 2-4
     - Network rank: 32-64

3. **使用 LoRA + IP-Adapter**
   - 加载训练好的 LoRA
   - 配合 Flux IP-Adapter 使用
   - 在生成时设置较高的 LoRA 权重（0.8-1.0）

### 方法2: 纯 IP-Adapter FaceID（如果可用）

如果找到 Flux 版本的 IP-Adapter FaceID Plus：
1. 下载权重
2. 使用一张清晰的科学主持人照片
3. 提取人脸 embedding
4. 在生成时传递 embedding

## ⚠️ 注意事项

1. **权重兼容性**
   - InstantID 的 IP-Adapter 权重（`ip-adapter.bin`）是为 SDXL 设计的
   - 不能直接用于 Flux
   - 必须使用 Flux 专用的 IP-Adapter 权重

2. **人脸 Embedding 格式**
   - InsightFace 提取的 embedding 是 512 维
   - Flux IP-Adapter 可能需要不同的格式
   - 代码中已经做了格式转换，但可能需要根据实际权重调整

3. **效果预期**
   - 标准 IP-Adapter 主要用于风格参考，人脸固定效果可能不如 FaceID
   - 如果需要强人脸固定，建议使用 LoRA + IP-Adapter 组合

## 🔍 检查清单

- [ ] 已下载 Flux 专用的 IP-Adapter 权重
- [ ] 权重文件放在正确的目录
- [ ] 代码可以成功加载权重
- [ ] 人脸特征提取成功
- [ ] 生成时 IP-Adapter 参数正确传递
- [ ] 生成的图像包含科学主持人特征

## 📚 参考资源

- Flux IP-Adapter: https://huggingface.co/XLabs-AI/flux-ip-adapter
- IP-Adapter FaceID: https://huggingface.co/h94/IP-Adapter-FaceID
- LoRA 训练: https://github.com/bmaltais/kohya_ss

## 💡 临时解决方案

如果暂时无法获取 Flux IP-Adapter，可以：
1. 使用 LoRA 固定科学主持人形象（推荐）
2. 使用 SDXL + InstantID 生成，然后进行风格迁移
3. 等待 Flux 版本的 IP-Adapter FaceID Plus 发布

