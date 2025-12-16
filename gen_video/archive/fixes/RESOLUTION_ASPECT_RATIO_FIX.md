# 图像和视频分辨率比例修复说明

## 🔍 问题描述

用户反馈：图片和视频的分辨率比例不一致，导致视频看起来有点横向压缩。

## ✅ 修复方案

### 1. 读取图像实际分辨率

在图像生成后，读取实际分辨率：
```python
generated_image = PILImage.open(image_path)
actual_image_width, actual_image_height = generated_image.size
width = actual_image_width
height = actual_image_height
```

### 2. 将分辨率传递给视频生成器

通过 `video_scene` 传递分辨率：
```python
video_scene['width'] = width  # 使用图像的实际宽度
video_scene['height'] = height  # 使用图像的实际高度
```

### 3. 视频生成器从scene读取分辨率

优先从scene中获取分辨率（确保与图像一致）：
```python
if scene and 'width' in scene and 'height' in scene:
    width = scene['width']
    height = scene['height']
    print(f"  ℹ 从scene获取分辨率: {width}x{height} (与图像一致)")
```

### 4. 调整分辨率到8的倍数时保持长宽比

**关键修复**：在调整到8的倍数时，保持长宽比：
```python
original_aspect = width / height

# 先调整宽度到8的倍数
width = (width // 8) * 8
# 根据原始长宽比计算高度（保持长宽比）
height = int(width / original_aspect)
# 再调整高度到8的倍数
height = (height // 8) * 8
# 重新计算宽度，确保长宽比一致
width = int(height * original_aspect)
width = (width // 8) * 8
```

### 5. 图像调整时保持长宽比

**关键修复**：如果图像分辨率与目标不一致，使用保持长宽比的方式调整：

```python
if image_aspect > target_aspect:
    # 图像更宽，先调整高度，然后居中裁剪宽度
    new_height = height
    new_width = int(image.size[0] * (height / image.size[1]))
    resized_image = image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    # 居中裁剪
    left = (new_width - width) // 2
    image = resized_image.crop((left, 0, left + width, height))
else:
    # 图像更高，先调整宽度，然后居中裁剪高度
    new_width = width
    new_height = int(image.size[1] * (width / image.size[0]))
    resized_image = image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    # 居中裁剪
    top = (new_height - height) // 2
    image = resized_image.crop((0, top, width, top + height))
```

## 📊 处理流程

1. **图像生成**：使用指定的width和height生成图像
2. **读取实际分辨率**：读取生成图像的实际分辨率
3. **更新分辨率变量**：更新width和height为实际分辨率
4. **传递给视频生成器**：通过video_scene传递分辨率
5. **调整到8的倍数**：保持长宽比，调整到8的倍数
6. **调整图像大小**：如果图像与目标不一致，使用保持长宽比的方式调整（居中裁剪）

## 🎯 关键改进

### 改进1：调整到8的倍数时保持长宽比
- **之前**：直接调整width和height到8的倍数，可能改变长宽比
- **现在**：先调整一个维度，根据长宽比计算另一个维度，再调整到8的倍数

### 改进2：图像调整时保持长宽比
- **之前**：直接resize，可能导致变形
- **现在**：使用保持长宽比的方式（resize + 居中裁剪），避免变形

## ⚠️ 注意事项

1. **8的倍数限制**：HunyuanVideo要求分辨率是8的倍数，调整时可能会略微改变长宽比（但已尽量保持接近）

2. **居中裁剪**：如果图像长宽比与目标不一致，会使用居中裁剪，可能会丢失边缘内容

3. **最佳实践**：建议图像生成时就使用与视频目标分辨率一致的长宽比，避免后续调整

## 📝 总结

- ✅ 图像和视频使用相同的分辨率（从图像实际分辨率读取）
- ✅ 调整到8的倍数时保持长宽比
- ✅ 图像调整时使用保持长宽比的方式（居中裁剪）
- ✅ 避免横向压缩或变形

现在图像和视频应该使用相同的分辨率，并且长宽比保持一致，不会出现横向压缩的问题。



