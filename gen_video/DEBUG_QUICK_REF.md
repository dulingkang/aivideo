# 调试快速参考

## 🚨 当前问题
**场景2生成失败**: `'unet'` KeyError  
**位置**: SDXL pipeline 调用时  
**状态**: 🔴 调试中

## 🔍 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 两阶段法入口 | `image_generator.py` | 2629-2720 |
| Stage A（人设图） | `image_generator.py` | 2640-2700 |
| Stage B（场景生成） | `image_generator.py` | 2702-2719 |
| IP-Adapter 加载 | `image_generator.py` | 1447-1565 |
| Pipeline 验证 | `image_generator.py` | 6461-6560 |

## 🛠️ 快速修复

### 1. 查看完整错误
```bash
proxychains4 python test_lingjie_scenes.py --scenes 1 2>&1 | tee debug.log
```

### 2. 临时禁用 CPU Offload
```yaml
# config.yaml
image:
  enable_cpu_offload: false
```

### 3. 检查人设图
```bash
ls -lh outputs/images/lingjie_test_scenes/character_cache/hanli_character.png
```

## 📋 两阶段法流程

```
场景2（top-down + lying）
  ↓
检测到 should_disable_instantid = True
  ↓
Stage A: 查找/生成人设图
  ├─ 检查缓存: character_cache/hanli_character.png
  ├─ 使用素材图: face_reference_dir/hanli_mid*.png
  └─ 生成人设图: InstantID (中景/半身)
  ↓
Stage B: 使用人设图生成场景
  ├─ 加载 SDXL pipeline
  ├─ 加载 IP-Adapter (scale=0.65)
  └─ 调用 _generate_image_sdxl
  ↓
❌ 错误: 'unet' KeyError
```

## 🔧 调试命令

```bash
# 测试场景2
cd /vepfs-dev/shawn/vid/fanren/gen_video
proxychains4 python test_lingjie_scenes.py --scenes 1

# 查看日志
tail -f debug.log | grep -E "(Stage|unet|IP-Adapter|SDXL)"
```

## 📝 待解决问题

- [ ] 定位 `'unet'` 错误的确切位置
- [ ] 确认 IP-Adapter 是否正确加载
- [ ] 验证 Stage A 是否成功
- [ ] 检查 pipeline 组件完整性

---
详细文档: `DEBUG_SUMMARY.md`

