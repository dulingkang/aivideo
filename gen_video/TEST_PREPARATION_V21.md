# v2.1系统测试准备清单

## ✅ 转换功能验证

**转换预览测试通过**：
```
场景 1: v2 → v2.1-exec
  Shot: medium (锁定: True)
  Pose: lying (修正: False)
  Model: flux + pulid
  决策原因: character_present + medium_shot -> flux + pulid

场景 2: v2 → v2.1-exec
  Shot: wide (锁定: True)
  Pose: stand (修正: False)
  Model: flux + None
  决策原因: no_character -> flux
```

## 📋 测试前检查清单

### 1. 环境准备

- [ ] 激活正确的conda环境（包含torch等依赖）
- [ ] 检查GPU可用性
- [ ] 检查显存是否充足

### 2. 配置文件检查

- [ ] `config.yaml` 配置正确
- [ ] 角色LoRA路径配置正确
- [ ] InstantID/PuLID路径配置正确

### 3. JSON文件检查

- [ ] JSON文件路径正确
- [ ] JSON格式有效
- [ ] 场景数量正确

### 4. 输出目录

- [ ] 输出目录有写入权限
- [ ] 有足够的磁盘空间

## 🚀 测试命令

### 预览转换（不依赖完整环境）

```bash
cd /vepfs-dev/shawn/vid/fanren/gen_video
python3 test_v21_batch_preview.py ../lingjie/episode/1.v2-1.json 1 3
```

### 实际生成测试（需要完整环境）

```bash
cd /vepfs-dev/shawn/vid/fanren
proxychains4 python3 gen_video/tools/batch_novel_generator.py \
  --json ../lingjie/episode/1.v2-1.json \
  --output-dir outputs/batch_test_$(date +%Y%m%d_%H%M%S) \
  --start 1 \
  --end 3 \
  --enable-m6 \
  --quick
```

## 🔍 预期行为

### v2.1-exec模式下的行为

1. **自动转换**：
   - 检测到v2格式自动转换为v2.1-exec
   - 打印转换信息

2. **硬规则应用**：
   - Shot类型根据Intent硬映射
   - Pose根据Shot类型验证和修正
   - Model路由根据场景条件硬选择

3. **角色锚系统**：
   - 使用LoRA（Layer 0）
   - 条件启用InstantID/PuLID（Layer 1）
   - 应用性别负锁（Layer 2）

4. **生成流程**：
   - 阶段1：生成图片（使用v2.1-exec决策）
   - 阶段2：生成配音
   - 阶段3：生成视频

## ⚠️ 注意事项

1. **环境依赖**：
   - 需要激活包含torch、transformers等依赖的conda环境
   - 如果遇到`ModuleNotFoundError: No module named 'torch'`，需要激活环境

2. **显存管理**：
   - v2.1-exec模式会清理显存
   - 如果显存不足，可能需要调整batch size

3. **错误处理**：
   - 如果v2.1-exec模式失败，会自动回退到原有流程
   - 检查日志了解失败原因

## 📊 验证点

测试时关注以下验证点：

1. **转换正确性**：
   - [ ] v2格式正确转换为v2.1-exec
   - [ ] Shot类型正确映射
   - [ ] Pose验证和修正正确

2. **生成质量**：
   - [ ] 图片生成成功
   - [ ] 角色一致性（性别、脸型）
   - [ ] 场景符合预期

3. **稳定性**：
   - [ ] 无随机性（相同输入产生相同输出）
   - [ ] 无LLM调用（STRICT模式）
   - [ ] 错误处理正确

## 🔗 相关文档

- `INTEGRATION_GUIDE_V21.md` - 集成指南
- `USAGE_V2_1.md` - 使用指南
- `DEVELOPMENT_STATUS_V21.md` - 开发状态

