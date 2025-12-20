# 修复总结 - 图片生成卡住问题

## 问题描述
批量生成图片时，第一张或第二张图片会在"回退到本地规则引擎"之后卡住。

## 根本原因分析
1. **`filter_duplicates` 方法可能阻塞**：在处理大量描述时，关键词提取和去重计算可能耗时过长
2. **`merge_prompt_parts` 方法可能阻塞**：在处理大量 prompt 部分时，去重逻辑可能卡住
3. **缺少异常保护**：没有超时机制和异常处理，导致卡住后无法恢复

## 修复方案

### 1. 优化 `filter_duplicates` 调用（execution_planner_v3.py）
- 添加异常保护，失败时回退到简单去重
- 当描述数量 ≤ 3 时，直接使用简单去重，避免复杂计算

### 2. 优化 `merge_prompt_parts` 调用（execution_planner_v3.py）
- 添加异常保护，失败时回退到简单合并
- 当部分数量 ≤ 5 时，直接合并，不进行复杂去重

### 3. 添加显存清理（batch_novel_generator.py）
- 每张图片生成后立即清理显存
- 多次调用 `torch.cuda.empty_cache()` 和 `gc.collect()`
- 显示清理后的显存状态

### 4. 修复语法错误
- 修复 `generate_novel_video.py` 中的缩进错误
- 修复 `pulid_engine.py` 中的 `try-except` 块结构错误

## 测试脚本

### 1. 完整流程测试
```bash
cd /vepfs-dev/shawn/vid/fanren
source /vepfs-dev/shawn/venv/py312/bin/activate
python3 gen_video/test_full_generation.py
```

### 2. 批量生成测试
```bash
cd /vepfs-dev/shawn/vid/fanren
source /vepfs-dev/shawn/venv/py312/bin/activate
bash gen_video/run_batch_test.sh
```

### 3. 手动运行批量生成
```bash
cd /vepfs-dev/shawn/vid/fanren
source /vepfs-dev/shawn/venv/py312/bin/activate
python3 gen_video/tools/batch_novel_generator.py \
    --json lingjie/episode/1.v2-1.json \
    --output-dir gen_video/outputs/batch_test_$(date +%Y%m%d_%H%M%S) \
    --start 0 \
    --end 2 \
    --enable-m6 \
    --quick
```

**注意**：JSON 文件路径应该是 `lingjie/episode/1.v2-1.json`（相对于项目根目录），不是 `gen_video/lingjie/episode/1.v2-1.json`

## 修改的文件

1. `gen_video/execution_planner_v3.py`
   - 优化 `filter_duplicates` 调用，添加异常保护
   - 优化 `merge_prompt_parts` 调用，添加异常保护

2. `gen_video/tools/batch_novel_generator.py`
   - 添加显存清理逻辑
   - 每张图片生成后立即清理显存

3. `gen_video/generate_novel_video.py`
   - 修复缩进错误

4. `gen_video/pulid_engine.py`
   - 修复 `try-except` 块结构错误

## 预期效果

1. **不再卡住**：所有异常都有保护，失败时会回退到简单处理
2. **显存管理**：每张图片生成后立即清理，避免显存累积
3. **更好的错误处理**：所有关键步骤都有异常保护，不会因为单个步骤失败而卡住

## 注意事项

1. 如果仍然卡住，检查日志中的具体位置
2. 确保虚拟环境已激活：`source /vepfs-dev/shawn/venv/py312/bin/activate`
3. 如果遇到 proxy 问题，可以临时禁用：`unset ALL_PROXY`

## 下一步优化建议

1. 添加更详细的日志，定位具体卡住位置
2. 考虑添加超时机制（使用 `signal` 或 `threading.Timer`）
3. 优化 `extract_keywords` 方法，提高性能
4. 考虑使用缓存，避免重复计算

