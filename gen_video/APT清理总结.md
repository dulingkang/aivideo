# APT 清理总结

## 当前系统状态

- **系统盘**: 20G
- **已使用**: 14G (68%)
- **可用空间**: 6.4G

## APT 相关空间占用分析

### 1. APT 缓存和列表

- **/var/cache/apt**: 20K（已清理，几乎为0）
- **/var/lib/apt/lists**: 63MB（包列表，必需）
- **总计**: 约 63MB

### 2. 已安装的包分析

**最大的包（CUDA 相关，开发必需）**:
- nsight-compute-2024.3.2: ~1.1GB
- libcublas-dev-12-6: ~886MB
- cudnn-local-repo-ubuntu2404-9.11.0: ~819MB
- libcudnn9-cuda-12: ~815MB
- 其他 CUDA 包: ~3-4GB

**这些包是开发必需的，不建议删除！**

### 3. 其他可清理项

- **/usr/share/doc**: 16MB
- **/usr/share/locale**: 1.5MB
- **/usr/share/man**: 300KB
- **总计**: 约 18MB

## 清理结果

### 已执行的清理

1. ✅ **APT 缓存**: 已清理（几乎为0）
2. ✅ **APT 列表**: 已更新（63MB，必需）
3. ✅ **自动删除**: 无需要删除的包
4. ✅ **包配置**: 无需要清理的配置
5. ✅ **编译缓存**: 已清理

### 可进一步清理的项目

#### 1. 文档和手册页（可选，约18MB）

```bash
# 清理压缩的文档
find /usr/share/doc -name "*.gz" -delete
find /usr/share/man -name "*.gz" -delete
```

**释放空间**: 约 5-10MB

#### 2. 语言包（可选，约1.5MB）

```bash
# 保留英语和中文，删除其他语言
find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name "en*" ! -name "zh*" ! -name "locale-archive" -type d -exec rm -rf {} +
```

**释放空间**: 约 1-1.5MB

#### 3. 日志文件（如果很大）

```bash
# 清理7天前的日志
find /var/log -type f -name "*.log" -mtime +7 -delete
find /var/log -type f -name "*.gz" -mtime +7 -delete
journalctl --vacuum-time=7d
```

**释放空间**: 取决于日志大小

## 结论

### APT 相关清理空间有限

APT 相关的清理可以释放的空间非常有限：
- **APT 缓存**: 已清理（几乎为0）
- **文档和语言包**: 约 18MB
- **总计可释放**: 约 20-30MB

### 主要空间占用

系统盘的主要空间占用是：
1. **CUDA 开发包**: ~5-6GB（开发必需，不能删除）
2. **系统文件**: ~3-4GB（系统必需）
3. **其他文件和缓存**: ~4-5GB

### 建议

#### 1. 保持当前状态

当前 6.4G 可用空间已经足够日常使用，APT 相关的清理空间有限，不建议过度清理。

#### 2. 定期维护

```bash
# 定期运行安全清理
bash cleanup_apt_safe.sh

# 定期清理系统
bash cleanup_system_disk.sh
```

#### 3. 如果确实需要更多空间

**选项A: 清理文档和语言包**（释放约20MB）
```bash
bash cleanup_apt.sh
# 选择清理文档和语言包
```

**选项B: 检查其他大文件**
```bash
# 查找大文件
du -sh /* 2>/dev/null | sort -hr | head -20

# 查找大文件（详细）
find / -type f -size +100M 2>/dev/null | head -20
```

**选项C: 移动 CUDA 开发包到挂载盘**（不推荐，可能导致问题）

## 清理脚本

### 1. cleanup_apt_safe.sh（推荐）

自动清理安全的项目，无需交互：
```bash
bash cleanup_apt_safe.sh
```

### 2. cleanup_apt.sh（完整版）

包含交互式选项，可以清理文档、语言包等：
```bash
bash cleanup_apt.sh
```

## 总结

**APT 清理空间**: 约 20-30MB（非常有限）

**当前状态**: 
- 系统盘使用率: 68%
- 可用空间: 6.4G
- 状态: 健康

**建议**: 
- 保持当前状态
- 定期运行清理脚本
- 如果确实需要更多空间，检查其他大文件而不是 APT 包

## 其他清理建议

如果确实需要更多空间，可以考虑：

1. **检查其他大文件**
   ```bash
   du -sh /* 2>/dev/null | sort -hr | head -20
   ```

2. **清理用户目录缓存**
   ```bash
   find /home -name ".cache" -type d -exec du -sh {} \; | sort -hr
   ```

3. **检查 Docker 相关**（如果有）
   ```bash
   docker system df
   docker system prune -a
   ```

4. **检查容器相关**（如果有）
   ```bash
   podman system prune -a
   ```

但是，当前 6.4G 可用空间已经足够日常使用，不建议过度清理。


