# AI视频生成平台 - Web前端

基于 Vue 3 + TypeScript + Element Plus 的现代化Web界面

## 🚀 快速开始

### 安装依赖

```bash
cd gen_video/api/web_ui_vue
npm install
```

### 开发模式

```bash
npm run dev
```

访问：http://localhost:3000

### 构建生产版本

```bash
npm run build
```

构建产物在 `dist/` 目录

## 📋 功能

- ✅ 用户登录（API Key认证）
- ✅ 图像生成界面
- ✅ 配额管理
- ✅ 响应式设计
- ⏳ 视频生成（开发中）
- ⏳ 任务历史（开发中）

## 🔧 配置

### API地址

编辑 `.env` 文件：

```
VITE_API_BASE_URL=http://localhost:8000
```

### 默认API Key

- 测试Key: `test-key-123` (免费版)
- 演示Key: `demo-key-456` (付费版)

## 📦 技术栈

- Vue 3 (Composition API)
- TypeScript
- Element Plus (UI组件库)
- Vue Router (路由)
- Pinia (状态管理)
- Axios (HTTP客户端)
- Vite (构建工具)

