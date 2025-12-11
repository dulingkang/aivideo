<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="welcome-card">
          <template #header>
            <div class="card-header">
              <span>欢迎使用AI视频生成平台</span>
            </div>
          </template>
          <div class="welcome-content">
            <h2>🎬 专业的AI图像和视频生成服务</h2>
            <p>使用最新的AI技术，快速生成高质量的图像和视频内容</p>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :xs="24" :sm="12" :md="8">
        <el-card class="feature-card" @click="$router.push('/image')">
          <div class="feature-icon">
            <el-icon :size="48"><Picture /></el-icon>
          </div>
          <h3>图像生成</h3>
          <p>使用AI生成高质量图像，支持多种风格和参数调整</p>
          <el-button type="primary" style="margin-top: 16px">开始生成</el-button>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="12" :md="8">
        <el-card class="feature-card" @click="$router.push('/video')">
          <div class="feature-icon">
            <el-icon :size="48"><VideoCamera /></el-icon>
          </div>
          <h3>视频生成</h3>
          <p>基于图像生成流畅的视频，支持多场景和自定义参数</p>
          <el-button type="primary" style="margin-top: 16px">开始生成</el-button>
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="12" :md="8">
        <el-card class="feature-card" @click="$router.push('/history')">
          <div class="feature-icon">
            <el-icon :size="48"><Document /></el-icon>
          </div>
          <h3>任务历史</h3>
          <p>查看历史生成任务，管理您的作品</p>
          <el-button type="primary" style="margin-top: 16px">查看历史</el-button>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="24">
        <el-card>
          <template #header>
            <span>快速开始</span>
          </template>
          <el-steps :active="0" finish-status="success">
            <el-step title="登录" description="使用API Key登录系统" />
            <el-step title="选择功能" description="选择图像生成或视频生成" />
            <el-step title="输入提示词" description="描述您想要生成的内容" />
            <el-step title="生成" description="等待AI生成，通常需要30-60秒" />
            <el-step title="下载" description="下载生成的结果" />
          </el-steps>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { Picture, VideoCamera, Document } from '@element-plus/icons-vue'
</script>

<style scoped>
.home {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.welcome-card {
  text-align: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.welcome-card :deep(.el-card__body) {
  color: white;
}

.welcome-content h2 {
  margin: 0 0 16px 0;
  font-size: 32px;
}

.welcome-content p {
  margin: 0;
  font-size: 16px;
  opacity: 0.9;
}

.feature-card {
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  height: 100%;
}

.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.feature-icon {
  margin-bottom: 16px;
  color: #409eff;
}

.feature-card h3 {
  margin: 0 0 12px 0;
  font-size: 20px;
}

.feature-card p {
  margin: 0;
  color: #909399;
  line-height: 1.6;
}
</style>

