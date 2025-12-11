<template>
  <div class="video-gen">
    <el-page-header @back="$router.push('/')" title="返回">
      <template #content>
        <span class="page-title">🎬 视频生成</span>
      </template>
    </el-page-header>

    <el-row :gutter="20" style="margin-top: 20px">
      <!-- 左侧：参数设置 -->
      <el-col :xs="24" :md="10">
        <el-card>
          <template #header>
            <span>视频参数</span>
          </template>

          <el-form :model="videoForm" label-width="120px" label-position="top">
            <el-row :gutter="12">
              <el-col :span="12">
                <el-form-item label="视频宽度">
                  <el-slider
                    v-model="videoForm.width"
                    :min="512"
                    :max="1920"
                    :step="64"
                    show-input
                    :format-tooltip="(val) => `${val}px`"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="视频高度">
                  <el-slider
                    v-model="videoForm.height"
                    :min="512"
                    :max="1080"
                    :step="64"
                    show-input
                    :format-tooltip="(val) => `${val}px`"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="帧率 (FPS)">
              <el-slider
                v-model="videoForm.fps"
                :min="15"
                :max="30"
                :step="1"
                show-input
              />
              <div class="form-tip">建议使用24fps，平衡质量和速度</div>
            </el-form-item>

            <el-divider content-position="left">场景列表</el-divider>

            <el-form-item>
              <div class="scene-list">
                <div
                  v-for="(scene, index) in scenes"
                  :key="index"
                  class="scene-item"
                >
                  <el-card shadow="hover">
                    <template #header>
                      <div class="scene-header">
                        <span>场景 {{ index + 1 }}</span>
                        <el-button
                          type="danger"
                          size="small"
                          text
                          @click="removeScene(index)"
                          :disabled="scenes.length <= 1"
                        >
                          <el-icon><Delete /></el-icon>
                        </el-button>
                      </div>
                    </template>

                    <el-form-item label="提示词" required>
                      <el-input
                        v-model="scene.prompt"
                        type="textarea"
                        :rows="3"
                        placeholder="描述这个场景，例如：一个美丽的风景，山峦起伏，云雾缭绕"
                        maxlength="500"
                        show-word-limit
                      />
                    </el-form-item>

                    <el-form-item label="时长（秒）">
                      <el-slider
                        v-model="scene.duration"
                        :min="1"
                        :max="30"
                        :step="0.5"
                        show-input
                        :format-tooltip="(val) => `${val}秒`"
                      />
                      <div class="form-tip">每个场景的播放时长</div>
                    </el-form-item>

                    <el-form-item label="预生成图像路径（可选）">
                      <el-input
                        v-model="scene.image_path"
                        placeholder="如果已有图像，可以输入路径"
                        clearable
                      />
                      <div class="form-tip">
                        如果为空，将根据提示词生成图像
                      </div>
                    </el-form-item>
                  </el-card>
                </div>

                <el-button
                  type="dashed"
                  style="width: 100%"
                  @click="addScene"
                  :disabled="scenes.length >= 10"
                >
                  <el-icon><Plus /></el-icon>
                  添加场景
                </el-button>
                <div class="form-tip" style="text-align: center">
                  最多10个场景，总时长: {{ totalDuration.toFixed(1) }}秒
                </div>
              </div>
            </el-form-item>

            <el-form-item>
              <el-button
                type="primary"
                size="large"
                :loading="generating"
                @click="handleGenerate"
                style="width: 100%"
                :disabled="!canGenerate"
              >
                <el-icon v-if="!generating"><VideoPlay /></el-icon>
                {{ generating ? '生成中...' : '生成视频' }}
              </el-button>
              <div class="form-tip" style="text-align: center; margin-top: 8px">
                预计耗时: {{ estimatedTime }}秒（约{{ Math.ceil(estimatedTime / 60) }}分钟）
              </div>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <!-- 右侧：结果展示 -->
      <el-col :xs="24" :md="14">
        <el-card>
          <template #header>
            <span>生成结果</span>
          </template>

          <div v-if="!result && !generating" class="empty-state">
            <el-empty description="生成的视频将显示在这里" />
          </div>

          <div v-if="generating" class="generating-state">
            <el-skeleton :rows="8" animated />
            <div class="generating-tip">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>正在生成视频，请稍候...（可能需要几分钟到十几分钟）</span>
            </div>
            <div v-if="errorMessage" class="error-message">
              <el-alert :title="errorMessage" type="error" :closable="false" />
            </div>
          </div>

          <div v-if="result" class="result-state">
            <video
              v-if="result.videoUrl"
              :src="result.videoUrl"
              controls
              style="width: 100%; max-height: 600px"
            />
            <div class="result-info">
              <el-descriptions :column="2" border size="small">
                <el-descriptions-item label="任务ID">{{ result.taskId }}</el-descriptions-item>
                <el-descriptions-item label="文件大小">
                  {{ (result.fileSize / 1024 / 1024).toFixed(2) }} MB
                </el-descriptions-item>
                <el-descriptions-item label="分辨率">
                  {{ result.width }} × {{ result.height }}
                </el-descriptions-item>
                <el-descriptions-item label="时长">
                  {{ result.duration.toFixed(1) }}秒
                </el-descriptions-item>
                <el-descriptions-item label="剩余配额" :span="2">
                  视频 {{ result.quota?.videos || 0 }} 个
                </el-descriptions-item>
              </el-descriptions>
              <div style="margin-top: 16px">
                <el-button type="primary" @click="downloadVideo">
                  <el-icon><Download /></el-icon>
                  下载视频
                </el-button>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useUserStore } from '../stores/user'
import api from '../utils/api'
import { VideoPlay, Loading, Download, Delete, Plus } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { checkApiHealth } from '../utils/debug'

const userStore = useUserStore()

interface Scene {
  prompt: string
  duration: number
  image_path?: string | null
}

const generating = ref(false)
const result = ref<any>(null)
const errorMessage = ref<string>('')

const videoForm = reactive({
  width: 1280,
  height: 768,
  fps: 24,
})

const scenes = ref<Scene[]>([
  {
    prompt: '',
    duration: 5.0,
    image_path: null,
  },
])

// 检查API服务
onMounted(async () => {
  const isHealthy = await checkApiHealth()
  if (!isHealthy) {
    ElMessage.warning('API服务未运行，请先启动API服务（python gen_video/api/mvp_main.py）')
  }
})

const totalDuration = computed(() => {
  return scenes.value.reduce((sum, scene) => sum + scene.duration, 0)
})

const estimatedTime = computed(() => {
  // 粗略估算：每个场景图像生成30秒 + 视频合成10秒/场景
  const imageTime = scenes.value.length * 30
  const videoTime = totalDuration.value * 10
  return Math.ceil(imageTime + videoTime)
})

const canGenerate = computed(() => {
  return (
    scenes.value.length > 0 &&
    scenes.value.every(scene => scene.prompt.trim().length > 0)
  )
})

const addScene = () => {
  if (scenes.value.length < 10) {
    scenes.value.push({
      prompt: '',
      duration: 5.0,
      image_path: null,
    })
  } else {
    ElMessage.warning('最多只能添加10个场景')
  }
}

const removeScene = (index: number) => {
  if (scenes.value.length > 1) {
    scenes.value.splice(index, 1)
  } else {
    ElMessage.warning('至少需要1个场景')
  }
}

const handleGenerate = async () => {
  console.log('开始生成视频...')

  if (!canGenerate.value) {
    ElMessage.warning('请填写所有场景的提示词')
    return
  }

  if (!userStore.user) {
    ElMessage.warning('请先登录')
    return
  }

  console.log('用户已登录:', userStore.user)
  console.log('生成参数:', {
    scenes: scenes.value,
    width: videoForm.width,
    height: videoForm.height,
    fps: videoForm.fps,
  })

  generating.value = true
  result.value = null
  errorMessage.value = ''

  try {
    console.log('调用API...')

    const requestData = {
      scenes: scenes.value.map(scene => ({
        prompt: scene.prompt.trim(),
        duration: scene.duration,
        image_path: scene.image_path || null,
      })),
      fps: videoForm.fps,
      width: videoForm.width,
      height: videoForm.height,
    }

    console.log('请求数据:', requestData)

    const data = await api.generateVideo(requestData)

    console.log('API响应:', data)

    if (data.status === 'completed') {
      const videoUrl = data.video_url 
        ? `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${data.video_url}`
        : data.video_path
      
      result.value = {
        taskId: data.task_id,
        videoUrl: videoUrl,
        fileSize: data.file_size,
        width: videoForm.width,
        height: videoForm.height,
        duration: data.duration || totalDuration.value,
        quota: data.quota_remaining,
      }
      console.log('生成成功，视频URL:', videoUrl)
      ElMessage.success('生成成功！')
    } else {
      console.warn('生成状态:', data.status)
      ElMessage.warning(`状态: ${data.status}`)
    }
  } catch (error: any) {
    console.error('生成失败:', error)
    const errorMsg = error?.message || error?.response?.data?.detail || '生成失败，请检查控制台'
    errorMessage.value = errorMsg
    ElMessage.error(errorMsg)
    console.error('完整错误信息:', error)
    
    // 显示详细错误信息（开发环境）
    if (import.meta.env.DEV) {
      console.error('错误详情:', {
        message: error?.message,
        response: error?.response,
        stack: error?.stack,
      })
    }
  } finally {
    generating.value = false
  }
}

const downloadVideo = () => {
  if (result.value?.videoUrl) {
    const link = document.createElement('a')
    link.href = result.value.videoUrl
    link.download = `video-${result.value.taskId}.mp4`
    link.click()
  }
}
</script>

<style scoped>
.video-gen {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.page-title {
  font-size: 20px;
  font-weight: bold;
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.empty-state,
.generating-state,
.result-state {
  min-height: 400px;
}

.generating-tip {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 20px;
  color: #409eff;
}

.error-message {
  margin-top: 20px;
}

.result-info {
  margin-top: 20px;
}

.scene-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.scene-item {
  width: 100%;
}

.scene-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-state video {
  border-radius: 4px;
  background: #000;
}
</style>
