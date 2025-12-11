<template>
  <div class="image-gen">
    <el-page-header @back="$router.push('/')" title="返回">
      <template #content>
        <span class="page-title">🎨 图像生成</span>
      </template>
    </el-page-header>

    <el-row :gutter="20" style="margin-top: 20px">
      <!-- 左侧：参数设置 -->
      <el-col :xs="24" :md="10">
        <el-card>
          <template #header>
            <span>生成参数</span>
          </template>

          <el-form :model="form" label-width="120px" label-position="top">
            <el-form-item label="提示词" required>
              <el-input
                v-model="form.prompt"
                type="textarea"
                :rows="4"
                placeholder="详细描述您想要生成的图像，例如：一个美丽的风景，山峦起伏，云雾缭绕，阳光透过云层"
                maxlength="500"
                show-word-limit
              />
            </el-form-item>

            <el-form-item label="负面提示词（可选）">
              <el-input
                v-model="form.negativePrompt"
                type="textarea"
                :rows="2"
                placeholder="描述不想要的内容，例如：模糊，低质量，变形"
                maxlength="500"
              />
            </el-form-item>

            <el-row :gutter="12">
              <el-col :span="12">
                <el-form-item label="宽度">
                  <el-slider
                    v-model="form.width"
                    :min="512"
                    :max="2048"
                    :step="64"
                    show-input
                    :format-tooltip="(val) => `${val}px`"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="高度">
                  <el-slider
                    v-model="form.height"
                    :min="512"
                    :max="2048"
                    :step="64"
                    show-input
                    :format-tooltip="(val) => `${val}px`"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-row :gutter="12">
              <el-col :span="12">
                <el-form-item label="推理步数">
                  <el-slider
                    v-model="form.numSteps"
                    :min="10"
                    :max="100"
                    :step="5"
                    show-input
                  />
                  <div class="form-tip">更多步数=更好质量，但更慢</div>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="引导尺度">
                  <el-slider
                    v-model="form.guidanceScale"
                    :min="1"
                    :max="20"
                    :step="0.5"
                    show-input
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="随机种子（可选）">
              <el-input-number
                v-model="form.seed"
                :min="0"
                :max="2147483647"
                :precision="0"
                placeholder="留空则随机生成"
                style="width: 100%"
              />
              <div class="form-tip">相同种子会生成相同图像</div>
            </el-form-item>

            <el-form-item label="参考图像（可选）">
              <el-upload
                v-model:file-list="referenceImageList"
                :auto-upload="false"
                :limit="1"
                :on-change="handleReferenceImageChange"
                :on-remove="handleReferenceImageRemove"
                accept="image/*"
                list-type="picture"
              >
                <el-button type="primary">
                  <el-icon><Upload /></el-icon>
                  选择参考图像
                </el-button>
                <template #tip>
                  <div class="el-upload__tip">
                    支持场景参考或面部参考，上传后会在生成时使用
                  </div>
                </template>
              </el-upload>
              <el-radio-group v-model="form.referenceImageType" style="margin-top: 8px">
                <el-radio label="scene">场景参考</el-radio>
                <el-radio label="face">面部参考</el-radio>
              </el-radio-group>
              <div class="form-tip">
                场景参考：用于控制整体风格和构图<br/>
                面部参考：用于控制角色面部特征（需要InstantID）
              </div>
            </el-form-item>

            <el-divider content-position="left">LoRA设置（可选）</el-divider>
            
            <el-form-item label="角色LoRA">
              <el-select
                v-model="form.characterLora"
                placeholder="选择角色LoRA（留空表示不使用）"
                clearable
                filterable
                style="width: 100%"
              >
                <el-option
                  v-for="lora in availableCharacterLoras"
                  :key="lora.name"
                  :label="`${lora.name} - ${lora.description}`"
                  :value="lora.name"
                />
              </el-select>
              <div class="form-tip">
                选择角色LoRA以固定人物形象（如：host_person_v2 用于科普主持人）
              </div>
            </el-form-item>

            <el-form-item label="风格LoRA">
              <el-select
                v-model="form.styleLora"
                placeholder="选择风格LoRA（留空表示不使用）"
                clearable
                filterable
                style="width: 100%"
              >
                <el-option
                  v-for="lora in availableStyleLoras"
                  :key="lora.name"
                  :label="`${lora.name} - ${lora.description}`"
                  :value="lora.name"
                />
              </el-select>
              <div class="form-tip">
                选择风格LoRA以控制图像风格（如：anime_style 用于动漫风格）
              </div>
            </el-form-item>

            <el-form-item>
              <el-button
                type="primary"
                size="large"
                :loading="generating"
                @click="handleGenerate"
                style="width: 100%"
              >
                <el-icon v-if="!generating"><MagicStick /></el-icon>
                {{ generating ? '生成中...' : '生成图像' }}
              </el-button>
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
            <el-empty description="生成的图像将显示在这里" />
          </div>

          <div v-if="generating" class="generating-state">
            <el-skeleton :rows="8" animated />
            <div class="generating-tip">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>正在生成，请稍候...（通常需要30-60秒）</span>
            </div>
            <div v-if="errorMessage" class="error-message">
              <el-alert :title="errorMessage" type="error" :closable="false" />
            </div>
          </div>

          <div v-if="result" class="result-state">
            <el-image
              :src="result.imageUrl"
              fit="contain"
              :preview-src-list="[result.imageUrl]"
              style="width: 100%; max-height: 600px"
            />
            <div class="result-info">
              <el-descriptions :column="2" border size="small">
                <el-descriptions-item label="任务ID">{{ result.taskId }}</el-descriptions-item>
                <el-descriptions-item label="文件大小">
                  {{ (result.fileSize / 1024).toFixed(1) }} KB
                </el-descriptions-item>
                <el-descriptions-item label="分辨率">
                  {{ result.width }} × {{ result.height }}
                </el-descriptions-item>
                <el-descriptions-item label="剩余配额">
                  图像 {{ result.quota?.images || 0 }} 张
                </el-descriptions-item>
              </el-descriptions>
              <div style="margin-top: 16px">
                <el-button type="primary" @click="downloadImage">
                  <el-icon><Download /></el-icon>
                  下载图像
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
import { ref, reactive, onMounted } from 'vue'
import { useUserStore } from '../stores/user'
import api from '../utils/api'
import { MagicStick, Loading, Download, Upload } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { checkApiHealth, debug } from '../utils/debug'
import type { UploadFile, UploadFiles } from 'element-plus'

const userStore = useUserStore()

// 可用的 LoRA 列表
const availableCharacterLoras = ref<Array<{name: string, description: string}>>([])
const availableStyleLoras = ref<Array<{name: string, description: string}>>([])

// 加载可用的 LoRA 列表
const loadLoras = async () => {
  try {
    const response = await api.getLoras()
    if (response) {
      availableCharacterLoras.value = response.character || []
      availableStyleLoras.value = response.style || []
      console.log('已加载 LoRA 列表:', {
        character: availableCharacterLoras.value,
        style: availableStyleLoras.value
      })
    }
  } catch (error) {
    console.warn('无法加载 LoRA 列表:', error)
    // 如果 API 不可用，使用默认值
    availableCharacterLoras.value = [
      { name: 'host_person_v2', description: '主持人/角色 LoRA' },
      { name: 'hanli', description: '角色 LoRA' }
    ]
    availableStyleLoras.value = [
      { name: 'anime_style', description: '风格 LoRA' }
    ]
  }
}

// 检查API服务
onMounted(async () => {
  const isHealthy = await checkApiHealth()
  if (!isHealthy) {
    ElMessage.warning('API服务未运行，请先启动API服务（python gen_video/api/mvp_main.py）')
  }
  // 加载 LoRA 列表
  await loadLoras()
})

const generating = ref(false)
const result = ref<any>(null)
const errorMessage = ref<string>('')
const referenceImageList = ref<UploadFile[]>([])
const referenceImageFile = ref<File | null>(null)

const form = reactive({
  prompt: '',
  negativePrompt: '',
  width: 1024,
  height: 1024,
  numSteps: 40,
  guidanceScale: 7.5,
  seed: null as number | null,
  referenceImageType: 'scene' as 'scene' | 'face',
  characterLora: null as string | null,  // 默认不使用角色LoRA
  styleLora: null as string | null,      // 默认不使用风格LoRA
})

const handleReferenceImageChange = (file: UploadFile) => {
  if (file.raw) {
    referenceImageFile.value = file.raw as File
    console.log('参考图像已选择:', file.name)
  }
}

const handleReferenceImageRemove = () => {
  referenceImageFile.value = null
  console.log('参考图像已移除')
}

const handleGenerate = async () => {
  console.log('开始生成图像...')
  
  if (!form.prompt.trim()) {
    ElMessage.warning('请输入提示词')
    return
  }

  if (!userStore.user) {
    ElMessage.warning('请先登录')
    return
  }

  console.log('用户已登录:', userStore.user)
  
  // 自动检测任务类型（用于日志和调试）
  const promptLower = form.prompt.toLowerCase()
  let detectedTask = null
  if (promptLower.includes('主持人') || promptLower.includes('host') || promptLower.includes('presenter')) {
    detectedTask = 'host_face'
  } else if (promptLower.includes('实验室') || promptLower.includes('lab') || promptLower.includes('医学') || promptLower.includes('medical')) {
    detectedTask = 'lab_scene'
  } else if (promptLower.includes('量子') || promptLower.includes('粒子') || promptLower.includes('太空') || promptLower.includes('quantum') || promptLower.includes('space')) {
    detectedTask = 'science_background'
  } else if (promptLower.includes('中国') || promptLower.includes('官方') || promptLower.includes('chinese') || promptLower.includes('official')) {
    detectedTask = 'official_style'
  }
  
  console.log('生成参数:', {
    prompt: form.prompt,
    width: form.width,
    height: form.height,
    numSteps: form.numSteps,
    referenceImage: referenceImageFile.value?.name,
    referenceImageType: form.referenceImageType,
    characterLora: form.characterLora || 'None（不使用）',
    styleLora: form.styleLora || 'None（不使用）',
    useModelManager: true,
    task: detectedTask || 'auto（自动检测）',
  })

  generating.value = true
  result.value = null
  errorMessage.value = ''

  try {
    console.log('调用API...')
    
    // 构建FormData（支持文件上传）
    const formData = new FormData()
    formData.append('prompt', form.prompt)
    if (form.negativePrompt) {
      formData.append('negative_prompt', form.negativePrompt)
    }
    formData.append('width', form.width.toString())
    formData.append('height', form.height.toString())
    formData.append('num_inference_steps', form.numSteps.toString())
    formData.append('guidance_scale', form.guidanceScale.toString())
    if (form.seed) {
      formData.append('seed', form.seed.toString())
    }
    if (referenceImageFile.value) {
      formData.append('reference_image', referenceImageFile.value)
      formData.append('reference_image_type', form.referenceImageType)
    }
    
    // 添加LoRA参数
    // 注意：
    // - 如果为null/undefined/空字符串，不添加到FormData中，后端会收到None（不使用LoRA，仅使用参考图）
    // - 如果有值，添加到FormData中，后端会使用指定的LoRA
    // 注意：空字符串会被trim后检查，如果为空则不添加
    const charLora = form.characterLora?.trim() || null
    const styleLora = form.styleLora?.trim() || null
    
    if (charLora && charLora !== '') {
      formData.append('character_lora', charLora)
    }
    if (styleLora && styleLora !== '') {
      formData.append('style_lora', styleLora)
    }
    
    // 添加 ModelManager 参数（默认启用，确保使用最优模型）
    formData.append('use_model_manager', 'true')
    
    // 自动检测任务类型（如果是主持人相关，明确指定 task）
    if (detectedTask) {
      formData.append('task', detectedTask)
      console.log(`✅ 检测到任务类型: ${detectedTask}`)
    } else {
      console.log('ℹ️  未检测到特定任务类型，使用后端自动检测')
    }
    
    const data = await api.generateImageWithFile(formData)

    console.log('API响应:', data)

    if (data.status === 'completed') {
      const imageUrl = data.image_url 
        ? `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}${data.image_url}`
        : data.image_path
      
      result.value = {
        taskId: data.task_id,
        imageUrl: imageUrl,
        fileSize: data.file_size,
        width: data.width,
        height: data.height,
        quota: data.quota_remaining,
      }
      console.log('生成成功，图像URL:', imageUrl)
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

const downloadImage = () => {
  if (result.value?.imageUrl) {
    const link = document.createElement('a')
    link.href = result.value.imageUrl
    link.download = `image-${result.value.taskId}.png`
    link.click()
  }
}
</script>

<style scoped>
.image-gen {
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
</style>

