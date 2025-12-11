<template>
  <div id="app">
    <el-container>
      <!-- 头部导航 -->
      <el-header class="header">
        <div class="header-content">
          <div class="logo">
            <el-icon :size="32"><VideoCamera /></el-icon>
            <span class="title">AI视频生成平台</span>
          </div>
          <div class="nav-right">
            <el-button v-if="!user" @click="showLogin = true" type="primary">登录</el-button>
            <el-dropdown v-else>
              <span class="user-info">
                <el-icon><User /></el-icon>
                {{ user.user_id }}
                <el-icon class="el-icon--right"><arrow-down /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="showQuota = true">配额信息</el-dropdown-item>
                  <el-dropdown-item @click="logout">退出登录</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
      </el-header>

      <!-- 主内容 -->
      <el-main>
        <router-view />
      </el-main>

      <!-- 登录对话框 -->
      <el-dialog v-model="showLogin" title="登录" width="400px">
        <el-form :model="loginForm" label-width="80px">
          <el-form-item label="API Key">
            <el-input
              v-model="loginForm.apiKey"
              type="password"
              placeholder="请输入API Key"
              show-password
            />
            <div class="form-tip">
              <el-text type="info" size="small">
                测试Key: test-key-123 (免费版)<br/>
                演示Key: demo-key-456 (付费版)
              </el-text>
            </div>
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="showLogin = false">取消</el-button>
          <el-button type="primary" @click="handleLogin">登录</el-button>
        </template>
      </el-dialog>

      <!-- 配额信息对话框 -->
      <el-dialog v-model="showQuota" title="配额信息" width="500px">
        <el-descriptions v-if="quotaInfo" :column="1" border>
          <el-descriptions-item label="用户ID">{{ quotaInfo.user_id }}</el-descriptions-item>
          <el-descriptions-item label="套餐">{{ quotaInfo.tier === 'free' ? '免费版' : '付费版' }}</el-descriptions-item>
          <el-descriptions-item label="图像配额">
            {{ quotaInfo.remaining.images }} / {{ quotaInfo.limits.images }} 张/天
          </el-descriptions-item>
          <el-descriptions-item label="视频配额">
            {{ quotaInfo.remaining.videos }} / {{ quotaInfo.limits.videos }} 个/天
          </el-descriptions-item>
          <el-descriptions-item label="重置时间">{{ quotaInfo.reset_at }}</el-descriptions-item>
        </el-descriptions>
        <el-empty v-else description="加载中..." />
      </el-dialog>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from './stores/user'
import { VideoCamera, User, ArrowDown } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import api from './utils/api'

const router = useRouter()
const userStore = useUserStore()

const showLogin = ref(false)
const showQuota = ref(false)
const quotaInfo = ref<any>(null)

const loginForm = ref({
  apiKey: 'test-key-123'
})

const user = computed(() => userStore.user)

const handleLogin = async () => {
  if (!loginForm.value.apiKey) {
    ElMessage.warning('请输入API Key')
    return
  }

  try {
    await userStore.login(loginForm.value.apiKey)
    showLogin.value = false
    ElMessage.success('登录成功')
    router.push('/')
  } catch (error: any) {
    ElMessage.error(error.message || '登录失败')
  }
}

const logout = () => {
  userStore.logout()
  ElMessage.success('已退出登录')
  router.push('/')
}

const loadQuota = async () => {
  if (!userStore.user) return
  
  try {
    const data = await api.getQuota()
    quotaInfo.value = {
      user_id: data.user_id,
      tier: data.tier,
      limits: data.limits,
      remaining: data.remaining,
      reset_at: data.reset_at,
    }
  } catch (error) {
    console.error('加载配额失败:', error)
  }
}

watch(showQuota, (val) => {
  if (val && userStore.user) {
    loadQuota()
  }
})

onMounted(() => {
  // 检查是否已登录
  if (userStore.user) {
    loadQuota()
  }
})
</script>

<style scoped>
#app {
  min-height: 100vh;
  background: #f5f7fa;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 0;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
  padding: 0 20px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 20px;
  font-weight: bold;
}

.nav-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: white;
}

.form-tip {
  margin-top: 8px;
  line-height: 1.6;
}
</style>

