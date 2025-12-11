import { defineStore } from 'pinia'
import { ref } from 'vue'
import api from '../utils/api'

export const useUserStore = defineStore('user', () => {
  const user = ref<any>(null)
  const apiKey = ref<string | null>(localStorage.getItem('api_key'))

  const login = async (key: string) => {
    try {
      // 验证API Key（通过查询配额）
      const quota = await api.getQuota(key)
      user.value = {
        user_id: quota.user_id,
        tier: quota.tier,
        api_key: key,
      }
      apiKey.value = key
      localStorage.setItem('api_key', key)
      return user.value
    } catch (error: any) {
      throw new Error(error.message || '登录失败，请检查API Key')
    }
  }

  const logout = () => {
    user.value = null
    apiKey.value = null
    localStorage.removeItem('api_key')
  }

  // 初始化时尝试登录
  if (apiKey.value) {
    login(apiKey.value).catch(() => {
      logout()
    })
  }

  return {
    user,
    apiKey,
    login,
    logout,
  }
})

