import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5分钟超时（图像生成可能需要较长时间）
})

// 请求拦截器：添加API Key
apiClient.interceptors.request.use((config) => {
  // 从localStorage读取apiKey（避免循环依赖）
  const apiKey = localStorage.getItem('api_key') || 'test-key-123'
  config.headers['X-API-Key'] = apiKey
  return config
})

// 响应拦截器：处理错误
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response) {
      const message = error.response.data?.detail || error.response.data?.error || '请求失败'
      return Promise.reject(new Error(message))
    }
    return Promise.reject(error)
  }
)

const api = {
  // 健康检查
  health: () => apiClient.get('/api/v1/health'),

  // 查询配额
  getQuota: (apiKey?: string) => {
    const headers = apiKey ? { 'X-API-Key': apiKey } : {}
    return apiClient.get('/api/v1/quota', { headers })
  },

  // 生成图像（JSON方式，不支持文件上传）
  generateImage: (data: any) => apiClient.post('/api/v1/images/generate', data),

  // 生成图像（FormData方式，支持文件上传）
  generateImageWithFile: (formData: FormData) => {
    // 注意：这里不能直接使用useUserStore，因为这是工具函数
    // 需要在调用时传入apiKey，或者从localStorage读取
    const apiKey = localStorage.getItem('api_key') || 'test-key-123'
    return axios.post(`${API_BASE_URL}/api/v1/images/generate`, formData, {
      headers: {
        'X-API-Key': apiKey,
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5分钟超时
    }).then(response => response.data)
  },

  // 生成视频
  generateVideo: (data: any) => apiClient.post('/api/v1/videos/generate', data),

  // 查询任务状态
  getTaskStatus: (taskId: string) => apiClient.get(`/api/v1/tasks/${taskId}`),

  // 获取可用的 LoRA 列表
  getLoras: () => apiClient.get('/api/v1/loras'),
}

export default api

