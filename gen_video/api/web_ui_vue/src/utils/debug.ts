// 调试工具

export const debug = {
  log: (...args: any[]) => {
    if (import.meta.env.DEV) {
      console.log('[DEBUG]', ...args)
    }
  },
  error: (...args: any[]) => {
    console.error('[ERROR]', ...args)
  },
  warn: (...args: any[]) => {
    console.warn('[WARN]', ...args)
  },
}

// 检查API服务是否可用
export const checkApiHealth = async (baseUrl: string = 'http://localhost:8000') => {
  try {
    const response = await fetch(`${baseUrl}/api/v1/health`)
    if (response.ok) {
      const data = await response.json()
      debug.log('API服务正常:', data)
      return true
    } else {
      debug.warn('API服务响应异常:', response.status)
      return false
    }
  } catch (error) {
    debug.error('API服务不可用:', error)
    return false
  }
}
