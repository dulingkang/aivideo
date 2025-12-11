import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import ImageGen from '../views/ImageGen.vue'
import VideoGen from '../views/VideoGen.vue'
import History from '../views/History.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
  },
  {
    path: '/image',
    name: 'ImageGen',
    component: ImageGen,
  },
  {
    path: '/video',
    name: 'VideoGen',
    component: VideoGen,
  },
  {
    path: '/history',
    name: 'History',
    component: History,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫
router.beforeEach((to, from, next) => {
  // 可以在这里添加认证检查
  next()
})

export default router

