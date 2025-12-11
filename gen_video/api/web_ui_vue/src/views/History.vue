<template>
  <div class="history">
    <el-page-header @back="$router.push('/')" title="返回">
      <template #content>
        <span class="page-title">📋 任务历史</span>
      </template>
    </el-page-header>

    <el-alert
      title="功能开发中"
      type="info"
      :closable="false"
      style="margin-top: 20px"
    >
      <template #default>
        <p>任务历史功能正在开发中，敬请期待！</p>
        <p>即将推出：查看历史任务、下载生成结果、任务管理等功能</p>
      </template>
    </el-alert>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.history {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}

.page-title {
  font-size: 20px;
  font-weight: bold;
}
</style>

