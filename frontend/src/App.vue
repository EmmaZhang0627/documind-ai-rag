<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { healthCheck } from './service/app'

const backendStatus = ref<string>('checking...')

onMounted(async () => {
  try {
    const data = await healthCheck()
    backendStatus.value = data.status
  } catch (error) {
    backendStatus.value = 'error'
  }
})
</script>

<template>
  <main class="app">
    <h1>DocuMind</h1>
    <p>RAG-based Enterprise Document Q&A System</p>

    <section class="status-card">
      <h2>Backend Status</h2>
      <p>{{ backendStatus }}</p>
    </section>
  </main>
</template>

<style scoped>
.app {
  max-width: 960px;
  margin: 80px auto;
  padding: 24px;
  font-family: Arial, sans-serif;
}

.status-card {
  margin-top: 24px;
  padding: 16px;
  border: 1px solid #ddd;
  border-radius: 8px;
}
</style>
