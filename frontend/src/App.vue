<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { healthCheck, uploadDocument } from './services/api'

const backendStatus = ref<string>('checking...')

const selectedFile = ref<File | null>(null)
const uploadResult = ref<any>(null)
const uploading = ref<boolean>(false)
const errorMessage = ref<string>('')

onMounted(async () => {
  try {
    const data = await healthCheck()
    backendStatus.value = data.status
  } catch (error) {
    backendStatus.value = 'error'
  }
})

const handleFileChange = (event: Event) => {
  const input = event.target as HTMLInputElement

  if (!input.files || input.files.length === 0) {
    selectedFile.value = null
    return
  }

  selectedFile.value = input.files[0]
  uploadResult.value = null
  errorMessage.value = ''
}

const handleUpload = async () => {
  if (!selectedFile.value) {
    errorMessage.value = 'Please select a PDF file first.'
    return
  }

  uploading.value = true
  uploadResult.value = null
  errorMessage.value = ''

  try {
    const result = await uploadDocument(selectedFile.value)
    uploadResult.value = result
  } catch (error: any) {
    errorMessage.value =
      error.response?.data?.detail || 'Upload failed. Please try again.'
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <main class="app">
    <header>
      <h1>DocuMind</h1>
      <p>RAG-based Enterprise Document Q&A System</p>
    </header>

    <section class="status-card">
      <h2>Backend Status</h2>
      <p>{{ backendStatus }}</p>
    </section>

    <section class="upload-card">
      <h2>Upload PDF Document</h2>

      <input type="file" accept="application/pdf" @change="handleFileChange" />

      <div v-if="selectedFile" class="file-info">
        <p><strong>Selected file:</strong> {{ selectedFile.name }}</p>
        <p><strong>Size:</strong> {{ selectedFile.size }} bytes</p>
      </div>

      <button :disabled="!selectedFile || uploading" @click="handleUpload">
        {{ uploading ? 'Uploading...' : 'Upload' }}
      </button>

      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

      <div v-if="uploadResult" class="result">
        <h3>Upload Result</h3>
        <pre>{{ uploadResult }}</pre>
      </div>
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

.status-card,
.upload-card {
  margin-top: 24px;
  padding: 16px;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.file-info {
  margin-top: 16px;
  padding: 12px;
  background: #f7f7f7;
  border-radius: 6px;
}

button {
  margin-top: 16px;
  padding: 8px 16px;
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.error {
  margin-top: 16px;
  color: #c00;
}

.result {
  margin-top: 16px;
  padding: 12px;
  background: #f7f7f7;
  border-radius: 6px;
}

pre {
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
