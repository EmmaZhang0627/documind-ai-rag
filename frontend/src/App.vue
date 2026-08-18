<script setup lang="ts">
import { onMounted, ref } from 'vue'
import CitationList from './components/CitationList.vue'
import PdfEvidenceViewer from './components/PdfEvidenceViewer.vue'
import {
  askQuestion,
  healthCheck,
  uploadDocument,
  type ChatResponse,
  type SourceMetadata,
} from './services/api'

const backendStatus = ref('checking...')
const selectedFile = ref<File | null>(null)
const uploadResult = ref<Record<string, unknown> | null>(null)
const uploading = ref(false)
const question = ref('')
const chatResult = ref<ChatResponse | null>(null)
const asking = ref(false)
const selectedCitation = ref<SourceMetadata | null>(null)
const errorMessage = ref('')

onMounted(async () => {
  try {
    backendStatus.value = (await healthCheck()).status
  } catch {
    backendStatus.value = 'error'
  }
})

const handleFileChange = (event: Event) => {
  const input = event.target as HTMLInputElement
  selectedFile.value = input.files?.[0] ?? null
  uploadResult.value = null
  errorMessage.value = ''
}

const handleUpload = async () => {
  if (!selectedFile.value) {
    errorMessage.value = 'Please select a PDF file first.'
    return
  }

  uploading.value = true
  errorMessage.value = ''
  try {
    uploadResult.value = await uploadDocument(selectedFile.value)
  } catch (error: any) {
    errorMessage.value = error.response?.data?.detail || 'Upload failed. Please try again.'
  } finally {
    uploading.value = false
  }
}

const handleAsk = async () => {
  const normalizedQuestion = question.value.trim()
  if (!normalizedQuestion) return

  asking.value = true
  errorMessage.value = ''
  selectedCitation.value = null
  try {
    chatResult.value = await askQuestion(normalizedQuestion)
  } catch (error: any) {
    errorMessage.value = error.response?.data?.detail || 'Chat request failed.'
  } finally {
    asking.value = false
  }
}
</script>

<template>
  <main class="app-shell">
    <header class="hero">
      <div>
        <p class="eyebrow">Enterprise document intelligence</p>
        <h1>DocuMind</h1>
        <p>Ask a question, inspect its citations, and verify the evidence in the PDF.</p>
      </div>
      <span class="status" :class="{ offline: backendStatus !== 'ok' }">
        API {{ backendStatus }}
      </span>
    </header>

    <div class="workspace">
      <div class="primary-column">
        <section class="panel upload-panel">
          <h2>1. Add a PDF</h2>
          <input type="file" accept="application/pdf" @change="handleFileChange" />
          <p v-if="selectedFile" class="muted">{{ selectedFile.name }}</p>
          <button :disabled="!selectedFile || uploading" @click="handleUpload">
            {{ uploading ? 'Indexing…' : 'Upload and index' }}
          </button>
          <p v-if="uploadResult" class="success">
            Indexed {{ uploadResult.source_file }} · {{ uploadResult.chunk_count }} chunks
          </p>
        </section>

        <section class="panel chat-panel">
          <h2>2. Ask DocuMind</h2>
          <form @submit.prevent="handleAsk">
            <textarea
              v-model="question"
              rows="3"
              placeholder="Ask a question grounded in your indexed documents…"
            />
            <button :disabled="asking || !question.trim()" type="submit">
              {{ asking ? 'Finding evidence…' : 'Ask question' }}
            </button>
          </form>

          <p v-if="errorMessage" class="error" role="alert">{{ errorMessage }}</p>

          <article v-if="chatResult" class="answer-card">
            <p class="eyebrow">AI answer</p>
            <p class="answer-text">{{ chatResult.answer }}</p>
            <p class="trace">Trace ID: {{ chatResult.trace_id }}</p>
            <CitationList
              :sources="chatResult.sources"
              @select="selectedCitation = $event"
            />
          </article>
        </section>
      </div>

      <PdfEvidenceViewer v-if="selectedCitation" :source="selectedCitation" />
      <aside v-else class="evidence-placeholder panel">
        <p class="eyebrow">Evidence viewer</p>
        <h2>Select a citation</h2>
        <p>The cited PDF page and supporting preview will appear here.</p>
      </aside>
    </div>
  </main>
</template>
