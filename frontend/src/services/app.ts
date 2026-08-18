import axios from 'axios'

export interface SourceMetadata {
  document_id?: string | null
  source_file?: string | null
  file_name?: string | null
  version?: string | null
  status?: string | null
  page_number?: number | null
  chunk_index?: number | null
  source_snippet?: string | null
}

export interface ChatResponse {
  trace_id: string
  question: string
  answer: string
  sources: SourceMetadata[]
  status: string
  fallback_reason?: string | null
}

export const backendBaseUrl =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'

const api = axios.create({
  baseURL: backendBaseUrl,
  timeout: 180000,
})

export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export const uploadDocument = async (file: File) => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await api.post('/api/documents/parse-pdf', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  return response.data
}

export const askQuestion = async (question: string): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>('/api/chat', { question })
  return response.data
}

export default api
