import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 100000,
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

export default api
