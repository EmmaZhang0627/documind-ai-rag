import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 10000,
})

export const healthCheck = async () => {
  const response = await api.get('/health')
  return response.data
}

export default api