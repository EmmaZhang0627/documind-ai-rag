import type { SourceMetadata } from '../services/app'
import { backendBaseUrl } from '../services/app'

const normalized = (value?: string | null) =>
  (value ?? '').trim().replace(/\s+/g, ' ').toLocaleLowerCase()

export const sourceLabel = (source: SourceMetadata) =>
  source.source_file || source.file_name || 'Unknown source'

export const deduplicateSources = (sources: SourceMetadata[]) => {
  const seen = new Set<string>()

  return sources.filter((source) => {
    const key = [
      normalized(source.document_id),
      normalized(sourceLabel(source)),
      normalized(source.version || '1'),
      source.page_number ?? 'unknown-page',
      normalized(source.source_snippet),
    ].join('|')

    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export const evidencePdfUrl = (
  source: SourceMetadata,
  baseUrl = backendBaseUrl,
) => {
  if (!source.document_id) return null

  const params = new URLSearchParams({
    document_id: source.document_id,
    version: source.version || '1',
  })
  const page = Math.max(1, source.page_number || 1)
  return `${baseUrl}/api/documents/evidence-pdf?${params.toString()}#page=${page}`
}
