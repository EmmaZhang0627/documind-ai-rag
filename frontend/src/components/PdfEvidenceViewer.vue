<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { SourceMetadata } from '../services/app'
import { evidencePdfUrl, sourceLabel } from '../utils/citations'

const props = defineProps<{ source: SourceMetadata }>()
const viewerFailed = ref(false)
const pdfUrl = computed(() => evidencePdfUrl(props.source))

watch(
  () => props.source,
  () => {
    viewerFailed.value = false
  },
)
</script>

<template>
  <aside class="evidence-viewer" aria-label="PDF evidence viewer">
    <div class="viewer-heading">
      <div>
        <p class="eyebrow">Source evidence</p>
        <h2>{{ sourceLabel(source) }}</h2>
        <p>Page {{ source.page_number || 1 }}</p>
      </div>
      <a v-if="pdfUrl" :href="pdfUrl" target="_blank" rel="noopener noreferrer">
        Open PDF in new tab
      </a>
    </div>

    <p v-if="!pdfUrl" class="viewer-warning">
      This citation has no document identity, so the PDF cannot be opened.
    </p>
    <p v-else-if="viewerFailed" class="viewer-warning">
      The PDF viewer could not load. The supporting snippet remains available below.
    </p>
    <iframe
      v-if="pdfUrl && !viewerFailed"
      :key="pdfUrl"
      class="pdf-frame"
      :src="pdfUrl"
      :title="`PDF evidence: ${sourceLabel(source)}`"
      @error="viewerFailed = true"
    />

    <section class="snippet-panel" aria-label="Supporting snippet">
      <h3>Supporting snippet</h3>
      <p class="snippet-note">
        This is a short evidence preview, not the complete source document.
      </p>
      <p v-if="source.source_snippet"><mark>{{ source.source_snippet }}</mark></p>
      <p v-else class="viewer-warning">
        No snippet was returned. Use the cited page to inspect the source manually.
      </p>
    </section>
  </aside>
</template>
