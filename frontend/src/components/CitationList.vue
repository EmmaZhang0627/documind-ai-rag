<script setup lang="ts">
import { computed } from 'vue'
import type { SourceMetadata } from '../services/app'
import { deduplicateSources, sourceLabel } from '../utils/citations'

const props = defineProps<{ sources: SourceMetadata[] }>()
const emit = defineEmits<{ select: [source: SourceMetadata] }>()
const citations = computed(() => deduplicateSources(props.sources))
</script>

<template>
  <section v-if="citations.length" class="citations" aria-label="Answer sources">
    <h3>Sources</h3>
    <button
      v-for="(source, index) in citations"
      :key="`${source.document_id}-${source.version}-${source.page_number}-${index}`"
      class="citation-card"
      type="button"
      @click="emit('select', source)"
    >
      <span class="citation-title">
        [{{ index + 1 }}] {{ sourceLabel(source) }}
        <span v-if="source.page_number"> · Page {{ source.page_number }}</span>
      </span>
      <span v-if="source.source_snippet" class="citation-snippet">
        {{ source.source_snippet }}
      </span>
      <span v-else class="citation-fallback">Supporting preview unavailable.</span>
    </button>
  </section>
</template>
