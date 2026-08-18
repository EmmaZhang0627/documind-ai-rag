import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import PdfEvidenceViewer from './PdfEvidenceViewer.vue'

describe('PdfEvidenceViewer', () => {
  it('navigates to the cited page and displays the supporting snippet', () => {
    const wrapper = mount(PdfEvidenceViewer, {
      props: {
        source: {
          document_id: 'policy-a',
          version: '2',
          source_file: 'Policy.pdf',
          page_number: 3,
          source_snippet: 'Appeals must be submitted within the stated period.',
        },
      },
    })

    expect(wrapper.get('iframe').attributes('src')).toContain('document_id=policy-a')
    expect(wrapper.get('iframe').attributes('src')).toContain('version=2')
    expect(wrapper.get('iframe').attributes('src')).toContain('#page=3')
    expect(wrapper.get('mark').text()).toContain('Appeals must be submitted')
  })

  it('keeps a visible fallback when no snippet is available', () => {
    const wrapper = mount(PdfEvidenceViewer, {
      props: {
        source: {
          document_id: 'policy-a',
          source_file: 'Policy.pdf',
          page_number: 2,
        },
      },
    })

    expect(wrapper.text()).toContain('No snippet was returned')
    expect(wrapper.find('iframe').exists()).toBe(true)
  })

  it('does not attempt to open a PDF without document identity', () => {
    const wrapper = mount(PdfEvidenceViewer, {
      props: {
        source: {
          source_file: 'Unavailable.pdf',
          page_number: 1,
          source_snippet: 'The answer remains visible in this preview.',
        },
      },
    })

    expect(wrapper.find('iframe').exists()).toBe(false)
    expect(wrapper.text()).toContain('PDF cannot be opened')
    expect(wrapper.get('mark').text()).toContain('answer remains visible')
  })
})
