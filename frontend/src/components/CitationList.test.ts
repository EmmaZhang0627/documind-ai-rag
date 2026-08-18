import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import CitationList from './CitationList.vue'

const sources = [
  {
    document_id: 'study-plan',
    version: '1',
    source_file: 'Study Plan.pdf',
    page_number: 1,
    source_snippet: 'The programme duration is 24 months.',
  },
  {
    document_id: 'study-plan',
    version: '1',
    source_file: 'Study Plan.pdf',
    page_number: 1,
    source_snippet: '  The programme duration is 24 months. ',
  },
  {
    document_id: 'study-plan',
    version: '1',
    source_file: 'Study Plan.pdf',
    page_number: 1,
    source_snippet: 'A different supporting fact on the same page.',
  },
]

describe('CitationList', () => {
  it('renders citations and removes only duplicate evidence', async () => {
    const wrapper = mount(CitationList, { props: { sources } })
    const cards = wrapper.findAll('button.citation-card')

    expect(cards).toHaveLength(2)
    expect(cards[0].text().replace(/\s+/g, ' ')).toContain(
      '[1] Study Plan.pdf · Page 1',
    )
    expect(cards[1].text()).toContain('different supporting fact')

    await cards[0].trigger('click')
    expect(wrapper.emitted('select')?.[0]?.[0]).toMatchObject({
      document_id: 'study-plan',
      page_number: 1,
    })
  })
})
