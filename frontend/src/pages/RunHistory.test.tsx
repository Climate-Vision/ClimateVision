import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ToastProvider } from '../contexts/ToastContext'
import RunHistory from './RunHistory'
import { listRuns } from '../api'

// Stub the network layer so the page renders deterministically. Other API
// exports (types, helpers) stay real via importActual.
vi.mock('../api', async (importActual) => {
  const actual = await importActual<typeof import('../api')>()
  return {
    ...actual,
    listRuns: vi.fn().mockResolvedValue([]),
    getRun: vi.fn().mockResolvedValue(null),
  }
})

function renderPage() {
  return render(
    <MemoryRouter>
      <ToastProvider>
        <RunHistory />
      </ToastProvider>
    </MemoryRouter>,
  )
}

describe('RunHistory page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page heading', async () => {
    renderPage()
    expect(await screen.findByText('Run History')).toBeInTheDocument()
  })

  it('fetches runs on mount and shows the empty state when there are none', async () => {
    renderPage()
    expect(await screen.findByText('No runs yet')).toBeInTheDocument()
    expect(listRuns).toHaveBeenCalled()
  })
})
