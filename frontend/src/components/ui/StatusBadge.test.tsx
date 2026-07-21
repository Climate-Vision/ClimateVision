import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { StatusBadge } from './StatusBadge'

describe('StatusBadge (run status)', () => {
  it('renders the label and an accessible status for a known status', () => {
    render(<StatusBadge status="completed" />)
    expect(screen.getByText('Completed')).toBeInTheDocument()
    expect(screen.getByLabelText('Status: Completed')).toBeInTheDocument()
  })

  it('renders the Failed status', () => {
    render(<StatusBadge status="failed" />)
    expect(screen.getByText('Failed')).toBeInTheDocument()
  })

  it('falls back to Pending for an unrecognised status', () => {
    render(<StatusBadge status="something-else" />)
    expect(screen.getByText('Pending')).toBeInTheDocument()
  })
})
