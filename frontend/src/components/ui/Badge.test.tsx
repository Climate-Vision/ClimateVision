import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import {
  Badge,
  StatusBadge,
  SeverityBadge,
  AnalysisTypeBadge,
} from './Badge'

describe('Badge', () => {
  it('renders its children', () => {
    render(<Badge>Hello</Badge>)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})

describe('StatusBadge', () => {
  it('maps a known status to its label', () => {
    render(<StatusBadge status="completed" />)
    expect(screen.getByText('Completed')).toBeInTheDocument()
  })

  it('falls back to the Pending label for an unknown status', () => {
    // @ts-expect-error deliberately passing an invalid status
    render(<StatusBadge status="bogus" />)
    expect(screen.getByText('Pending')).toBeInTheDocument()
  })
})

describe('SeverityBadge', () => {
  it('maps a severity to its label', () => {
    render(<SeverityBadge severity="high" />)
    expect(screen.getByText('High')).toBeInTheDocument()
  })
})

describe('AnalysisTypeBadge', () => {
  it('maps an analysis type to a human-readable label', () => {
    render(<AnalysisTypeBadge type="deforestation" />)
    expect(screen.getByText('Deforestation')).toBeInTheDocument()
  })
})
