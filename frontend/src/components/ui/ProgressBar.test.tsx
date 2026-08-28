import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ProgressBar } from './ProgressBar'

describe('ProgressBar', () => {
  it('shows the percentage label when showLabel is set', () => {
    render(<ProgressBar value={42} showLabel />)
    expect(screen.getByText('42.0%')).toBeInTheDocument()
  })

  it('renders a custom label', () => {
    render(<ProgressBar value={10} label="Coverage" />)
    expect(screen.getByText('Coverage')).toBeInTheDocument()
  })

  it('clamps values above the maximum to 100%', () => {
    render(<ProgressBar value={150} showLabel />)
    expect(screen.getByText('100.0%')).toBeInTheDocument()
  })

  it('clamps negative values to 0%', () => {
    render(<ProgressBar value={-10} showLabel />)
    expect(screen.getByText('0.0%')).toBeInTheDocument()
  })
})
