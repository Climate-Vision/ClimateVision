import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useRunPolling, POLL_INTERVAL_MS } from './useRunPolling'
import type { Run } from '../api'

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  static readonly OPEN = 1
  static readonly CLOSED = 3

  readyState = FakeWebSocket.OPEN
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
  }

  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  fail() {
    this.onerror?.()
  }
}

const originalWebSocket = globalThis.WebSocket

function run(id: number, status: Run['status']): Run {
  return {
    id,
    kind: 'demo',
    status,
    analysis_type: 'deforestation',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  }
}

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.stubGlobal('WebSocket', originalWebSocket)
})

describe('useRunPolling', () => {
  it('does not poll while the WebSocket path is healthy', () => {
    const fetchRuns = vi.fn()
    renderHook(() => useRunPolling([run(1, 'running')], fetchRuns))

    act(() => {
      vi.advanceTimersByTime(POLL_INTERVAL_MS * 3)
    })

    expect(fetchRuns).not.toHaveBeenCalled()
  })

  it('refetches as soon as a run reports a terminal status', () => {
    const fetchRuns = vi.fn()
    renderHook(() => useRunPolling([run(1, 'running')], fetchRuns))

    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'status', run_id: 1, status: 'completed' })
    })

    expect(fetchRuns).toHaveBeenCalledTimes(1)
  })

  it('falls back to polling once a socket errors', () => {
    const fetchRuns = vi.fn()
    renderHook(() => useRunPolling([run(1, 'running')], fetchRuns))

    // The error flips the hook to unhealthy, which starts the interval.
    // `act` flushes that state update synchronously, so no `waitFor` is
    // needed — and `waitFor` would deadlock against the fake timers here.
    act(() => {
      FakeWebSocket.instances[0].fail()
    })

    act(() => {
      vi.advanceTimersByTime(POLL_INTERVAL_MS * 2)
    })

    expect(fetchRuns).toHaveBeenCalledTimes(2)
  })

  it('neither polls nor connects when nothing is running', () => {
    const fetchRuns = vi.fn()
    renderHook(() => useRunPolling([run(1, 'completed')], fetchRuns))

    act(() => {
      vi.advanceTimersByTime(POLL_INTERVAL_MS * 3)
    })

    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(fetchRuns).not.toHaveBeenCalled()
  })

  it('still reports the running → completed transition', () => {
    const onCompleted = vi.fn()
    const { rerender } = renderHook(
      ({ runs }) => useRunPolling(runs, vi.fn(), onCompleted),
      { initialProps: { runs: [run(1, 'running')] } },
    )

    rerender({ runs: [run(1, 'completed')] })

    expect(onCompleted).toHaveBeenCalledTimes(1)
    expect(onCompleted.mock.calls[0][0].id).toBe(1)
  })
})
