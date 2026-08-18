import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import {
  buildRunWebSocketUrl,
  isWebSocketSupported,
  useRunWebSocket,
} from './useRunWebSocket'

/** Minimal scriptable WebSocket stand-in. */
class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3

  readyState = FakeWebSocket.OPEN
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  closed = false

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
  }

  close() {
    this.closed = true
    this.readyState = FakeWebSocket.CLOSED
  }

  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  emitRaw(data: string) {
    this.onmessage?.({ data })
  }

  fail() {
    this.onerror?.()
  }
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
})

afterEach(() => {
  vi.stubGlobal('WebSocket', originalWebSocket)
  vi.unstubAllEnvs()
})

describe('buildRunWebSocketUrl', () => {
  it('derives a ws:// URL from the page origin when no API base is set', () => {
    expect(buildRunWebSocketUrl(7)).toBe(`${window.location.origin.replace(/^http/, 'ws')}/ws/runs/7`)
  })

  it('upgrades an https API base to wss', () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com')
    expect(buildRunWebSocketUrl(3)).toBe('wss://api.example.com/ws/runs/3')
  })

  it('does not double up the slash when the base has a trailing one', () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/')
    expect(buildRunWebSocketUrl(3)).toBe('wss://api.example.com/ws/runs/3')
  })
})

describe('useRunWebSocket', () => {
  it('opens one socket per running run', () => {
    renderHook(() => useRunWebSocket([1, 2], vi.fn()))
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(FakeWebSocket.instances.map((s) => s.url)).toEqual([
      buildRunWebSocketUrl(1),
      buildRunWebSocketUrl(2),
    ])
  })

  it('opens nothing when there are no running runs', () => {
    renderHook(() => useRunWebSocket([], vi.fn()))
    expect(FakeWebSocket.instances).toHaveLength(0)
  })

  it('invokes the callback on a terminal status', () => {
    const onTerminal = vi.fn()
    renderHook(() => useRunWebSocket([1], onTerminal))

    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'status', run_id: 1, status: 'completed' })
    })

    expect(onTerminal).toHaveBeenCalledTimes(1)
    expect(onTerminal.mock.calls[0][0]).toMatchObject({ run_id: 1, status: 'completed' })
  })

  it('treats a failed run as terminal', () => {
    const onTerminal = vi.fn()
    renderHook(() => useRunWebSocket([1], onTerminal))

    act(() => {
      FakeWebSocket.instances[0].emit({
        type: 'status',
        run_id: 1,
        status: 'failed',
        error: 'boom',
      })
    })

    expect(onTerminal).toHaveBeenCalledTimes(1)
  })

  it('ignores a non-terminal status', () => {
    const onTerminal = vi.fn()
    renderHook(() => useRunWebSocket([1], onTerminal))

    act(() => {
      FakeWebSocket.instances[0].emit({ type: 'status', run_id: 1, status: 'running' })
    })

    expect(onTerminal).not.toHaveBeenCalled()
  })

  it('survives an unparseable frame', () => {
    const onTerminal = vi.fn()
    renderHook(() => useRunWebSocket([1], onTerminal))

    expect(() => {
      act(() => {
        FakeWebSocket.instances[0].emitRaw('not json')
      })
    }).not.toThrow()
    expect(onTerminal).not.toHaveBeenCalled()
  })

  it('reports unhealthy once a socket errors', async () => {
    const { result } = renderHook(() => useRunWebSocket([1], vi.fn()))
    expect(result.current.healthy).toBe(true)

    act(() => {
      FakeWebSocket.instances[0].fail()
    })

    await waitFor(() => expect(result.current.healthy).toBe(false))
  })

  it('reports unhealthy when the environment has no WebSocket', async () => {
    vi.stubGlobal('WebSocket', undefined)
    expect(isWebSocketSupported()).toBe(false)

    const { result } = renderHook(() => useRunWebSocket([1], vi.fn()))
    await waitFor(() => expect(result.current.healthy).toBe(false))
  })

  it('closes its sockets on unmount', () => {
    const { unmount } = renderHook(() => useRunWebSocket([1, 2], vi.fn()))
    unmount()
    expect(FakeWebSocket.instances.every((s) => s.closed)).toBe(true)
  })

  it('does not reopen sockets when the id list is unchanged', () => {
    const { rerender } = renderHook(({ ids }) => useRunWebSocket(ids, vi.fn()), {
      initialProps: { ids: [1] },
    })
    expect(FakeWebSocket.instances).toHaveLength(1)

    // A fresh array with equal contents must not churn the connection.
    rerender({ ids: [1] })
    expect(FakeWebSocket.instances).toHaveLength(1)
  })
})
