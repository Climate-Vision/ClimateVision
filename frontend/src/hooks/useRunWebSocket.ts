import { useEffect, useRef, useState } from 'react'
import { getApiBaseUrl } from '../api'
import type { RunStatus } from '../api'

export interface RunStatusEvent {
  type: 'status' | 'error'
  run_id: number
  status?: RunStatus
  result?: Record<string, unknown>
  error?: string
}

const TERMINAL_STATUSES: ReadonlySet<string> = new Set(['completed', 'failed'])

/**
 * Resolve the WebSocket URL for a run.
 *
 * `getApiBaseUrl()` is empty by default, meaning "same origin", so fall back to
 * the current page's host and swap http(s) for ws(s).
 */
export function buildRunWebSocketUrl(runId: number): string {
  const base = getApiBaseUrl()
  const origin = base !== '' ? base : window.location.origin
  const wsOrigin = origin.replace(/^http/, 'ws')
  return `${wsOrigin.replace(/\/$/, '')}/ws/runs/${runId}`
}

/** True when the environment provides a WebSocket implementation. */
export function isWebSocketSupported(): boolean {
  return typeof WebSocket !== 'undefined'
}

/**
 * Watch a set of runs over WebSockets.
 *
 * Opens one socket per run id and calls `onTerminal` when a run reports
 * `completed` or `failed`. `healthy` is false when the browser has no
 * WebSocket implementation, or when any socket errored — the caller uses that
 * to decide whether polling still needs to run.
 */
export function useRunWebSocket(
  runIds: number[],
  onTerminal: (event: RunStatusEvent) => void,
): { healthy: boolean } {
  const [failed, setFailed] = useState(false)
  const onTerminalRef = useRef(onTerminal)

  // Keep the latest callback without making it a effect dependency, so a new
  // inline function on each render does not tear down and reopen the sockets.
  useEffect(() => {
    onTerminalRef.current = onTerminal
  }, [onTerminal])

  // Stable key so re-renders with an equal id list do not reopen sockets.
  const key = runIds.join(',')

  useEffect(() => {
    if (!isWebSocketSupported()) {
      setFailed(true)
      return
    }
    const ids = key === '' ? [] : key.split(',').map(Number)
    if (ids.length === 0) return

    const sockets: WebSocket[] = []
    let cancelled = false

    for (const runId of ids) {
      let socket: WebSocket
      try {
        socket = new WebSocket(buildRunWebSocketUrl(runId))
      } catch {
        // Constructor throws on a malformed URL or a blocked scheme.
        setFailed(true)
        continue
      }

      socket.onmessage = (message) => {
        if (cancelled) return
        let event: RunStatusEvent
        try {
          event = JSON.parse(String(message.data)) as RunStatusEvent
        } catch {
          // A frame we cannot parse is not fatal; polling still covers us.
          return
        }
        if (event.status && TERMINAL_STATUSES.has(event.status)) {
          onTerminalRef.current(event)
        }
      }

      socket.onerror = () => {
        if (!cancelled) setFailed(true)
      }

      sockets.push(socket)
    }

    return () => {
      cancelled = true
      for (const socket of sockets) {
        // Detach handlers first so a close-triggered error cannot flip
        // `failed` after this effect has been torn down.
        socket.onmessage = null
        socket.onerror = null
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          socket.close()
        }
      }
    }
  }, [key])

  return { healthy: !failed }
}
