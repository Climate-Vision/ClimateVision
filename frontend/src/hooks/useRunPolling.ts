import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { Run } from '../api'
import { useRunWebSocket } from './useRunWebSocket'

/** Interval used when falling back to polling, in milliseconds. */
export const POLL_INTERVAL_MS = 5000

/**
 * Keep `runs` fresh while any of them is still running.
 *
 * Prefers a WebSocket per in-flight run and refetches as soon as one reports a
 * terminal status. Polling is only started when the WebSocket path is
 * unavailable — no WebSocket implementation, or a socket that errored — so the
 * hook degrades to its previous behaviour rather than losing updates.
 */
export function useRunPolling(
  runs: Run[],
  fetchRuns: () => void,
  onCompleted?: (run: Run) => void,
) {
  const prevRunsRef = useRef<Map<number, string>>(new Map())
  const fetchRunsRef = useRef(fetchRuns)

  useEffect(() => {
    fetchRunsRef.current = fetchRuns
  }, [fetchRuns])

  const runningIds = useMemo(
    () => runs.filter((r) => r.status === 'running').map((r) => r.id),
    [runs],
  )

  const handleTerminal = useCallback(() => {
    fetchRunsRef.current()
  }, [])

  const { healthy } = useRunWebSocket(runningIds, handleTerminal)

  useEffect(() => {
    if (runningIds.length === 0) return
    // A healthy socket already pushes updates, so polling would be redundant.
    if (healthy) return

    const interval = setInterval(() => {
      fetchRunsRef.current()
    }, POLL_INTERVAL_MS)

    return () => clearInterval(interval)
  }, [runningIds, healthy])

  // Detect transitions from running → completed
  useEffect(() => {
    const prev = prevRunsRef.current
    for (const run of runs) {
      const prevStatus = prev.get(run.id)
      if (prevStatus === 'running' && run.status === 'completed' && onCompleted) {
        onCompleted(run)
      }
    }
    prevRunsRef.current = new Map(runs.map((r) => [r.id, r.status]))
  }, [runs, onCompleted])
}
