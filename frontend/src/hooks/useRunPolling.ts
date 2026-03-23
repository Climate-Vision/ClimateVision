import { useEffect, useRef } from 'react'
import type { Run } from '../api'

export function useRunPolling(
  runs: Run[],
  fetchRuns: () => void,
  onCompleted?: (run: Run) => void,
) {
  const prevRunsRef = useRef<Map<number, string>>(new Map())

  useEffect(() => {
    const hasRunning = runs.some((r) => r.status === 'running')
    if (!hasRunning) return

    const interval = setInterval(() => {
      fetchRuns()
    }, 5000)

    return () => clearInterval(interval)
  }, [runs, fetchRuns])

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
