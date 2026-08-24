import { useState, useEffect, useMemo, useCallback } from 'react'
import { RefreshCw, Search, Grid, List, Table as TableIcon } from 'lucide-react'
import { listRuns, getRun } from '../api'
import type { Run, RunWithResult } from '../api'
import { RunCard } from '../components/runs/RunCard'
import { StatusBadge } from '../components/ui/StatusBadge'
import { ResultsPanel } from '../components/results/ResultsPanel'
import { SkeletonCard, SkeletonRow } from '../components/ui/SkeletonCard'
import { EmptyState, SatelliteIllustration } from '../components/ui/EmptyState'
import { ErrorBoundary } from '../components/ui/ErrorBoundary'
import { useToast } from '../contexts/ToastContext'
import { useRunPolling } from '../hooks/useRunPolling'
import { useNavigate } from 'react-router-dom'

type ViewMode = 'grid' | 'table'
type StatusFilter = 'all' | 'completed' | 'failed' | 'running' | 'pending'

const ANALYSIS_LABEL: Record<string, string> = {
  deforestation: 'Deforestation',
  ice_melting: 'Ice Melting',
  flooding: 'Flooding',
  drought: 'Drought',
  wildfire: 'Wildfire',
}

export default function RunHistory() {
  const { showToast } = useToast()
  const navigate = useNavigate()

  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const [selectedRunData, setSelectedRunData] = useState<RunWithResult | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [search, setSearch] = useState('')
  const [lastRefreshed, setLastRefreshed] = useState(new Date())

  const fetchRuns = useCallback(async () => {
    try {
      const data = await listRuns()
      setRuns(data)
      setLastRefreshed(new Date())
    } catch (e) {
      showToast('error', String(e))
    }
  }, [showToast])

  useEffect(() => {
    setLoading(true)
    fetchRuns().finally(() => setLoading(false))
  }, [fetchRuns])

  useEffect(() => {
    if (selectedRunId == null) { setSelectedRunData(null); return }
    setLoadingDetail(true)
    getRun(selectedRunId)
      .then(setSelectedRunData)
      .catch((e) => showToast('error', String(e)))
      .finally(() => setLoadingDetail(false))
  }, [selectedRunId, showToast])

  const onRunCompleted = useCallback((run: Run) => {
    showToast('success', `✓ Run #${run.id} complete — ${ANALYSIS_LABEL[run.analysis_type] ?? run.analysis_type}`, {
      label: 'View',
      onClick: () => setSelectedRunId(run.id),
    })
  }, [showToast])

  useRunPolling(runs, fetchRuns, onRunCompleted)

  const stats = useMemo(() => ({
    total: runs.length,
    completed: runs.filter((r) => r.status === 'completed').length,
    failed: runs.filter((r) => r.status === 'failed').length,
    running: runs.filter((r) => r.status === 'running').length,
  }), [runs])

  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      if (statusFilter !== 'all' && r.status !== statusFilter) return false
      if (search) {
        const q = search.toLowerCase()
        return String(r.id).includes(q) || r.analysis_type.includes(q) || r.status.includes(q)
      }
      return true
    })
  }, [runs, statusFilter, search])

  return (
    <div className="px-6 py-8 space-y-6 max-w-7xl mx-auto">
      {/* Page header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-xl font-bold text-cv-text-primary">Run History</h2>
          {/* Stats chips */}
          <div className="flex flex-wrap gap-2 mt-3">
            {([
              { label: `Total ${stats.total}`, filter: 'all' as StatusFilter, classes: 'border-cv-border text-cv-text-secondary' },
              { label: `Completed ${stats.completed}`, filter: 'completed' as StatusFilter, classes: 'border-green-300 text-green-700' },
              { label: `Failed ${stats.failed}`, filter: 'failed' as StatusFilter, classes: 'border-red-300 text-red-700' },
              { label: `Running ${stats.running}`, filter: 'running' as StatusFilter, classes: 'border-amber-300 text-amber-700' },
            ]).map(({ label, filter, classes }) => (
              <button
                key={filter}
                onClick={() => setStatusFilter(statusFilter === filter ? 'all' : filter)}
                className={`text-xs px-3 py-1 rounded-full border transition ${classes} ${statusFilter === filter ? 'bg-cv-card-hover' : 'hover:bg-cv-card'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 justify-between">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-cv-text-dim" />
          <input
            type="text"
            placeholder="Search runs…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-cv-card border border-cv-border rounded-lg pl-9 pr-3 py-2 text-sm text-cv-text-primary placeholder:text-cv-text-dim focus:outline-none focus:border-cv-primary w-52 transition"
          />
        </div>

        <div className="flex items-center gap-2">
          {/* View toggle */}
          <div className="flex border border-cv-border rounded-lg overflow-hidden">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 transition ${viewMode === 'grid' ? 'bg-cv-card text-cv-primary' : 'text-cv-text-dim hover:text-cv-text-secondary'}`}
              title="Grid view"
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`p-2 transition ${viewMode === 'table' ? 'bg-cv-card text-cv-primary' : 'text-cv-text-dim hover:text-cv-text-secondary'}`}
              title="Table view"
            >
              <TableIcon className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={fetchRuns}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-cv-card border border-cv-border text-sm text-cv-text-secondary hover:text-cv-text-primary transition"
            title={`Last refreshed ${lastRefreshed.toLocaleTimeString()}`}
          >
            <RefreshCw className="w-4 h-4" />
            <span className="hidden sm:inline">Refresh</span>
          </button>
        </div>
      </div>

      <div className="flex gap-6">
        {/* Run list */}
        <div className="flex-1 min-w-0">
          {loading ? (
            <div className={viewMode === 'grid' ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4' : 'space-y-2'}>
              {Array.from({ length: 6 }).map((_, i) =>
                viewMode === 'grid' ? <SkeletonCard key={i} /> : <SkeletonRow key={i} />
              )}
            </div>
          ) : filteredRuns.length === 0 ? (
            <EmptyState
              icon={<SatelliteIllustration />}
              heading={runs.length === 0 ? 'No runs yet' : 'No matching runs'}
              subtext={runs.length === 0 ? 'Create your first analysis to get started' : 'Try adjusting your search or filters'}
              action={
                runs.length === 0 ? (
                  <button
                    onClick={() => navigate('/app')}
                    className="px-4 py-2 rounded-lg bg-cv-primary-muted text-cv-primary text-sm font-medium hover:bg-green-800/40 transition"
                  >
                    New Analysis
                  </button>
                ) : undefined
              }
            />
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredRuns.map((run) => (
                <ErrorBoundary key={run.id}>
                  <RunCard
                    run={run}
                    selected={selectedRunId === run.id}
                    onClick={() => setSelectedRunId(selectedRunId === run.id ? null : run.id)}
                    confidence={undefined}
                  />
                </ErrorBoundary>
              ))}
            </div>
          ) : (
            /* Table view */
            <div className="rounded-xl border border-cv-border overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-cv-border bg-cv-surface">
                    <th className="text-left px-4 py-3 text-xs font-medium text-cv-text-dim">#</th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-cv-text-dim">Type</th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-cv-text-dim">Date</th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-cv-text-dim">Status</th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-cv-text-dim">Kind</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRuns.map((run) => (
                    <tr
                      key={run.id}
                      onClick={() => setSelectedRunId(selectedRunId === run.id ? null : run.id)}
                      className={`border-b border-cv-border cursor-pointer transition ${
                        selectedRunId === run.id ? 'bg-cv-primary-muted/20' : 'hover:bg-cv-card'
                      }`}
                    >
                      <td className="px-4 py-3 font-semibold text-cv-text-primary">#{run.id}</td>
                      <td className="px-4 py-3 text-cv-text-secondary">{ANALYSIS_LABEL[run.analysis_type] ?? run.analysis_type}</td>
                      <td className="px-4 py-3 text-cv-text-dim">{new Date(run.created_at).toLocaleDateString()}</td>
                      <td className="px-4 py-3"><StatusBadge status={run.status} /></td>
                      <td className="px-4 py-3 text-cv-text-dim capitalize">{run.kind}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Run detail slide-over */}
        {selectedRunId && (
          <div className="w-80 shrink-0 hidden lg:block">
            <div className="sticky top-4 bg-cv-card border border-cv-border rounded-xl p-5">
              {loadingDetail ? (
                <div className="space-y-3">
                  <SkeletonCard />
                </div>
              ) : selectedRunData ? (
                <ErrorBoundary section="Results">
                  <ResultsPanel
                    run={selectedRunData.run}
                    payload={selectedRunData.result?.payload as Record<string, unknown> | null ?? null}
                    onRunAgain={() => navigate('/app')}
                  />
                </ErrorBoundary>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
