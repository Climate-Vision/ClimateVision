import { useState, useEffect, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { listRuns } from '../api'
import type { Run } from '../api'
import { SkeletonCard } from '../components/ui/SkeletonCard'
import { useToast } from '../contexts/ToastContext'

const COLORS: Record<string, string> = {
  deforestation: '#22c55e',
  ice_melting: '#06b6d4',
  flooding: '#3b82f6',
  drought: '#f59e0b',
  wildfire: '#ef4444',
}

const STATUS_COLORS: Record<string, string> = {
  completed: '#22c55e',
  failed: '#ef4444',
  running: '#f59e0b',
  pending: '#6b7280',
}

const LABEL: Record<string, string> = {
  deforestation: 'Deforestation',
  ice_melting: 'Ice Melting',
  flooding: 'Flooding',
  drought: 'Drought',
  wildfire: 'Wildfire',
}

type Period = '7d' | '30d' | '90d'

function KPICard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-cv-card border border-cv-border rounded-xl p-5">
      <div className="text-xs font-medium text-cv-text-secondary uppercase tracking-wide mb-1">{label}</div>
      <div className="text-3xl font-bold text-cv-text-primary">{value}</div>
      {sub && <div className="text-xs text-cv-text-dim mt-1">{sub}</div>}
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-cv-card border border-cv-border rounded-xl p-5">
      <h3 className="text-sm font-semibold text-cv-text-primary mb-4">{title}</h3>
      {children}
    </div>
  )
}

export default function Analytics() {
  const { showToast } = useToast()
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [period, setPeriod] = useState<Period>('30d')

  useEffect(() => {
    listRuns({ limit: 200 })
      .then(setRuns)
      .catch((e) => showToast('error', String(e)))
      .finally(() => setLoading(false))
  }, [showToast])

  const kpis = useMemo(() => {
    const total = runs.length
    const completed = runs.filter((r) => r.status === 'completed').length
    const successRate = total ? Math.round((completed / total) * 100) : 0
    const typeCounts = runs.reduce<Record<string, number>>((acc, r) => {
      acc[r.analysis_type] = (acc[r.analysis_type] ?? 0) + 1
      return acc
    }, {})
    const mostCommon = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—'
    return { total, successRate, mostCommon: LABEL[mostCommon] ?? mostCommon }
  }, [runs])

  const typeData = useMemo(() => {
    const counts: Record<string, number> = {}
    runs.forEach((r) => { counts[r.analysis_type] = (counts[r.analysis_type] ?? 0) + 1 })
    return Object.entries(counts).map(([type, count]) => ({ type: LABEL[type] ?? type, count, fill: COLORS[type] ?? '#6b7280' }))
  }, [runs])

  const statusData = useMemo(() => {
    const counts: Record<string, number> = {}
    runs.forEach((r) => { counts[r.status] = (counts[r.status] ?? 0) + 1 })
    return Object.entries(counts).map(([name, value]) => ({ name, value, fill: STATUS_COLORS[name] ?? '#6b7280' }))
  }, [runs])

  const timelineData = useMemo(() => {
    const days = period === '7d' ? 7 : period === '30d' ? 30 : 90
    const cutoff = new Date()
    cutoff.setDate(cutoff.getDate() - days)
    const recent = runs.filter((r) => new Date(r.created_at) >= cutoff)
    const byDay: Record<string, number> = {}
    recent.forEach((r) => {
      const day = r.created_at.split('T')[0]
      byDay[day] = (byDay[day] ?? 0) + 1
    })
    return Object.entries(byDay).sort().map(([date, count]) => ({ date: date.slice(5), count }))
  }, [runs, period])

  const failedRuns = useMemo(() => runs.filter((r) => r.status === 'failed').slice(0, 10), [runs])

  if (loading) {
    return (
      <div className="px-6 py-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    )
  }

  return (
    <div className="px-6 py-8 max-w-7xl mx-auto space-y-6">
      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <KPICard label="Total Runs" value={kpis.total} />
        <KPICard label="Success Rate" value={`${kpis.successRate}%`} sub="completed / total" />
        <KPICard label="Most Common Type" value={kpis.mostCommon} />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Runs by type */}
        <ChartCard title="Runs by Analysis Type">
          {typeData.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-cv-text-dim text-sm">No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={typeData} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={90} />
                <Tooltip
                  contentStyle={{ background: '#162019', border: '1px solid #1f3024', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#86efac' }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {typeData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        {/* Status donut */}
        <ChartCard title="Status Distribution">
          {statusData.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-cv-text-dim text-sm">No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={statusData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={75} paddingAngle={3}>
                  {statusData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Pie>
                <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: '#86efac', fontSize: 12 }}>{v}</span>} />
                <Tooltip
                  contentStyle={{ background: '#162019', border: '1px solid #1f3024', borderRadius: 8, fontSize: 12 }}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </ChartCard>
      </div>

      {/* Timeline */}
      <ChartCard title="Run Timeline">
        <div className="flex gap-2 mb-4">
          {(['7d', '30d', '90d'] as Period[]).map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition ${
                period === p
                  ? 'bg-cv-primary-muted text-cv-primary'
                  : 'bg-cv-surface border border-cv-border text-cv-text-secondary hover:text-cv-text-primary'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
        {timelineData.length === 0 ? (
          <div className="h-40 flex items-center justify-center text-cv-text-dim text-sm">No runs in this period</div>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={timelineData}>
              <CartesianGrid stroke="#1f3024" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#162019', border: '1px solid #1f3024', borderRadius: 8, fontSize: 12 }}
              />
              <Line type="monotone" dataKey="count" stroke="#22c55e" strokeWidth={2} dot={{ fill: '#22c55e', r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      {/* Failed runs table */}
      {failedRuns.length > 0 && (
        <ChartCard title="Failed Runs">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-cv-border">
                <th className="text-left pb-2 text-xs font-medium text-cv-text-dim">Run</th>
                <th className="text-left pb-2 text-xs font-medium text-cv-text-dim">Type</th>
                <th className="text-left pb-2 text-xs font-medium text-cv-text-dim">Date</th>
              </tr>
            </thead>
            <tbody>
              {failedRuns.map((r) => (
                <tr key={r.id} className="border-b border-cv-border/50">
                  <td className="py-2 text-cv-text-primary font-medium">#{r.id}</td>
                  <td className="py-2 text-cv-text-secondary">{LABEL[r.analysis_type] ?? r.analysis_type}</td>
                  <td className="py-2 text-cv-text-dim">{new Date(r.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ChartCard>
      )}
    </div>
  )
}
