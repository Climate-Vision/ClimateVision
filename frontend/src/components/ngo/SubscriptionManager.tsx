import { useState } from 'react'
import { Card } from '../ui/Card'
import { Badge, AnalysisTypeBadge } from '../ui/Badge'
import type { AnalysisType } from '../ui/Badge'

function cx(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(' ')
}

export interface Subscription {
  id: number
  organization_id: number
  name?: string
  description?: string
  bbox: number[]
  analysis_types: AnalysisType[]
  alert_threshold: number
  notification_channel: string
  webhook_url?: string
  active: boolean
  last_checked_at?: string
  created_at: string
}

export interface SubscriptionManagerProps {
  subscriptions: Subscription[]
  onAdd?: () => void
  onEdit?: (subscription: Subscription) => void
  onDelete?: (subscriptionId: number) => void
  onToggle?: (subscriptionId: number, active: boolean) => void
  loading?: boolean
  className?: string
}

// Format bbox for display
function formatBBox(bbox: number[]): string {
  if (!bbox || bbox.length !== 4) return 'Invalid region'
  return `${bbox[0].toFixed(3)}°, ${bbox[1].toFixed(3)}° to ${bbox[2].toFixed(3)}°, ${bbox[3].toFixed(3)}°`
}

// Calculate approximate area from bbox
function calculateArea(bbox: number[]): string {
  if (!bbox || bbox.length !== 4) return 'N/A'
  const [minLon, minLat, maxLon, maxLat] = bbox
  const latDiff = Math.abs(maxLat - minLat)
  const lonDiff = Math.abs(maxLon - minLon)
  // Rough approximation: 1 degree ≈ 111 km at equator
  const avgLat = (minLat + maxLat) / 2
  const lonKm = lonDiff * 111 * Math.cos((avgLat * Math.PI) / 180)
  const latKm = latDiff * 111
  const areaKm2 = lonKm * latKm
  
  if (areaKm2 < 1) return `${(areaKm2 * 1000000).toFixed(0)} m²`
  if (areaKm2 < 100) return `${areaKm2.toFixed(2)} km²`
  if (areaKm2 < 10000) return `${areaKm2.toFixed(0)} km²`
  return `${(areaKm2 / 1000).toFixed(1)}k km²`
}

export function SubscriptionManager({
  subscriptions,
  onAdd,
  onEdit,
  onDelete,
  onToggle,
  loading = false,
  className,
}: SubscriptionManagerProps) {
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const activeCount = subscriptions.filter((s) => s.active).length

  return (
    <Card
      title="Monitored Regions"
      right={
        onAdd && (
          <button
            onClick={onAdd}
            className="px-3 py-1.5 text-xs font-medium text-white bg-brand-600 hover:bg-brand-500 rounded-lg transition"
          >
            + Add Region
          </button>
        )
      }
      className={className}
    >
      {/* Stats */}
      <div className="flex items-center gap-4 pb-4 border-b border-base-800 mb-4">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-base-400">Total:</span>
          <span className="font-medium text-base-100">{subscriptions.length}</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-brand-600">Active:</span>
          <span className="font-medium text-base-100">{activeCount}</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-base-500">Paused:</span>
          <span className="font-medium text-base-100">{subscriptions.length - activeCount}</span>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="text-center py-8 text-base-400">
          <div className="animate-spin w-6 h-6 border-2 border-base-600 border-t-brand-500 rounded-full mx-auto" />
          <p className="mt-2 text-sm">Loading subscriptions...</p>
        </div>
      )}

      {/* Empty state */}
      {!loading && subscriptions.length === 0 && (
        <div className="text-center py-8 text-base-400">
          <svg className="w-12 h-12 mx-auto mb-3 text-base-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
          </svg>
          <p className="text-sm">No monitored regions yet</p>
          {onAdd && (
            <button
              onClick={onAdd}
              className="mt-3 px-4 py-2 text-sm font-medium text-brand-600 hover:text-brand-700 bg-brand-600/10 hover:bg-brand-600/20 rounded-lg transition"
            >
              Add your first region
            </button>
          )}
        </div>
      )}

      {/* Subscriptions list */}
      {!loading && subscriptions.length > 0 && (
        <div className="space-y-3">
          {subscriptions.map((sub) => (
            <div
              key={sub.id}
              className={cx(
                'rounded-xl border transition',
                sub.active
                  ? 'bg-base-900/50 border-base-700'
                  : 'bg-base-900/20 border-base-800 opacity-70',
              )}
            >
              {/* Header */}
              <button
                onClick={() => setExpandedId(expandedId === sub.id ? null : sub.id)}
                className="w-full text-left p-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-base-100">
                        {sub.name || `Region #${sub.id}`}
                      </span>
                      {!sub.active && (
                        <Badge variant="neutral" size="sm">Paused</Badge>
                      )}
                    </div>
                    <div className="mt-1 text-xs text-base-400">
                      {formatBBox(sub.bbox)}
                    </div>
                    <div className="mt-2 flex items-center gap-2 flex-wrap">
                      {sub.analysis_types.map((type) => (
                        <AnalysisTypeBadge key={type} type={type} size="sm" />
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <div className="text-right">
                      <div className="text-xs text-base-400">Area</div>
                      <div className="text-sm font-medium text-base-200">
                        {calculateArea(sub.bbox)}
                      </div>
                    </div>
                    <svg
                      className={cx(
                        'w-5 h-5 text-base-500 transition-transform',
                        expandedId === sub.id && 'rotate-180',
                      )}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </button>

              {/* Expanded details */}
              {expandedId === sub.id && (
                <div className="px-4 pb-4 pt-2 border-t border-base-800">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-base-400">Alert Threshold: </span>
                      <span className="text-base-200">{sub.alert_threshold}% change</span>
                    </div>
                    <div>
                      <span className="text-base-400">Notification: </span>
                      <span className="text-base-200 capitalize">{sub.notification_channel}</span>
                    </div>
                    {sub.webhook_url && (
                      <div className="col-span-2">
                        <span className="text-base-400">Webhook: </span>
                        <span className="text-base-200 font-mono text-xs break-all">
                          {sub.webhook_url}
                        </span>
                      </div>
                    )}
                    {sub.last_checked_at && (
                      <div>
                        <span className="text-base-400">Last Checked: </span>
                        <span className="text-base-200">
                          {new Date(sub.last_checked_at).toLocaleDateString()}
                        </span>
                      </div>
                    )}
                    <div>
                      <span className="text-base-400">Created: </span>
                      <span className="text-base-200">
                        {new Date(sub.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  {sub.description && (
                    <p className="mt-3 text-sm text-base-300">{sub.description}</p>
                  )}

                  {/* Actions */}
                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {onToggle && (
                        <button
                          onClick={() => onToggle(sub.id, !sub.active)}
                          className={cx(
                            'px-3 py-1.5 text-xs font-medium rounded-lg transition',
                            sub.active
                              ? 'text-amber-700 bg-amber-600/10 hover:bg-amber-600/20'
                              : 'text-brand-600 bg-brand-600/10 hover:bg-brand-600/20',
                          )}
                        >
                          {sub.active ? 'Pause Monitoring' : 'Resume Monitoring'}
                        </button>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {onEdit && (
                        <button
                          onClick={() => onEdit(sub)}
                          className="px-3 py-1.5 text-xs font-medium text-base-300 hover:text-base-100 bg-base-800/50 hover:bg-base-800 rounded-lg transition"
                        >
                          Edit
                        </button>
                      )}
                      {onDelete && (
                        <button
                          onClick={() => {
                            if (window.confirm('Are you sure you want to delete this subscription?')) {
                              onDelete(sub.id)
                            }
                          }}
                          className="px-3 py-1.5 text-xs font-medium text-danger-600 hover:text-danger-700 bg-danger-600/10 hover:bg-danger-600/20 rounded-lg transition"
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

// Form for creating/editing subscriptions
export interface SubscriptionFormData {
  name: string
  description: string
  bbox: string
  analysis_types: AnalysisType[]
  alert_threshold: number
  notification_channel: string
  webhook_url: string
}

export interface SubscriptionFormProps {
  initialData?: Partial<SubscriptionFormData>
  onSubmit: (data: SubscriptionFormData) => void
  onCancel: () => void
  loading?: boolean
}

export function SubscriptionForm({
  initialData,
  onSubmit,
  onCancel,
  loading = false,
}: SubscriptionFormProps) {
  const [formData, setFormData] = useState<SubscriptionFormData>({
    name: initialData?.name || '',
    description: initialData?.description || '',
    bbox: initialData?.bbox || '[-62.0, -3.1, -61.8, -2.9]',
    analysis_types: initialData?.analysis_types || ['deforestation'],
    alert_threshold: initialData?.alert_threshold || 5,
    notification_channel: initialData?.notification_channel || 'email',
    webhook_url: initialData?.webhook_url || '',
  })

  const analysisTypeOptions: { value: AnalysisType; label: string }[] = [
    { value: 'deforestation', label: 'Deforestation Detection' },
    { value: 'ice_melting', label: 'Arctic Ice Melting' },
    { value: 'flooding', label: 'Flood Detection' },
  ]

  const handleAnalysisTypeToggle = (type: AnalysisType) => {
    setFormData((prev) => ({
      ...prev,
      analysis_types: prev.analysis_types.includes(type)
        ? prev.analysis_types.filter((t) => t !== type)
        : [...prev.analysis_types, type],
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-base-200 mb-1">
          Name (optional)
        </label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          placeholder="e.g., Amazon Basin Watch"
          className="w-full rounded-lg border border-base-700 bg-base-900/50 px-3 py-2 text-sm text-base-100 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-base-200 mb-1">
          Bounding Box
        </label>
        <input
          type="text"
          value={formData.bbox}
          onChange={(e) => setFormData({ ...formData, bbox: e.target.value })}
          placeholder="[minLon, minLat, maxLon, maxLat]"
          className="w-full rounded-lg border border-base-700 bg-base-900/50 px-3 py-2 text-sm text-base-100 font-mono focus:outline-none focus:ring-2 focus:ring-brand-500/40"
          required
        />
        <p className="mt-1 text-xs text-base-400">
          Format: [minLongitude, minLatitude, maxLongitude, maxLatitude]
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-base-200 mb-2">
          Analysis Types
        </label>
        <div className="flex flex-wrap gap-2">
          {analysisTypeOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleAnalysisTypeToggle(option.value)}
              className={cx(
                'px-3 py-1.5 text-xs font-medium rounded-lg border transition',
                formData.analysis_types.includes(option.value)
                  ? 'bg-brand-600/20 border-brand-600/40 text-brand-600'
                  : 'bg-base-800/50 border-base-700 text-base-400 hover:text-base-200',
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-base-200 mb-1">
          Alert Threshold (%)
        </label>
        <input
          type="number"
          value={formData.alert_threshold}
          onChange={(e) => setFormData({ ...formData, alert_threshold: Number(e.target.value) })}
          min={0}
          max={100}
          step={0.5}
          className="w-full rounded-lg border border-base-700 bg-base-900/50 px-3 py-2 text-sm text-base-100 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        />
        <p className="mt-1 text-xs text-base-400">
          Alert when change exceeds this percentage
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-base-200 mb-1">
          Notification Channel
        </label>
        <select
          value={formData.notification_channel}
          onChange={(e) => setFormData({ ...formData, notification_channel: e.target.value })}
          className="w-full rounded-lg border border-base-700 bg-base-900/50 px-3 py-2 text-sm text-base-100 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        >
          <option value="email">Email</option>
          <option value="webhook">Webhook</option>
          <option value="api">API (polling)</option>
        </select>
      </div>

      {formData.notification_channel === 'webhook' && (
        <div>
          <label className="block text-sm font-medium text-base-200 mb-1">
            Webhook URL
          </label>
          <input
            type="url"
            value={formData.webhook_url}
            onChange={(e) => setFormData({ ...formData, webhook_url: e.target.value })}
            placeholder="https://your-server.com/webhook"
            className="w-full rounded-lg border border-base-700 bg-base-900/50 px-3 py-2 text-sm text-base-100 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
          />
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-base-200 mb-1">
          Description (optional)
        </label>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
          rows={2}
          placeholder="Notes about this monitoring region..."
          className="w-full rounded-lg border border-base-700 bg-base-900/50 px-3 py-2 text-sm text-base-100 focus:outline-none focus:ring-2 focus:ring-brand-500/40"
        />
      </div>

      <div className="flex items-center justify-end gap-3 pt-4">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 text-sm font-medium text-base-300 hover:text-base-100 transition"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={loading || formData.analysis_types.length === 0}
          className="px-4 py-2 text-sm font-medium text-white bg-brand-600 hover:bg-brand-500 rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Saving...' : 'Save Subscription'}
        </button>
      </div>
    </form>
  )
}
