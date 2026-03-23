export function SkeletonCard() {
  return (
    <div className="rounded-xl border border-cv-border bg-cv-card p-5 space-y-3">
      <div className="skeleton h-28 rounded-lg" />
      <div className="flex items-center justify-between">
        <div className="skeleton h-4 w-12 rounded" />
        <div className="skeleton h-5 w-20 rounded-full" />
      </div>
      <div className="skeleton h-4 w-36 rounded" />
      <div className="skeleton h-3 w-24 rounded" />
      <div className="skeleton h-1.5 w-full rounded-full" />
    </div>
  )
}

export function SkeletonRow() {
  return (
    <div className="flex items-center gap-4 px-4 py-3 border-b border-cv-border">
      <div className="skeleton h-4 w-8 rounded" />
      <div className="skeleton h-4 w-28 rounded" />
      <div className="skeleton h-4 w-20 rounded" />
      <div className="skeleton h-5 w-16 rounded-full ml-auto" />
    </div>
  )
}
