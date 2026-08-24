import { Component, ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'

interface Props { children: ReactNode; section?: string }
interface State { hasError: boolean; error?: Error }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
        <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mb-4">
          <AlertTriangle className="w-6 h-6 text-red-600" />
        </div>
        <h3 className="text-base font-semibold text-cv-text-primary mb-1">
          Something went wrong{this.props.section ? ` in ${this.props.section}` : ''}
        </h3>
        <p className="text-sm text-cv-text-secondary mb-4">
          {this.state.error?.message ?? 'An unexpected error occurred'}
        </p>
        <button
          onClick={() => this.setState({ hasError: false, error: undefined })}
          className="px-4 py-2 rounded-lg bg-cv-primary-muted text-cv-primary text-sm font-medium hover:bg-green-800/40 transition"
        >
          Retry
        </button>
      </div>
    )
  }
}
