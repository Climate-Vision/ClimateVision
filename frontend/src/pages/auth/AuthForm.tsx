import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Mail, Lock, User, Eye, EyeOff, Loader2, AlertTriangle } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'

function friendlyError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err)
  if (msg.includes('auth/invalid-credential') || msg.includes('auth/wrong-password'))
    return 'Incorrect email or password.'
  if (msg.includes('auth/user-not-found')) return 'No account found with that email.'
  if (msg.includes('auth/email-already-in-use')) return 'An account with that email already exists.'
  if (msg.includes('auth/weak-password')) return 'Password should be at least 6 characters.'
  if (msg.includes('auth/invalid-email')) return 'That email address looks invalid.'
  if (msg.includes('auth/popup-closed-by-user')) return 'Google sign-in was cancelled.'
  return msg
}

function GoogleIcon() {
  return (
    <svg className="h-5 w-5" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.16-3.16A10.96 10.96 0 0 0 12 1 11 11 0 0 0 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
      />
    </svg>
  )
}

const inputClass =
  'w-full rounded-xl border border-cv-border bg-cv-surface py-3 pl-11 pr-11 text-sm text-cv-text-primary placeholder-cv-text-dim outline-none transition focus:border-cv-primary focus:ring-1 focus:ring-cv-primary'

export function AuthForm({ mode }: { mode: 'signin' | 'signup' }) {
  const { enabled, signIn, signUp, signInWithGoogle } = useAuth()
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setBusy(true)
    try {
      if (mode === 'signup') await signUp(name, email, password)
      else await signIn(email, password)
      navigate('/app')
    } catch (err) {
      setError(friendlyError(err))
    } finally {
      setBusy(false)
    }
  }

  const google = async () => {
    setError(null)
    setBusy(true)
    try {
      await signInWithGoogle()
      navigate('/app')
    } catch (err) {
      setError(friendlyError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      {!enabled && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-cv-warning/40 bg-cv-warning-muted/30 p-4 text-sm text-cv-text-secondary">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-cv-warning" />
          <span>
            Demo mode — Firebase keys are not configured yet. Add <code>VITE_FIREBASE_*</code>{' '}
            values to <code>frontend/.env</code> to enable real sign-in.
          </span>
        </div>
      )}

      <button
        type="button"
        onClick={google}
        disabled={busy}
        className="flex w-full items-center justify-center gap-3 rounded-xl border border-cv-border bg-cv-card py-3 text-sm font-semibold text-cv-text-primary transition hover:border-cv-border-strong hover:bg-cv-card-hover disabled:opacity-50"
      >
        <GoogleIcon />
        Continue with Google
      </button>

      <div className="my-6 flex items-center gap-4">
        <div className="h-px flex-1 bg-cv-border" />
        <span className="text-xs uppercase tracking-wider text-cv-text-dim">or</span>
        <div className="h-px flex-1 bg-cv-border" />
      </div>

      <form onSubmit={submit} className="space-y-4">
        {mode === 'signup' && (
          <div className="relative">
            <User className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-cv-text-dim" />
            <input
              type="text"
              placeholder="Full name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="name"
              className={inputClass}
            />
          </div>
        )}
        <div className="relative">
          <Mail className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-cv-text-dim" />
          <input
            type="email"
            required
            placeholder="Email address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            className={inputClass}
          />
        </div>
        <div className="relative">
          <Lock className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-cv-text-dim" />
          <input
            type={showPassword ? 'text' : 'password'}
            required
            minLength={6}
            placeholder={mode === 'signup' ? 'Password (min. 6 characters)' : 'Password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
            className={inputClass}
          />
          <button
            type="button"
            onClick={() => setShowPassword((s) => !s)}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-cv-text-dim transition hover:text-cv-text-secondary"
            aria-label={showPassword ? 'Hide password' : 'Show password'}
          >
            {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>

        {error && (
          <p className="rounded-lg border border-cv-danger/40 bg-cv-danger-muted/30 px-4 py-3 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-cv-primary py-3 text-sm font-semibold text-cv-bg transition hover:bg-cv-primary-hover disabled:opacity-50"
        >
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}
          {mode === 'signup' ? 'Create account' : 'Sign in'}
        </button>
      </form>
    </div>
  )
}
