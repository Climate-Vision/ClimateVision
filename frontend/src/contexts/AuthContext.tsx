import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  signOut as firebaseSignOut,
  updateProfile,
  type User,
} from 'firebase/auth'
import { auth, googleProvider, firebaseEnabled } from '../lib/firebase'

interface AuthContextValue {
  user: User | null
  loading: boolean
  /** false when VITE_FIREBASE_* env vars are missing (demo mode) */
  enabled: boolean
  signIn: (email: string, password: string) => Promise<void>
  signUp: (name: string, email: string, password: string) => Promise<void>
  signInWithGoogle: () => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

const DEMO_ERROR =
  'Authentication is not configured yet. Add your Firebase keys to frontend/.env (see .env.example).'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(firebaseEnabled)

  useEffect(() => {
    if (!auth) return
    const unsubscribe = onAuthStateChanged(auth, (u) => {
      setUser(u)
      setLoading(false)
    })
    return unsubscribe
  }, [])

  const signIn = async (email: string, password: string) => {
    if (!auth) throw new Error(DEMO_ERROR)
    await signInWithEmailAndPassword(auth, email, password)
  }

  const signUp = async (name: string, email: string, password: string) => {
    if (!auth) throw new Error(DEMO_ERROR)
    const cred = await createUserWithEmailAndPassword(auth, email, password)
    if (name.trim()) await updateProfile(cred.user, { displayName: name.trim() })
  }

  const signInWithGoogle = async () => {
    if (!auth) throw new Error(DEMO_ERROR)
    await signInWithPopup(auth, googleProvider)
  }

  const signOut = async () => {
    if (!auth) return
    await firebaseSignOut(auth)
  }

  return (
    <AuthContext.Provider
      value={{ user, loading, enabled: firebaseEnabled, signIn, signUp, signInWithGoogle, signOut }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
