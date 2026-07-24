// Firebase initialization.
// Reads config from Vite env vars (VITE_FIREBASE_*). If they are not set,
// the app runs in "demo mode": auth UI renders but sign-in is disabled
// with a friendly notice instead of crashing.
import { initializeApp, type FirebaseApp } from 'firebase/app'
import { getAuth, GoogleAuthProvider, type Auth } from 'firebase/auth'

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
}

export const firebaseEnabled = Boolean(config.apiKey && config.projectId && config.appId)

let app: FirebaseApp | null = null
let authInstance: Auth | null = null

if (firebaseEnabled) {
  app = initializeApp(config)
  authInstance = getAuth(app)
}

export const auth = authInstance
export const googleProvider = new GoogleAuthProvider()
