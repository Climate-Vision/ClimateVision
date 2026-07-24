import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { ToastProvider } from './contexts/ToastContext'
import { AppProvider } from './contexts/AppContext'
import { AuthProvider } from './contexts/AuthContext'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import Landing from './pages/Landing'
import SignIn from './pages/auth/SignIn'
import SignUp from './pages/auth/SignUp'
import NewAnalysis from './pages/NewAnalysis'
import Upload from './pages/Upload'
import RunHistory from './pages/RunHistory'
import Analytics from './pages/Analytics'
import Settings from './pages/Settings'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppProvider>
      <ToastProvider>
        <AuthProvider>
          <BrowserRouter>
            <ErrorBoundary section="App">
              <Routes>
                {/* Public marketing + auth */}
                <Route path="/" element={<Landing />} />
                <Route path="/signin" element={<SignIn />} />
                <Route path="/signup" element={<SignUp />} />

                {/* Dashboard */}
                <Route path="/app" element={<Layout />}>
                  <Route index element={<NewAnalysis />} />
                  <Route path="upload" element={<Upload />} />
                  <Route path="runs" element={<RunHistory />} />
                  <Route path="analytics" element={<Analytics />} />
                  <Route path="settings" element={<Settings />} />
                </Route>

                {/* Legacy paths → new dashboard locations */}
                <Route path="/upload" element={<Navigate to="/app/upload" replace />} />
                <Route path="/runs" element={<Navigate to="/app/runs" replace />} />
                <Route path="/analytics" element={<Navigate to="/app/analytics" replace />} />
                <Route path="/settings" element={<Navigate to="/app/settings" replace />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </ErrorBoundary>
          </BrowserRouter>
        </AuthProvider>
      </ToastProvider>
    </AppProvider>
  </React.StrictMode>,
)
