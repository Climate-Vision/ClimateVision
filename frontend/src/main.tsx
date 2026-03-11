import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { ToastProvider } from './contexts/ToastContext'
import { AppProvider } from './contexts/AppContext'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
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
        <BrowserRouter>
          <ErrorBoundary section="App">
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<NewAnalysis />} />
                <Route path="/upload" element={<Upload />} />
                <Route path="/runs" element={<RunHistory />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/settings" element={<Settings />} />
              </Route>
            </Routes>
          </ErrorBoundary>
        </BrowserRouter>
      </ToastProvider>
    </AppProvider>
  </React.StrictMode>,
)
