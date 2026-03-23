import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { useApp } from '../../contexts/AppContext'

export function Layout() {
  const { theme, toggleTheme } = useApp()

  return (
    <div className="flex h-full min-h-screen bg-cv-bg">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar theme={theme} onToggleTheme={toggleTheme} />
        <main className="flex-1 overflow-auto pb-20 md:pb-0">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
