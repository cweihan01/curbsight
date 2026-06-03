import { BrowserRouter, NavLink, useLocation } from 'react-router-dom'
import { ControlProvider } from './context/ControlContext'
import { InferenceProvider } from './context/InferenceContext'
import { DashboardPage } from './pages/DashboardPage'
import { SignPage } from './pages/SignPage'

function NavBar() {
  const base = 'px-4 py-2 text-sm font-medium rounded-lg transition-colors'
  const active = `${base} bg-slate-700 text-slate-100`
  const inactive = `${base} text-slate-400 hover:text-slate-200 hover:bg-slate-800`

  return (
    <header className="flex items-center justify-between mb-6">
      <h1 className="text-xl font-semibold tracking-tight text-slate-100">CurbSight</h1>
      <nav className="flex gap-1 bg-slate-900 border border-slate-700 rounded-xl p-1">
        <NavLink to="/" end className={({ isActive }) => (isActive ? active : inactive)}>
          Dashboard
        </NavLink>
        <NavLink to="/sign" className={({ isActive }) => (isActive ? active : inactive)}>
          Street Sign
        </NavLink>
      </nav>
    </header>
  )
}

/** Keep both tab views mounted so WebSocket, controls, and live frames persist across navigation. */
function AppShell() {
  const { pathname } = useLocation()
  const showDashboard = pathname === '/' || pathname === ''
  const showSign = pathname === '/sign'

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6 flex flex-col flex-1">
      <NavBar />
      <div className={showDashboard ? 'flex flex-1 flex-col min-h-0' : 'hidden'} aria-hidden={!showDashboard}>
        <DashboardPage />
      </div>
      <div className={showSign ? 'flex flex-1 flex-col min-h-0' : 'hidden'} aria-hidden={!showSign}>
        <SignPage />
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <InferenceProvider>
        <ControlProvider>
          <AppShell />
        </ControlProvider>
      </InferenceProvider>
    </BrowserRouter>
  )
}
