import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
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
        <NavLink to="/" end className={({ isActive }) => isActive ? active : inactive}>
          Dashboard
        </NavLink>
        <NavLink to="/sign" className={({ isActive }) => isActive ? active : inactive}>
          Street Sign
        </NavLink>
      </nav>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-900 text-slate-100 p-6 flex flex-col">
        <NavBar />
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/sign" element={<SignPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
