import { useEffect, useState } from 'react'
import { getSessionRegions, getSessions, startInference, stopInference } from '../api/client'
import type { InferenceState, ParkingRegion } from '../types'
import { RegionSelector } from './RegionSelector'

interface ControlPanelProps {
  status: InferenceState
  notifyStart: (sessionId: string) => void
}

export function ControlPanel({ status, notifyStart }: ControlPanelProps) {
  const [sessions, setSessions] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [stride, setStride] = useState(60)
  const [conf, setConf] = useState(0.1)
  const [iou, setIou] = useState(0.7)
  const [error, setError] = useState<string | null>(null)

  const [regions, setRegions] = useState<ParkingRegion[]>([])
  const [selectedIndices, setSelectedIndices] = useState<number[]>([])
  const [showSelector, setShowSelector] = useState(false)

  const isRunning = status === 'running' || status === 'started'
  const isStarting = status === 'started'

  useEffect(() => {
    getSessions()
      .then((ids) => {
        setSessions(ids)
        if (ids.length > 0) setSelected(ids[0])
      })
      .catch(() => setError('Could not load sessions — is the backend running?'))
  }, [])

  useEffect(() => {
    if (!selected) {
      setRegions([])
      setSelectedIndices([])
      return
    }
    let cancelled = false
    getSessionRegions(selected)
      .then((regs) => {
        if (cancelled) return
        setRegions(regs)
        setSelectedIndices(regs.map((_, i) => i))
      })
      .catch(() => {
        if (cancelled) return
        setRegions([])
        setSelectedIndices([])
      })
    return () => {
      cancelled = true
    }
  }, [selected])

  async function handleStart() {
    if (!selected) return
    if (regions.length > 0 && selectedIndices.length === 0) {
      setError('Select at least one parking box before starting.')
      return
    }
    setError(null)
    const keptRegions =
      regions.length > 0 ? selectedIndices.map((i) => regions[i]) : undefined
    try {
      await startInference({
        session_id: selected,
        stride,
        publish_every: 1,
        conf,
        iou,
        ...(keptRegions ? { regions: keptRegions } : {}),
      })
      notifyStart(selected)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start inference')
    }
  }

  async function handleStop() {
    setError(null)
    try {
      await stopInference()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to stop inference')
    }
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6 flex flex-col gap-5">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Controls</h2>

      <div className="flex flex-col gap-2">
        <label className="text-slate-400 text-xs">Session</label>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={isRunning}
          className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
        >
          {sessions.length === 0 && <option value="">No sessions found</option>}
          {sessions.map((id) => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-slate-400 text-xs">Parking boxes</label>
        <button
          onClick={() => setShowSelector(true)}
          disabled={isRunning || regions.length === 0}
          className="bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:hover:bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 transition-colors text-left"
        >
          {regions.length === 0
            ? 'No boxes for session'
            : `Select boxes (${selectedIndices.length}/${regions.length} kept)`}
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-slate-400 text-xs">Stride</label>
          <input
            type="number"
            min={1}
            value={stride}
            onChange={(e) => setStride(Number(e.target.value))}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-slate-400 text-xs">Conf</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={conf}
            onChange={(e) => setConf(Number(e.target.value))}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-slate-400 text-xs">IoU</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={iou}
            onChange={(e) => setIou(Number(e.target.value))}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
      </div>

      {error && (
        <p className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      <div className="flex gap-3">
        <button
          onClick={handleStart}
          disabled={isRunning || !selected}
          className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg py-2 text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          {isStarting && (
            <span className="w-3 h-3 border border-white/40 border-t-white rounded-full animate-spin" />
          )}
          {isStarting ? 'Starting...' : 'Start'}
        </button>
        <button
          onClick={handleStop}
          disabled={!isRunning}
          className="flex-1 bg-red-600 hover:bg-red-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg py-2 text-sm font-medium transition-colors"
        >
          Stop
        </button>
      </div>

      {showSelector && selected && (
        <RegionSelector
          sessionId={selected}
          regions={regions}
          initialSelected={selectedIndices}
          onApply={(next) => {
            setSelectedIndices(next)
            setShowSelector(false)
          }}
          onClose={() => setShowSelector(false)}
        />
      )}
    </div>
  )
}
