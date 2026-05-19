import { useEffect, useState } from 'react'
import { getVideos, startInference, stopInference } from '../api/client'
import type { InferenceState } from '../types'

interface Props {
  status: InferenceState
  onStart: () => void
  notifyStart: () => void
}

export function ControlPanel({ status, onStart, notifyStart }: Props) {
  const [videos, setVideos] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [stride, setStride] = useState(30)
  const [conf, setConf] = useState(0.1)
  const [iou, setIou] = useState(0.7)
  const [error, setError] = useState<string | null>(null)

  const isRunning = status === 'running' || status === 'started'
  const isStarting = status === 'started'

  useEffect(() => {
    getVideos()
      .then((v) => {
        setVideos(v)
        if (v.length > 0) setSelected(v[0])
      })
      .catch(() => setError('Could not load videos — is the backend running?'))
  }, [])

  async function handleStart() {
    if (!selected) return
    setError(null)
    try {
      await startInference({ video_filename: selected, stride, publish_every: 1, conf, iou })
      notifyStart()
      onStart()
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
        <label className="text-slate-400 text-xs">Video</label>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={isRunning}
          className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
        >
          {videos.length === 0 && <option value="">No videos found</option>}
          {videos.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
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
    </div>
  )
}
