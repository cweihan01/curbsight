import { useEffect, useState } from 'react'
import { getSessionRegions, getSessions, startInference, stopInference } from '../api/client'
import type { InferenceState, ParkingRegion } from '../types'
import { INFERENCE_DEFAULTS } from '../constants'
import { RegionSelector } from './RegionSelector'

interface ControlPanelProps {
  status: InferenceState
  notifyStart: (sessionId: string) => void
}

function TooltipLabel({ label, tip }: { label: string; tip: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-slate-400 text-xs">{label}</span>
      <span className="relative group inline-flex">
        <span className="w-4 h-4 rounded-full border border-slate-500 text-slate-300 text-[10px] leading-none inline-flex items-center justify-center">
          ?
        </span>
        <span className="pointer-events-none absolute left-0 top-[125%] z-20 hidden group-hover:block w-56 max-w-[75vw] rounded-md border border-slate-600 bg-slate-900 px-2 py-1.5 text-[11px] leading-4 text-slate-200 shadow-xl">
          {tip}
        </span>
      </span>
    </div>
  )
}

export function ControlPanel({ status, notifyStart }: ControlPanelProps) {
  const [sessions, setSessions] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [stride, setStride] = useState<string>(String(INFERENCE_DEFAULTS.stride))
  const [voteRadius, setVoteRadius] = useState<string>(String(INFERENCE_DEFAULTS.voteRadius))
  const [voteFrameStep, setVoteFrameStep] = useState<string>(String(INFERENCE_DEFAULTS.voteFrameStep))
  const [conf, setConf] = useState<string>(String(INFERENCE_DEFAULTS.conf))
  const [iou, setIou] = useState<string>(String(INFERENCE_DEFAULTS.iou))
  const [error, setError] = useState<string | null>(null)

  const [regions, setRegions] = useState<ParkingRegion[]>([])
  const [selectedIndices, setSelectedIndices] = useState<number[]>([])
  const [showSelector, setShowSelector] = useState(false)

  const isRunning = status === 'running' || status === 'started'
  const isStarting = status === 'started'

  function parseNumberField(
    raw: string,
    {
      name,
      min,
      max,
      integer = false,
    }: { name: string; min: number; max?: number; integer?: boolean },
  ): number {
    const value = raw.trim()
    if (!value) throw new Error(`${name} is required.`)
    const n = Number(value)
    if (!Number.isFinite(n)) throw new Error(`${name} must be a valid number.`)
    if (integer && !Number.isInteger(n)) throw new Error(`${name} must be an integer.`)
    if (n < min) throw new Error(`${name} must be >= ${min}.`)
    if (max !== undefined && n > max) throw new Error(`${name} must be <= ${max}.`)
    return n
  }

  function normalizeOnBlur(
    raw: string,
    setter: (v: string) => void,
    fallback: number,
    opts: { min: number; max?: number; integer?: boolean },
  ) {
    try {
      const n = parseNumberField(raw, { name: 'Value', ...opts })
      setter(opts.integer ? String(Math.trunc(n)) : String(n))
    } catch {
      setter(String(fallback))
    }
  }

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
    let strideValue: number
    let voteRadiusValue: number
    let voteFrameStepValue: number
    let confValue: number
    let iouValue: number
    try {
      strideValue = parseNumberField(stride, { name: 'Stride', min: 1, integer: true })
      voteRadiusValue = parseNumberField(voteRadius, { name: 'Vote radius', min: 0, integer: true })
      voteFrameStepValue = parseNumberField(voteFrameStep, {
        name: 'Vote step',
        min: 1,
        integer: true,
      })
      confValue = parseNumberField(conf, { name: 'Conf', min: 0, max: 1 })
      iouValue = parseNumberField(iou, { name: 'IoU', min: 0, max: 1 })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Invalid inference parameters.')
      return
    }
    try {
      await startInference({
        session_id: selected,
        stride: strideValue,
        vote_radius: voteRadiusValue,
        vote_frame_step: voteFrameStepValue,
        publish_every: INFERENCE_DEFAULTS.publishEvery,
        conf: confValue,
        iou: iouValue,
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

  function resetParamsToDefault() {
    setStride(String(INFERENCE_DEFAULTS.stride))
    setVoteRadius(String(INFERENCE_DEFAULTS.voteRadius))
    setVoteFrameStep(String(INFERENCE_DEFAULTS.voteFrameStep))
    setConf(String(INFERENCE_DEFAULTS.conf))
    setIou(String(INFERENCE_DEFAULTS.iou))
    setError(null)
  }

  return (
    <div className="bg-slate-800 rounded-xl p-4 flex flex-col gap-4">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Controls</h2>

      <div className="flex flex-col gap-2">
        <TooltipLabel
          label="Session"
          tip="Select which recording and base region file to run. Changing session switches the reference frame and available parking boxes."
        />
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
        <TooltipLabel
          label="Parking boxes"
          tip="Open the selector to keep/remove predefined boxes. Kept boxes are sent to backend as regions for this run."
        />
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

      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Stride"
            tip="Run inference every N frames. Increase for less frequent updates; decrease for more frequent updates. At 30 FPS, setting stride to 120 runs inference every 4 seconds."
          />
          <input
            type="text"
            inputMode="numeric"
            value={stride}
            onChange={(e) => setStride(e.target.value)}
            onBlur={() =>
              normalizeOnBlur(stride, setStride, INFERENCE_DEFAULTS.stride, {
                min: 1,
                integer: true,
              })
            }
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Vote radius"
            tip="Number of samples on each side of the anchor frame. Higher radius smooths transient occlusions but adds more per-inference work. A vote radius of 3 corresponds to a majority vote over 7 frames per inference."
          />
          <input
            type="text"
            inputMode="numeric"
            value={voteRadius}
            onChange={(e) => setVoteRadius(e.target.value)}
            onBlur={() =>
              normalizeOnBlur(voteRadius, setVoteRadius, INFERENCE_DEFAULTS.voteRadius, {
                min: 0,
                integer: true,
              })
            }
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Vote step"
            tip="Frame gap between voting samples per inference. Increase to spread samples farther in time, making the vote more robust to brief occlusions (also requires a higher stride); decrease for tighter temporal voting. A vote step of 15 means each inference samples the frame f, f+15, f-15, f+30, f-30, etc depending on the vote radius."
          />
          <input
            type="text"
            inputMode="numeric"
            value={voteFrameStep}
            onChange={(e) => setVoteFrameStep(e.target.value)}
            onBlur={() =>
              normalizeOnBlur(voteFrameStep, setVoteFrameStep, INFERENCE_DEFAULTS.voteFrameStep, {
                min: 1,
                integer: true,
              })
            }
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Conf"
            tip="Detection confidence threshold in range [0, 1]. Increase to reduce false positives (but can miss cars); decrease to catch more detections (but may add noise)."
          />
          <input
            type="text"
            inputMode="decimal"
            value={conf}
            onChange={(e) => setConf(e.target.value)}
            onBlur={() =>
              normalizeOnBlur(conf, setConf, INFERENCE_DEFAULTS.conf, {
                min: 0,
                max: 1,
              })
            }
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="IoU"
            tip="Overlap threshold in range [0, 1]. Higher IoU keeps more overlapping detections; lower IoU suppresses duplicates more aggressively."
          />
          <input
            type="text"
            inputMode="decimal"
            value={iou}
            onChange={(e) => setIou(e.target.value)}
            onBlur={() =>
              normalizeOnBlur(iou, setIou, INFERENCE_DEFAULTS.iou, {
                min: 0,
                max: 1,
              })
            }
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

      <button
        type="button"
        onClick={resetParamsToDefault}
        disabled={isRunning}
        className="w-full bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:hover:bg-slate-700 text-slate-200 rounded-lg py-1 text-sm font-medium transition-colors border border-slate-600"
      >
        Reset to default parameters
      </button>

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
