import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { getSessionRegions, getSessions, startInference, stopInference } from '../api/client'
import { INFERENCE_DEFAULTS } from '../constants'
import type { ParkingRegion } from '../types'
import { useInference } from './InferenceContext'

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

interface ControlContextValue {
  sessions: string[]
  selected: string
  setSelected: (id: string) => void
  stride: string
  setStride: (v: string) => void
  voteRadius: string
  setVoteRadius: (v: string) => void
  voteFrameStep: string
  setVoteFrameStep: (v: string) => void
  conf: string
  setConf: (v: string) => void
  regions: ParkingRegion[]
  selectedIndices: number[]
  setSelectedIndices: (indices: number[]) => void
  showSelector: boolean
  setShowSelector: (open: boolean) => void
  error: string | null
  isRunning: boolean
  isStarting: boolean
  normalizeStride: () => void
  normalizeVoteRadius: () => void
  normalizeVoteFrameStep: () => void
  normalizeConf: () => void
  resetParamsToDefault: () => void
  handleStart: () => Promise<void>
  handleStop: () => Promise<void>
}

const ControlContext = createContext<ControlContextValue | null>(null)

export function ControlProvider({ children }: { children: ReactNode }) {
  const { status, notifyStart } = useInference()

  const [sessions, setSessions] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [stride, setStride] = useState(String(INFERENCE_DEFAULTS.stride))
  const [voteRadius, setVoteRadius] = useState(String(INFERENCE_DEFAULTS.voteRadius))
  const [voteFrameStep, setVoteFrameStep] = useState(String(INFERENCE_DEFAULTS.voteFrameStep))
  const [conf, setConf] = useState(String(INFERENCE_DEFAULTS.conf))
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
        setSelected((prev) => prev || (ids.length > 0 ? ids[0] : ''))
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

  const handleStart = useCallback(async () => {
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
    try {
      strideValue = parseNumberField(stride, { name: 'Stride', min: 1, integer: true })
      voteRadiusValue = parseNumberField(voteRadius, { name: 'Vote radius', min: 0, integer: true })
      voteFrameStepValue = parseNumberField(voteFrameStep, {
        name: 'Vote step',
        min: 1,
        integer: true,
      })
      confValue = parseNumberField(conf, { name: 'Conf', min: 0, max: 1 })
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
        ...(keptRegions ? { regions: keptRegions } : {}),
      })
      notifyStart(selected)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start inference')
    }
  }, [
    selected,
    regions,
    selectedIndices,
    stride,
    voteRadius,
    voteFrameStep,
    conf,
    notifyStart,
  ])

  const handleStop = useCallback(async () => {
    setError(null)
    try {
      await stopInference()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to stop inference')
    }
  }, [])

  const resetParamsToDefault = useCallback(() => {
    setStride(String(INFERENCE_DEFAULTS.stride))
    setVoteRadius(String(INFERENCE_DEFAULTS.voteRadius))
    setVoteFrameStep(String(INFERENCE_DEFAULTS.voteFrameStep))
    setConf(String(INFERENCE_DEFAULTS.conf))
    setError(null)
  }, [])

  const value = useMemo<ControlContextValue>(
    () => ({
      sessions,
      selected,
      setSelected,
      stride,
      setStride,
      voteRadius,
      setVoteRadius,
      voteFrameStep,
      setVoteFrameStep,
      conf,
      setConf,
      regions,
      selectedIndices,
      setSelectedIndices,
      showSelector,
      setShowSelector,
      error,
      isRunning,
      isStarting,
      normalizeStride: () =>
        normalizeOnBlur(stride, setStride, INFERENCE_DEFAULTS.stride, { min: 1, integer: true }),
      normalizeVoteRadius: () =>
        normalizeOnBlur(voteRadius, setVoteRadius, INFERENCE_DEFAULTS.voteRadius, {
          min: 0,
          integer: true,
        }),
      normalizeVoteFrameStep: () =>
        normalizeOnBlur(voteFrameStep, setVoteFrameStep, INFERENCE_DEFAULTS.voteFrameStep, {
          min: 1,
          integer: true,
        }),
      normalizeConf: () =>
        normalizeOnBlur(conf, setConf, INFERENCE_DEFAULTS.conf, { min: 0, max: 1 }),
      resetParamsToDefault,
      handleStart,
      handleStop,
    }),
    [
      sessions,
      selected,
      stride,
      voteRadius,
      voteFrameStep,
      conf,
      regions,
      selectedIndices,
      showSelector,
      error,
      isRunning,
      isStarting,
      resetParamsToDefault,
      handleStart,
      handleStop,
    ],
  )

  return <ControlContext.Provider value={value}>{children}</ControlContext.Provider>
}

export function useControl(): ControlContextValue {
  const ctx = useContext(ControlContext)
  if (!ctx) {
    throw new Error('useControl must be used within ControlProvider')
  }
  return ctx
}
