import { useEffect } from 'react'
import type { InferenceEvent } from '../types'

interface FrameViewerProps {
  currentFrame: string | null
  currentEvent: InferenceEvent | null
  isLoading: boolean
  currentIndex: number
  frameCount: number
  isLive: boolean
  canGoPrev: boolean
  canGoNext: boolean
  onPrev: () => void
  onNext: () => void
  onGoToIndex: (index: number) => void
  onLatest: () => void
}

export function FrameViewer({
  currentFrame,
  currentEvent,
  isLoading,
  currentIndex,
  frameCount,
  isLive,
  canGoPrev,
  canGoNext,
  onPrev,
  onNext,
  onGoToIndex,
  onLatest,
}: FrameViewerProps) {
  const showNav = frameCount > 0 && !isLoading

  useEffect(() => {
    if (!showNav) return
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'ArrowLeft') onPrev()
      else if (e.key === 'ArrowRight') onNext()
      else if (e.key === 'End') onLatest()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [showNav, onPrev, onNext, onLatest])

  return (
    <div className="bg-slate-800 rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Live Frame</h2>
        {showNav && isLive && (
          <span className="text-xs font-medium text-green-400 bg-green-400/10 px-2 py-0.5 rounded-full">
            Live
          </span>
        )}
      </div>

      <div className="relative w-full aspect-video bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center">
        {isLoading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-slate-600 border-t-green-400 rounded-full animate-spin" />
            <p className="text-slate-500 text-sm">Initializing inference...</p>
          </div>
        ) : currentFrame ? (
          <img
            key={currentFrame}
            src={currentFrame}
            alt="Annotated parking frame"
            className="w-full h-full object-contain"
          />
        ) : (
          <p className="text-slate-600 text-sm">No frame available</p>
        )}
      </div>

      {showNav && (
        <div className="flex flex-col gap-2">
          <input
            type="range"
            min={0}
            max={Math.max(0, frameCount - 1)}
            value={currentIndex}
            onChange={(e) => onGoToIndex(Number(e.target.value))}
            className="w-full accent-green-500"
            aria-label="Frame position"
          />
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={onPrev}
              disabled={!canGoPrev}
              className="px-3 py-1.5 text-sm rounded-lg bg-slate-700 text-slate-200 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              ← Prev
            </button>
            <span className="text-slate-400 text-xs tabular-nums">
              {currentIndex + 1} / {frameCount}
            </span>
            <button
              type="button"
              onClick={onNext}
              disabled={!canGoNext}
              className="px-3 py-1.5 text-sm rounded-lg bg-slate-700 text-slate-200 hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Next →
            </button>
          </div>
          {!isLive && (
            <button
              type="button"
              onClick={onLatest}
              className="w-full py-1.5 text-sm rounded-lg bg-green-600/20 text-green-400 hover:bg-green-600/30 border border-green-600/30"
            >
              Jump to live
            </button>
          )}
        </div>
      )}

      {currentEvent && !isLoading && (
        <p className="text-slate-600 text-xs">
          Frame {currentEvent.frame_index} · Inference {currentEvent.inference_index} ·{' '}
          {currentEvent.timestamp_iso.slice(11, 19)}
        </p>
      )}
    </div>
  )
}
