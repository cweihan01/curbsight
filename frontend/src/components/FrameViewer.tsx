import type { InferenceEvent } from '../types'

interface Props {
  currentFrame: string | null
  currentEvent: InferenceEvent | null
  isLoading: boolean
}

export function FrameViewer({ currentFrame, currentEvent, isLoading }: Props) {
  return (
    <div className="bg-slate-800 rounded-xl p-4 flex flex-col gap-3">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Live Frame</h2>
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
      {currentEvent && !isLoading && (
        <p className="text-slate-600 text-xs">
          Frame {currentEvent.frame_index} · Inference {currentEvent.inference_index} · {currentEvent.timestamp_iso.slice(11, 19)}
        </p>
      )}
    </div>
  )
}
