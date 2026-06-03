import type { InferenceEvent } from '../types'

interface Props {
  latest: InferenceEvent | null
  isLoading: boolean
}

function ratioColor(ratio: number): string {
  if (ratio >= 0.9) return 'text-red-400'
  if (ratio >= 0.7) return 'text-yellow-400'
  return 'text-green-400'
}

export function OccupancyGauge({ latest, isLoading }: Props) {
  if (!latest) {
    return (
      <div className="bg-slate-800 rounded-xl p-4 flex flex-col items-center justify-center gap-2 min-h-[120px]">
        {isLoading ? (
          <>
            <div className="w-8 h-8 border-2 border-slate-600 border-t-green-400 rounded-full animate-spin" />
            <p className="text-slate-500 text-sm">Waiting for first event...</p>
          </>
        ) : (
          <p className="text-slate-500 text-sm">Waiting for inference...</p>
        )}
      </div>
    )
  }

  const { occupied_spots, available_spots, total_spots, occupancy_ratio } = latest
  const pct = Math.round(occupancy_ratio * 100)

  return (
    <div className="bg-slate-800 rounded-xl p-4 flex flex-col gap-4">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Occupancy</h2>
      <div className="flex items-end gap-3">
        <span className={`text-5xl font-bold tabular-nums ${ratioColor(occupancy_ratio)}`}>
          {pct}%
        </span>
        <span className="text-slate-400 text-lg pb-1">
          {occupied_spots} / {total_spots} spots
        </span>
      </div>
      <div className="flex gap-6 text-sm">
        <div>
          <span className="text-red-400 font-semibold">{occupied_spots}</span>
          <span className="text-slate-500 ml-1">occupied</span>
        </div>
        <div>
          <span className="text-green-400 font-semibold">{available_spots}</span>
          <span className="text-slate-500 ml-1">available</span>
        </div>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            occupancy_ratio >= 0.9 ? 'bg-red-400' : occupancy_ratio >= 0.7 ? 'bg-yellow-400' : 'bg-green-400'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
