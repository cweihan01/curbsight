import type { InferenceEvent } from '../types'
import { OccupancyChart } from './OccupancyChart'

interface Props {
  latest: InferenceEvent | null
  events: InferenceEvent[]
  isLoading: boolean
}

function ratioColor(ratio: number): string {
  if (ratio >= 0.9) return 'text-red-400'
  if (ratio >= 0.7) return 'text-yellow-400'
  return 'text-green-400'
}

export function OccupancyGauge({ latest, events, isLoading }: Props) {
  const gaugeBody = !latest ? (
    <div className="flex flex-col items-center justify-center gap-2 min-h-[80px]">
      {isLoading ? (
        <>
          <div className="w-8 h-8 border-2 border-slate-600 border-t-green-400 rounded-full animate-spin" />
          <p className="text-slate-500 text-sm">Waiting for first event...</p>
        </>
      ) : (
        <p className="text-slate-500 text-sm">Waiting for inference...</p>
      )}
    </div>
  ) : (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <span
          className={`text-5xl font-bold tabular-nums leading-none ${ratioColor(latest.occupancy_ratio)}`}
        >
          {Math.round(latest.occupancy_ratio * 100)}%
        </span>
        <span className="text-slate-400 text-xl">
          {latest.occupied_spots} / {latest.total_spots} spots
        </span>
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-base">
        <div>
          <span className="text-red-400 font-semibold">{latest.occupied_spots}</span>
          <span className="text-slate-500 ml-1">occupied</span>
        </div>
        <div>
          <span className="text-green-400 font-semibold">{latest.available_spots}</span>
          <span className="text-slate-500 ml-1">available</span>
        </div>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            latest.occupancy_ratio >= 0.9
              ? 'bg-red-400'
              : latest.occupancy_ratio >= 0.7
                ? 'bg-yellow-400'
                : 'bg-green-400'
          }`}
          style={{ width: `${Math.round(latest.occupancy_ratio * 100)}%` }}
        />
      </div>
    </div>
  )

  return (
    <div className="bg-slate-800 rounded-xl p-4 flex flex-col h-full min-h-0">
      <div className="shrink-0 flex flex-col gap-3 pb-3">
        <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">
          Occupancy
        </h2>
        {gaugeBody}
      </div>
      <div className="border-t border-slate-700 pt-3 flex flex-col flex-1 min-h-0 gap-2">
        <h3 className="text-slate-500 text-xs font-medium uppercase tracking-wider shrink-0">
          History
        </h3>
        <OccupancyChart events={events} isLoading={isLoading} embedded />
      </div>
    </div>
  )
}
