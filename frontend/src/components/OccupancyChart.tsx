import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { InferenceEvent } from '../types'

const MAX_DISPLAY = 100

interface Props {
  events: InferenceEvent[]
  isLoading: boolean
  /** When true, render only the chart (no outer card). */
  embedded?: boolean
}

export function OccupancyChart({ events, isLoading, embedded = false }: Props) {
  const data = events.slice(-MAX_DISPLAY).map((e) => ({
    time: e.timestamp_iso.slice(11, 19),
    occupancy: Math.round(e.occupancy_ratio * 100),
  }))

  const chartMargin = embedded
    ? { top: 8, right: 12, left: -12, bottom: 8 }
    : undefined

  const lineChart = (
    <LineChart data={data} margin={chartMargin}>
      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
      <XAxis
        dataKey="time"
        tick={{ fill: '#94a3b8', fontSize: 11 }}
        tickMargin={8}
        interval="preserveStartEnd"
      />
      <YAxis
        domain={[0, 100]}
        tick={{ fill: '#94a3b8', fontSize: 11 }}
        tickMargin={8}
        tickFormatter={(v) => `${v}%`}
      />
      <Tooltip
        allowEscapeViewBox={{ x: false, y: false }}
        wrapperStyle={{ pointerEvents: 'none', outline: 'none', zIndex: 1 }}
        contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
        labelStyle={{ color: '#94a3b8' }}
        formatter={(value) => [`${value}%`, 'Occupancy']}
        isAnimationActive={false}
      />
      <Line
        type="monotone"
        dataKey="occupancy"
        stroke="#4ade80"
        strokeWidth={2}
        dot={false}
        isAnimationActive={false}
      />
    </LineChart>
  )

  const emptyMessage = isLoading ? 'Waiting for first event...' : 'No data yet'

  if (embedded) {
    if (data.length === 0) {
      return (
        <div className="flex-1 min-h-0 flex items-center justify-center">
          <p className="text-slate-500 text-sm">{emptyMessage}</p>
        </div>
      )
    }
    return (
      <div className="flex-1 min-h-0 overflow-hidden">
        <ResponsiveContainer width="100%" height="100%">
          {lineChart}
        </ResponsiveContainer>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-4 flex items-center justify-center min-h-[140px]">
        <p className="text-slate-500 text-sm">{emptyMessage}</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-xl p-4">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-2">
        Occupancy Over Time
      </h2>
      <ResponsiveContainer width="100%" height={180}>
        {lineChart}
      </ResponsiveContainer>
    </div>
  )
}
