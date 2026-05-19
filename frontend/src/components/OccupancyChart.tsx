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
}

export function OccupancyChart({ events, isLoading }: Props) {
  const data = events.slice(-MAX_DISPLAY).map((e) => ({
    time: e.timestamp_iso.slice(11, 19),
    occupancy: Math.round(e.occupancy_ratio * 100),
  }))

  if (data.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-6 flex items-center justify-center min-h-[200px]">
        <p className="text-slate-500 text-sm">
          {isLoading ? 'Waiting for first event...' : 'No data yet'}
        </p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-xl p-6">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-4">
        Occupancy Over Time
      </h2>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="time"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(value) => [`${value}%`, 'Occupancy']}
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
      </ResponsiveContainer>
    </div>
  )
}
