import type { InferenceState } from '../types'

const config: Record<InferenceState, { label: string; classes: string }> = {
  running: { label: 'Running', classes: 'bg-green-500/20 text-green-400 border-green-500/30' },
  started: { label: 'Starting', classes: 'bg-blue-500/20 text-blue-400 border-blue-500/30' },
  stopped: { label: 'Stopped', classes: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
  idle:    { label: 'Idle',    classes: 'bg-slate-500/20 text-slate-400 border-slate-500/30' },
}

interface Props {
  status: InferenceState
  connected: boolean
}

export function StatusBadge({ status, connected }: Props) {
  const { label, classes } = config[status]
  return (
    <div className="flex items-center gap-2">
      {!connected && (
        <span className="text-xs border rounded-full px-2 py-0.5 bg-red-500/20 text-red-400 border-red-500/30">
          Disconnected
        </span>
      )}
      <span className={`text-xs border rounded-full px-2 py-0.5 ${classes}`}>
        {label}
      </span>
    </div>
  )
}
