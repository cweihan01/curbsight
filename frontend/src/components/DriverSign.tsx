import type { StreetSignRow } from '../hooks/useStreetSignBoard'

interface DriverSignProps {
  rows: StreetSignRow[]
}

function spotColor(available: number | null, isLoading: boolean): string {
  if (isLoading || available === null) return '#6b7280'
  if (available === 0) return '#ef4444'
  if (available <= 3) return '#f59e0b'
  return '#4ade80'
}

function SpotCount({ available, isLoading }: { available: number | null; isLoading: boolean }) {
  const color = spotColor(available, isLoading)
  if (isLoading) {
    return (
      <span
        style={{ color, fontFamily: 'monospace' }}
        className="text-2xl font-bold tracking-widest"
      >
        --
      </span>
    )
  }
  return (
    <span
      style={{ color, fontFamily: 'monospace' }}
      className="text-2xl font-bold tabular-nums tracking-widest"
    >
      {available ?? '--'}
    </span>
  )
}

export function DriverSign({ rows }: DriverSignProps) {
  const liveRow = rows.find((r) => r.isLive)
  const footerTotal = liveRow?.totalSpots ?? rows.find((r) => r.totalSpots !== null)?.totalSpots

  return (
    <div
      className="w-full rounded-xl overflow-hidden select-none"
      style={{
        background: '#0a1a0a',
        border: '3px solid #1a3a1a',
        boxShadow: '0 0 40px rgba(74, 222, 128, 0.08), inset 0 0 60px rgba(0,0,0,0.4)',
        fontFamily: "'Courier New', Courier, monospace",
      }}
    >
      <div
        className="flex items-center justify-between px-6 py-4"
        style={{ borderBottom: '2px solid #1a3a1a', background: '#0d220d' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="flex items-center justify-center rounded font-black text-sm"
            style={{
              width: 32,
              height: 32,
              background: '#4ade80',
              color: '#0a1a0a',
              letterSpacing: 0,
            }}
          >
            P
          </div>
          <span
            className="text-lg font-bold tracking-widest uppercase"
            style={{ color: '#4ade80', letterSpacing: '0.15em' }}
          >
            Westwood Village Parking
          </span>
        </div>
        <span
          className="text-xs font-bold tracking-widest uppercase"
          style={{ color: '#22543d', letterSpacing: '0.2em' }}
        >
          {liveRow ? 'LIVE' : 'OPEN'}
        </span>
      </div>

      {rows.map((row, i) => (
        <div
          key={row.streetId}
          className="flex items-center justify-between px-6 py-5"
          style={{
            borderBottom: i < rows.length - 1 ? '1px solid #112211' : undefined,
            background: i % 2 === 0 ? 'transparent' : 'rgba(74,222,128,0.02)',
          }}
        >
          <div className="flex items-center gap-3">
            {row.isLive && (
              <span
                className="text-xs font-bold rounded px-1.5 py-0.5"
                style={{ background: '#14532d', color: '#4ade80', letterSpacing: '0.1em' }}
              >
                LIVE
              </span>
            )}
            <span
              className="text-base tracking-wide"
              style={{ color: '#86efac', letterSpacing: '0.05em' }}
            >
              {row.displayName}
            </span>
          </div>
          <SpotCount available={row.availableSpots} isLoading={row.isLoading} />
        </div>
      ))}

      <div
        className="px-6 py-2 text-right"
        style={{ borderTop: '1px solid #112211' }}
      >
        <span className="text-xs" style={{ color: '#1a4a1a', letterSpacing: '0.1em' }}>
          {footerTotal != null
            ? `${footerTotal} TOTAL SPACES${liveRow ? ' (LIVE)' : ''}`
            : 'CURBSIGHT'}
        </span>
      </div>
    </div>
  )
}
