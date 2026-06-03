import { useState } from 'react'
import { sessionReferenceFrameUrl } from '../api/client'
import type { ParkingRegion } from '../types'

interface RegionSelectorProps {
  sessionId: string
  regions: ParkingRegion[]
  /** Indices initially selected (kept). */
  initialSelected: number[]
  onApply: (selected: number[]) => void
  onClose: () => void
}

function polygonPoints(region: ParkingRegion): string {
  return region.points.map(([x, y]) => `${x},${y}`).join(' ')
}

function centroid(region: ParkingRegion): [number, number] {
  const n = region.points.length || 1
  const sum = region.points.reduce(
    (acc, [x, y]) => [acc[0] + x, acc[1] + y],
    [0, 0],
  )
  return [sum[0] / n, sum[1] / n]
}

export function RegionSelector({
  sessionId,
  regions,
  initialSelected,
  onApply,
  onClose,
}: RegionSelectorProps) {
  const [selected, setSelected] = useState<Set<number>>(() => new Set(initialSelected))
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)

  function toggle(index: number) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  const selectedCount = selected.size

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-2">
      <div className="bg-slate-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[94vh] flex flex-col">
        <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-slate-700">
          <div>
            <h2 className="text-slate-200 text-base font-semibold">Select parking boxes</h2>
            <p className="text-slate-400 text-xs mt-0.5">
              Click a box to keep or remove it. Only kept boxes are sent to inference.
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-2xl leading-none px-2"
            aria-label="Close"
          >
            x
          </button>
        </div>

        <div className="flex-1 min-h-0 p-3 flex items-center justify-center overflow-hidden bg-slate-900/50">
          {regions.length === 0 ? (
            <p className="text-slate-500 text-sm">No boxes defined for this session.</p>
          ) : (
            <div className="relative inline-block max-h-full max-w-full">
              <img
                src={sessionReferenceFrameUrl(sessionId)}
                alt="Session reference frame"
                className="block max-h-[calc(94vh-9rem)] max-w-full w-auto h-auto rounded-lg"
                onLoad={(e) =>
                  setDims({
                    w: e.currentTarget.naturalWidth,
                    h: e.currentTarget.naturalHeight,
                  })
                }
              />
              {dims && (
                <svg
                  className="absolute inset-0 w-full h-full"
                  viewBox={`0 0 ${dims.w} ${dims.h}`}
                  preserveAspectRatio="none"
                >
                  {regions.map((region, i) => {
                    const isKept = selected.has(i)
                    const [cx, cy] = centroid(region)
                    return (
                      <g
                        key={i}
                        onClick={() => toggle(i)}
                        className="cursor-pointer"
                      >
                        <polygon
                          points={polygonPoints(region)}
                          fill={isKept ? 'rgba(34, 197, 94, 0.35)' : 'rgba(148, 163, 184, 0.08)'}
                          stroke={isKept ? '#22c55e' : '#94a3b8'}
                          strokeWidth={3}
                          strokeDasharray={isKept ? undefined : '8 6'}
                          vectorEffect="non-scaling-stroke"
                        />
                        <text
                          x={cx}
                          y={cy}
                          fill={isKept ? '#dcfce7' : '#cbd5e1'}
                          fontSize={Math.max(dims.w, dims.h) * 0.022}
                          textAnchor="middle"
                          dominantBaseline="central"
                          className="select-none pointer-events-none font-semibold"
                        >
                          {i + 1}
                        </text>
                      </g>
                    )
                  })}
                </svg>
              )}
            </div>
          )}
        </div>

        <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-t border-slate-700">
          <div className="flex items-center gap-3">
            <span className="text-slate-400 text-sm">
              {selectedCount} of {regions.length} kept
            </span>
            <button
              onClick={() => setSelected(new Set(regions.map((_, i) => i)))}
              className="text-xs text-slate-300 hover:text-white underline"
            >
              Select all
            </button>
            <button
              onClick={() => setSelected(new Set())}
              className="text-xs text-slate-300 hover:text-white underline"
            >
              Clear
            </button>
          </div>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => onApply([...selected].sort((a, b) => a - b))}
              disabled={selectedCount === 0}
              className="bg-green-600 hover:bg-green-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              Apply ({selectedCount})
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
