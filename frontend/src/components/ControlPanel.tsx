import { useControl } from '../context/ControlContext'
import { RegionSelector } from './RegionSelector'

function TooltipLabel({ label, tip }: { label: string; tip: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-slate-400 text-xs">{label}</span>
      <span className="relative group inline-flex">
        <span className="w-4 h-4 rounded-full border border-slate-500 text-slate-300 text-[10px] leading-none inline-flex items-center justify-center">
          ?
        </span>
        <span className="pointer-events-none absolute left-0 top-[125%] z-20 hidden group-hover:block w-56 max-w-[75vw] rounded-md border border-slate-600 bg-slate-900 px-2 py-1.5 text-[11px] leading-4 text-slate-200 shadow-xl">
          {tip}
        </span>
      </span>
    </div>
  )
}

export function ControlPanel() {
  const {
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
    normalizeStride,
    normalizeVoteRadius,
    normalizeVoteFrameStep,
    normalizeConf,
    resetParamsToDefault,
    handleStart,
    handleStop,
  } = useControl()

  return (
    <div className="bg-slate-800 rounded-xl py-3 px-4 flex flex-col gap-3">
      <h2 className="text-slate-400 text-sm font-medium uppercase tracking-wider">Controls</h2>

      <div className="flex flex-col gap-2">
        <TooltipLabel
          label="Session"
          tip="Select which recording and base region file to run. Changing session switches the reference frame and available parking boxes."
        />
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={isRunning}
          className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
        >
          {sessions.length === 0 && <option value="">No sessions found</option>}
          {sessions.map((id) => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <TooltipLabel
          label="Parking boxes"
          tip="Open the selector to keep/remove predefined boxes. Kept boxes are sent to backend as regions for this run."
        />
        <button
          onClick={() => setShowSelector(true)}
          disabled={isRunning || regions.length === 0}
          className="bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:hover:bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 transition-colors text-left"
        >
          {regions.length === 0
            ? 'No boxes for session'
            : `Select boxes (${selectedIndices.length}/${regions.length} kept)`}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Stride"
            tip="Run inference every N frames. Increase for less frequent updates; decrease for more frequent updates. At 30 FPS, setting stride to 120 runs inference every 4 seconds."
          />
          <input
            type="text"
            inputMode="numeric"
            value={stride}
            onChange={(e) => setStride(e.target.value)}
            onBlur={normalizeStride}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Vote radius"
            tip="Number of samples on each side of the anchor frame. Higher radius smooths transient occlusions but adds more per-inference work. A vote radius of 3 corresponds to a majority vote over 7 frames per inference."
          />
          <input
            type="text"
            inputMode="numeric"
            value={voteRadius}
            onChange={(e) => setVoteRadius(e.target.value)}
            onBlur={normalizeVoteRadius}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Vote step"
            tip="Frame gap between voting samples per inference. Increase to spread samples farther in time, making the vote more robust to brief occlusions (also requires a higher stride); decrease for tighter temporal voting. A vote step of 15 means each inference samples the frame f, f+15, f-15, f+30, f-30, etc depending on the vote radius."
          />
          <input
            type="text"
            inputMode="numeric"
            value={voteFrameStep}
            onChange={(e) => setVoteFrameStep(e.target.value)}
            onBlur={normalizeVoteFrameStep}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
        <div className="flex flex-col gap-1">
          <TooltipLabel
            label="Conf"
            tip="Detection confidence threshold in range [0, 1]. Increase to reduce false positives (but can miss cars); decrease to catch more detections (but may add noise)."
          />
          <input
            type="text"
            inputMode="decimal"
            value={conf}
            onChange={(e) => setConf(e.target.value)}
            onBlur={normalizeConf}
            disabled={isRunning}
            className="bg-slate-700 text-slate-200 rounded-lg px-3 py-2 text-sm border border-slate-600 focus:outline-none focus:border-slate-400 disabled:opacity-50"
          />
        </div>
      </div>

      {error && (
        <p className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      <button
        type="button"
        onClick={resetParamsToDefault}
        disabled={isRunning}
        className="w-full bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:hover:bg-slate-700 text-slate-200 rounded-lg py-1 text-sm font-medium transition-colors border border-slate-600"
      >
        Reset to default parameters
      </button>

      <div className="flex gap-3">
        <button
          onClick={() => void handleStart()}
          disabled={isRunning || !selected}
          className="flex-1 bg-green-600 hover:bg-green-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg py-2 text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          {isStarting && (
            <span className="w-3 h-3 border border-white/40 border-t-white rounded-full animate-spin" />
          )}
          {isStarting ? 'Starting...' : 'Start'}
        </button>
        <button
          onClick={() => void handleStop()}
          disabled={!isRunning}
          className="flex-1 bg-red-600 hover:bg-red-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-lg py-2 text-sm font-medium transition-colors"
        >
          Stop
        </button>
      </div>

      {showSelector && selected && (
        <RegionSelector
          sessionId={selected}
          regions={regions}
          initialSelected={selectedIndices}
          onApply={(next) => {
            setSelectedIndices(next)
            setShowSelector(false)
          }}
          onClose={() => setShowSelector(false)}
        />
      )}
    </div>
  )
}
