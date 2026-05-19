import { useInferenceSocket } from '../hooks/useInferenceSocket'
import { useFramePlayer } from '../hooks/useFramePlayer'
import { ControlPanel } from '../components/ControlPanel'
import { FrameViewer } from '../components/FrameViewer'
import { DriverSign } from '../components/DriverSign'

export function SignPage() {
  const { events, status, notifyStart, clearEvents } = useInferenceSocket()
  const { currentFrame, currentEvent, reset } = useFramePlayer(events)

  const isLoading = status === 'started' || (status === 'running' && currentEvent === null)

  function handleStart() {
    clearEvents()
    reset()
  }

  return (
    <div className="grid grid-cols-[280px_1fr_1fr] gap-6 flex-1 items-start">
      <aside className="flex flex-col gap-4">
        <ControlPanel status={status} onStart={handleStart} notifyStart={notifyStart} />
      </aside>

      <FrameViewer currentFrame={currentFrame} currentEvent={currentEvent} isLoading={isLoading} />

      <DriverSign
        availableSpots={currentEvent?.available_spots ?? null}
        totalSpots={currentEvent?.total_spots ?? null}
        isLoading={isLoading}
      />
    </div>
  )
}
