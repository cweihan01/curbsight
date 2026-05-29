import { useInferenceSocket } from '../hooks/useInferenceSocket'
import { useFramePlayer } from '../hooks/useFramePlayer'
import { ControlPanel } from '../components/ControlPanel'
import { OccupancyGauge } from '../components/OccupancyGauge'
import { OccupancyChart } from '../components/OccupancyChart'
import { FrameViewer } from '../components/FrameViewer'

export function DashboardPage() {
  const { events, status, notifyStart } = useInferenceSocket()
  const { currentFrame, currentEvent, displayedEvents } = useFramePlayer(events)

  const isLoading = status === 'started' || (status === 'running' && currentEvent === null)

  return (
    <div className="grid grid-cols-[280px_1fr] gap-6 flex-1">
      <aside className="flex flex-col gap-4">
        <ControlPanel status={status} notifyStart={notifyStart} />
      </aside>
      <main className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-4">
          <FrameViewer currentFrame={currentFrame} currentEvent={currentEvent} isLoading={isLoading} />
          <OccupancyGauge latest={currentEvent} isLoading={isLoading} />
        </div>
        <OccupancyChart events={displayedEvents} isLoading={isLoading} />
      </main>
    </div>
  )
}
