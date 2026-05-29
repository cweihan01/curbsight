import { useInferenceSocket } from '../hooks/useInferenceSocket'
import { useFramePlayer } from '../hooks/useFramePlayer'
import { ControlPanel } from '../components/ControlPanel'
import { FrameViewer } from '../components/FrameViewer'
import { DriverSign } from '../components/DriverSign'

export function SignPage() {
  const { events, status, notifyStart } = useInferenceSocket()
  const frame = useFramePlayer(events)

  const isLoading = status === 'started' || (status === 'running' && frame.currentEvent === null)

  return (
    <div className="grid grid-cols-[280px_1fr_1fr] gap-6 flex-1 items-start">
      <aside className="flex flex-col gap-4">
        <ControlPanel status={status} notifyStart={notifyStart} />
      </aside>

      <FrameViewer
        currentFrame={frame.currentFrame}
        currentEvent={frame.currentEvent}
        isLoading={isLoading}
        currentIndex={frame.currentIndex}
        frameCount={frame.frameCount}
        isLive={frame.isLive}
        canGoPrev={frame.canGoPrev}
        canGoNext={frame.canGoNext}
        onPrev={frame.goPrev}
        onNext={frame.goNext}
        onGoToIndex={frame.goToIndex}
        onLatest={frame.goLatest}
      />

      <DriverSign
        availableSpots={frame.currentEvent?.available_spots ?? null}
        totalSpots={frame.currentEvent?.total_spots ?? null}
        isLoading={isLoading}
      />
    </div>
  )
}
