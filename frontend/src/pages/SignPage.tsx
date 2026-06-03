import { useInference } from '../context/InferenceContext'
import { useStreetSignBoard } from '../hooks/useStreetSignBoard'
import { ControlPanel } from '../components/ControlPanel'
import { FrameViewer } from '../components/FrameViewer'
import { DriverSign } from '../components/DriverSign'

export function SignPage() {
  const { events, status, frame, isLoading } = useInference()
  const signRows = useStreetSignBoard(
    events,
    status,
    isLoading,
    frame.currentIndex,
    frame.isLive,
  )

  return (
    <div className="grid grid-cols-[280px_1fr_1fr] gap-6 flex-1 items-start">
      <aside className="flex flex-col gap-4">
        <ControlPanel />
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

      <DriverSign rows={signRows} />
    </div>
  )
}
