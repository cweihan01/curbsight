import { useInference } from '../context/InferenceContext'
import { ControlPanel } from '../components/ControlPanel'
import { OccupancyGauge } from '../components/OccupancyGauge'
import { OccupancyChart } from '../components/OccupancyChart'
import { FrameViewer } from '../components/FrameViewer'

export function DashboardPage() {
  const { frame, isLoading } = useInference()

  return (
    <div className="grid grid-cols-[280px_1fr] gap-4">
      <aside className="flex flex-col gap-3">
        <ControlPanel />
      </aside>
      <main className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
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
          <OccupancyGauge latest={frame.currentEvent} isLoading={isLoading} />
        </div>
        <OccupancyChart events={frame.displayedEvents} isLoading={isLoading} />
      </main>
    </div>
  )
}
