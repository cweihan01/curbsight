import { useEffect, useRef, useState } from 'react'
import type { InferenceEvent } from '../types'

const ASSUMED_FPS = 30
const MAX_QUEUE = 50

interface QueueEntry {
  path: string
  event: InferenceEvent
}

export function useFramePlayer(events: InferenceEvent[]) {
  const [currentFrame, setCurrentFrame] = useState<string | null>(null)
  const [currentEvent, setCurrentEvent] = useState<InferenceEvent | null>(null)
  const [displayedEvents, setDisplayedEvents] = useState<InferenceEvent[]>([])

  const queueRef = useRef<QueueEntry[]>([])
  const processedRef = useRef(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const strideRef = useRef(30)

  useEffect(() => {
    const newEvents = events.slice(processedRef.current)
    for (const e of newEvents) {
      if (e.inferred_image_path) {
        strideRef.current = e.stride
        if (queueRef.current.length < MAX_QUEUE) {
          queueRef.current.push({ path: e.inferred_image_path, event: e })
        }
      }
    }
    processedRef.current = events.length
  }, [events])

  useEffect(() => {
    function tick() {
      const entry = queueRef.current.shift()
      if (!entry) return
      const filename = entry.path.split('/').pop()
      setCurrentFrame(`/frames/${filename}`)
      setCurrentEvent(entry.event)
      setDisplayedEvents((prev) => [...prev, entry.event])
    }

    function startInterval() {
      if (intervalRef.current) clearInterval(intervalRef.current)
      const ms = (strideRef.current / ASSUMED_FPS) * 1000
      intervalRef.current = setInterval(tick, ms)
    }

    startInterval()
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  function reset() {
    queueRef.current = []
    processedRef.current = 0
    setCurrentFrame(null)
    setCurrentEvent(null)
    setDisplayedEvents([])
  }

  return { currentFrame, currentEvent, displayedEvents, reset }
}
