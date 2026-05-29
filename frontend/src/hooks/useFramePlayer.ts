import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { sessionFrameUrl } from '../api/client'
import type { InferenceEvent } from '../types'

function frameUrlForEvent(event: InferenceEvent | null): string | null {
  if (!event?.inferred_image_filename) return null
  return sessionFrameUrl(event.session_id, event.inferred_image_filename)
}

export function useFramePlayer(events: InferenceEvent[]) {
  const [index, setIndex] = useState(-1)
  const followLiveRef = useRef(true)

  useEffect(() => {
    if (events.length === 0) {
      followLiveRef.current = true
      setIndex(-1)
      return
    }

    setIndex((prev) => {
      const last = events.length - 1
      if (followLiveRef.current) return last
      return Math.min(prev < 0 ? 0 : prev, last)
    })
  }, [events])

  const currentIndex = events.length === 0 ? -1 : Math.max(0, index)
  const currentEvent = currentIndex >= 0 ? events[currentIndex] : null
  const isLive = events.length > 0 && currentIndex === events.length - 1

  const currentFrame = useMemo(
    () => frameUrlForEvent(currentEvent),
    [currentEvent],
  )

  const goPrev = useCallback(() => {
    followLiveRef.current = false
    setIndex((prev) => {
      const current = prev < 0 ? events.length - 1 : prev
      return Math.max(0, current - 1)
    })
  }, [events.length])

  const goNext = useCallback(() => {
    setIndex((prev) => {
      const current = prev < 0 ? events.length - 1 : prev
      const next = Math.min(events.length - 1, current + 1)
      followLiveRef.current = next === events.length - 1
      return next
    })
  }, [events.length])

  const goToIndex = useCallback(
    (i: number) => {
      if (events.length === 0) return
      const clamped = Math.max(0, Math.min(events.length - 1, i))
      followLiveRef.current = clamped === events.length - 1
      setIndex(clamped)
    },
    [events.length],
  )

  const goLatest = useCallback(() => {
    followLiveRef.current = true
    if (events.length > 0) setIndex(events.length - 1)
  }, [events.length])

  return {
    currentFrame,
    currentEvent,
    displayedEvents: events,
    currentIndex,
    frameCount: events.length,
    isLive,
    canGoPrev: currentIndex > 0,
    canGoNext: currentIndex < events.length - 1,
    goPrev,
    goNext,
    goToIndex,
    goLatest,
  }
}
