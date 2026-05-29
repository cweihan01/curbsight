import { useMemo } from 'react'
import { sessionFrameUrl } from '../api/client'
import type { InferenceEvent } from '../types'

export function useFramePlayer(events: InferenceEvent[]) {
  const latest = events.length > 0 ? events[events.length - 1] : null

  const currentFrame = useMemo(() => {
    if (!latest?.inferred_image_filename) return null
    return sessionFrameUrl(latest.session_id, latest.inferred_image_filename)
  }, [latest])

  return {
    currentFrame,
    currentEvent: latest,
    displayedEvents: events,
  }
}
