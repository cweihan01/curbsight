import { useMemo } from 'react'
import { SIGN_STREETS } from '../constants'
import type { InferenceEvent, InferenceState } from '../types'

export interface StreetSignRow {
  streetId: string
  displayName: string
  availableSpots: number | null
  totalSpots: number | null
  isLive: boolean
  isLoading: boolean
}

export function useStreetSignBoard(
  events: InferenceEvent[],
  status: InferenceState,
  isLoading: boolean,
  currentIndex: number,
  frameIsLive: boolean,
): StreetSignRow[] {
  const isRunning = status === 'running' || status === 'started'

  return useMemo(() => {
    const visibleEvents =
      currentIndex < 0 ? [] : events.slice(0, currentIndex + 1)

    const latestByStreet = new Map<
      string,
      { availableSpots: number; totalSpots: number }
    >()

    for (const event of visibleEvents) {
      const streetId = event.street_id
      if (!streetId) continue
      latestByStreet.set(streetId, {
        availableSpots: event.available_spots,
        totalSpots: event.total_spots,
      })
    }

    const liveStreetId =
      frameIsLive && isRunning && events.length > 0
        ? events[events.length - 1].street_id ?? null
        : null

    return SIGN_STREETS.map((street) => {
      const latest = latestByStreet.get(street.id)
      const isLive = isRunning && liveStreetId === street.id
      return {
        streetId: street.id,
        displayName: street.displayName,
        availableSpots: latest?.availableSpots ?? null,
        totalSpots: latest?.totalSpots ?? null,
        isLive,
        isLoading: isLive && isLoading,
      }
    })
  }, [events, isRunning, isLoading, currentIndex, frameIsLive])
}
