import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { useFramePlayer } from '../hooks/useFramePlayer'
import { useInferenceSocket } from '../hooks/useInferenceSocket'
import type { InferenceEvent, InferenceState } from '../types'

export interface FramePlayerState {
  currentFrame: string | null
  currentEvent: InferenceEvent | null
  displayedEvents: InferenceEvent[]
  currentIndex: number
  frameCount: number
  isLive: boolean
  canGoPrev: boolean
  canGoNext: boolean
  goPrev: () => void
  goNext: () => void
  goToIndex: (index: number) => void
  goLatest: () => void
}

interface InferenceContextValue {
  events: InferenceEvent[]
  latest: InferenceEvent | null
  status: InferenceState
  connected: boolean
  notifyStart: (sessionId: string) => void
  frame: FramePlayerState
  isLoading: boolean
}

const InferenceContext = createContext<InferenceContextValue | null>(null)

export function InferenceProvider({ children }: { children: ReactNode }) {
  const socket = useInferenceSocket()
  const frame = useFramePlayer(socket.events)

  const isLoading =
    socket.status === 'started' ||
    (socket.status === 'running' && frame.currentEvent === null)

  const value = useMemo<InferenceContextValue>(
    () => ({
      events: socket.events,
      latest: socket.latest,
      status: socket.status,
      connected: socket.connected,
      notifyStart: socket.notifyStart,
      frame,
      isLoading,
    }),
    [socket, frame, isLoading],
  )

  return (
    <InferenceContext.Provider value={value}>{children}</InferenceContext.Provider>
  )
}

export function useInference(): InferenceContextValue {
  const ctx = useContext(InferenceContext)
  if (!ctx) {
    throw new Error('useInference must be used within InferenceProvider')
  }
  return ctx
}
