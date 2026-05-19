import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useFramePlayer } from '../hooks/useFramePlayer'
import { makeEvent } from '../test/factories'

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('Bug 3: occupancy updates in sync with frame player', () => {
  it('currentEvent is null immediately after events arrive — does not race ahead', () => {
    const events = [makeEvent({ frame_index: 0, inference_index: 0 })]
    const { result } = renderHook(() => useFramePlayer(events))

    // Events arrived but interval hasn't fired yet
    expect(result.current.currentEvent).toBeNull()
    expect(result.current.currentFrame).toBeNull()
    expect(result.current.displayedEvents).toHaveLength(0)
  })

  it('currentEvent and currentFrame advance together on each interval tick', () => {
    const event1 = makeEvent({ frame_index: 0, inference_index: 0, inferred_image_path: 'frames/f0.jpg' })
    const event2 = makeEvent({ frame_index: 30, inference_index: 1, inferred_image_path: 'frames/f1.jpg' })

    const { result, rerender } = renderHook(
      ({ events }) => useFramePlayer(events),
      { initialProps: { events: [event1] } }
    )

    // Tick once — should show first event
    act(() => { vi.advanceTimersByTime(1000) })
    expect(result.current.currentEvent?.frame_index).toBe(0)
    expect(result.current.currentFrame).toContain('f0.jpg')
    expect(result.current.displayedEvents).toHaveLength(1)

    // Add second event and tick
    rerender({ events: [event1, event2] })
    act(() => { vi.advanceTimersByTime(1000) })
    expect(result.current.currentEvent?.frame_index).toBe(30)
    expect(result.current.currentFrame).toContain('f1.jpg')
    expect(result.current.displayedEvents).toHaveLength(2)
  })

  it('displayedEvents only contains events that have been shown', () => {
    const events = [
      makeEvent({ frame_index: 0, inferred_image_path: 'frames/f0.jpg' }),
      makeEvent({ frame_index: 30, inferred_image_path: 'frames/f1.jpg' }),
      makeEvent({ frame_index: 60, inferred_image_path: 'frames/f2.jpg' }),
    ]

    const { result } = renderHook(() => useFramePlayer(events))

    // No ticks yet
    expect(result.current.displayedEvents).toHaveLength(0)

    // One tick
    act(() => { vi.advanceTimersByTime(1000) })
    expect(result.current.displayedEvents).toHaveLength(1)

    // Two ticks
    act(() => { vi.advanceTimersByTime(1000) })
    expect(result.current.displayedEvents).toHaveLength(2)
  })

  it('reset clears all state', () => {
    const events = [makeEvent({ inferred_image_path: 'frames/f0.jpg' })]
    const { result } = renderHook(() => useFramePlayer(events))

    act(() => { vi.advanceTimersByTime(1000) })
    expect(result.current.currentFrame).not.toBeNull()

    act(() => { result.current.reset() })
    expect(result.current.currentFrame).toBeNull()
    expect(result.current.currentEvent).toBeNull()
    expect(result.current.displayedEvents).toHaveLength(0)
  })
})
