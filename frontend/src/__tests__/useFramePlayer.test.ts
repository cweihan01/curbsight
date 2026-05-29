import { renderHook, act } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { useFramePlayer } from '../hooks/useFramePlayer'
import { makeEvent } from '../test/factories'

describe('useFramePlayer', () => {
  it('reflects the latest event immediately', () => {
    const event = makeEvent({ frame_index: 0, inferred_image_filename: 'f0.jpg' })
    const { result } = renderHook(() => useFramePlayer([event]))

    expect(result.current.currentEvent?.frame_index).toBe(0)
    expect(result.current.currentFrame).toContain('f0.jpg')
    expect(result.current.displayedEvents).toHaveLength(1)
  })

  it('updates when a new event is appended', () => {
    const event1 = makeEvent({ frame_index: 0, inferred_image_filename: 'f0.jpg' })
    const event2 = makeEvent({ frame_index: 30, inferred_image_filename: 'f1.jpg' })

    const { result, rerender } = renderHook(
      ({ events }) => useFramePlayer(events),
      { initialProps: { events: [event1] } },
    )

    rerender({ events: [event1, event2] })

    expect(result.current.currentEvent?.frame_index).toBe(30)
    expect(result.current.currentFrame).toContain('f1.jpg')
    expect(result.current.displayedEvents).toHaveLength(2)
  })

  it('auto-advances when live and a new event arrives', () => {
    const event1 = makeEvent({ frame_index: 0, inferred_image_filename: 'f0.jpg' })
    const event2 = makeEvent({ frame_index: 30, inferred_image_filename: 'f1.jpg' })

    const { result, rerender } = renderHook(
      ({ events }) => useFramePlayer(events),
      { initialProps: { events: [event1] } },
    )

    expect(result.current.isLive).toBe(true)

    rerender({ events: [event1, event2] })

    expect(result.current.isLive).toBe(true)
    expect(result.current.currentEvent?.frame_index).toBe(30)
  })

  it('navigates to previous frames without losing new live events at the end', () => {
    const event1 = makeEvent({ frame_index: 0, inferred_image_filename: 'f0.jpg' })
    const event2 = makeEvent({ frame_index: 30, inferred_image_filename: 'f1.jpg' })
    const event3 = makeEvent({ frame_index: 60, inferred_image_filename: 'f2.jpg' })

    const { result, rerender } = renderHook(
      ({ events }) => useFramePlayer(events),
      { initialProps: { events: [event1, event2, event3] } },
    )

    act(() => { result.current.goPrev() })
    expect(result.current.currentEvent?.frame_index).toBe(30)
    expect(result.current.isLive).toBe(false)

    rerender({ events: [event1, event2, event3, makeEvent({ frame_index: 90, inferred_image_filename: 'f3.jpg' })] })
    expect(result.current.currentEvent?.frame_index).toBe(30)

    act(() => { result.current.goLatest() })
    expect(result.current.currentEvent?.frame_index).toBe(90)
    expect(result.current.isLive).toBe(true)
  })

  it('clears when events array is emptied', () => {
    const events = [makeEvent({ inferred_image_filename: 'f0.jpg' })]
    const { result, rerender } = renderHook(
      ({ events }) => useFramePlayer(events),
      { initialProps: { events } },
    )

    expect(result.current.currentFrame).not.toBeNull()

    rerender({ events: [] })
    expect(result.current.currentFrame).toBeNull()
    expect(result.current.currentEvent).toBeNull()
    expect(result.current.displayedEvents).toHaveLength(0)
  })
})
