import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useInferenceSocket } from '../hooks/useInferenceSocket'
import { installMockWebSocket, getMockWs } from '../test/mockWebSocket'
import { makeEvent } from '../test/factories'

beforeEach(() => {
  installMockWebSocket()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Bug 1: events are ignored before notifyStart()', () => {
  it('does not update events or latest when receiving events before notifyStart', async () => {
    const { result } = renderHook(() => useInferenceSocket())
    const ws = getMockWs()

    await act(async () => {
      ws.simulateOpen()
      ws.simulateMessage(makeEvent())
    })

    expect(result.current.events).toHaveLength(0)
    expect(result.current.latest).toBeNull()
  })

  it('processes events after notifyStart and reconnect', async () => {
    const { result } = renderHook(() => useInferenceSocket())

    await act(async () => {
      getMockWs().simulateOpen()
      result.current.notifyStart('test/session')
    })

    await act(async () => {
      getMockWs().simulateOpen()
      getMockWs().simulateMessage(makeEvent())
    })

    expect(result.current.events).toHaveLength(1)
    expect(result.current.latest).not.toBeNull()
  })
})

describe('Bug 2: Stop button enabled after start', () => {
  it('status becomes started immediately after notifyStart', async () => {
    const { result } = renderHook(() => useInferenceSocket())
    const ws = getMockWs()

    await act(async () => {
      ws.simulateOpen()
      result.current.notifyStart('test/session')
    })

    expect(result.current.status).toBe('started')
  })

  it('status becomes running after first inference event', async () => {
    const { result } = renderHook(() => useInferenceSocket())

    await act(async () => {
      getMockWs().simulateOpen()
      result.current.notifyStart('test/session')
    })
    await act(async () => {
      getMockWs().simulateOpen()
      getMockWs().simulateMessage(makeEvent())
    })

    expect(result.current.status).toBe('running')
  })

  it('status returns to idle and stops processing events after idle status message', async () => {
    const { result } = renderHook(() => useInferenceSocket())

    await act(async () => {
      getMockWs().simulateOpen()
      result.current.notifyStart('test/session')
    })
    await act(async () => {
      const ws = getMockWs()
      ws.simulateOpen()
      ws.simulateMessage(makeEvent())
      ws.simulateMessage({ type: 'status', state: 'idle' })
      ws.simulateMessage(makeEvent({ frame_index: 99 }))
    })

    expect(result.current.status).toBe('idle')
    expect(result.current.events).toHaveLength(1)
  })
})
