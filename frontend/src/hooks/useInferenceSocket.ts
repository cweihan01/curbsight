import { useCallback, useEffect, useRef, useState } from 'react'
import type { InferenceEvent, InferenceState, SocketMessage } from '../types'

const MAX_EVENTS = 500
const WS_URL = 'ws://localhost:8000/ws/events'

export function useInferenceSocket() {
  const [events, setEvents] = useState<InferenceEvent[]>([])
  const [latest, setLatest] = useState<InferenceEvent | null>(null)
  const [status, setStatus] = useState<InferenceState>('idle')
  const [connected, setConnected] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const retryDelay = useRef(1000)
  const retryTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
  const unmounted = useRef(false)
  const isActive = useRef(false)

  const clearEvents = useCallback(() => {
    setEvents([])
    setLatest(null)
  }, [])

  const notifyStart = useCallback(() => {
    isActive.current = true
    setStatus('started')
    clearEvents()
    // The inference subprocess clears the JSONL file on start. Reconnect after
    // a short delay so the backend reopens the file from position 0, otherwise
    // the existing file handle stays past the truncated content and misses new events.
    setTimeout(() => {
      retryDelay.current = 100
      wsRef.current?.close()
    }, 1500)
  }, [clearEvents])

  const connect = useCallback(() => {
    if (unmounted.current) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      retryDelay.current = 1000
    }

    ws.onmessage = (e) => {
      let msg: SocketMessage
      try {
        msg = JSON.parse(e.data)
      } catch {
        return
      }

      if ('type' in msg) {
        if (msg.type === 'status') {
          setStatus(msg.state)
          if (msg.state === 'idle' || msg.state === 'stopped') {
            isActive.current = false
          }
        }
        return
      }

      if (!isActive.current) return

      const event = msg as InferenceEvent
      if (status !== 'running') setStatus('running')
      setLatest(event)
      setEvents((prev) => {
        const next = [...prev, event]
        return next.length > MAX_EVENTS ? next.slice(next.length - MAX_EVENTS) : next
      })
    }

    ws.onclose = () => {
      setConnected(false)
      if (!unmounted.current) {
        retryTimeout.current = setTimeout(() => {
          retryDelay.current = Math.min(retryDelay.current * 2, 30000)
          connect()
        }, retryDelay.current)
      }
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    unmounted.current = false
    connect()
    return () => {
      unmounted.current = true
      if (retryTimeout.current) clearTimeout(retryTimeout.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { events, latest, status, connected, clearEvents, notifyStart }
}
