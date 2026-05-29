import { useCallback, useEffect, useRef, useState } from 'react'
import type { InferenceEvent, InferenceState, SocketMessage } from '../types'

const MAX_EVENTS = 500

function eventsWebSocketUrl(sessionId: string | null): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const base = `${protocol}//${window.location.host}/ws/events`
  if (!sessionId) return base
  return `${base}?session_id=${encodeURIComponent(sessionId)}`
}

export function useInferenceSocket() {
  const [events, setEvents] = useState<InferenceEvent[]>([])
  const [latest, setLatest] = useState<InferenceEvent | null>(null)
  const [status, setStatus] = useState<InferenceState>('idle')
  const [connected, setConnected] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const runEpochRef = useRef(0)
  const wsEpochRef = useRef(0)
  const retryDelay = useRef(1000)
  const retryTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
  const unmounted = useRef(false)
  const isActive = useRef(false)

  const clearEvents = useCallback(() => {
    setEvents([])
    setLatest(null)
  }, [])

  const notifyStart = useCallback((sessionId: string) => {
    runEpochRef.current += 1
    sessionIdRef.current = sessionId
    isActive.current = false
    setStatus('started')
    clearEvents()
    retryDelay.current = 100
    wsRef.current?.close()
  }, [clearEvents])

  const connect = useCallback(() => {
    if (unmounted.current) return

    const ws = new WebSocket(eventsWebSocketUrl(sessionIdRef.current))
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      retryDelay.current = 1000
      wsEpochRef.current = runEpochRef.current
      if (sessionIdRef.current) {
        isActive.current = true
      }
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

      if (!isActive.current || wsEpochRef.current !== runEpochRef.current) return

      const event = msg as InferenceEvent
      setStatus((prev) => (prev !== 'running' ? 'running' : prev))
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
