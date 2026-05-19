export class MockWebSocket {
  static instance: MockWebSocket | null = null

  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  readyState = 1

  constructor() {
    MockWebSocket.instance = this
  }

  send(_data: string) {}
  close() {
    this.onclose?.()
  }

  // Test helpers
  simulateOpen() {
    this.onopen?.()
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }

  simulateClose() {
    this.onclose?.()
  }
}

export function installMockWebSocket() {
  MockWebSocket.instance = null
  vi.stubGlobal('WebSocket', MockWebSocket)
}

export function getMockWs(): MockWebSocket {
  if (!MockWebSocket.instance) throw new Error('No MockWebSocket instance')
  return MockWebSocket.instance
}
