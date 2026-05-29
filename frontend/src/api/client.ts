import type { InferenceState, StartInferenceRequest } from '../types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, options)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function getSessions(): Promise<string[]> {
  const data = await request<{ session_ids: string[] }>('/sessions')
  return data.session_ids
}

/** URL for GET /sessions/{session_id}/frames/{image_name} (via Vite /api proxy). */
export function sessionFrameUrl(sessionId: string, imageName: string): string {
  const sessionPath = sessionId.split('/').map(encodeURIComponent).join('/')
  return `/api/sessions/${sessionPath}/frames/${encodeURIComponent(imageName)}`
}

export async function startInference(req: StartInferenceRequest): Promise<void> {
  await request('/inference/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
}

export async function stopInference(): Promise<void> {
  await request('/inference/stop', { method: 'POST' })
}

export async function getStatus(): Promise<InferenceState> {
  const data = await request<{ status: InferenceState }>('/inference/status')
  return data.status
}
