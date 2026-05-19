import type { InferenceState, StartInferenceRequest } from '../types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, options)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function getVideos(): Promise<string[]> {
  const data = await request<{ filenames: string[] }>('/videos')
  return data.filenames
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
