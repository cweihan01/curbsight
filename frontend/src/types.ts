export interface InferenceEvent {
  timestamp_iso: string
  source_id: string
  frame_index: number
  inference_index: number
  stride: number
  occupied_spots: number
  available_spots: number
  total_spots: number
  occupancy_ratio: number
  total_tracks: number
  inferred_image_path: string | null
}

export interface StatusMessage {
  type: 'status'
  state: InferenceState
}

export type SocketMessage = InferenceEvent | StatusMessage

export type InferenceState = 'running' | 'idle' | 'started' | 'stopped'

export interface StartInferenceRequest {
  video_filename: string
  stride: number
  publish_every: number
  max_frames?: number
  conf: number
  iou: number
}
