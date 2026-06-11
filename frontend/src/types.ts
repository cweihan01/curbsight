export interface InferenceEvent {
  timestamp_iso: string
  session_id: string
  street_id?: string
  street_display_name?: string
  frame_index: number
  inference_index: number
  stride: number
  occupied_spots: number
  available_spots: number
  total_spots: number
  occupancy_ratio: number
  total_tracks: number
  inferred_image_filename: string | null
}

export interface StatusMessage {
  type: 'status'
  state: InferenceState
}

export interface WarningMessage {
  type: 'warning'
  message: string
}

export type SocketMessage = InferenceEvent | StatusMessage | WarningMessage

export type InferenceState = 'running' | 'idle' | 'started' | 'stopped'

/** One parking slot polygon (bounding_boxes.json format: 4 [x, y] points). */
export interface ParkingRegion {
  points: number[][]
}

export interface StartInferenceRequest {
  session_id: string
  stride: number
  vote_radius: number
  vote_frame_step: number
  publish_every: number
  max_frames?: number
  conf: number
  regions?: ParkingRegion[]
}
