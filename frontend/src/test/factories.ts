import type { InferenceEvent } from '../types'

export function makeEvent(overrides: Partial<InferenceEvent> = {}): InferenceEvent {
  return {
    timestamp_iso: '2026-05-18T17:00:00',
    session_id: 'test/session',
    street_id: 'gayley',
    street_display_name: 'Gayley Ave',
    frame_index: 0,
    inference_index: 0,
    stride: 30,
    occupied_spots: 2,
    available_spots: 8,
    total_spots: 10,
    occupancy_ratio: 0.2,
    total_tracks: 2,
    inferred_image_filename: 'inferred_001_frame_000000.jpg',
    ...overrides,
  }
}
