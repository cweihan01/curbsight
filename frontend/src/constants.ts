/** Fixed rows on the driver street sign (order preserved). */
export const SIGN_STREETS = [
  { id: 'leconte', displayName: 'Le Conte Ave' },
  { id: 'gayley', displayName: 'Gayley Ave' },
  { id: 'westwood', displayName: 'Westwood Blvd' },
] as const

export type SignStreetId = (typeof SIGN_STREETS)[number]['id']

export const INFERENCE_DEFAULTS = {
  stride: 120,
  voteRadius: 3,
  voteFrameStep: 15,
  publishEvery: 1,
  conf: 0.1,
  iou: 0.7,
} as const
