"""ParkingManagement subclass with per-region occupancy and majority voting."""

from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np
from ultralytics import solutions
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults


def sample_frame_indices(
    anchor: int,
    vote_radius: int,
    vote_frame_step: int,
    frame_count: int,
) -> list[int]:
    """
    Return (2*vote_radius + 1) frame indices around anchor in steps of vote_frame_step,
    clamped/shifted at edges.

    With vote_radius=2 and vote_frame_step=12, anchor f yields [f-24, f-12, f, f+12, f+24].
    """
    if vote_radius <= 0:
        return [anchor]

    indices = [
        anchor + k * vote_frame_step for k in range(-vote_radius, vote_radius + 1)
    ]

    if frame_count <= 0:
        return [max(0, i) for i in indices]

    # Clamp/shift the indices at the edges of the video
    max_idx = frame_count - 1
    if indices[0] < 0:
        shift = -indices[0]
        indices = [i + shift for i in indices]
    if indices[-1] > max_idx:
        shift = indices[-1] - max_idx
        indices = [i - shift for i in indices]

    return [max(0, min(max_idx, i)) for i in indices]


class VotingParkingManagement(solutions.ParkingManagement):
    """Extends Ultralytics ParkingManagement with per-slot occupancy helpers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_region_occupied: list[bool] | None = None

    @property
    def last_region_occupied(self) -> list[bool] | None:
        """Per-region occupancy from the most recent process() call (voted when active)."""
        return self._last_region_occupied

    def region_occupancy(self, im0: np.ndarray) -> list[bool]:
        """
        Return one occupied flag per parking region/bounding box (no annotation)
        as defined in the JSON file.
        This is functionally identical to the standard Ultralytics ParkingManagement
        `process()` method, but without any annotations.
        """
        self.extract_tracks(im0)
        flags: list[bool] = []

        # Compute the occupancy for each bounding box
        for region in self.json:
            region_polygon = np.array(
                region["points"], dtype=np.int32).reshape((-1, 1, 2))
            region_occupied = False
            for box, _cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                inside_distance = cv2.pointPolygonTest(region_polygon, (xc, yc), False)
                if inside_distance >= 0:
                    region_occupied = True
                    break
            flags.append(region_occupied)

        return flags

    def majority_region_occupancy(self, samples: list[list[bool]]) -> list[bool]:
        """
        Majority vote per region across multiple occupancy snapshots.
        `samples` is a list of lists of booleans, where each inner list contains the
        occupancy for each bounding box in a frame, and there are as many such lists
        as there are frames in the vote window.

        Returns a list of booleans, where each boolean is the occupancy for a
        parking region/bounding box.
        """
        if not samples:
            return []

        n_regions = len(samples[0])
        threshold = (len(samples) + 1) // 2  # Threshold is half the number of samples
        votes = [0] * n_regions

        # Count the number of occupied regions for each bounding box
        for flags in samples:
            if len(flags) != n_regions:
                raise ValueError("All samples must have the same number of regions")
            for i, occupied in enumerate(flags):
                votes[i] += int(occupied)

        # Return a list of booleans, where each boolean is the occupancy for a parking region/bounding box
        return [votes[i] >= threshold for i in range(n_regions)]

    def majority_vote_occupancy(
        self,
        cap: cv2.VideoCapture,
        anchor_index: int,
        vote_radius: int,
        vote_frame_step: int,
        frame_count: int,
    ) -> list[bool]:
        """
        Run inference on frames near anchor and majority-vote per slot.
        Returns a list of booleans, where each boolean is the occupancy for a
        parking region/bounding box.
        """
        # Get the frame indices to sample around the anchor frame
        indices = sample_frame_indices(
            anchor_index,
            vote_radius=vote_radius,
            vote_frame_step=vote_frame_step,
            frame_count=frame_count,
        )

        # Read the frames and compute the occupancy for each bounding box
        # This stores a list of lists of booleans, where each inner list is the occupancy for a bounding box
        # and each outer list is the occupancy for a frame
        samples: list[list[bool]] = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            samples.append(self.region_occupancy(frame))

        # Majority vote the occupancy for each bounding box
        majority = self.majority_region_occupancy(samples)
        # print(
        #     f"[voting] anchor={anchor_index}: sampled frames={indices} "
        #     f"(read {len(samples)}/{len(indices)}), voted occupied="
        #     f"{sum(majority)}/{len(majority)} -> {[int(o) for o in majority]}",
        #     file=sys.stderr,
        # )
        return majority

    def process_from_occupancy(
        self, im0: np.ndarray, region_occupied: list[bool]
    ) -> SolutionResults:
        """
        Annotate frame using precomputed per-region occupancy after majority vote.
        This is functionally identical to the standard Ultralytics ParkingManagement
        `process()` method, but with the occupancy precomputed by the majority vote.
        As such, the object labels (car, truck, etc.) are not added to the overlay
        to avoid mismatches between the anchor frame and the majority vote.

        Returns a SolutionResults object with the occupancy precomputed by the majority
        vote, same as the standard Ultralytics ParkingManagement `process()` method.
        """
        self.extract_tracks(im0)
        n_regions = len(self.json)
        if len(region_occupied) != n_regions:
            raise ValueError(
                f"region_occupied length {len(region_occupied)} != {n_regions} regions"
            )

        occupied_slots = sum(region_occupied)
        available_slots = n_regions - occupied_slots
        annotator = SolutionAnnotator(im0, self.line_width)

        for region, occupied in zip(self.json, region_occupied):
            region_polygon = np.array(
                region["points"], dtype=np.int32).reshape((-1, 1, 2))
            # Here, unlike the standard Ultralytics ParkingManagement, the object labels
            # (car, truck, etc.) are not added to the overlay as the majority vote
            # may not align with the anchor frame
            cv2.polylines(
                im0,
                [region_polygon],
                isClosed=True,
                color=self.occ if occupied else self.arc,
                thickness=2,
            )

        self.pr_info["Occupancy"], self.pr_info["Available"] = occupied_slots, available_slots
        self._last_region_occupied = list(region_occupied)

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output with base class function

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            filled_slots=occupied_slots,
            available_slots=available_slots,
            total_tracks=len(self.track_ids),  # Note this is not accurate, but unused
        )

    def process(
        self,
        im0: np.ndarray,
        *,
        vote_radius: int = 0,
        vote_frame_step: int = 1,
        stride: int = 1,
        cap: cv2.VideoCapture | None = None,
        frame_index: int | None = None,
    ) -> SolutionResults:
        """
        Process one frame; optionally majority-vote occupancy from nearby frames.

        Args:
            im0: The input frame.
            vote_radius: The radius of the vote window.
            vote_frame_step: Spacing in frames between samples in the vote window.
            stride: The stride of the vote window.
            cap: The video capture object.
            frame_index: The index of the frame to process.

        Returns a SolutionResults object, same as the standard Ultralytics
        ParkingManagement `process()` method.
        """
        # Call the standard Ultralytics ParkingManagement process method if no voting
        window = 2 * vote_radius * vote_frame_step
        if vote_radius <= 0 or stride <= window:
            # reason = (
            #     "vote_radius<=0"
            #     if vote_radius <= 0
            #     else f"stride({stride}) <= 2*vote_radius*vote_frame_step({window})"
            # )
            # print(
            #     f"[voting] frame={frame_index}: voting OFF ({reason}) -> base process() "
            #     f"with class labels [vote_radius={vote_radius}, vote_frame_step={vote_frame_step}, "
            #     f"stride={stride}]",
            #     file=sys.stderr,
            # )
            results = super().process(im0)
            self._last_region_occupied = self.region_occupancy(im0)
            return results

        # Check if the required video capture and frame index are provided
        if cap is None or frame_index is None:
            raise ValueError(
                "When majority vote is active, process() requires cap and frame_index."
            )

        # Majority vote occupancy on the frames around the anchor frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(
        #     f"[voting] frame={frame_index}: voting ON [vote_radius={vote_radius}, "
        #     f"vote_frame_step={vote_frame_step}, stride={stride}, window={window}]",
        #     file=sys.stderr,
        # )
        voted = self.majority_vote_occupancy(
            cap,
            frame_index,
            vote_radius=vote_radius,
            vote_frame_step=vote_frame_step,
            frame_count=frame_count,
        )

        # Reset the video capture to the next frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1)

        # Process the frame using the precomputed per-region occupancy
        return self.process_from_occupancy(im0, voted)
