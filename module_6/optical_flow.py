"""
==============================================================================
CSc 8830: Computer Vision - Assignment 6, Part A (Step 1)
Dense Optical Flow Computation and Visualization

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Computes dense optical flow (Farneback's method, a classical
    polynomial-expansion-based algorithm - no ML/DL) between every
    consecutive pair of frames in a ~30-second video clip, and renders the
    flow field as a color video: HUE encodes flow DIRECTION (angle) and
    VALUE/BRIGHTNESS encodes flow MAGNITUDE (speed). This is the standard
    HSV optical-flow visualization convention.

WHAT OPTICAL FLOW TELLS YOU (see report for full discussion + evidence):
    - Direction of motion of each pixel/region between frames (hue)
    - Speed of motion (brightness/saturation of the color)
    - Which regions are moving vs. static (static regions -> near-zero flow
      -> appear black in the visualization)
    - Can reveal camera motion (global, uniform flow pattern across the
      whole frame) vs. independent object motion (localized flow regions
      that differ from the background's flow)

REQUIREMENTS:
    pip install opencv-python numpy

USAGE:
    python optical_flow.py --video my_video.mp4 --out flow_visualization.mp4

OUTPUT:
    Saves an .mp4 showing the color-coded optical flow for every frame
    pair, and prints summary statistics (mean/max flow magnitude per
    frame) useful for the report's "evidence" discussion.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Assignment 6
==============================================================================
"""

import argparse
import cv2
import numpy as np


def compute_and_visualize_flow(video_path, out_path, max_seconds=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame.")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255  # full saturation; hue=direction, value=magnitude

    frame_idx = 0
    mean_mags, max_mags = [], []

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Farneback dense optical flow: classical, polynomial-expansion
        # based method (Gunnar Farneback, 2003) - not a learned model.
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2       # hue = direction
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        writer.write(bgr_flow)
        mean_mags.append(float(np.mean(mag)))
        max_mags.append(float(np.max(mag)))

        prev_gray = gray
        frame_idx += 1

    cap.release()
    writer.release()
    return mean_mags, max_mags


def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualize dense optical flow (Farneback) for a video clip.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="flow_visualization.mp4")
    parser.add_argument("--max_seconds", type=float, default=30.0)
    args = parser.parse_args()

    mean_mags, max_mags = compute_and_visualize_flow(args.video, args.out, args.max_seconds)

    print(f"Processed {len(mean_mags)} frame pairs.")
    print(f"Mean flow magnitude across clip: {np.mean(mean_mags):.3f} px")
    print(f"Peak flow magnitude observed:    {np.max(max_mags):.3f} px")
    print(f"Saved flow visualization to: {args.out}")


if __name__ == "__main__":
    main()
