"""
==============================================================================
CSc 8830: Computer Vision - Assignment 6, Part A (Step 1)
Dense Optical Flow Computation and Visualization (Standalone Script)

PURPOSE:
    Computes dense optical flow using Gunnar Farneback's classical 
    polynomial-expansion algorithm (2003) between consecutive frame pairs 
    of a video clip. It visualizes motion using the standard HSV convention:
    - HUE: Encodes motion direction (angle).
    - VALUE / BRIGHTNESS: Encodes motion magnitude (speed).
==============================================================================
"""

import argparse
import sys
import cv2
import numpy as np


def compute_and_visualize_flow(video_path, out_path, max_seconds=30):
    """Computes dense optical flow for a video clip and saves the result as an MP4.

    Args:
        video_path (str): Path to the input video file.
        out_path (str): Path where the output visualization video will be saved.
        max_seconds (float): Maximum duration of the video to process.

    Returns:
        tuple: Lists containing mean and maximum flow magnitudes per frame pair.
    """
    # 1. Initialize video capture from the source file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    # Extract video metadata (FPS, width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback FPS if metadata is missing

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid video dimensions detected: {w}x{h}. Check the video file.")

    max_frames = int(max_seconds * fps)

    # 2. Initialize video writer with fallback codecs for cross-platform compatibility
    fourcc = 0
    writer = None
    for codec in ["mp4v", "XVID", "MJPG"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"Using video codec: {codec}")
            break
        writer.release()

    if writer is None or not writer.isOpened():
        raise RuntimeError("Failed to initialize OpenCV VideoWriter with available codecs.")

    # 3. Read the initial frame and convert to grayscale for optical flow analysis
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame from the video.")
    
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    # Pre-allocate HSV visualization matrix (Hue=Direction, Saturation=Maximum, Value=Magnitude)
    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255  

    frame_idx = 0
    mean_mags, max_mags = [], []

    print(f"Processing video (Target FPS: {fps:.1f}, Max Frames: {max_frames})...")

    # 4. Main frame-by-frame processing loop
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Farneback dense optical flow across all pixels
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        # Convert Cartesian flow vectors (dx, dy) to Polar coordinates (Magnitude, Angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map optical flow properties to HSV color space
        hsv[..., 0] = ang * 180 / np.pi / 2       # Hue corresponds to motion direction
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value corresponds to speed
        
        # Convert HSV visualization frame back to BGR format for video writing
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Write frame to output and collect summary statistics for reporting
        writer.write(bgr_flow)
        mean_mags.append(float(np.mean(mag)))
        max_mags.append(float(np.max(mag)))

        # Update previous frame state
        prev_gray = gray
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    # 5. Clean up resources
    cap.release()
    writer.release()
    return mean_mags, max_mags


def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Compute and visualize dense optical flow (Farneback) for a video clip."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default="flow_visualization.mp4", help="Path to output video file")
    parser.add_argument("--max_seconds", type=float, default=30.0, help="Max duration to process")
    args = parser.parse_args()

    try:
        # Execute processing pipeline and print performance statistics
        mean_mags, max_mags = compute_and_visualize_flow(args.video, args.out, args.max_seconds)

        print("\n--- Processing Complete ---")
        print(f"Processed {len(mean_mags)} frame pairs.")
        print(f"Mean flow magnitude across clip: {np.mean(mean_mags):.3f} px")
        print(f"Peak flow magnitude observed:    {np.max(max_mags):.3f} px")
        print(f"Saved flow visualization to: {args.out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
