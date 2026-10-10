"""
==============================================================================
CSc 8830: Computer Vision - Module 6: Optical Flow, Lucas-Kanade & Planar SfM
Part A: Dense Optical Flow Web Implementation
==============================================================================
"""

import tempfile
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Module 6: Optical Flow & SfM", page_icon="🎥", layout="wide"
)

st.title("Module 6: Optical Flow, Lucas-Kanade & Planar SfM")

# Navigation tabs for Module 6
tab1, tab2, tab3 = st.tabs(
    [
        "1. Optical Flow (Dense)",
        "2. Lucas-Kanade Tracking (Sparse)",
        "3. Planar Structure from Motion (SfM)",
    ]
)

with tab1:
  st.subheader("Dense Optical Flow")
  st.write(
      "Compute dense motion vectors across all pixels using Farneback's"
      " classical polynomial-expansion algorithm[cite: 2]."
  )

  uploaded_file = st.file_uploader(
      "Upload video for Dense Optical Flow", type=["mp4", "mov", "avi"]
  )

  if uploaded_file is not None:
    st.info("Video loaded for dense optical flow processing.")

    # Save uploaded file temporarily so OpenCV can read it frame-by-frame
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      st.error("Could not open the uploaded video.")
    else:
      fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
      max_frames = int(30.0 * fps)  # Cap at 30 seconds for web responsiveness

      ret, prev = cap.read()
      if not ret:
        st.error("Could not read the first frame from the video.")
      else:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        
        # Pre-allocate HSV frame (Hue=Direction, Saturation=255, Value=Magnitude)[cite: 2]
        hsv = np.zeros_like(prev)
        hsv[..., 1] = 255  

        st.write("Processing frames and rendering live in the web interface...")
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()

        frame_idx = 0
        mean_mags, max_mags = [], []

        while frame_idx < max_frames:
          ret, frame = cap.read()
          if not ret:
            break
          
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          # Compute Farneback dense optical flow[cite: 2]
          flow = cv2.calcOpticalFlowFarneback(
              prev_gray,
              gray,
              None,
              pyr_scale=0.5,
              levels=3,
              winsize=15,
              iterations=3,
              poly_n=5,
              poly_sigma=1.2,
              flags=0,
          )

          # Convert cartesian flow vectors to polar coordinates (magnitude & angle)
          mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
          
          # Map to HSV color space[cite: 2]
          hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction[cite: 2]
          hsv[..., 2] = cv2.normalize(
              mag, None, 0, 255, cv2.NORM_MINMAX
          )  # Value = magnitude[cite: 2]
          
          bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

          # Render the processed frame live directly into the browser interface
          frame_placeholder.image(
              bgr_flow,
              channels="BGR",
              caption=f"Dense Optical Flow - Frame {frame_idx}",
          )

          mean_mags.append(float(np.mean(mag)))
          max_mags.append(float(np.max(mag)))

          prev_gray = gray
          frame_idx += 1
          progress_bar.progress(min(frame_idx / max_frames, 1.0))

        cap.release()
        st.success("Optical flow processing complete!")

        # Display performance statistics for report evidence[cite: 2]
        col1, col2 = st.columns(2)
        with col1:
          st.metric("Mean Flow Magnitude", f"{np.mean(mean_mags):.3f} px")
        with col2:
          st.metric("Peak Flow Magnitude", f"{np.max(max_mags):.3f} px")

with tab2:
  st.subheader("Lucas-Kanade Tracking (Sparse)")
  st.write("Implement sparse feature tracking using iterative Lucas-Kanade.")

with tab3:
  st.subheader("Planar Structure from Motion (SfM)")
  st.write("Recover camera motion from planar feature correspondences.")
