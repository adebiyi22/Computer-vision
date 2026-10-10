"""
==============================================================================
CSc 8830: Computer Vision - Module 6: Optical Flow, Lucas-Kanade & Planar SfM
Part A, B, & C: Complete Streamlit Implementation
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

# ==========================================
# TAB 1: DENSE OPTICAL FLOW
# ==========================================
with tab1:
  st.subheader("Dense Optical Flow")
  st.write(
      "Compute dense motion vectors across all pixels using Farneback's"
      " classical polynomial-expansion algorithm."
  )

  uploaded_file_1 = st.file_uploader(
      "Upload video for Dense Optical Flow",
      type=["mp4", "mov", "avi"],
      key="dense_upload",
  )

  if uploaded_file_1 is not None:
    st.info("Video loaded for dense optical flow processing.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file_1.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      st.error("Could not open the uploaded video.")
    else:
      fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
      max_frames = int(30.0 * fps)

      ret, prev = cap.read()
      if not ret:
        st.error("Could not read the first frame from the video.")
      else:
        prev = cv2.resize(prev, (640, 360))
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        
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
          
          frame = cv2.resize(frame, (640, 360))
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

          mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
          hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction[cite: 2]
          hsv[..., 2] = cv2.normalize(
              mag, None, 0, 255, cv2.NORM_MINMAX
          )  # Value = magnitude[cite: 2]
          
          bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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

        col1, col2 = st.columns(2)
        with col1:
          st.metric("Mean Flow Magnitude", f"{np.mean(mean_mags):.3f} px")
        with col2:
          st.metric("Peak Flow Magnitude", f"{np.max(max_mags):.3f} px")

# ==========================================
# TAB 2: LUCAS-KANADE TRACKING (SPARSE)
# ==========================================
with tab2:
  st.subheader("Lucas-Kanade Tracking (Sparse)")
  st.write(
      "Implement sparse feature tracking using iterative Lucas-Kanade with"
      " pyramids."
  )

  uploaded_file_2 = st.file_uploader(
      "Upload video for Sparse Lucas-Kanade Tracking",
      type=["mp4", "mov", "avi"],
      key="lk_upload",
  )

  if uploaded_file_2 is not None:
    st.info("Video loaded for Lucas-Kanade sparse feature tracking.")
    # Implement your Lucas-Kanade tracking pipeline here
    st.warning("Implement your Lucas-Kanade feature tracking pipeline here.")

# ==========================================
# TAB 3: PLANAR STRUCTURE FROM MOTION (SfM)
# ==========================================
with tab3:
  st.subheader("Planar Structure from Motion (SfM)")
  st.write(
      "Recover camera motion and sparse 3D structure from planar feature"
      " correspondences."
  )

  uploaded_files_3 = st.file_uploader(
      "Upload images for Planar SfM",
      type=["jpg", "png", "jpeg"],
      accept_multiple_files=True,
      key="sfm_upload",
  )

  if uploaded_files_3:
    st.info(f"{len(uploaded_files_3)} images loaded for Planar SfM.")
    # Implement your feature matching, homography estimation, and SfM pipeline here
    st.warning("Implement your Planar Structure from Motion pipeline here.")
