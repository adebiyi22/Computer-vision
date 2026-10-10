"""
==============================================================================
CSc 8830: Computer Vision - Module 6: Optical Flow, Lucas-Kanade & Planar SfM
Complete Portfolio Implementation (Tabs 1, 2, and 3)
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
# TAB 1: DENSE OPTICAL FLOW (FARNEBACK)
# ==========================================
with tab1:
  st.subheader("Dense Optical Flow")
  st.write(
      "Compute dense motion vectors across all pixels using Farneback's"
      " classical polynomial-expansion algorithm[cite: 2]."
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

    tfile_lk = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_lk.write(uploaded_file_2.read())
    lk_video_path = tfile_lk.name

    cap_lk = cv2.VideoCapture(lk_video_path)
    if not cap_lk.isOpened():
      st.error("Could not open the uploaded video for Lucas-Kanade.")
    else:
      fps_lk = cap_lk.get(cv2.CAP_PROP_FPS) or 30.0
      max_frames_lk = int(30.0 * fps_lk)

      feature_params = dict(
          maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
      )
      lk_params = dict(
          winSize=(15, 15),
          maxLevel=2,
          criteria=(
              cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              10,
              0.03,
          ),
      )

      color = np.random.randint(0, 255, (100, 3))

      ret, old_frame = cap_lk.read()
      if not ret:
        st.error("Could not read the first frame.")
      else:
        old_frame = cv2.resize(old_frame, (640, 360))
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, mask=None, **feature_params
        )

        mask = np.zeros_like(old_frame)

        st.write("Processing Sparse Lucas-Kanade Tracking live...")
        progress_bar_lk = st.progress(0)
        frame_placeholder_lk = st.empty()

        frame_idx_lk = 0

        while cap_lk.isOpened() and frame_idx_lk < max_frames_lk:
          ret, frame = cap_lk.read()
          if not ret:
            break

          frame = cv2.resize(frame, (640, 360))
          frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          p1, st_flow, err = cv2.calcOpticalFlowPyrLK(
              old_gray, frame_gray, p0, None, **lk_params
          )

          if p1 is not None:
            good_new = p1[st_flow == 1]
            good_old = p0[st_flow == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
              a, b = new.ravel()
              c, d = old.ravel()
              mask = cv2.line(
                  mask,
                  (int(a), int(b)),
                  (int(c), int(d)),
                  color[i].tolist(),
                  2,
              )
              frame = cv2.circle(
                  frame, (int(a), int(b)), 5, color[i].tolist(), -1
              )

            img_display = cv2.add(frame, mask)

            frame_placeholder_lk.image(
                img_display,
                channels="BGR",
                caption=f"Lucas-Kanade Frame {frame_idx_lk}",
            )

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

          frame_idx_lk += 1
          progress_bar_lk.progress(min(frame_idx_lk / max_frames_lk, 1.0))

        cap_lk.release()
        st.success("Lucas-Kanade feature tracking complete!")

# ==========================================
# TAB 3: PLANAR STRUCTURE FROM MOTION (SFM)
# ==========================================
with tab3:
  st.subheader("Planar Structure from Motion (SfM)")
  st.write(
      "Recover camera motion and planar feature correspondences between"
      " sequential images."
  )

  uploaded_files_3 = st.file_uploader(
      "Upload image sequence for Planar SfM (2 or more images)",
      type=["jpg", "png", "jpeg"],
      accept_multiple_files=True,
      key="sfm_upload",
  )

  if uploaded_files_3 and len(uploaded_files_3) >= 2:
    st.info(
        f"Successfully loaded {len(uploaded_files_3)} images for Planar SfM"
        " analysis."
    )

    if st.button("Run Planar SfM Pipeline", key="run_sfm_btn"):
      with st.spinner("Detecting features and computing homographies..."):
        images = []
        for uploaded_file in sorted(uploaded_files_3, key=lambda x: x.name):
          file_bytes = np.asarray(
              bytearray(uploaded_file.read()), dtype=np.uint8
          )
          img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
          if img is not None:
            img = cv2.resize(img, (640, 480))
            images.append(img)

        if len(images) < 2:
          st.error(
              "Please upload at least 2 valid images that can be processed."
          )
        else:
          orb = cv2.ORB_create(nfeatures=500)
          bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

          st.write("---")
          st.write("### Frame-to-Frame Homography & Feature Matches")

          for i in range(len(images) - 1):
            img1 = images[i]
            img2 = images[i + 1]

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)

            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
              matches = bf.match(des1, des2)
              matches = sorted(matches, key=lambda x: x.distance)

              src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
              dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

              H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

              match_img = cv2.drawMatches(
                  img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
              )

              st.write(f"**Image Pair {i+1} $\rightarrow$ Image {i+2}**")
              st.image(
                  match_img,
                  channels="BGR",
                  caption=f"ORB Feature Correspondences (Pair {i+1} & {i+2})",
              )

              if H is not None:
                with st.expander(f"View Estimated Homography Matrix (Pair {i+1} & {i+2})"):
                  st.code(str(H))
            else:
              st.warning(f"Not enough features detected between image {i+1} and {i+2}.")

          st.success("Planar Structure from Motion processing complete!")
  elif uploaded_files_3 and len(uploaded_files_3) < 2:
    st.warning("Please upload at least 2 images to perform planar SfM and homography matching.")
