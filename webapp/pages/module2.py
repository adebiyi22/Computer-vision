import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from camera_calibration import calibrate_camera
from perspective_dimension import measure_distance

st.set_page_config(
    page_title="Module 2 — Calibration & Measurement", layout="wide"
)
st.header("Camera Calibration & Perspective-Projection Measurement")

sub_calib, sub_measure = st.tabs(["1. Calibrate camera", "2. Measure an object"])

with sub_calib:
  st.write("Upload 8–20 checkerboard photos taken from varied angles/distances.")
  c1, c2, c3 = st.columns(3)
  cols = c1.number_input("Board columns (internal corners)", min_value=3, value=9)
  rows = c2.number_input("Board rows (internal corners)", min_value=3, value=6)
  square_size = c3.number_input("Square size (mm)", min_value=1.0, value=25.0)

  calib_files = st.file_uploader(
      "Checkerboard images",
      type=["png", "jpg", "jpeg"],
      accept_multiple_files=True,
      key="calib_files",
  )

  if st.button("Run calibration") and calib_files:
    with tempfile.TemporaryDirectory() as tmpdir:
      for f in calib_files:
        with open(os.path.join(tmpdir, f.name), "wb") as out:
          out.write(f.getbuffer())
      try:
        results = calibrate_camera(
            tmpdir,
            int(cols),
            int(rows),
            float(square_size),
            save_corner_images=False,
        )
        st.session_state["K"] = results["K"]
        st.session_state["dist"] = results["dist"]
        st.success(f"Calibrated using {len(results['used_images'])} images.")
        st.write("**Intrinsic matrix K:**")
        st.code(np.array2string(results["K"], precision=3))
        st.write(f"**Distortion coefficients:** {results['dist'].ravel()}")
        st.write(f"**Mean reprojection error:** {results['mean_error']:.4f} px")
      except Exception as e:
        st.error(str(e))

with sub_measure:
  if "K" not in st.session_state:
    st.warning("Run calibration in the first tab before measuring.")
  else:
    distance_mm = st.number_input(
        "Known camera-to-object distance (mm)", min_value=1.0, value=2500.0
    )
    obj_file = st.file_uploader(
        "Object photo", type=["png", "jpg", "jpeg"], key="obj_file"
    )
    if obj_file:
      image = Image.open(obj_file).convert("RGB")
      st.write(
          "Click two points on the image (first click = point 1, second click ="
          " point 2). Click again to reset."
      )

      if "click_points" not in st.session_state:
        st.session_state["click_points"] = []

      coords = streamlit_image_coordinates(image, key="measure_click")

      if coords is not None:
        pt = (coords["x"], coords["y"])
        pts = st.session_state["click_points"]
        if len(pts) >= 2:
          pts.clear()
        if not pts or pts[-1] != pt:
          pts.append(pt)

      pts = st.session_state["click_points"]
      st.write(f"Selected points: {pts}")

      if len(pts) == 2:
        result = measure_distance(
            pts[0],
            pts[1],
            distance_mm,
            st.session_state["K"],
            st.session_state["dist"],
        )
        st.success(
            f"Real-world distance: **{result['distance_mm']:.2f} mm**"
            f" ({result['distance_mm']/10:.2f} cm)"
        )