"""
==============================================================================
CSc 8830: Computer Vision — Combined Web Application (Modules 2, 3, 4)

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    A single Streamlit app exposing the web-demo requirement for three
    assignments in one place:
      - Module 2: camera calibration + perspective-projection dimension
        measurement
      - Module 3: image blurring, spatial convolution vs. Fourier-domain
        multiplication
      - Module 4: classical (non-ML) human boundary segmentation for RGB
        and thermal images

    Each module's underlying logic is imported unchanged from its own
    assignment script (camera_calibration.py, perspective_dimension.py,
    rgb_segmentation.py, thermal_segmentation.py) - this file only adds the
    Streamlit UI layer and wires them together, it does not duplicate or
    alter any of the actual computer-vision implementations.

REQUIREMENTS:
    pip install -r requirements.txt

USAGE (local):
    streamlit run app.py

DEPLOYMENT:
    Push this whole folder to GitHub, then deploy on Streamlit Community
    Cloud (share.streamlit.io) with the main file path set to app.py. Use
    the resulting public URL for all three modules' "web application" links.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision
==============================================================================
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2
from streamlit_image_coordinates import streamlit_image_coordinates

from camera_calibration import calibrate_camera
from perspective_dimension import measure_distance
from rgb_segmentation import segment_human_rgb
from thermal_segmentation import segment_human_thermal

st.set_page_config(page_title="CSc 8830 - Computer Vision", layout="wide")
st.title("CSc 8830: Computer Vision")
st.caption("Combined demo for Modules 2, 3, and 4")

tab2, tab3, tab4 = st.tabs([
    "Module 2 — Calibration & Measurement",
    "Module 3 — Image Blurring",
    "Module 4 — Human Segmentation",
])

# ---------------------------------------------------------------------------
# MODULE 2
# ---------------------------------------------------------------------------
with tab2:
    st.header("Camera Calibration & Perspective-Projection Measurement")
    sub_calib, sub_measure = st.tabs(["1. Calibrate camera", "2. Measure an object"])

    with sub_calib:
        st.write("Upload 8–20 checkerboard photos taken from varied angles/distances.")
        c1, c2, c3 = st.columns(3)
        cols = c1.number_input("Board columns (internal corners)", min_value=3, value=9)
        rows = c2.number_input("Board rows (internal corners)", min_value=3, value=6)
        square_size = c3.number_input("Square size (mm)", min_value=1.0, value=25.0)

        calib_files = st.file_uploader(
            "Checkerboard images", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True, key="calib_files")

        if st.button("Run calibration") and calib_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in calib_files:
                    with open(os.path.join(tmpdir, f.name), "wb") as out:
                        out.write(f.getbuffer())
                try:
                    results = calibrate_camera(
                        tmpdir, int(cols), int(rows), float(square_size),
                        save_corner_images=False)
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
            distance_mm = st.number_input("Known camera-to-object distance (mm)",
                                           min_value=1.0, value=2500.0)
            obj_file = st.file_uploader("Object photo", type=["png", "jpg", "jpeg"],
                                         key="obj_file")
            if obj_file:
                image = Image.open(obj_file).convert("RGB")
                st.write("Click two points on the image (first click = point 1, "
                         "second click = point 2). Click again to reset.")

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
                        pts[0], pts[1], distance_mm,
                        st.session_state["K"], st.session_state["dist"])
                    st.success(
                        f"Real-world distance: **{result['distance_mm']:.2f} mm** "
                        f"({result['distance_mm']/10:.2f} cm)")

# ---------------------------------------------------------------------------
# MODULE 3
# ---------------------------------------------------------------------------
with tab3:
    st.header("Image Blurring: Spatial vs. Fourier-Domain Filtering")

    def gaussian_kernel(size, sigma):
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def spatial_filter(image, kernel):
        return convolve2d(image, kernel, mode="same", boundary="wrap")

    def fourier_filter(image, kernel):
        h, w = image.shape
        kh, kw = kernel.shape
        padded = np.zeros_like(image, dtype=np.float64)
        padded[:kh, :kw] = kernel
        padded = np.roll(padded, -(kh // 2), axis=0)
        padded = np.roll(padded, -(kw // 2), axis=1)
        F = fft2(image)
        H = fft2(padded)
        return np.real(ifft2(F * H))

    def normalize(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    blur_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="blur_file")
    sigma = st.slider("Gaussian blur sigma", 0.5, 10.0, 2.0, 0.5)

    if blur_file:
        image = Image.open(blur_file).convert("L")
        arr = np.asarray(image, dtype=np.float64) / 255.0

        kernel_size = int(max(3, 2 * round(3 * sigma) + 1))
        kernel = gaussian_kernel(kernel_size, sigma)

        spatial = spatial_filter(arr, kernel)
        fourier = fourier_filter(arr, kernel)
        diff = spatial - fourier

        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        max_error = np.max(np.abs(diff))

        c1, c2, c3 = st.columns(3)
        c1.image(arr, caption="Original", clamp=True, use_container_width=True)
        c2.image(np.clip(spatial, 0, 1), caption="Spatial convolution", clamp=True,
                  use_container_width=True)
        c3.image(np.clip(fourier, 0, 1), caption="Fourier multiplication", clamp=True,
                  use_container_width=True)

        st.subheader("Validation")
        st.write(f"Kernel size: **{kernel_size} × {kernel_size}**")
        st.write(f"MAE: **{mae:.3e}**  |  RMSE: **{rmse:.3e}**  |  Max error: **{max_error:.3e}**")
        st.image(normalize(np.abs(diff)), caption="Absolute difference (normalized)",
                  clamp=True, use_container_width=True)

# ---------------------------------------------------------------------------
# MODULE 4
# ---------------------------------------------------------------------------
with tab4:
    st.header("Classical Human Boundary Segmentation (no ML/DL)")
    sub_rgb, sub_thermal = st.tabs(["RGB image", "Thermal image"])

    with sub_rgb:
        rgb_file = st.file_uploader("Upload an RGB photo of a person",
                                     type=["png", "jpg", "jpeg"], key="rgb_seg_file")
        if rgb_file:
            file_bytes = np.asarray(bytearray(rgb_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            try:
                mask, boundary, rect = segment_human_rgb(img)
                vis = img.copy()
                cv2.drawContours(vis, [boundary], -1, (0, 255, 0), 3)
                c1, c2 = st.columns(2)
                c1.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Boundary overlay",
                          use_container_width=True)
                c2.image(mask, caption="Binary mask", use_container_width=True)
                st.write(f"Foreground pixels: **{int(np.sum(mask == 255))}**")
                st.write(f"Bounding box (x,y,w,h): **{rect}**")
            except Exception as e:
                st.error(str(e))

    with sub_thermal:
        thermal_file = st.file_uploader("Upload a thermal image of a person",
                                         type=["png", "jpg", "jpeg"], key="thermal_seg_file")
        false_color = st.checkbox("Image uses a false-color palette (not grayscale)")
        if thermal_file:
            file_bytes = np.asarray(bytearray(thermal_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            try:
                mask, boundary, rect = segment_human_thermal(img, false_color=false_color)
                vis = img.copy()
                cv2.drawContours(vis, [boundary], -1, (0, 255, 0), 3)
                c1, c2 = st.columns(2)
                c1.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Boundary overlay",
                          use_container_width=True)
                c2.image(mask, caption="Binary mask", use_container_width=True)
                st.write(f"Foreground pixels: **{int(np.sum(mask == 255))}**")
                st.write(f"Bounding box (x,y,w,h): **{rect}**")
            except Exception as e:
                st.error(str(e))
