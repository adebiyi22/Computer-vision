"""
Module 4: Classical Human Boundary Segmentation (no ML/DL)
"""

import cv2
import numpy as np
import streamlit as st

from rgb_segmentation import segment_human_rgb
from thermal_segmentation import segment_human_thermal

st.set_page_config(page_title="Module 4 — Human Segmentation", layout="wide")
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