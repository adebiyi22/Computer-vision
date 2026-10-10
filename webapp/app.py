"""
==============================================================================
CSc 8830: Computer Vision — Main Portfolio Hub
==============================================================================
"""

import streamlit as st

st.set_page_config(
    page_title="CSc 8830 - Computer Vision Portfolio",
    page_icon="👁️",
    layout="wide",
)

st.title("CSc 8830: Computer Vision Portfolio 👁️🚀")
st.caption("Interactive web applications for weekly labs and modules")

st.markdown("""
Welcome to my computer vision web application! You can use the **sidebar navigation on the left** 
(click the arrow in the top-left if it's hidden) or use the quick links below to jump straight to any module.
""")

st.subheader("Quick Navigation Links:")

# Direct clickable buttons to your module pages
st.page_link("pages/module2.py", label="Module 2: Camera Calibration & Measurement", icon="📐")
st.page_link("pages/module3.py", label="Module 3: Image Blurring (Spatial vs. Fourier)", icon="🌊")
st.page_link("pages/module4.py", label="Module 4: Classical Human Boundary Segmentation", icon="👤")
st.page_link("pages/module6.py", label="Module 6: Optical Flow, Lucas-Kanade & Planar SfM", icon="🎥")

st.info("👈 You can also use the sidebar menu at any time to switch between modules.")
