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
Welcome to my computer vision web application! 

### Available Modules:
* **Module 2:** Camera Calibration & Perspective-Projection Measurement
* **Module 3:** Image Blurring (Spatial vs. Fourier-Domain)
* **Module 4:** Classical Human Boundary Segmentation
* **Module 6:** Optical Flow, Lucas-Kanade Tracking, & Planar SfM
* *More weekly modules coming soon...*
""")

st.info(
    "👉 **How to navigate:** Click on any of the module pages in the **sidebar menu on the left** "
    "to open its interactive workspace."
)
