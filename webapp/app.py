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
Welcome to my computer vision web application! Use the **sidebar navigation on the left** 
to switch between weekly modules and assignments.

### Current Modules Available:
- **Module 2:** Camera Calibration & Perspective-Projection Measurement[cite: 1]
- **Module 3:** Image Blurring (Spatial vs. Fourier-Domain)[cite: 1]
- **Module 4:** Classical Human Boundary Segmentation[cite: 1]
- **Module 6:** Optical Flow, Lucas-Kanade Tracking, & Planar SfM
- *More weekly modules coming soon...*
""")

st.info("👈 Click any module in the sidebar to open its interactive workspace.")
