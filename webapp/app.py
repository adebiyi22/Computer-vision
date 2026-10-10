"""
==============================================================================
CSc 8830: Computer Vision — Main Portfolio Hub (Single-Page Router)
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

# Create a clean selector box on the main page to choose a module
st.markdown("### Select a Module to Enter:")
selected_module = st.selectbox(
    "Choose from your weekly computer vision labs:",
    [
        "-- Select a Module --",
        "Module 2: Camera Calibration & Measurement",
        "Module 3: Image Blurring (Spatial vs. Fourier)",
        "Module 4: Classical Human Boundary Segmentation",
        "Module 6: Optical Flow, Lucas-Kanade & Planar SfM",
    ],
)

st.divider()

# Load the respective module based on user selection
if selected_module == "-- Select a Module --":
  st.info("👆 Please select a module from the dropdown above to get started.")

elif selected_module == "Module 2: Camera Calibration & Measurement":
  st.header("Module 2: Camera Calibration & Measurement")
  # Put your Module 2 code or import it here
  st.write("Loading Module 2 interface...")
  # from module2 import run_module_2
  # run_module_2()

elif selected_module == "Module 3: Image Blurring (Spatial vs. Fourier)":
  st.header("Module 3: Image Blurring: Spatial vs. Fourier-Domain Filtering")
  # Put your Module 3 code or import it here
  st.write("Loading Module 3 interface...")
  # from module3 import run_module_3
  # run_module_3()

elif selected_module == "Module 4: Classical Human Boundary Segmentation":
  st.header("Module 4: Classical Human Boundary Segmentation")
  # Put your Module 4 code or import it here
  st.write("Loading Module 4 interface...")

elif (
    selected_module == "Module 6: Optical Flow, Lucas-Kanade & Planar SfM"
):
  st.header("Module 6: Optical Flow, Lucas-Kanade & Planar SfM")
  # Put your Module 6 code or import it here
  st.write("Loading Module 6 interface...")
