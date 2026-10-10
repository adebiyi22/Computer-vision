"""
==============================================================================
CSc 8830: Computer Vision — Apple-Inspired Portfolio Hub
==============================================================================
"""

import streamlit as st

st.set_page_config(
    page_title="Computer Vision Portfolio",
    page_icon="👁️",
    layout="wide",
)

# Custom Apple-inspired CSS injection
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fbfbfd;
        color: #1d1d1f;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 3rem;
        max-width: 1000px;
    }
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 3rem 1rem;
    }
    .hero-eyebrow {
        font-size: 13px;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 52px;
        font-weight: 700;
        color: #1d1d1f;
        letter-spacing: -0.015em;
        margin-bottom: 1rem;
        line-height: 1.08;
    }
    .hero-subtitle {
        font-size: 21px;
        font-weight: 400;
        color: #515154;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.45;
        letter-spacing: 0.01em;
    }
    .section-header {
        font-size: 28px;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.008em;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HERO SECTION ---
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-eyebrow">CSc 8830 • Advanced Computer Vision</div>
        <div class="hero-title">Vision. Precision. Code.</div>
        <div class="hero-subtitle">
            An exploration of spatial filtering, classical segmentation, and advanced motion tracking, engineered for performance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-header">Select a Module to Open</div>',
    unsafe_allow_html=True,
)

# --- DIRECT CLICKABLE MODULE LINKS ---
# Using 2 columns to present clean, clickable cards that route directly to your pages
col1, col2 = st.columns(2, gap="large")

with col1:
  st.markdown("### 📐 Module 02")
  st.write("Camera Calibration & Perspective-Projection Measurement.")
  st.page_link(
      "pages/module2.py",
      label="Open Module 2 Workspace",
      icon="🚀",
  )

  st.markdown("---")

  st.markdown("### 👤 Module 04")
  st.write(
      "Classical computer vision boundary extraction for RGB and thermal"
      " imaging."
  )
  st.page_link(
      "pages/module4.py",
      label="Open Module 4 Workspace",
      icon="🚀",
  )

with col2:
  st.markdown("### 🌊 Module 03")
  st.write("Spatial convolution vs. optimized frequency-domain filtering.")
  st.page_link(
      "pages/module3.py",
      label="Open Module 3 Workspace",
      icon="🚀",
  )

  st.markdown("---")

  st.markdown("### 🎥 Module 06")
  st.write("Optical Flow, Lucas-Kanade sparse tracking, and Planar SfM.")
  st.page_link(
      "pages/module6.py",
      label="Open Module 6 Workspace",
      icon="🚀",
  )
