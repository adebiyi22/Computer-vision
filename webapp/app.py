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
    /* Global Apple-style typography and background */
    .stApp {
        background-color: #fbfbfd;
        color: #1d1d1f;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    
    /* Hide default Streamlit padding at the top for a cleaner hero look */
    .block-container {
        padding-top: 3rem;
        max-width: 1000px;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 4rem 1rem;
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

    /* Section Header */
    .section-header {
        font-size: 28px;
        font-weight: 600;
        color: #1d1d1f;
        letter-spacing: -0.008em;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Apple-style minimalist cards */
    .module-card {
        background-color: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 20px;
        padding: 2rem;
        height: 100%;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.02);
        transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1);
    }
    .module-card:hover {
        border-color: #d2d2d7;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        transform: translateY(-3px);
    }
    .module-tag {
        font-size: 12px;
        font-weight: 600;
        color: #0066cc;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }
    .module-title {
        font-size: 19px;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 0.5rem;
        letter-spacing: -0.005em;
    }
    .module-desc {
        font-size: 14px;
        color: #515154;
        line-height: 1.5;
        margin: 0;
    }

    /* Footer instruction banner */
    .nav-footer {
        background-color: #f5f5f7;
        border-radius: 14px;
        padding: 1.25rem 2rem;
        text-align: center;
        margin-top: 4rem;
        font-size: 15px;
        color: #1d1d1f;
        font-weight: 500;
        border: 1px solid #e5e5ea;
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

st.markdown('<div class="section-header">Explore Modules</div>', unsafe_allow_html=True)

# --- MODULE GRID (Apple Product-Tile Style Layout) ---
col1, col2 = st.columns(2, gap="large")

with col1:
  st.markdown(
      """
        <div class="module-card">
            <div class="module-tag">Module 02</div>
            <div class="module-title">Camera Calibration & Measurement</div>
            <p class="module-desc">Calibrate camera parameters and execute precise perspective-projection measurements in real-world coordinates.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )
  st.write("")  # spacing
  st.markdown(
      """
        <div class="module-card">
            <div class="module-tag">Module 04</div>
            <p class="module-title">Human Boundary Segmentation</p>
            <p class="module-desc">Classical computer vision boundary extraction pipelines built for RGB and thermal imaging without heavy machine learning dependencies.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

with col2:
  st.markdown(
      """
        <div class="module-card">
            <div class="module-tag">Module 03</div>
            <div class="module-title">Spatial vs. Fourier Blurring</div>
            <p class="module-desc">Analyze image blurring behaviors by comparing direct spatial convolution against optimized frequency-domain filtering.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )
  st.write("")  # spacing
  st.markdown(
      """
        <div class="module-card">
            <div class="module-tag">Module 06</div>
            <div class="module-title">Optical Flow, Lucas-Kanade & SfM</div>
            <p class="module-desc">Compute dense optical flow, track sparse features iteratively, and reconstruct planar structures from motion.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

# --- NAVIGATION FOOTER ---
st.markdown(
    """
    <div class="nav-footer">
        👉 <b>To begin:</b> Open the sidebar menu on the left to navigate directly into any interactive module workspace.
    </div>
    """,
    unsafe_allow_html=True,
)
