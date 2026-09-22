"""
CSc 8830 - Module 3 Assignment
Image Blurring Using Spatial and Fourier-Domain Filtering

Run:
    pip install -r requirements.txt
    python -m streamlit run app.py

The application demonstrates:
1. Spatial Gaussian filtering (convolution).
2. Fourier-domain filtering (FFT -> multiplication -> inverse FFT).
3. Numerical comparison of the two outputs.
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2

st.set_page_config(page_title="Module 3 - Image Blurring", layout="wide")
st.title("CSc 8830 — Module 3: Image Blurring")
st.write(
    "Compare spatial-domain convolution with equivalent Fourier-domain "
    "multiplication."
)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
sigma = st.slider("Gaussian blur σ", 0.5, 10.0, 2.0, 0.5)

def gaussian_kernel(size, sigma):
    """Create a normalized 2-D Gaussian kernel."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def spatial_filter(image, kernel):
    """Direct spatial-domain convolution with circular (wrap) boundaries."""
    return convolve2d(
        image,
        kernel,
        mode="same",
        boundary="wrap",
    )

def fourier_filter(image, kernel):
    """Fourier-domain filtering: FFT(image) * FFT(kernel), then inverse FFT."""
    h, w = image.shape
    kh, kw = kernel.shape

    # Embed and circularly shift the kernel so its center is at (0, 0).
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

if uploaded:
    image = Image.open(uploaded).convert("L")
    arr = np.asarray(image, dtype=np.float64) / 255.0

    kernel_size = int(max(3, 2 * round(3 * sigma) + 1))
    kernel = gaussian_kernel(kernel_size, sigma)

    spatial = spatial_filter(arr, kernel)
    fourier = fourier_filter(arr, kernel)

    diff = spatial - fourier
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.image(arr, caption="Original", clamp=True, use_container_width=True)
    c2.image(np.clip(spatial, 0, 1), caption="Spatial convolution", clamp=True,
             use_container_width=True)
    c3.image(np.clip(fourier, 0, 1), caption="Fourier multiplication", clamp=True,
             use_container_width=True)

    st.subheader("Validation")
    st.write(f"Kernel size: **{kernel_size} × {kernel_size}**")
    st.write(f"Mean Absolute Error (MAE): **{mae:.12e}**")
    st.write(f"Root Mean Square Error (RMSE): **{rmse:.12e}**")
    st.write(f"Maximum absolute difference: **{max_error:.12e}**")

    st.image(
        normalize(np.abs(diff)),
        caption="Absolute difference (normalized for visualization)",
        clamp=True,
        use_container_width=True,
    )

    st.subheader("Gaussian Kernel")
    st.dataframe(kernel, use_container_width=True)

    st.download_button(
        "Download spatial result",
        data=io.BytesIO(np.uint8(np.clip(spatial, 0, 1) * 255)).getvalue(),
        file_name="spatial_blur.png",
        mime="image/png",
    )
    st.download_button(
        "Download Fourier result",
        data=io.BytesIO(np.uint8(np.clip(fourier, 0, 1) * 255)).getvalue(),
        file_name="fourier_blur.png",
        mime="image/png",
    )
else:
    st.info("Upload a grayscale-compatible image to begin.")
