"""
Module 3: Image Blurring — Spatial vs. Fourier-Domain Filtering
"""

import numpy as np
import streamlit as st
from PIL import Image
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2

st.set_page_config(page_title="Module 3 — Image Blurring", layout="wide")
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