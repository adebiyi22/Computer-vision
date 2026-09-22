# Theory: Equivalence of Spatial Convolution and Fourier-Domain Multiplication

## 1. Statement of what is to be shown

Let `f(x,y)` be an image and `h(x,y)` a filter (point spread function). Spatial
filtering (2D convolution) is defined as:

```
g(x,y) = (f * h)(x,y) = sum_m sum_n f(m,n) h(x-m, y-n)
```

(discrete form; the continuous form replaces sums with integrals).

We want to show:

```
F{ f * h } = F{f} · F{h}
```

that is, convolving two functions in the spatial domain is equivalent to
taking the Fourier transform of each, multiplying them point-by-point, and
that this product equals the Fourier transform of the convolution. Then, by
applying the inverse transform:

```
g(x,y) = F^-1{ F(u,v) H(u,v) }
```

## 2. Definitions

The 2D discrete Fourier transform (DFT) of an M x N image f(x,y) is:

```
F(u,v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x,y) * exp(-j2*pi*(ux/M + vy/N))
```

and the inverse DFT is:

```
f(x,y) = (1/MN) * sum_{u=0}^{M-1} sum_{v=0}^{N-1} F(u,v) * exp(+j2*pi*(ux/M + vy/N))
```

Circular (periodic) convolution of f and h (both size M x N, with h
zero-padded/embedded into an M x N array as needed) is defined as:

```
g(x,y) = sum_{m=0}^{M-1} sum_{n=0}^{N-1} f(m,n) * h(x-m mod M, y-n mod N)
```

## 3. Derivation

Take the DFT of g:

```
G(u,v) = sum_x sum_y g(x,y) exp(-j2*pi*(ux/M + vy/N))

       = sum_x sum_y [ sum_m sum_n f(m,n) h(x-m, y-n) ] exp(-j2*pi*(ux/M + vy/N))
```

Exchange the order of summation (finite sums, so this is always valid):

```
       = sum_m sum_n f(m,n) [ sum_x sum_y h(x-m, y-n) exp(-j2*pi*(ux/M + vy/N)) ]
```

Substitute `x' = x - m` (mod M) and `y' = y - n` (mod N). Because the
summation over x and y is over one full period (0..M-1, 0..N-1) and h is
treated as periodic (circular convolution), the inner sum over the shifted
variables x', y' still runs over one full period:

```
sum_x sum_y h(x-m, y-n) exp(-j2*pi*(ux/M + vy/N))
    = sum_x' sum_y' h(x', y') exp(-j2*pi*(u(x'+m)/M + v(y'+n)/N))
    = exp(-j2*pi*(um/M + vn/N)) * sum_x' sum_y' h(x', y') exp(-j2*pi*(ux'/M + vy'/N))
    = exp(-j2*pi*(um/M + vn/N)) * H(u,v)
```

Substituting back:

```
G(u,v) = sum_m sum_n f(m,n) * exp(-j2*pi*(um/M + vn/N)) * H(u,v)

       = H(u,v) * sum_m sum_n f(m,n) exp(-j2*pi*(um/M + vn/N))

       = H(u,v) * F(u,v)
```

Therefore:

```
G(u,v) = F(u,v) * H(u,v)
```

which is exactly the claim: the Fourier transform of the (circular)
convolution equals the pointwise product of the individual Fourier
transforms. Applying the inverse DFT to both sides:

```
g(x,y) = F^-1{ G(u,v) } = F^-1{ F(u,v) H(u,v) }
```

This is the **Convolution Theorem**.

## 4. Why the implementation is required to match this exactly

The derivation above assumes **circular (periodic) convolution**, because
the DFT inherently treats a signal as periodic. This is why the
implementation:

1. Uses `boundary="wrap"` in the spatial-domain convolution
   (`scipy.signal.convolve2d(..., boundary="wrap")`), rather than the
   default zero-padding boundary. Zero-padding at the edges corresponds to
   **linear** convolution, not circular convolution, and would not exactly
   match the FFT-domain result at the image borders.
2. Zero-pads the kernel `h` to the same size as the image before taking its
   FFT, and circularly shifts it (`np.roll`) so its center sits at index
   (0,0). This matches the assumption in the derivation that `h(x,y)` is
   defined (and periodic) over the same M x N domain as `f(x,y)`, with its
   origin at (0,0) rather than at the middle of the kernel array. Without
   this shift, the Fourier-domain result would be spatially shifted
   relative to the spatial-domain result (a correct product, but shifted in
   position by half the kernel size).

With both of these matched, the numerical experiment (Section 5) confirms
the spatial and Fourier results agree to floating-point precision — proving
the theorem holds not just abstractly, but for this exact implementation.

## 5. Experimental validation (evidence)

Using the Gaussian-blur implementation from this assignment
(`gaussian_kernel`, `spatial_filter`, `fourier_filter` in `app.py`), a
256x256 synthetic test image containing sharp edges (rectangle, ellipse,
triangle, and a striped region) was filtered both ways across ten different
blur strengths (sigma = 0.5 to 10), and the pixelwise difference between the
spatial-domain result and the Fourier-domain result was measured:

| Sigma | Kernel size | MAE | RMSE | Max abs error |
|---|---|---|---|---|
| 0.5 | 5x5 | 1.00e-16 | 1.34e-16 | 6.94e-16 |
| 1.0 | 7x7 | 1.02e-16 | 1.39e-16 | 6.66e-16 |
| 1.5 | 11x11 | 1.02e-16 | 1.28e-16 | 7.08e-16 |
| 2.0 | 13x13 | 1.25e-16 | 1.65e-16 | 7.77e-16 |
| 3.0 | 19x19 | 1.48e-16 | 2.09e-16 | 1.11e-15 |
| 4.0 | 25x25 | 1.13e-16 | 1.53e-16 | 7.77e-16 |
| 5.0 | 31x31 | 1.47e-16 | 2.19e-16 | 1.33e-15 |
| 6.0 | 37x37 | 1.44e-16 | 1.88e-16 | 1.11e-15 |
| 8.0 | 49x49 | 2.44e-16 | 3.46e-16 | 1.55e-15 |
| 10.0 | 61x61 | 2.95e-16 | 4.53e-16 | 2.11e-15 |

All errors sit at the level of IEEE-754 double-precision floating point
round-off (~1e-16), regardless of blur strength or kernel size. This is
exactly the behavior predicted by the theorem: the spatial and Fourier
results are the *same* function evaluated two different ways, so any
observed difference is attributable purely to floating-point arithmetic
error accumulated during the FFT/IFFT and summation operations, not to any
real discrepancy between the two approaches.

See `figures/original.png`, `figures/spatial.png`, `figures/fourier.png`,
`figures/difference.png`, and `figures/error_vs_sigma.png` for the visual
evidence (the spatial and Fourier outputs are visually indistinguishable,
and the difference map is uniform near-zero noise at the limits of display
precision).
