# Theory: Edge Detection and Region Segmentation via Fourier (Frequency) Domain Analysis

## 1. Overview

An image f(x,y) can be decomposed into a sum of sinusoidal basis functions
via its 2D Fourier transform F(u,v). Slowly-varying regions of roughly
constant intensity (the interior of objects/background) are represented by
**low spatial frequencies**, while sharp intensity discontinuities (object
boundaries/edges) are represented by **high spatial frequencies**. This
theory section derives that relationship and shows how it leads directly to
frequency-domain edge detection (high-pass filtering) and region
segmentation (low-pass filtering / power-spectrum analysis).

## 2. The Differentiation Theorem (why edges = high frequencies)

The 2D continuous Fourier transform pair is:

```
F(u,v) = integral integral f(x,y) exp(-j2*pi*(ux+vy)) dx dy
f(x,y) = integral integral F(u,v) exp(+j2*pi*(ux+vy)) du dv
```

Differentiate f(x,y) with respect to x under the inverse transform integral:

```
df/dx = integral integral F(u,v) * d/dx[exp(j2*pi*(ux+vy))] du dv
      = integral integral F(u,v) * (j2*pi*u) * exp(j2*pi*(ux+vy)) du dv
```

This is exactly the inverse Fourier transform of `(j2*pi*u) * F(u,v)`.
Therefore:

```
F{ df/dx } = (j2*pi*u) * F(u,v)
```

and by the same argument in y:

```
F{ df/dy } = (j2*pi*v) * F(u,v)
```

**This is the key result**: differentiation in the spatial domain
corresponds to multiplication by a factor proportional to frequency (u or
v) in the Fourier domain. Since the magnitude of the multiplier
`|j2*pi*u| = 2*pi*|u|` grows linearly with frequency, differentiation acts
as a **high-pass filter** - it amplifies high-frequency content
(sharp transitions = edges) and suppresses/zeroes low-frequency content
(flat regions, where the derivative is near zero).

This directly explains why classical *spatial* edge detectors (Sobel,
Prewitt, gradient operators) are equivalent, by the Convolution Theorem
(established in the Module 3 theory), to multiplying the image's spectrum
by a frequency-domain high-pass filter shaped like `j2*pi*u` (or `j2*pi*v`).

## 3. The Laplacian and the Marr-Hildreth Edge Detector

The Laplacian operator is the sum of second derivatives:

```
Laplacian(f) = d^2f/dx^2 + d^2f/dy^2
```

Applying the differentiation theorem twice:

```
F{ d^2f/dx^2 } = (j2*pi*u)^2 * F(u,v) = -(2*pi*u)^2 * F(u,v)
F{ d^2f/dy^2 } = -(2*pi*v)^2 * F(u,v)

F{ Laplacian(f) } = -(2*pi)^2 * (u^2 + v^2) * F(u,v)
```

The multiplier `-(2*pi)^2(u^2+v^2)` is a **radially symmetric high-pass
filter**: it is zero at the origin (u=v=0, the DC/average-intensity term)
and grows quadratically with distance from the origin in frequency space.
This is exactly the frequency-domain characterization of the Laplacian
edge/zero-crossing detector (the basis of the Marr-Hildreth edge detection
algorithm, which convolves the image with a Laplacian-of-Gaussian kernel
and finds zero-crossings of the result).

## 4. Practical Frequency-Domain High-Pass Filters for Edge Detection

Rather than using the unbounded/noise-sensitive ideal differentiator
directly, practical frequency-domain edge detection multiplies F(u,v) by a
smooth high-pass filter H(u,v), such as a Gaussian high-pass filter:

```
H_HP(u,v) = 1 - exp( -D(u,v)^2 / (2*sigma^2) )

where D(u,v) = sqrt(u^2 + v^2) is the distance from the frequency origin.
```

The edge-enhanced image is obtained by:

```
G(u,v) = F(u,v) * H_HP(u,v)
g(x,y) = F^-1{ G(u,v) }
```

By the Convolution Theorem, this is identical to spatially convolving
f(x,y) with the (spatial-domain) kernel corresponding to H_HP - i.e., a
high-pass spatial filter, of which the Sobel/Laplacian kernels are discrete
approximations. This unifies spatial-domain gradient/Laplacian edge
detectors and frequency-domain high-pass filtering as the same operation
viewed in two domains (exactly as shown for general convolution in Module
3's theory).

## 5. Region Segmentation via Low-Pass Filtering

Symmetrically, a **low-pass filter** attenuates high frequencies and
retains low frequencies:

```
H_LP(u,v) = exp( -D(u,v)^2 / (2*sigma^2) )   (Gaussian low-pass)
```

Applying this to F(u,v) and inverse-transforming smooths the image,
suppressing fine texture and edge detail while preserving the coarse
region structure (the slowly-varying average intensity of each object/
background area). This is the frequency-domain justification for why
smoothing (blurring) is a standard **pre-processing step for region-based
segmentation** methods (e.g., thresholding, watershed, region growing):
removing high-frequency noise/texture makes regions more uniform
internally, so a single threshold or seed-growing criterion better
separates one region from another without being confused by high-frequency
noise pixels.

## 6. Band-Pass Filtering and Texture-Based Region Segmentation

Regions can also be distinguished by *texture* - patterns that repeat at a
characteristic spatial frequency (e.g., a knit fabric vs. smooth skin).
The power spectrum |F(u,v)|^2 reveals this: a textured region produces
energy concentrated in a specific frequency band (not just near DC), while
a smooth region concentrates almost all energy near the origin (u=v=0).

A band-pass filter:

```
H_BP(u,v) = H_LP,high_cutoff(u,v) - H_LP,low_cutoff(u,v)
```

(a Gaussian low-pass at a high cutoff minus a Gaussian low-pass at a lower
cutoff) isolates a specific frequency band. Applying several band-pass
filters at different center frequencies and orientations (e.g., a bank of
Gabor filters, which are Gaussian-windowed sinusoids with a well-defined
center frequency and orientation) and comparing the local energy response
of each filter at every pixel allows texture-based region segmentation:
pixels whose dominant response band matches are grouped into the same
region, even when their raw intensity values are similar to a
neighboring, differently-textured region.

## 7. Experimental Validation

A Gaussian low-pass filter `H_LP(u,v)` and its complement high-pass filter
`H_HP(u,v) = 1 - H_LP(u,v)` were applied in the frequency domain to a test
image (the same synthetic human-silhouette image used to validate Parts 1-2)
and compared against their classical spatial-domain counterparts:

- **High-pass result vs. Sobel gradient magnitude:** the FFT high-pass
  output shows a bright ring exactly along the silhouette boundary with a
  dark (near-zero) interior and background - visually and functionally
  equivalent to a spatial gradient-magnitude edge map, confirming the
  Differentiation Theorem's prediction that edges correspond to high
  spatial frequencies. (See `figures/freq_highpass.png` vs.
  `figures/spatial_sobel.png`.)
- **Low-pass result vs. Gaussian blur:** the FFT low-pass output is a
  smoothed version of the original with the same overall region structure
  preserved and high-frequency noise removed - visually indistinguishable
  from spatial Gaussian blurring, consistent with (and, per the Convolution
  Theorem proven in the Module 3 theory, mathematically identical to) the
  spatial-domain Gaussian blur operation. (See `figures/freq_lowpass.png`
  vs. `figures/spatial_blur.png`.)

This confirms both halves of the edge/region duality: attenuating low
frequencies isolates edges, and attenuating high frequencies isolates
region structure, exactly as predicted by the Differentiation Theorem and
Convolution Theorem derived above.

## 8. Summary: the Edge/Region Duality in the Frequency Domain

| Operation | Spatial domain | Frequency domain |
|---|---|---|
| Edge detection | Gradient / Laplacian convolution | High-pass filter (H(u,v) large at high frequency, ~0 at DC) |
| Region smoothing / segmentation pre-processing | Gaussian blur | Low-pass filter (H(u,v) ~1 at DC, decays at high frequency) |
| Texture-based region segmentation | Gabor filter bank convolution | Band-pass filter at a specific center frequency/orientation |

All of these follow from the same two facts proven above: (1) the
Convolution Theorem (spatial convolution = frequency-domain multiplication,
derived in the Module 3 theory) and (2) the Differentiation Theorem
(spatial differentiation = multiplication by j2*pi*(frequency) in the
Fourier domain, derived in Section 2 here). Edge detectors are high-pass
filters because edges are, by definition, high-frequency content (rapid
spatial intensity change); region interiors are low-frequency content
(slowly varying intensity), which is why low-pass filtering is the natural
pre-processing step for region-based segmentation, and why texture-specific
band-pass filtering can separate regions that share similar mean intensity
but differ in spatial frequency content.
