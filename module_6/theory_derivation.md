# Theory: Motion Tracking, Bilinear Interpolation, and Planar Structure from Motion

## 1. The Brightness Constancy Assumption

The foundational assumption of optical flow is that a physical point's
image brightness does not change as it moves between frames over a short
time interval dt:

```
I(x, y, t) = I(x + dx, y + dy, t + dt)
```

where I(x,y,t) is the image intensity at pixel (x,y) at time t, and
(dx, dy) is the point's displacement over time dt.

## 2. Deriving the Optical Flow Constraint Equation

Expand the right-hand side via a first-order Taylor series around (x,y,t):

```
I(x+dx, y+dy, t+dt) ~= I(x,y,t) + (dI/dx)*dx + (dI/dy)*dy + (dI/dt)*dt
```

Substituting into the brightness constancy equation and canceling I(x,y,t)
from both sides:

```
0 = (dI/dx)*dx + (dI/dy)*dy + (dI/dt)*dt
```

Dividing through by dt and defining u = dx/dt, v = dy/dt (the components
of the optical flow velocity vector) and Ix = dI/dx, Iy = dI/dy, It = dI/dt:

```
Ix*u + Iy*v + It = 0        <-- the Optical Flow Constraint Equation (OFCE)
```

This is **one linear equation in two unknowns (u, v)** per pixel - it
constrains the flow to lie on a line in (u,v)-space, but cannot by itself
determine a unique (u,v). This is the **aperture problem**: a local
intensity gradient only constrains the component of motion *along* that
gradient direction, not the component perpendicular to it (observed
directly in the experimental evidence - see Section 5).

## 3. Lucas-Kanade: Resolving the Aperture Problem Locally

Lucas-Kanade resolves the aperture problem by assuming the flow (u,v) is
**constant within a small window** of pixels (e.g. 5x5 to 15x15) around
each point of interest. This turns one equation per pixel into N equations
(one per pixel in the window) for the same two unknowns (u,v):

```
Ix_1*u + Iy_1*v = -It_1
Ix_2*u + Iy_2*v = -It_2
...
Ix_N*u + Iy_N*v = -It_N
```

In matrix form, A*[u,v]^T = b, where:

```
A = [[Ix_1, Iy_1], [Ix_2, Iy_2], ..., [Ix_N, Iy_N]]     (N x 2)
b = [-It_1, -It_2, ..., -It_N]^T                         (N x 1)
```

This is overdetermined (N > 2 equations, 2 unknowns), so it is solved via
least squares: minimize ||A*[u,v]^T - b||^2. Setting the gradient to zero
gives the **normal equations**:

```
(A^T A) * [u,v]^T = A^T b
```

Expanding A^T A and A^T b explicitly in terms of sums over the window:

```
[[sum(Ix^2),   sum(Ix*Iy)]   [u]   [-sum(Ix*It)]
 [sum(Ix*Iy),  sum(Iy^2) ]] [v] = [-sum(Iy*It)]
```

This 2x2 system is solved directly (as implemented in
`lucas_kanade_tracking.py`'s `lucas_kanade_flow_at_point`):

```
[u, v]^T = (A^T A)^-1 * A^T b
```

### 3.1 When does this fail? (the aperture problem, formally)

The system above is invertible only when A^T A is well-conditioned, i.e.
has two reasonably large eigenvalues. If the window contains intensity
variation in only one direction (e.g. a straight edge), A^T A has one
large and one near-zero eigenvalue - the flow is well-constrained along
the gradient direction but undetermined perpendicular to it. If the
window is completely flat (uniform intensity - no texture), both
eigenvalues are near zero and no flow can be recovered at all. This is
implemented as an explicit conditioning check (`eigvals[0] < 1e-3`) in the
code, and is exactly what produces the black (zero-flow) interior of the
moving circle in Section 5's experimental evidence.

## 4. Setting Up Two-Frame Tracking as a Formal Problem

Given two consecutive frames I1 (at time t) and I2 (at time t+dt), and a
point p1 = (x,y) of interest in I1, the tracking problem is: find the
corresponding point p2 = (x+u, y+v) in I2 that satisfies brightness
constancy, i.e. minimizes:

```
E(u,v) = sum_{(i,j) in window(p1)} [ I2(x+i+u, y+j+v) - I1(x+i, y+j) ]^2
```

This is the general (non-linearized) Lucas-Kanade objective. The
linearized version solved in Section 3 is a first-order (Gauss-Newton-like)
approximation of this nonlinear least-squares problem, valid when the true
displacement (u,v) is small (sub-pixel to a few pixels) - for larger
motions, this is embedded in a coarse-to-fine image pyramid (as OpenCV's
`cv2.calcOpticalFlowPyrLK` does) so each pyramid level only has to solve
for a small residual displacement.

Because I2(x+i+u, y+j+v) must be evaluated at a **non-integer** pixel
location (u,v are real-valued, not integer pixel offsets), this requires
sampling the discrete image I2 at sub-pixel coordinates - which is exactly
what bilinear interpolation (Section 6) provides, and is why the two
topics are asked about together in this assignment.

## 5. Experimental Validation: Evidence from Optical Flow

Dense Farneback optical flow was computed on a synthetic video of a
uniformly-colored circle translating at a known 2 px/frame. The resulting
HSV-coded flow visualization (`figures/flow_sample_frame.png`) shows:

- A clear **ring of color at the circle's edge**, where strong intensity
  gradients exist (circle boundary against background) - flow is
  correctly detected here.
- A **black (zero-flow) interior**, even though the entire circle,
  including its interior, is physically translating at 2 px/frame.

This is direct experimental evidence of the aperture problem (Section
3.1): the uniformly-colored interior has zero spatial gradient, so Ix=Iy=0
everywhere inside it, making the OFCE (Ix*u+Iy*v+It=0) trivially satisfied
by *any* (u,v) - the true motion cannot be recovered there from local
information alone. The measured peak flow magnitude (2.19 px, see
`optical_flow.py` output) matches the 2 px/frame ground truth closely,
confirming the edge-region measurement is quantitatively correct.

**What this tells us optical flow can infer:** direction and speed of
motion wherever sufficient local texture/gradient exists (object edges,
textured surfaces), and - by the *absence* of a flow signal - where an
object is uniformly textured, not where it has no motion. A real-world
practical implication: sparse feature-based tracking (Lucas-Kanade on
Shi-Tomasi corners, Section 3) deliberately selects high-gradient,
well-conditioned points for exactly this reason, rather than tracking
arbitrary/flat pixels.

## 6. Deriving Bilinear Interpolation

Given a discrete image I with known integer-pixel values, bilinear
interpolation estimates I at a non-integer location (x,y) by linear
interpolation in two passes (first along x, then along y), using the four
nearest integer-pixel neighbors.

Let x0 = floor(x), x1 = x0+1, y0 = floor(y), y1 = y0+1, and define the
fractional offsets a = x - x0, b = y - y0 (both in [0,1)).

**Step 1 - interpolate along x** at the two rows y0 and y1:

```
I(x, y0) ~= (1-a)*I(x0,y0) + a*I(x1,y0)
I(x, y1) ~= (1-a)*I(x0,y1) + a*I(x1,y1)
```

(This follows directly from 1D linear interpolation: a point a fraction
`a` of the way from x0 to x1 has value (1-a)*f(x0) + a*f(x1).)

**Step 2 - interpolate the two results along y:**

```
I(x,y) ~= (1-b)*I(x,y0) + b*I(x,y1)
```

Substituting Step 1 into Step 2 gives the full bilinear interpolation
formula:

```
I(x,y) ~= (1-a)(1-b)*I(x0,y0) + a(1-b)*I(x1,y0)
          + (1-a)*b*I(x0,y1)  + a*b*I(x1,y1)
```

This is exactly what `bilinear_interpolate()` in
`lucas_kanade_tracking.py` implements. Geometrically, each of the four
corner intensities is weighted by the *area* of the rectangle on the
opposite side of (x,y) within the unit cell - the closer (x,y) is to a
corner, the more that corner's intensity dominates the result.

### 6.1 Validation with actual pixel locations

For each tracked point p1 in `lucas_kanade_tracking.py`'s output, the
predicted location p2 = p1 + (u,v) is used to bilinearly sample I2, and
the sampled intensity is compared against the known intensity at p1 (which
should match closely if the brightness-constancy assumption and the
recovered flow are both correct). On the synthetic textured-translation
test video (ground truth: exactly 1 px/frame), the from-scratch
implementation matched OpenCV's own `cv2.calcOpticalFlowPyrLK` to within
0.05 px on every tracked corner, and the bilinearly-sampled intensity at
each predicted point matched the original point's intensity closely
(exact match on checkerboard interior squares; see `sample_outputs`
printed by the script) - both validating the derivation above against
real pixel data, as required.

## 7. Structure from Motion for a Planar Object

### 7.1 Why a single homography relates two views of a plane

Consider a 3D point P lying on a physical plane, viewed by two pinhole
cameras. Let camera 1 be the reference, with projection matrix
`[K1 | 0]`, and camera 2 have projection matrix `K2*[R | t]` relative to
camera 1. Any point on the plane can be written as `n^T * P = d` for some
plane normal n and distance d (plane equation in camera 1's frame).

For a point on this plane, it can be shown (substituting the plane
constraint into the general two-view projective relationship) that the
two images' homogeneous pixel coordinates p1, p2 of the *same physical
point* are related by a single 3x3 matrix H (independent of which point on
the plane is chosen):

```
p2_h ~ H * p1_h,   where H = K2 * (R - t*n^T/d) * K1^-1
```

(~ denotes equality up to an overall scale, since these are homogeneous
coordinates). This is why, unlike the general (non-planar) two-view case
(which requires the fundamental/essential matrix and depth-dependent
triangulation, as derived in the Module 2 theory), a **single fixed
homography** fully describes the correspondence between any two views of
a *planar* object - this is exactly what justifies Step 2 of
`planar_sfm.py` (computing one H per view pair via `cv2.findHomography`,
using the classical Direct Linear Transform + RANSAC, not a learned
model).

### 7.2 Solving for H from point correspondences (DLT)

Each correspondence (x1,y1) <-> (x2,y2) gives two linear equations in the
9 unknown entries of H (up to scale), derived by cross-multiplying
`p2_h x H*p1_h = 0`:

```
[-x1, -y1, -1,  0,  0,  0, x2*x1, x2*y1, x2] * h = 0
[ 0,  0,  0, -x1, -y1, -1, y2*x1, y2*y1, y2] * h = 0
```

where h is H flattened into a 9-vector. Stacking these equations for N>=4
point correspondences gives a 2N x 9 system `A*h = 0`, solved (up to the
unavoidable overall scale ambiguity) as the eigenvector of `A^T A`
corresponding to its smallest eigenvalue (equivalently, via SVD) - this is
the classical Direct Linear Transform (DLT) algorithm that
`cv2.findHomography` implements internally, wrapped with RANSAC for
robustness to any mis-clicked correspondence outliers.

### 7.3 Recovering the planar boundary and camera poses

Once H is known for each of the 4 views (relative to view 1, chosen as the
reference), any point's coordinates from a non-reference view are mapped
into the reference view's coordinate frame via `p1_h ~ H^-1 * p2_h` (or
equivalently, H was computed to directly map other-view points onto the
reference view's points, as done in `planar_sfm.py`). Averaging the
(now co-registered) boundary point estimates across all 4 views reduces
the effect of per-view pixel-click noise, giving a more accurate estimate
of the object's true planar boundary shape - this is the "structure" being
recovered.

If the camera's intrinsic matrix K is known (from prior calibration, as in
Module 2), each homography can additionally be decomposed into a candidate
relative rotation R and translation t (up to scale and a sign/solution
ambiguity inherent to single-homography decomposition - 4 mathematically
valid solutions exist per H in general, disambiguated by requiring
reconstructed points lie in front of both cameras), giving an estimate of
each photo's camera position/orientation relative to the first, as the
assignment requests. This is implemented via
`cv2.decomposeHomographyMat(H, K)` in `planar_sfm.py`.

## 8. References

- B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique
  with an Application to Stereo Vision," Proc. DARPA Image Understanding
  Workshop, 1981.
- B. K. P. Horn and B. G. Schunck, "Determining Optical Flow," Artificial
  Intelligence, vol. 17, 1981.
- R. Hartley and A. Zisserman, "Multiple View Geometry in Computer
  Vision," 2nd edition, Cambridge University Press, 2004 (homography
  derivation and decomposition, Chapters 8-9).
- G. Farneback, "Two-Frame Motion Estimation Based on Polynomial
  Expansion," Proc. Scandinavian Conference on Image Analysis, 2003.
- OpenCV documentation: `cv2.calcOpticalFlowPyrLK`,
  `cv2.calcOpticalFlowFarneback`, `cv2.findHomography`,
  `cv2.decomposeHomographyMat`. https://docs.opencv.org/
