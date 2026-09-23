# Theory: Relating Image Coordinates of Point P Across Two Cameras

## 1. Setup and Assumptions

- **Camera 1 (C1)** is static and defines the world coordinate frame: its
  optical center is the world origin, and its optical axis is the world
  Z-axis.
- **Camera 2 (C2)** is displaced from C1 by a translation and rotated by
  some oblique orientation relative to C1.
- A 3D point **P = (X, Y, Z)** (expressed in C1's coordinate frame) is
  visible in both cameras' fields of view.
- Both cameras are modeled as ideal pinhole cameras (after lens-distortion
  correction using each camera's calibration, per Step 1).
- Each camera has been individually intrinsically calibrated, so **K1**
  and **K2** (the intrinsic matrices) are known.
- The extrinsic relationship between the two cameras — rotation **R** and
  translation **t** taking points from C1's frame into C2's frame — is
  fixed (the cameras are rigidly mounted, or their relative pose is
  determined once, e.g. via stereo calibration) and can be assumed known
  or determined via calibration.

## 2. Camera 1 Projection

Since C1 defines the world frame, its projection of P is the standard
pinhole projection:

```
Z1 * p1_h = K1 * [I | 0] * P_h
```

where `P_h = (X, Y, Z, 1)^T` is P in homogeneous world coordinates,
`p1_h = (u1, v1, 1)^T` is the homogeneous image coordinate in camera 1,
and `Z1 = Z` is the depth of P along C1's optical axis.

Expanded:

```
u1 = fx1 * (X / Z) + cx1
v1 = fy1 * (Y / Z) + cy1
```

where `fx1, fy1, cx1, cy1` are the intrinsic parameters of C1 (from its
individual calibration).

## 3. Transforming P into Camera 2's Frame

Camera 2 sees the same physical point P, but expressed in its own
coordinate frame after applying the rigid body transform (R, t) from C1
to C2:

```
P2 = R * P + t
```

where **R** is a 3x3 rotation matrix (encoding C2's oblique orientation
relative to C1) and **t** is a 3x1 translation vector (the position of
C2's origin, expressed in C1's frame, negated/transformed appropriately
depending on convention — here we define R, t such that P2 above directly
gives P's coordinates in C2's frame).

So:

```
X2 = r11*X + r12*Y + r13*Z + tx
Y2 = r21*X + r22*Y + r23*Z + ty
Z2 = r31*X + r32*Y + r33*Z + tz
```

## 4. Camera 2 Projection

Camera 2 then projects P2 using its own intrinsics K2:

```
Z2 * p2_h = K2 * [R | t] * P_h
```

Expanded:

```
u2 = fx2 * (X2 / Z2) + cx2
v2 = fy2 * (Y2 / Z2) + cy2
```

## 5. The Full Relationship Between (u1, v1) and (u2, v2)

Combining the above: given a pixel (u1, v1) in camera 1 and a known depth
Z (from C1), we can recover P in C1's frame via the inverse pinhole
relation (Step 2's approach):

```
X = (u1 - cx1) * Z / fx1
Y = (v1 - cy1) * Z / fy1
```

Substituting X, Y, Z into the rigid transform and then into camera 2's
projection equations gives (u2, v2) directly in terms of (u1, v1, Z):

```
X2 = r11*(u1-cx1)*Z/fx1 + r12*(v1-cy1)*Z/fy1 + r13*Z + tx
Y2 = r21*(u1-cx1)*Z/fx1 + r22*(v1-cy1)*Z/fy1 + r23*Z + ty
Z2 = r31*(u1-cx1)*Z/fx1 + r32*(v1-cy1)*Z/fy1 + r33*Z + tz

u2 = fx2 * (X2 / Z2) + cx2
v2 = fy2 * (Y2 / Z2) + cy2
```

This is the general relationship, parameterized by the unknown depth Z.
If Z is **not** known, (u1, v1) alone does not determine a unique
(u2, v2); instead, as Z varies, (u2, v2) traces out a line in image 2 —
this is the **epipolar line** corresponding to (u1, v1), and the
constraint that (u2, v2) must lie on it is the epipolar constraint,
expressed compactly as:

```
p2_h^T * F * p1_h = 0
```

where **F** is the 3x3 fundamental matrix, related to the essential
matrix **E** (which encodes R and t directly) by `F = K2^-T * E * K1^-1`,
and `E = [t]_x * R` (with `[t]_x` the skew-symmetric cross-product matrix
of t). This is the standard two-view epipolar geometry result, and it
follows directly from requiring that P, and the two camera centers, all
lie in a single plane (the epipolar plane).

## 6. Static Parameters vs. Variables

**Static (fixed once, determined by one-time calibration):**
- `K1 = [[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]]` — intrinsics of camera 1.
  Determined via the Step 1 checkerboard calibration for camera 1.
- `K2` — intrinsics of camera 2, determined the same way for camera 2.
- `R, t` — the extrinsic relative pose between camera 1 and camera 2.
  Determined via **stereo calibration**: capturing simultaneous images of
  the same checkerboard from both cameras and running `cv2.stereoCalibrate`
  (or the MATLAB equivalent), which solves for R and t given each
  camera's individually-known intrinsics.
- Lens distortion coefficients for each camera (used to undistort raw
  pixel coordinates before applying the pinhole equations above).

**Variables (change per point / per frame):**
- `(u1, v1)`, `(u2, v2)` — the observed pixel coordinates of P in each
  camera; these change for every 3D point observed.
- `(X, Y, Z)` — the 3D coordinates of P in camera 1's frame; unknown in
  general unless depth Z is independently supplied (as in Step 2) or
  recovered via triangulation.
- `Z2` — the depth of P as seen from camera 2; derived from Z, R, t.

## 7. Recovering P by Triangulation (if Z is unknown)

If neither camera's depth is known ahead of time but the point is
observed in **both** images simultaneously, P can be recovered (rather
than assumed) via triangulation: each image ray (from each camera's
optical center through its respective pixel) is a line in 3D space, and
P is the point where these two rays intersect (or, due to noise, the
point that minimizes the sum of squared reprojection distances to both
rays — solved via linear least squares, e.g. `cv2.triangulatePoints`).
This requires R and t to already be known, i.e. the two cameras must have
already been stereo-calibrated as described above.

## 8. Justification of Assumptions

- **Rigid relative pose (R, t fixed):** valid as long as neither camera
  moves after stereo calibration; if either camera can move independently,
  R and t must be re-estimated (e.g. via a shared visible calibration
  target, or via matched feature points and essential matrix estimation
  through the 8-point algorithm and RANSAC).
- **Pinhole model post-undistortion:** valid for typical consumer cameras
  after applying the radial/tangential distortion correction from
  individual calibration; wide-angle/fisheye lenses would require a
  different distortion model.
- **Point P visible in both cameras' FoV:** required for any two-view
  relationship (epipolar constraint or triangulation) to apply at all;
  points outside the overlap region can only be handled by one camera's
  single-view equations (Step 2's approach) if depth is otherwise known.
