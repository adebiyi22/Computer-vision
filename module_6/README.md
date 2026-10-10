# CSc 8830 — Assignment 6: Optical Flow, Motion Tracking & Planar SfM

## Contents
| File | Purpose |
|---|---|
| `optical_flow.py` | Part A: dense optical flow (Farneback) + HSV visualization |
| `lucas_kanade_tracking.py` | Part A: from-scratch Lucas-Kanade + bilinear interpolation, validated vs. OpenCV |
| `planar_sfm.py` | Part B: planar structure-from-motion from 4 viewpoints (homography-based) |
| `theory_derivation.md` | Full math: OFCE, Lucas-Kanade, bilinear interpolation, planar SfM, with citations |
| `build_report.py` | Generates `Assignment_6_Report.pdf` from your real videos/results |
| `figures/` | Evidence images from testing (aperture problem demo, tracking validation) |

## Tested and verified
- `optical_flow.py`: confirmed on a synthetic moving-circle video — measured peak flow (2.19 px) matched the 2 px/frame ground truth, and visually demonstrated the aperture problem (flat interior shows no flow).
- `lucas_kanade_tracking.py`: **caught and fixed a real bug** — OpenCV's unnormalized 3×3 Sobel kernel was silently shrinking the recovered flow by ~8×. After fixing, the from-scratch implementation matches OpenCV's own LK to within 0.05 px on a synthetic video with known 1 px/frame translation.
- `planar_sfm.py`: homography + averaging logic validated against synthetic ground-truth plane points (reconstruction error ~5×10⁻⁶ px).

## Before submitting

1. **Capture 2 real videos** (≥30 seconds each, with motion).
2. Run `optical_flow.py` on both → produces the required flow visualization videos.
3. Run `lucas_kanade_tracking.py` on two consecutive frames from each video → gives the "validate with actual pixel locations" evidence.
4. **Photograph a flat/planar object from 4 viewpoints**, and run `planar_sfm.py` on those images (click the object's boundary points in each window when prompted).
5. If you have a calibrated camera (e.g. from Module 2), pass `--calib calibration_result.npz` to also get camera pose estimates per viewpoint.
6. Copy your real output videos/images into `figures/`, fill in `GITHUB_LINK` at the top of `build_report.py`, then run `python3 build_report.py` to generate the final PDF with your real results.
7. Push code to GitHub, record your demo video, and submit the PDF + video to Classroom.

## Important: citations required
This assignment explicitly requires **all references to be cited** in your
report. `theory_derivation.md` Section 8 includes citations for Lucas-Kanade,
Horn-Schunck, Hartley & Zisserman (homography), and Farneback's method —
carry these into your final PDF citation list, and add any additional
sources you personally consulted.
