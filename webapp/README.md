# CSc 8830 — Combined Web App (Modules 2, 3, 4)

One Streamlit app covering the web-demo requirement for three assignments.

## Contents
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — three tabs, one per module |
| `camera_calibration.py` | Module 2 Step 1 (imported, unchanged) |
| `perspective_dimension.py` | Module 2 Step 2 (imported, unchanged) |
| `rgb_segmentation.py` | Module 4 Part 1 (imported, unchanged) |
| `thermal_segmentation.py` | Module 4 Part 2 (imported, unchanged) |
| `requirements.txt` | Dependencies |

Module 3's blur logic is embedded directly in `app.py` (it was originally a
single self-contained script, so there's nothing separate to import).

## What each tab does
- **Module 2:** upload checkerboard photos → calibrate → upload an object
  photo → click two points on it → get a real-world distance measurement.
- **Module 3:** upload any image, adjust blur sigma, compare spatial vs.
  Fourier-domain blurring with MAE/RMSE/max-error stats.
- **Module 4:** upload an RGB or thermal photo of a person → see the
  classical (non-ML) segmentation boundary and mask.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly (Streamlit Community Cloud, free)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → sign in with GitHub.
3. New app → select this repo/branch → main file path: `app.py`.
4. Deploy → you get a URL like `https://your-app.streamlit.app`.
5. Use that one URL as the "web application" link for Modules 2, 3, and 4.

## Note on Module 2's calibration
Each calibration run in this app is temporary (in-memory for that browser
session) — the resulting K matrix is only used for measurements you make in
that same session's "Measure" tab. Re-running calibration overwrites it.
