# CSc 8830 — Module 2: Camera Calibration & Perspective Measurement

## Contents
| File | Purpose |
|---|---|
| `camera_calibration.py` | Step 1: checkerboard camera calibration |
| `perspective_dimension.py` | Step 2: real-world 2D dimension from an image + known depth |
| `validation_experiment.py` | Step 3: 20-trial validation + error statistics |
| `theory_derivation.md` | Theory: two-camera epipolar geometry |
| `webapp/` | Flask app exposing Steps 1–2 through a browser |
| `build_report.py` | Generates `Module_2_Report.pdf` from your real results |

## Before submitting

1. Run `camera_calibration.py` on real checkerboard photos → `calibration_result.npz`
2. Collect 20 real object measurements → fill `trials_template.csv` → run `validation_experiment.py` → `results.csv`
3. Copy both files next to `build_report.py`
4. Set `GITHUB_LINK` and `WEBAPP_LINK` at the top of `build_report.py`
5. Run `python3 build_report.py` → produces `Module_2_Report.pdf` with your real data
6. Deploy `webapp/` publicly (see below)
7. Submit the code zip + `Module_2_Report.pdf` to Classroom, then click **Hand In**

## Deploying the web app (Render.com, free)
1. Push code to GitHub
2. render.com → sign in with GitHub → New → Web Service → select repo
3. Build command: `pip install -r webapp/requirements.txt`
4. Start command: `gunicorn --chdir webapp app:app`
5. Deploy → use the resulting `https://your-app.onrender.com` URL in your PDF/video

(`webapp/requirements.txt` and `app.py`'s port handling are already set up for this.)
