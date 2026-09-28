# CSc 8830 — Module 4: Human Boundary Segmentation (RGB + Thermal) vs. SAM2

## Contents
| File | Purpose |
|---|---|
| `rgb_segmentation.py` | Part 1: classical (no ML/DL) RGB human boundary segmentation |
| `thermal_segmentation.py` | Part 2: classical (no ML/DL) thermal human boundary segmentation |
| `compare_with_sam2.py` | Runs SAM2 (external baseline) and computes IoU/Dice vs. your classical mask |
| `theory_derivation.md` | Fourier-domain edge detection & region segmentation derivation |
| `webapp/` | Flask app demoing Parts 1–2 through a browser |
| `build_report.py` | Generates `Module_4_Report.pdf` from your real results |
| `figures/` | Evidence images (frequency-domain theory validation) |

## Current status: running on localhost only
Deploy the web app (see below) before submitting — a localhost link does
not satisfy the assignment's "accessible via webpage" requirement.

## Before submitting

1. Run `rgb_segmentation.py` on a real RGB photo of a person
2. Run `thermal_segmentation.py` on a real thermal photo of a person
3. Run `compare_with_sam2.py` **on your own machine with a GPU** — SAM2
   needs PyTorch + a checkpoint download not available in this sandboxed
   environment (setup instructions are in that script's header)
4. Copy your real result images into `figures/` (`rgb_result.png`,
   `thermal_result.png`) and fill in the IoU/Dice numbers plus your
   `GITHUB_LINK`/`WEBAPP_LINK` at the top of `build_report.py`
5. Run `python3 build_report.py` → produces `Module_4_Report.pdf` with your
   real data
6. Deploy `webapp/` publicly (Render.com, same steps as Module 2 — see below)
7. Submit the code zip + `Module_4_Report.pdf` to Classroom, then click **Hand In**

## Deploying the web app (Render.com, free)
1. Push code to GitHub
2. render.com → sign in with GitHub → New → Web Service → select repo
3. Build command: `pip install -r webapp/requirements.txt`
4. Start command: `gunicorn --chdir webapp app:app`
5. Deploy → use the resulting `https://your-app.onrender.com` URL — not `localhost` — in your PDF/video

## Important constraint reminder
`rgb_segmentation.py` and `thermal_segmentation.py` use ONLY classical CV
(edge detection, thresholding, morphology, GrabCut's per-image graph-cut) —
no trained/learned models, per the assignment's rules. SAM2 is used
exclusively as an external comparison baseline in `compare_with_sam2.py`,
never inside the two segmentation scripts themselves.
