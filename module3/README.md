# CSc 8830 — Module 3 Assignment
## Image Blurring: Spatial Filtering vs. Fourier-Domain Filtering

## Contents
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app: Gaussian blur via spatial convolution and via FFT multiplication, with numerical comparison |
| `requirements.txt` | Python dependencies |
| `theory_derivation.md` | Full mathematical derivation of the Convolution Theorem |
| `Module_3_Report.pdf` | Assignment write-up: implementation, theory, and experimental evidence |
| `figures/` | Generated evidence images (original/spatial/fourier/difference, error-vs-sigma chart) |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## What's still needed before you submit

### 1. Fill in the PDF placeholders
Open `build_report.py`, replace:
- `GITHUB_LINK_PLACEHOLDER` with your actual repo URL
- `WEBAPP_LINK_PLACEHOLDER` with your deployed app's URL

then rerun `python3 build_report.py` to regenerate `Module_3_Report.pdf` with the real links filled in.

### 2. Push to GitHub
```bash
cd module_3
git init
git add .
git commit -m "Module 3: image blurring, spatial vs Fourier equivalence"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Make sure the repo is public (or shared with your instructor) so the link in your PDF is accessible.

### 3. Deploy the web app so it's accessible via a webpage
The assignment requires the app be reachable through a browser, not just run locally. The
fastest free option for a Streamlit app is **Streamlit Community Cloud**:
1. Push your code to GitHub (step 2 above).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select your repo/branch, and set the main file path to `app.py`.
4. Deploy — you'll get a public URL like `https://your-app-name.streamlit.app`.
5. Paste that URL into the PDF (step 1) and into your video.

(Alternatives: Render, Railway, or Hugging Face Spaces also host Streamlit apps for free.)

### 4. Record the demonstration video
Suggested flow for the screen recording:
1. Open your deployed app URL in a browser (show the address bar so it's clearly a live webpage).
2. Upload a test image with visible edges/detail.
3. Move the sigma slider through a few values (e.g. 1, 3, 6).
4. Point out the three side-by-side images: original, spatial result, Fourier result — note they look identical.
5. Point out the MAE / RMSE / max-error numbers and explain they're at floating-point precision (~1e-16), which is the experimental proof of the Convolution Theorem.
6. Briefly show the difference image (uniform noise, no structure).
7. Mention where the mathematical derivation is in your PDF/report.

### 5. Assemble the final PDF for submission
`Module_3_Report.pdf` already contains the assignment restatement, implementation
description with code excerpt, the full typed theory derivation, and the
experimental validation figures/table — this satisfies the "convert your hand
solving into typed/digital format and append to the final PDF" requirement,
since the theory here is a mathematical derivation rather than a hand sketch.
Just make sure the GitHub and web app links (step 1) are filled in before
your final submission.

## Notes on the theory answer
The convolution theorem derivation in `theory_derivation.md` (also included in
the PDF) shows step-by-step why `F{f*h} = F{f}·F{h}` holds for the discrete
2D case, and explains why the implementation specifically uses circular
("wrap") boundary conditions and shifts the kernel to have its origin at
(0,0) before taking its FFT — both are necessary for the spatial and
Fourier results to match to floating-point precision, which is what the
experimental results in Section 4 of the PDF confirm.
