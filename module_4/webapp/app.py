"""
==============================================================================
CSc 8830 - Computer Vision - Module 4 Assignment
Web Application: Human Boundary Segmentation (RGB + Thermal)

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    A Flask web app demonstrating Parts 1-2 of the assignment through a
    browser: upload an RGB photo or a thermal image containing a person,
    and the app runs the corresponding classical (non-ML) segmentation
    pipeline and displays the detected boundary overlay + binary mask.

REQUIREMENTS:
    pip install -r requirements.txt

USAGE:
    cd webapp
    python app.py
    Open http://localhost:5000

DEPLOYMENT:
    See ../README.md for deploying this publicly (Render.com), which is
    required by the assignment ("all assignments must be accessible via
    that webpage" - localhost alone does not satisfy this).

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 4 Assignment
==============================================================================
"""

import os
import sys
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rgb_segmentation import segment_human_rgb          # noqa: E402
from thermal_segmentation import segment_human_thermal   # noqa: E402

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segment", methods=["POST"])
def api_segment():
    mode = request.form.get("mode", "rgb")  # "rgb" or "thermal"
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "No image uploaded."}), 400

    uid = uuid.uuid4().hex
    in_path = os.path.join(UPLOAD_DIR, f"{uid}_in.png")
    f.save(in_path)

    img = cv2.imread(in_path)
    if img is None:
        return jsonify({"error": "Could not read uploaded image."}), 400

    try:
        if mode == "thermal":
            false_color = request.form.get("false_color") == "true"
            mask, boundary, rect = segment_human_thermal(img, false_color=false_color)
        else:
            mask, boundary, rect = segment_human_rgb(img)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    vis = img.copy()
    cv2.drawContours(vis, [boundary], -1, (0, 255, 0), 3)

    result_path = os.path.join(UPLOAD_DIR, f"{uid}_result.png")
    mask_path = os.path.join(UPLOAD_DIR, f"{uid}_mask.png")
    cv2.imwrite(result_path, vis)
    cv2.imwrite(mask_path, mask)

    fg_pixels = int(np.sum(mask == 255))
    x, y, w, h = rect

    return jsonify({
        "result_url": f"/uploads/{uid}_result.png",
        "mask_url": f"/uploads/{uid}_mask.png",
        "foreground_pixels": fg_pixels,
        "bounding_box": [x, y, w, h],
    })


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
