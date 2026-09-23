"""
==============================================================================
CSc 8830 - Computer Vision - Module 2 Assignment
Web Application: Camera Calibration + Real-World Dimension Measurement

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    A minimal Flask web app that demonstrates Steps 1-2 of the assignment
    end-to-end through a browser:
      1. Upload 8+ checkerboard photos -> runs OpenCV calibration -> shows
         the intrinsic matrix K, distortion coefficients, and reprojection
         error.
      2. Upload a photo of an object + enter the known camera-to-object
         distance -> click two points on the image in the browser -> the
         app computes and displays the real-world distance between them.

REQUIREMENTS:
    pip install flask opencv-python-headless numpy

USAGE:
    cd webapp
    python app.py
    Then open http://localhost:5000 in a browser.

    For a public/deployed demo (e.g. for your assignment video), you can
    run this on any machine reachable over the network, or deploy it to a
    free host (Render, PythonAnywhere, Railway, etc.) and share that URL -
    the assignment requires the app be "accessible via a webpage".

FILES:
    app.py              - this file: Flask routes and glue logic
    templates/index.html- single-page UI (upload forms + canvas click UI)
    static/style.css     - styling
    ../camera_calibration.py   - Step 1 calibration logic (imported)
    ../perspective_dimension.py- Step 2 measurement logic (imported)

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Module 2 Assignment
==============================================================================
"""

import os
import sys
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from camera_calibration import calibrate_camera  # noqa: E402
from perspective_dimension import measure_distance  # noqa: E402

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory store of the most recent calibration result (K, dist), keyed by
# session id. For a class demo this simple in-memory approach is sufficient;
# a production app would persist this per-user in a database.
CALIBRATION_STORE = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    """Accept multiple checkerboard images, run calibration, return results."""
    files = request.files.getlist("images")
    cols = int(request.form.get("cols", 9))
    rows = int(request.form.get("rows", 6))
    square_size = float(request.form.get("square_size", 25.0))

    if len(files) < 5:
        return jsonify({"error": "Please upload at least 5 checkerboard images."}), 400

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for f in files:
        f.save(os.path.join(session_dir, f.filename))

    try:
        results = calibrate_camera(session_dir, cols, rows, square_size,
                                    save_corner_images=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    CALIBRATION_STORE[session_id] = {
        "K": results["K"],
        "dist": results["dist"],
    }

    return jsonify({
        "session_id": session_id,
        "K": results["K"].tolist(),
        "dist": results["dist"].ravel().tolist(),
        "mean_error_px": results["mean_error"],
        "overall_rms_px": results["overall_rms"],
        "n_images_used": len(results["used_images"]),
    })


@app.route("/api/upload_object_image", methods=["POST"])
def api_upload_object_image():
    """Accept a single object photo for the measurement demo, return a URL to display it."""
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "No image uploaded."}), 400

    filename = f"{uuid.uuid4()}_{f.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)
    return jsonify({"image_url": f"/uploads/{filename}"})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/measure", methods=["POST"])
def api_measure():
    """
    Given a session_id (from a prior /api/calibrate call), two clicked pixel
    points, and a known distance Z, compute the real-world distance.
    """
    data = request.get_json()
    session_id = data.get("session_id")
    p1 = tuple(data.get("p1"))
    p2 = tuple(data.get("p2"))
    distance_mm = float(data.get("distance_mm"))

    calib = CALIBRATION_STORE.get(session_id)
    if calib is None:
        return jsonify({"error": "No calibration found for this session. "
                                  "Please run calibration first."}), 400

    result = measure_distance(p1, p2, distance_mm, calib["K"], calib["dist"])

    return jsonify({
        "distance_mm": result["distance_mm"],
        "distance_cm": result["distance_mm"] / 10.0,
        "real_p1_mm": list(result["real_p1_mm"]),
        "real_p2_mm": list(result["real_p2_mm"]),
    })


if __name__ == "__main__":
    # Render (and most cloud hosts) assign the port dynamically via the PORT
    # environment variable - fall back to 5000 for local development.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
