"""
==============================================================================
CSc 8830: Computer Vision - Assignment 6, Part B
Structure from Motion for a Planar (2D) Object, from 4 Viewpoints

README / HOW TO RUN
------------------------------------------------------------------------------
PURPOSE:
    Reconstructs the boundary of a flat/planar object from 4 photos taken
    from different viewpoints, using homographies (since all points on a
    single physical plane seen by a pinhole camera are related to any
    reference view of that plane by a single 3x3 homography - see
    theory_derivation.md Section 4 for the full derivation).

PIPELINE:
    1. For each of the 4 images, click the 4+ corner/boundary points of the
       planar object (interactive OpenCV window).
    2. Choose image 1 as the reference view. Compute the homography H_i
       mapping each other image i's clicked points to image 1's clicked
       points (cv2.findHomography - classical DLT + RANSAC, not learned).
    3. Warp each image's full set of boundary points into the reference
       frame via H_i, then average the (now aligned) point estimates
       across all 4 views to get a refined boundary estimate - this is
       the "structure" (the object's shape in a common reference plane).
    4. If camera intrinsics K are supplied (from a prior calibration, e.g.
       Module 2's camera_calibration.py), additionally decomposes each H_i
       into a candidate (R, t) relative pose using cv2.decomposeHomographyMat,
       to report camera position/orientation per viewpoint as the
       assignment requests.

REQUIREMENTS:
    pip install opencv-python numpy

USAGE:
    python planar_sfm.py --images view1.jpg view2.jpg view3.jpg view4.jpg \
        --out reconstructed_boundary.png [--calib calibration_result.npz]

    Click the boundary corners of your planar object in each image window,
    in the SAME physical order each time (e.g. always top-left, top-right,
    bottom-right, bottom-left first, then any additional boundary points).
    Press 'n' to move to the next image once you've clicked that image's
    points, or 'q' to finish early.

OUTPUT:
    Prints each view's homography matrix relative to view 1, and (if
    --calib is supplied) the decomposed relative camera rotation/
    translation per view. Saves a visualization of the reconstructed
    (averaged, reference-frame) boundary.

AUTHOR: [Your Name]
COURSE: CSc 8830 - Computer Vision, Assignment 6
==============================================================================
"""

import argparse
import cv2
import numpy as np

_clicked = []


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked.append((x, y))


def click_points(image, window_name):
    global _clicked
    _clicked = []
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_cb)
    while True:
        vis = image.copy()
        for i, p in enumerate(_clicked):
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
            cv2.putText(vis, str(i), (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('n') or key == ord('q'):
            break
    cv2.destroyWindow(window_name)
    return list(_clicked)


def reconstruct_planar_boundary(image_paths, calib_path=None):
    images = [cv2.imread(p) for p in image_paths]
    if any(img is None for img in images):
        raise FileNotFoundError("Could not read one or more input images.")

    all_points = []
    for i, img in enumerate(images):
        print(f"Click boundary points on image {i+1}/{len(images)} "
              f"(same order each time), then press 'n'.")
        pts = click_points(img, f"Image {i+1}")
        if len(pts) < 4:
            raise RuntimeError(f"Need at least 4 points on image {i+1}; got {len(pts)}.")
        all_points.append(np.array(pts, dtype=np.float64))

    ref_points = all_points[0]
    homographies = [np.eye(3)]
    warped_points = [ref_points]

    for i in range(1, len(images)):
        H, mask = cv2.findHomography(all_points[i], ref_points, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError(f"Could not compute homography for image {i+1}.")
        homographies.append(H)

        pts_h = np.hstack([all_points[i], np.ones((len(all_points[i]), 1))])
        warped = (H @ pts_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        warped_points.append(warped)

    # Average the (now co-registered) boundary point estimates across all views
    reconstructed = np.mean(np.stack(warped_points, axis=0), axis=0)

    camera_poses = None
    if calib_path:
        data = np.load(calib_path, allow_pickle=True)
        K = data["K"]
        camera_poses = []
        for i, H in enumerate(homographies):
            if i == 0:
                camera_poses.append({"R": np.eye(3), "t": np.zeros(3)})
                continue
            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
            # Multiple decompositions are mathematically valid from a single
            # homography; report the first (in practice, disambiguated with
            # an additional constraint such as points being in front of
            # both cameras - see theory_derivation.md Section 4.3).
            camera_poses.append({"R": Rs[0], "t": Ts[0].ravel()})

    return {
        "images": images,
        "clicked_points": all_points,
        "homographies": homographies,
        "reconstructed_boundary": reconstructed,
        "camera_poses": camera_poses,
    }


def visualize_reconstruction(reference_image, reconstructed_points, out_path):
    vis = reference_image.copy()
    pts = reconstructed_points.astype(int)
    for i, p in enumerate(pts):
        cv2.circle(vis, tuple(p), 5, (0, 255, 0), -1)
    for i in range(len(pts)):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser(
        description="Planar structure-from-motion from 4 viewpoints via homography.")
    parser.add_argument("--images", nargs=4, required=True,
                         metavar=("VIEW1", "VIEW2", "VIEW3", "VIEW4"))
    parser.add_argument("--calib", default=None,
                         help="Optional calibration_result.npz (from Module 2's "
                              "camera_calibration.py) to additionally decompose "
                              "camera poses.")
    parser.add_argument("--out", default="reconstructed_boundary.png")
    args = parser.parse_args()

    result = reconstruct_planar_boundary(args.images, args.calib)

    print("\nHomographies relative to view 1:")
    for i, H in enumerate(result["homographies"]):
        print(f"View {i+1}:\n{H}\n")

    print("Reconstructed (averaged, reference-frame) boundary points:")
    print(result["reconstructed_boundary"])

    if result["camera_poses"]:
        print("\nDecomposed camera poses relative to view 1:")
        for i, pose in enumerate(result["camera_poses"]):
            print(f"View {i+1}: R=\n{pose['R']}\nt={pose['t']}\n")

    visualize_reconstruction(result["images"][0], result["reconstructed_boundary"], args.out)
    print(f"Saved reconstruction visualization to: {args.out}")


if __name__ == "__main__":
    main()
