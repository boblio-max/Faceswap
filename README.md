Enhanced Face Swap (MediaPipe + OpenCV)

A minimal, robust face-swap utility that uses MediaPipe Face Mesh for landmark detection and OpenCV for warping and blending. Designed for live webcam swapping using a static source image.

Features

Real-time face landmark detection (MediaPipe Face Mesh).

Delaunay triangulation + optimized triangle warping.

Face mask generation with Gaussian blur for smooth edges.

Color correction and seamlessClone fallback to alpha blending.

FPS tracking, landmark overlay toggle, screenshot save, and simple controls.

Robust error handling and fallback triangulation for edge cases.

Requirements

Python 3.9+ recommended

Packages:

opencv-python

mediapipe

numpy

Install:

python -m pip install opencv-python mediapipe numpy

Files

face_swap.py (the script you provided) — contains draw_face_landmarks() and FaceSwap class plus main().

Configuration

Edit the top of the script or set these variables before running:

SOURCE_IMAGE_PATH = "C:\\path\\to\\your\\source.jpg"   # REQUIRED: source face image
CAM_WIDTH = 640
CAM_HEIGHT = 480


Make sure SOURCE_IMAGE_PATH points to a readable image file with a clear frontal face.

Usage

Run the script directly:

python face_swap.py


Controls shown in console and active during the webcam loop:

q — quit

s — save screenshot (face_swap_<timestamp>.jpg)

l — toggle landmark visualization

f — toggle FPS overlay

Notes:

Camera index 0 is used. Change to other indices if you have multiple cameras.

The script mirrors the frame (cv2.flip(frame, 1)) to produce a selfie view.

Tips & Troubleshooting

If the source image does not detect a face:

Use a clear, well-lit, frontal image.

Resize source image so the face is reasonably large (script attempts to resize to max 800).

If triangulation fails or result looks broken:

Try a different source image.

Increase min_detection_confidence in face_mesh_static_config for better landmark quality.

Low FPS:

Reduce camera resolution.

Close other GPU/CPU heavy apps.

If cv2.imread() returns None ensure the path is correct and accessible.

If MediaPipe import fails, reinstall mediapipe or use a compatible Python version.

Internal Notes (for maintainers)

FaceSwap.prepare_source_face() caches source landmarks and triangulation to avoid recompute.

enhanced_delaunay_triangulation() inserts image corners to stabilize triangulation and falls back to fallback_triangulation() when needed.

warp_triangle_optimized() includes bounds checks to avoid OpenCV errors and silently skips problematic triangles.

apply_color_correction() uses masked channel means for a basic lighting match. This is simple and may be improved with histogram matching or LAB-space transfer.

Potential Improvements

Add CLI args (argparse) to set SOURCE_IMAGE_PATH, camera index, and confidence thresholds.

Persist triangle indices and source landmarks to disk for faster startup.

Replace simple color matching with Reinhard or adaptive histogram matching.

GPU acceleration for warping and cloning for higher FPS.

Add multi-face support and source selection UI.

Attribution

MediaPipe Face Mesh for landmarks.

OpenCV for image processing and blending.
