import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
from typing import List, Tuple, Optional, Dict
import time

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def draw_face_landmarks():
    # Open webcam
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Convert image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image
            results = face_mesh.process(image_rgb)

            # Draw landmarks
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())



            # Show
            cv2.imshow('Face Landmarks', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()


class FaceSwap:
    def __init__(self, source_img_path: str, cam_width: int = 640, cam_height: int = 480):
        """
        Initialize the FaceSwap class with improved configuration and validation.
        """
        self.source_img_path = source_img_path
        self.cam_width = cam_width
        self.cam_height = cam_height

        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Face detection settings for video stream
        self.face_mesh_config = {
            'max_num_faces': 2,
            'refine_landmarks': True,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.7
        }

        # Face detection settings for static images
        self.face_mesh_static_config = {
            'static_image_mode': True,
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': 0.5
        }

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Cache for source data
        self.src_img_bgr = None
        self.src_points = None
        self.triangles_idx = None
        self.src_prepared = False

        # Key facial landmark indices for better face detection
        self.FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        self.LEFT_EYE = [
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
            386, 385, 384, 398
        ]

        self.RIGHT_EYE = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
            159, 160, 161, 246
        ]

        self.LIPS = [
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82
        ]

    def validate_source_image(self) -> bool:
        """Validate source image exists and is readable."""
        if not os.path.exists(self.source_img_path):
            raise FileNotFoundError(f"Source image not found: {self.source_img_path}")

        img = cv2.imread(self.source_img_path)
        if img is None:
            raise ValueError(f"Cannot read source image: {self.source_img_path}")
        return True

    def mp_to_pixel_coords(self, landmarks, img_w: int, img_h: int) -> np.ndarray:
        """Convert MediaPipe normalized landmarks to pixel coordinates with bounds checking."""
        pts = []
        for lm in landmarks.landmark:
            x = max(0, min(int(lm.x * img_w), img_w - 1))
            y = max(0, min(int(lm.y * img_h), img_h - 1))
            pts.append((x, y))
        return np.array(pts, dtype=np.int32)

    def get_face_mask(self, points: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """Create an improved face mask using facial landmarks."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Create convex hull from face oval points
        face_points = points[self.FACE_OVAL]
        hull = cv2.convexHull(face_points)
        cv2.fillConvexPoly(mask, hull, 255)

        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask

    def enhanced_delaunay_triangulation(self, points: np.ndarray, img_shape: Tuple[int, int]) -> List[
        Tuple[int, int, int]]:
        """Enhanced Delaunay triangulation with better error handling."""
        h, w = img_shape[:2]

        # Add corner points for better triangulation
        corners = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
            [w // 2, 0], [w - 1, h // 2], [w // 2, h - 1], [0, h // 2]
        ])

        all_points = np.vstack([points, corners])

        # Create subdivision
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        # Insert points with error handling
        for i, point in enumerate(all_points):
            try:
                subdiv.insert((float(point[0]), float(point[1])))
            except cv2.error:
                continue

        # Get triangles
        try:
            triangle_list = subdiv.getTriangleList()
        except cv2.error:
            # Fallback to simple triangulation
            return self.fallback_triangulation(points)

        triangles_idx = []

        for t in triangle_list:
            # Extract triangle points
            pts_tri = [
                (int(t[0]), int(t[1])),
                (int(t[2]), int(t[3])),
                (int(t[4]), int(t[5]))
            ]

            # Check if triangle is within bounds
            if not all(0 <= x < w and 0 <= y < h for x, y in pts_tri):
                continue

            # Map to nearest landmark indices
            indices = []
            for pt in pts_tri:
                distances = np.sum((all_points - np.array(pt)) ** 2, axis=1)
                nearest_idx = np.argmin(distances)

                # Only use original face landmarks, not corner points
                if nearest_idx < len(points):
                    indices.append(nearest_idx)

            if len(set(indices)) == 3:
                triangles_idx.append(tuple(indices))

        # Remove duplicates
        triangles_idx = list(set(triangles_idx))

        if len(triangles_idx) < 50:  # Ensure we have enough triangles
            return self.fallback_triangulation(points)

        return triangles_idx

    def fallback_triangulation(self, points: np.ndarray) -> List[Tuple[int, int, int]]:
        """Fallback triangulation method for edge cases."""
        hull = cv2.convexHull(points, returnPoints=False).flatten()
        triangles = []

        # Simple fan triangulation from centroid
        centroid_idx = len(points) // 2  # Approximate centroid index

        for i in range(len(hull) - 1):
            triangles.append((hull[i], hull[i + 1], centroid_idx))

        return triangles

    def warp_triangle_optimized(self, src_img: np.ndarray, dst_img: np.ndarray,
                                src_tri: List[Tuple[int, int]], dst_tri: List[Tuple[int, int]]) -> None:
        """Optimized triangle warping with better error handling."""
        try:
            # Calculate bounding rectangles
            src_rect = cv2.boundingRect(np.float32(src_tri))
            dst_rect = cv2.boundingRect(np.float32(dst_tri))

            # Check for valid rectangles
            if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
                return

            # Offset triangle points
            src_tri_offset = [(p[0] - src_rect[0], p[1] - src_rect[1]) for p in src_tri]
            dst_tri_offset = [(p[0] - dst_rect[0], p[1] - dst_rect[1]) for p in dst_tri]

            # Extract source region
            y1, y2 = src_rect[1], src_rect[1] + src_rect[3]
            x1, x2 = src_rect[0], src_rect[0] + src_rect[2]

            if y1 >= src_img.shape[0] or x1 >= src_img.shape[1] or y2 <= 0 or x2 <= 0:
                return

            y1, y2 = max(0, y1), min(src_img.shape[0], y2)
            x1, x2 = max(0, x1), min(src_img.shape[1], x2)

            if y2 <= y1 or x2 <= x1:
                return

            src_cropped = src_img[y1:y2, x1:x2]
            if src_cropped.size == 0:
                return

            # Calculate affine transform
            M = cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset))

            # Apply warp
            warped = cv2.warpAffine(src_cropped, M, (dst_rect[2], dst_rect[3]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Create triangle mask
            mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), 255)

            # Apply to destination
            y1_dst, y2_dst = dst_rect[1], dst_rect[1] + dst_rect[3]
            x1_dst, x2_dst = dst_rect[0], dst_rect[0] + dst_rect[2]

            if (y1_dst >= dst_img.shape[0] or x1_dst >= dst_img.shape[1] or
                    y2_dst <= 0 or x2_dst <= 0):
                return

            y1_dst, y2_dst = max(0, y1_dst), min(dst_img.shape[0], y2_dst)
            x1_dst, x2_dst = max(0, x1_dst), min(dst_img.shape[1], x2_dst)

            if y2_dst <= y1_dst or x2_dst <= x1_dst:
                return

            # Adjust mask and warped image if needed
            mask_h, mask_w = y2_dst - y1_dst, x2_dst - x1_dst
            if mask_h != mask.shape[0] or mask_w != mask.shape[1]:
                mask = cv2.resize(mask, (mask_w, mask_h))
                warped = cv2.resize(warped, (mask_w, mask_h))

            dst_roi = dst_img[y1_dst:y2_dst, x1_dst:x2_dst]
            mask_bool = mask > 127
            dst_roi[mask_bool] = warped[mask_bool]

        except Exception:
            # Silently skip problematic triangles
            pass

    def prepare_source_face(self) -> bool:
        """Prepare source face data with caching."""
        if self.src_prepared:
            return True

        try:
            # Load and resize source image
            self.src_img_bgr = cv2.imread(self.source_img_path)
            if self.src_img_bgr is None:
                return False

            # Resize if too large
            h, w = self.src_img_bgr.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                self.src_img_bgr = cv2.resize(self.src_img_bgr, (new_w, new_h))
                h, w = new_h, new_w

            # Detect face landmarks
            with self.mp_face_mesh.FaceMesh(**self.face_mesh_static_config) as face_mesh:
                rgb = cv2.cvtColor(self.src_img_bgr, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    print("ERROR: No face detected in source image")
                    return False

                landmarks = results.multi_face_landmarks[0]
                self.src_points = self.mp_to_pixel_coords(landmarks, w, h)

            # Generate triangulation
            self.triangles_idx = self.enhanced_delaunay_triangulation(self.src_points, (h, w))

            print(f"INFO: Source prepared with {len(self.triangles_idx)} triangles")
            self.src_prepared = True
            return True

        except Exception as e:
            print(f"ERROR preparing source: {e}")
            return False

    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time

    def apply_color_correction(self, src_img: np.ndarray, dst_img: np.ndarray,
                               mask: np.ndarray) -> np.ndarray:
        """Apply color correction to match lighting conditions."""
        try:
            # Calculate color statistics for both images
            src_masked = cv2.bitwise_and(src_img, src_img, mask=mask)
            dst_masked = cv2.bitwise_and(dst_img, dst_img, mask=mask)

            src_mean = cv2.mean(src_masked, mask=mask)[:3]
            dst_mean = cv2.mean(dst_masked, mask=mask)[:3]

            # Apply simple color correction
            corrected = src_img.copy().astype(np.float32)
            for i in range(3):
                if src_mean[i] > 0:
                    corrected[:, :, i] *= dst_mean[i] / src_mean[i]

            return np.clip(corrected, 0, 255).astype(np.uint8)
        except:
            return src_img

    def run(self, show_fps: bool = True, show_landmarks: bool = False,
            save_key: str = 's', quit_key: str = 'q'):
        """Main execution loop with enhanced features."""
        if not self.prepare_source_face():
            print("ERROR: Failed to prepare source face")
            return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Face Swap Active - Controls:")
        print(f"  '{quit_key}' - Quit")
        print(f"  '{save_key}' - Save screenshot")
        print("  'l' - Toggle landmarks")
        print("  'f' - Toggle FPS display")

        with self.mp_face_mesh.FaceMesh(**self.face_mesh_config) as face_mesh:
            with mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=1) as selfie_segmentation:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)  # Mirror effect
                    h_tgt, w_tgt = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    output_frame = frame.copy()

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        tgt_points = self.mp_to_pixel_coords(landmarks, w_tgt, h_tgt)

                        # Create warped face
                        warped_face = np.zeros_like(output_frame)

                        # Warp triangles
                        for tri_idx in self.triangles_idx:
                            if len(tri_idx) != 3:
                                continue

                            src_tri = [tuple(self.src_points[i]) for i in tri_idx]
                            tgt_tri = [tuple(tgt_points[i]) for i in tri_idx]

                            # Skip invalid triangles
                            if any(x < 0 or x >= w_tgt or y < 0 or y >= h_tgt for x, y in tgt_tri):
                                continue

                            self.warp_triangle_optimized(self.src_img_bgr, warped_face, src_tri, tgt_tri)

                        # Create face mask
                        mask = self.get_face_mask(tgt_points, (h_tgt, w_tgt))

                        # Apply color correction
                        warped_face = self.apply_color_correction(warped_face, output_frame, mask)

                        # Seamless blending
                        try:
                            center = tuple(np.mean(tgt_points[self.FACE_OVAL], axis=0).astype(int))
                            output_frame = cv2.seamlessClone(warped_face, output_frame, mask, center, cv2.NORMAL_CLONE)
                        except:
                            # Fallback to alpha blending
                            alpha = (mask.astype(float) / 255.0)[:, :, None]
                            output_frame = (warped_face * alpha + output_frame * (1 - alpha)).astype(np.uint8)

                        # Optional landmark visualization
                        if show_landmarks:
                            for point in tgt_points:
                                cv2.circle(output_frame, tuple(point), 1, (0, 255, 0), -1)

                    # Update and display FPS
                    self.update_fps()
                    if show_fps:
                        cv2.putText(output_frame, f'FPS: {self.current_fps:.1f}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow('Enhanced Face Swap', output_frame)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(quit_key):
                        break
                    elif key == ord(save_key):
                        filename = f'face_swap_{int(time.time())}.jpg'
                        cv2.imwrite(filename, output_frame)
                        print(f'Saved: {filename}')
                    elif key == ord('l'):
                        show_landmarks = not show_landmarks
                        print(f'Landmarks: {"ON" if show_landmarks else "OFF"}')
                    elif key == ord('f'):
                        show_fps = not show_fps
                        print(f'FPS display: {"ON" if show_fps else "OFF"}')

        cap.release()
        cv2.destroyAllWindows()


# ------------- CONFIG - CHANGE YOUR IMAGE PATH HERE -------------
SOURCE_IMAGE_PATH = "C:\\Users\\smile\\Downloads\\neymar.jpg"  # PUT YOUR IMAGE PATH HERE
CAM_WIDTH = 640
CAM_HEIGHT = 480


# ----------------------------------------------------------------

def main():
    """Main function - now uses hardcoded path from config above."""
    global SOURCE_IMAGE_PATH
    try:
        #prompt = input("Enter person(PERSON.jpg): ")

        #SOURCE_IMAGE_PATH1 = SOURCE_IMAGE_PATH + prompt
        face_swap = FaceSwap(SOURCE_IMAGE_PATH, CAM_WIDTH, CAM_HEIGHT)
        face_swap.validate_source_image()
        #draw_face_landmarks()
        face_swap.run(show_fps=True, show_landmarks=False)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    while True:
        main()
