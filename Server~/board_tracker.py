import cv2
from pupil_apriltags import Detector
import numpy as np


# Open webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Failed to open camera")
#     exit()

# self.fx = 950.0
# self.fy = 950.0
# self.cx = 640.0
# self.cy = 360.0


class BoardTracker:
    def __init__(self):
        # Camera intrinsics (you may need to calibrate your camera)
        self.fx = 950.0
        self.fy = 950.0
        self.cx = 640.0
        self.cy = 360.0
        self.params_initialized = False

        self.tag_size = 0.056  # In meters

        # Initialize detector
        self.detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

        #board dimensions:
        board_width = 0.28
        board_height = 0.19

        self.board_coords = np.array([
            [self.tag_size/2, -self.tag_size/2 - 0.007, 0],
            [board_width + self.tag_size/2 + 0.005, -self.tag_size/2 - 0.007, 0],
            [board_width + self.tag_size/2 + 0.005, board_height - self.tag_size/2, 0],
            [self.tag_size/2, board_height - self.tag_size/2, 0]
        ], dtype=np.float32)

    def assign_camera_params(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.params_initialized = True

    def project_2D(self, points3D, fx, fy, cx, cy):
        points_2D = []
        for point in points3D:
            X , Y, Z = point
            u = (X/Z) * fx + cx
            v = (Y/Z) * fy + cy
            points_2D.append([u, v])
        return np.array(points_2D, dtype=np.float32)

    def get_grid_coords(self, row, col):
        pass

    def get_board_segment(self, frame):
        camera_params = (self.fx, self.fy, self.cx, self.cy)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=True,
                            camera_params=camera_params, tag_size=self.tag_size)
        cropped_img = frame.copy()
        for tag in tags:
            # Draw corners
            # for corner in tag.corners:
            #     pt = tuple(map(int, corner))
            #     cv2.circle(frame, pt, 5, (0, 255, 0), -1)

            # # Draw center and ID
            # center = tuple(map(int, tag.center))
            # cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # cv2.putText(frame, f"ID: {tag.tag_id}", (center[0] + 10, center[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Extract pose (translation and rotation)
            rvec = tag.pose_R  # 3x3 rotation matrix
            tvec = tag.pose_t  # 3x1 translation vector

            #board in world coordinates
            board_world_coords = rvec @ self.board_coords.T + tvec

            print("board_world_coords", board_world_coords)
            #get camera coordinates
            board_camera = self.project_2D(board_world_coords.T, self.fx, self.fy, self.cx, self.cy)
            # Draw board
            # for i in range(len(board_camera)):
            #     pt = tuple(map(int, board_camera[i]))
            #     cv2.circle(frame, pt, 5, (255, 0, 255), -1)
            # # Draw lines between corners
            # for i in range(len(board_camera)):
            #     pt1 = tuple(map(int, board_camera[i]))
            #     pt2 = tuple(map(int, board_camera[(i + 1) % len(board_camera)]))
            #     cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

            #crop image to board
            x_min = int(min(board_camera[:, 0]))
            x_max = int(max(board_camera[:, 0]))
            y_min = int(min(board_camera[:, 1]))
            y_max = int(max(board_camera[:, 1]))

            # Ensure the coordinates are within the image bounds
            x_min = max(0, x_min)
            x_max = min(frame.shape[1], x_max)
            y_min = max(0, y_min)
            y_max = min(frame.shape[0], y_max)
            cropped_img = frame[y_min:y_max, x_min:x_max]

        cv2.imwrite("cropped_img.png", cropped_img)
        return cropped_img
