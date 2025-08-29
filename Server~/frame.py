import time
import numpy as np

class Frame:
    def __init__(self, client_id, img, cam_mat, proj_mat, dist_mat, timestamp=time.time()):
        self.client_id = int(client_id)
        self.img = np.array(img)
        self.cam_mat = cam_mat
        self.proj_mat = proj_mat
        self.dist_mat = dist_mat
        self.timestamp = timestamp
