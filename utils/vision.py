# Functions for camera distortion and 3D keypoint calculation
import cv2
import pyjson5
import numpy as np
import camera.binding as camera

with open("calib.json", "r") as f:
    calib = pyjson5.load(f)

def get_cam(type, id):
    if type == "PS3 Eye Camera":
        return camera.Camera(id, (640, 480), 50, camera.ps3eye_format.PS3EYE_FORMAT_BGR)

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def read_camera_parameters(camera_id):
    return np.array(calib["cameras"][int(camera_id)]["intrinsics"]["cmtx"]), np.array(calib["cameras"][int(camera_id)]["intrinsics"]["dist"])

def read_rotation_translation(camera_id):
    return np.array(calib["cameras"][int(camera_id)]["extrinsics"]["rvec"]).squeeze(), np.array(calib["cameras"][int(camera_id)]["extrinsics"]["tvec"])

def get_projection_matrix(camera_id):
    #read camera parameters
    cmtx, dist = read_camera_parameters(int(camera_id))
    rvec, tvec = read_rotation_translation(int(camera_id))

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def get_depth(proj0, proj1, points0, points1):
    point3d = cv2.triangulatePoints(proj0, proj1, points0, points1)
    point3d = cv2.convertPointsFromHomogeneous(point3d.T)
    return point3d