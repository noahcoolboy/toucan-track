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

def triangulate(proj, points):
    n_views = len(proj)
    A = np.zeros((2 * n_views, 4))
    for j in range(n_views):
        A[j * 2 + 0] = points[j][0] * proj[j][2, :] - proj[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj[j][2, :] - proj[j][1, :]

    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    #point_3d = homogeneous_to_euclidean(point_3d_homo)
    point_3d = point_3d_homo

    return point_3d

def get_depth(oncm, values, multicam_val=0.75):
    num_views = len(oncm)
    num_keypoints = len(values[0][1])

    proj = np.array([oncm[i][5] for i in range(len(oncm))])
    points_2d = np.array([values[i][1] for i in range(len(values))]).transpose(1, 0, 2)

    points = np.zeros((num_keypoints, 4))
    for i in range(num_keypoints):
        idx = np.arange(num_views)
        if num_views > 2:
            if multicam_val > 1:
                # Get the best of N cameras
                idx = np.argsort(points_2d[i, :, 3])[:multicam_val]
            else:
                # Get the cameras with confidence above a threshold
                idx = np.where(points_2d[i, :, 3] > multicam_val)[0]
        
        if len(idx) > 1:
            points[i] = triangulate(proj[idx], points_2d[i][idx])
        else:
            points[i] = triangulate(proj, points_2d[i])

    points3d = cv2.convertPointsFromHomogeneous(points)
    return points3d