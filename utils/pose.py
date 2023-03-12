# Calculate and send pose tracker position and rotation based on keypoints location
import numpy as np
from . import owotrack
from scipy.spatial.transform import Rotation as R

def get_foot_rot(knee, ankle, direction):
    # Calculate the unit vector from the ankle to the knee
    ankle_to_knee = knee - ankle
    ankle_to_knee /= np.linalg.norm(ankle_to_knee)

    # Calculate the vector that points forward while being perpendicular to the ankle to knee vector (the lower leg)
    forward_vector = np.array([0, 0, 1.])
    forward_vector -= np.dot(forward_vector, ankle_to_knee) * ankle_to_knee
    forward_vector /= np.linalg.norm(forward_vector)

    # Rotate the forward vetor around the ankle to knee vector by the direction
    angle = direction / 180 * np.pi
    point = forward_vector
    point = np.cos(angle) * point + np.sin(angle) * np.cross(ankle_to_knee, point) + np.dot(ankle_to_knee, point) * ankle_to_knee
    point = np.dot(R.from_euler("y", angle).as_matrix(), point)

    # Convert forward up right to rotation vector
    rot = R.from_matrix(np.column_stack((np.cross(ankle_to_knee, point), ankle_to_knee, point)))
    return np.rad2deg(rot.as_rotvec())


def get_hip_rot(left_shoulder, right_shoulder, left_hip, right_hip):
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2
    spine_direction = hip_midpoint - shoulder_midpoint
    spine_direction /= np.linalg.norm(spine_direction)
    hip_direction = right_hip - left_hip
    hip_direction /= np.linalg.norm(hip_direction)
    axis_of_rotation = np.cross(spine_direction, hip_direction)
    angle_of_rotation = np.arccos(np.dot(spine_direction, hip_direction) / (np.linalg.norm(spine_direction) * np.linalg.norm(hip_direction)))
    rotation_vector = axis_of_rotation * angle_of_rotation
    return np.rad2deg(rotation_vector)

owotrack_server = None
def start_owotrack_server():
    global owotrack_server

    if owotrack_server is None:
        owotrack_server = owotrack.OwoTrackServer(6969)


def calc_pose(points, client, send_rot=False):
    # Hips
    direction = 0
    hip_center = points[33]
    client.send_pos(3, hip_center)

    if owotrack_server and owotrack_server.connected:
        #direction = owotrack_server.rotation[0]
        client.send_rot(3, owotrack_server.rotation)
    if send_rot:
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_hip = points[23]
        right_hip = points[24]
        hip_rot = get_hip_rot(left_shoulder, right_shoulder, left_hip, right_hip)
        direction = hip_rot[0]
        client.send_rot(3, hip_rot[[1, 0, 2]])
    else:
        client.send_rot(3)

    # Head
    head_center = (points[7] + points[8]) / 2
    client.send_pos("head", head_center)
    client.send_rot("head", [0, 0, 0])
    
    # Ankles
    left_knee, right_knee, left_ankle, right_ankle = points[25:29]
    client.send_pos(1, left_ankle)
    client.send_pos(2, right_ankle)

    client.send_rot(1, get_foot_rot(left_knee, left_ankle, direction))
    client.send_rot(2, get_foot_rot(right_knee, right_ankle, direction))

    # Knees
    client.send_pos(4, left_knee)
    client.send_pos(5, right_knee)
    client.send_rot(4)
    client.send_rot(5)