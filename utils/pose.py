# Calculate and send pose tracker position and rotation based on keypoints location
import numpy as np

def calc_pose(points, client):
    head_center = (points[7] + points[8]) / 2
    rot = np.rad2deg(np.arctan2(points[8][2] - points[7][2], points[8][0] - points[7][0]))
    client.send_pos("head", head_center)
    client.send_rot("head", [0, 0, 0])
    
    points[:, 2] += 0.35
    client.send_pos(1, points[27]) # left ankle
    client.send_pos(2, points[28]) # right ankle

    left_heel = points[29]
    left_toe = points[31]
    
    rot = np.rad2deg(np.arctan2(left_heel[2] - left_toe[2], left_heel[0] - left_toe[0])) + 90
    rot2 = np.rad2deg(np.arctan2(left_heel[2] - left_toe[2], left_heel[1] - left_toe[1])) + 90
    #rot = np.rad2deg(np.arctan2(left_heel[2] - left_toe[2], left_heel[0] - left_toe[0])) + 90
    
    client.send_rot(1, [rot2,-rot,.0])

    right_heel = points[30]
    right_toe = points[32]

    rot = np.rad2deg(np.arctan2(right_heel[2] - right_toe[2], right_heel[0] - right_toe[0])) + 90
    rot2 = np.rad2deg(np.arctan2(right_heel[2] - right_toe[2], right_heel[1] - right_toe[1])) + 90
    #rot = np.rad2deg(np.arctan2(left_heel[2] - left_toe[2], left_heel[0] - left_toe[0])) + 90

    client.send_rot(2, [rot2,-rot,.0])

    left_knee = points[25]
    right_knee = points[26]
    client.send_pos(4, left_knee)
    client.send_pos(5, right_knee)
    client.send_rot(4)
    client.send_rot(5)

    hip_center = points[33]
    client.send_pos(3, hip_center)

    right_hip = points[24]
    rot = np.rad2deg(np.arctan2(right_hip[2] - hip_center[2], right_hip[0] - hip_center[0]))
    client.send_rot(3, [0, -rot, 0])