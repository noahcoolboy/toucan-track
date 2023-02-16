# Functions for drawing debug results
import cv2
import numpy as np

#region Camera Debug
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def line(input_img, landmarks, flags, point1, point2):
    threshold = 0.2
    
    if flags[0] >= threshold and landmarks[point1, 3] >= threshold and landmarks[point2, 3] >= threshold:
        color = hsv_to_rgb(255 * point1 / 33, 255, 255)
        line_width = 2

        x1 = int(landmarks[point1, 0])
        y1 = int(landmarks[point1, 1])
        x2 = int(landmarks[point2, 0])
        y2 = int(landmarks[point2, 1])
        cv2.line(input_img, (x1, y1), (x2, y2), color, line_width)

def display_result(img, landmarks, flags, roi):
    connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                   (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                   (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                   (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                   (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                   (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                   (29, 31), (30, 32), (27, 31), (28, 32)]

    for connection in connections:
        line(img, landmarks, flags, connection[0], connection[1])
    
    if(len(landmarks) > 32):
        for i in range(1,7):
            cv2.circle(img, (int(landmarks[32+i, 0]), int(landmarks[32+i, 1])), 5, (255*(i%2), 255*((i//2)%2), 255*((i//4)%2)), -1)

    if roi:
        scale = roi[2] / 2
        corners = np.array([
            [-scale, scale],
            [scale, scale],
            [scale, -scale],
            [-scale, -scale]
        ])

        theta = roi[3]
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]],)

        corners = np.dot(corners.reshape(4, 1, 2), R.reshape(2, 2, 1)).reshape(4, 2)
        corners[:, 0] += roi[0]
        corners[:, 1] += roi[1]
        
        cv2.polylines(img, [np.int32(corners)], True, (0, 255, 0), 2)

    return img
#endregion

#region Pose Debug
from matplotlib import pyplot as plt
lines_3d = []
parents = np.array([-1, 0, 1, 2, 0, 4, 5, 3, 6, -1, 9, -1, 11, 11, 12, 13, 14, 15, 16, 15, 16, 15, 16, 11, 12, 23, 24, 25, 26, 27, 28])
joints_right = []

def init_pose_plot(size = 6, radius = 2.5):
    plt.ioff()
    fig = plt.figure(figsize=(size, size))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=15., azim=np.array(70., dtype=np.float32))
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 12.5
    ax.set_title("Visualisation")
    
    for j, j_parent in enumerate(parents):
        #if j_parent == -1:
        #    continue

        col = 'red' if j in joints_right else 'black'
        lines_3d.append(ax.plot([0, 0], [0, 0], [0, 0], zdir='y', c=col))
    
    plt.show(block=False)

def update_pose_plot(pos):
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        
        lines_3d[j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]])) # Hotfix matplotlib's bug. https://github.com/matplotlib/matplotlib/pull/20555
        lines_3d[j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
        lines_3d[j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='y')

    #plt.draw()
#endregion