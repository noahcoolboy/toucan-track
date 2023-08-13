from math import floor
import cv2
import numpy as np

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def display_result(
  image, landmarks):
  # draw using cv2
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  for con in KEYPOINT_EDGE_INDS_TO_COLOR:
    p0 = tuple(landmarks[con[0]])
    p1 = tuple(landmarks[con[1]])
    cv2.line(image, (floor(p0[0]), floor(p0[1])),
             (floor(p1[0]), floor(p1[1])), (0, 255, 0), 2)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return image

#region Pose Debug
from matplotlib import pyplot as plt
# Believe it or not, this is the most efficient way to draw the skeleton
connections_body = [16, 14, 12, 6, 8, 10, 8, 6, 5, 7, 9, 7, 5, 11, 12, 11, 13, 15]
connections_face = [4, 2, 0, 1, 3]
lines_body = None
lines_face = None
fig = None

def init_pose_plot(size = 6, radius = 2.5):
    global lines_body, lines_face, fig

    plt.ioff()
    fig = plt.figure(figsize=(size, size))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=15., azim=np.array(75., dtype=np.float32))
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 12.5
    ax.set_title("Visualisation")
    
    lines_body = ax.plot([0, 0], [0, 0], [0, 0], zdir='y', c='black')
    lines_face = ax.plot([0, 0], [0, 0], [0, 0], zdir='y', c='red')
    
    plt.show(block=False)

def update_pose_plot(pos):
    lines_body[0].set_xdata(pos[connections_body, 0])
    lines_body[0].set_ydata(pos[connections_body, 1])
    lines_body[0].set_3d_properties(pos[connections_body, 2], zdir='y')

    lines_face[0].set_xdata(pos[connections_face, 0])
    lines_face[0].set_ydata(pos[connections_face, 1])
    lines_face[0].set_3d_properties(pos[connections_face, 2], zdir='y')

def draw_plot():
    fig.canvas.draw()
    fig.canvas.flush_events()
#endregion