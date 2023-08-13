from queue import Queue
import numpy as np
import onnxruntime
import threading
import pyjson5
import time
import cv2

import utils.inference as inference
import utils.filters as filters
import utils.vision as vision
import utils.client as client
import utils.pose as pose
import utils.draw as draw
import camera.binding as camera

settings = pyjson5.decode_io(open("settings.json", "r"))
client = client.OSCClient(settings["ip"], settings.get("port", 9000))

calib = pyjson5.decode_io(open("calib.json", "r"))

model = ["lightning", "thunder"][settings.get("model", 1)]
size = [192, 256][settings.get("model", 1)]
suppress_warnings = onnxruntime.SessionOptions()
suppress_warnings.log_severity_level = 3
landmark_sess = onnxruntime.InferenceSession(f"models/{model}.onnx", suppress_warnings, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

running = True

#region Camera Initialization
fps = settings.get("fps", 50)
res = (640, 480)

cam_count = len(calib["cameras"])

cameras = []
for i in range(len(calib["cameras"])):
    cameras.append(vision.get_cam(calib["cameras"][i]["type"], calib["cameras"][i]["id"]))

oncm = []
for i in range(cam_count):
    cmtx, dist = vision.read_camera_parameters(i)
    rvec, tvec = vision.read_rotation_translation(i)
    optimal_cmtx, roi0 = cv2.getOptimalNewCameraMatrix(cmtx, dist, res, 1, res)
    proj = vision.get_projection_matrix(i)
    oncm.append((cmtx, dist, optimal_cmtx, rvec, tvec, proj))
#endregion

#region Multithreading Setup
roi = None
cam_queue = Queue(maxsize=cam_count)
pose_landmark_queue = Queue(maxsize=1)
pose_landmark_post_queue = Queue(maxsize=1)

cam_sync = threading.Event()
#endregion

# Fetch frame of camera, and undistorts it.
# A cam_thread gets spun up for each camera for being able to fetch frames in parallel
def cam_thread(id):
    while running:
        frame = cameras[id].get_frame()
        if settings.get("undistort", True):
            frame = cv2.undistort(frame, oncm[id][0], oncm[id][1], None, oncm[id][2])
        frame.flags.writeable = False

        cam_queue.put((id, frame), block=True)
        cam_sync.wait()

# Post processing thread for the detection model
def pose_landmark_thread():
    global roi
    while running:
        frames = [cam_queue.get(block=True) for i in range(cam_count)]
        frames.sort(key=lambda x: x[0])
        frames = [frame[1] for frame in frames]

        cam_sync.set()
        cam_sync.clear()

        imgs = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in frames]
        results = np.zeros((cam_count, 1, 1, 17, 3), dtype=np.float32)
        confs = []

        for i in range(cam_count):
            crop_region = inference.init_crop_region(res[1], res[0]) if roi is None else roi[i]
            imgs[i] = inference.crop_and_resize(imgs[i], crop_region, size)

            out = landmark_sess.run(None, {"input": np.expand_dims(np.array(imgs[i]).astype(np.int32), axis=0)})[0]

            image_height, image_width, _ = frames[i].shape
            for idx in range(17):
                out[0, 0, idx, 0] = (
                    crop_region['y_min'] * image_height +
                    crop_region['height'] * image_height *
                    out[0, 0, idx, 0]) / image_height
                out[0, 0, idx, 1] = (
                    crop_region['x_min'] * image_width +
                    crop_region['width'] * image_width *
                    out[0, 0, idx, 1]) / image_width

            results[i] = out
            conf = np.sum(out[0][0][:, 2]) / 17
            confs.append(conf)

            if(conf < settings.get("pose_lm_min_score", 0.35)):
                roi = None
                break
        else:
            pose_landmark_queue.put((results, confs, frames), block=True)
            continue

        

# Post processing for the landmarks
def pose_landmark_post_thread():
    global roi

    smoothing = [[filters.get_filter(settings.get("2d_filter"), fps, 2) for _ in range(17)] for _ in range(cam_count)]

    while running and (not settings.get("debug", False) or cv2.waitKey(1) != 27):
        landmarks, flags, imgs = pose_landmark_queue.get(block=True)
        values = []

        roi = [inference.determine_crop_region(landmarks[i], imgs[i].shape[0], imgs[i].shape[1]) for i in range(cam_count)]

        landmarks = landmarks.reshape((cam_count, 17, 3))
        t = time.time() * 1000
        for i in range(cam_count):
            landmarks[i, :, 0] *= imgs[i].shape[0]
            landmarks[i, :, 1] *= imgs[i].shape[1]
            landmarks[i, :, [0, 1]] = landmarks[i, :, [1, 0]]
            for j in range(17):
                landmarks[i][j][:2] = smoothing[i][j].filter(landmarks[i][j][:2], t)

            values.append((imgs[i], landmarks[i], flags[i]))

            if settings.get("debug", False):
                frame = imgs[i]
                frame = draw.display_result(frame, landmarks[i])
                cv2.imshow("Pose{}".format(i), frame)
        
        pose_landmark_post_queue.put(values, block=True)


if settings.get("draw_pose", False) and settings.get("debug", False):
    draw.init_pose_plot()

if settings.get("owotrack", False):
    pose.start_owotrack_server()

# Calculate pose from 3d points and send it to the OSC server
def triangulation_thread():
    start = time.time()
    frames = 0

    smoothing = [filters.get_filter(settings.get("3d_filter"), fps, 3) for _ in range(17)]

    while running:
        values = pose_landmark_post_queue.get(block=True)

        # Display FPS
        frames += 1
        if frames == 100:
            print("FPS: {}".format(100 / (time.time() - start)))
            start = time.time()
            frames = 0
        
        # Calculate and smooth 3D points
        points = vision.get_depth(oncm, values, multicam_val=settings.get("multicam_val", 0.75))
        points = points.squeeze() / 100 # (17, 3)
        
        points = points * settings.get("scale_multiplier", 1)
        if settings.get("flip_x", False):
            points[:, 0] = -points[:, 0]
        if settings.get("flip_y", False):
            points[:, 1] = -points[:, 1]
        if settings.get("flip_z", False):
            points[:, 2] = -points[:, 2]
        if settings.get("swap_xz", False):
            points[:, [0, 2]] = points[:, [2, 0]]
        
        t = time.time() * 1000
        for i in range(17):
            points[i] = smoothing[i].filter(points[i], t)

        pose.calc_pose(points, client, settings.get("send_rot", False))

        if settings.get("draw_pose", False) and settings.get("debug", False):
            draw.update_pose_plot(points)

if __name__ == "__main__":
    #region Thread Management
    threads = [
        threading.Thread(target=pose_landmark_thread),
        threading.Thread(target=pose_landmark_post_thread),
        threading.Thread(target=triangulation_thread)
    ]

    for i in range(cam_count):
        threads.append(threading.Thread(target=cam_thread, args=(i,)))

    for thread in threads:
        thread.daemon = True
        thread.start()
    #endregion

    # do nothing until keyboard interrupt
    try:
        if settings.get("draw_pose", False) and settings.get("debug", False):
            while True:
                draw.draw_plot()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        running = False
        time.sleep(0.1)
        for camera in cameras: # Gracefully close all cameras
            del camera
            time.sleep(0.2)