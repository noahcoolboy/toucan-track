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

model = ["lite", "full", "heavy"][settings.get("model", 1)]
suppress_warnings = onnxruntime.SessionOptions()
suppress_warnings.log_severity_level = 3
det_sess = onnxruntime.InferenceSession("models/pose_detection.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
landmark_sess = onnxruntime.InferenceSession(f"models/pose_landmark_{model}_batched.onnx", suppress_warnings, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

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
pose_det_pre_queue = Queue(maxsize=1)
pose_det_queue = Queue(maxsize=1)
pose_det_post_queue = Queue(maxsize=1)
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


# Preprocessing thread for the pose detection model
def pose_det_pre_thread():
    while running:
        # Fetch the frames
        # Frames get sorted to avoid race condition
        frames = [cam_queue.get(block=True) for i in range(cam_count)]
        frames.sort(key=lambda x: x[0])
        frames = [frame[1] for frame in frames]

        cam_sync.set()
        cam_sync.clear()

        imgs = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in frames]

        values = []

        # If we have a ROI, we can skip the detection step
        # This ROI is given be the landmark detection
        if roi is not None:
            # A copy of the roi variable is made to avoid the ROI being nullified while the image is being cropped
            roib = roi.copy()
            for i in range(cam_count):
                xc, yc, scale, theta = roib[i]
                # Images get cropped according to the ROI
                img2, affine, _ = inference.extract_roi(imgs[i], 
                    np.array([xc]),
                    np.array([yc]),
                    np.array([theta]),
                    np.array([scale])
                )
                values.append((img2[0], affine[0], imgs[i]))

            # Images get put on the pose_det_post_queue directly, skipping the pose detection step
            # As the images have already been cropped to fit the person
            pose_det_post_queue.put(values, block=True)
            continue
        
        values = []
        # Crops the image to 224x224 for a round of pose detection
        for i in range(cam_count):
            img224, scale, pad = inference.resize_pad(imgs[i])
            img224 = img224.astype('float32') / 128. - 1.
            values.append((img224, scale, pad, imgs[i]))
        pose_det_pre_queue.put(values, block=True)


# Run inference on the pose detection model
def pose_det_thread():
    while running:
        values = pose_det_pre_queue.get(block=True)
        for i in range(cam_count):
            img224, scale, pad, img = values[i]
            img224s = np.expand_dims(img224, axis=0)
            # TODO: figure out how to batch this
            pred_onnx = det_sess.run(["Identity", "Identity_1"], {"input_1": img224s})
            values[i] = (pred_onnx, img, scale, pad)
        
        pose_det_queue.put(values, block=True)


# Post processing thread for the detection model
def pose_det_post_thread():
    while running:
        values = pose_det_queue.get(block=True)

        for i in range(cam_count):
            pred_onnx, img, scale, pad = values[i]
            post = inference.detector_postprocess(pred_onnx, min_score_thresh=settings.get("pose_det_min_score", 0.75))
            count = len(post) if post[0].size != 0 else 0

            # If no person is detected on one of the cameras, we can't continue
            if count < 1:
                break

            imgs, affine, _ = inference.estimator_preprocess(img, post, scale, pad)  
            values[i] = (imgs[0], affine[0], img)
        else:
            pose_det_post_queue.put(values, block=True)


# Thread to run inference on the pose landmark model
def pose_landmark_thread():
    global roi

    prev_landmarks = None
    prev_t = None
    
    while running:
        values = pose_det_post_queue.get(block=True)
        output = landmark_sess.run(["Identity", "Identity_1", "Identity_3"], {"input_1": [values[i][0].transpose(2, 0, 1) for i in range(cam_count)]})
        normalized_landmarks, f, heatmap = output

        # If the confidence of the pose detection (on any of the images) is too low, we can't continue
        # The ROI is also removed as no one was found in it
        if((f[:, 0] < settings.get("pose_lm_min_score", 0.3)).any()):
            roi = None
            continue

        normalized_landmarks = inference.landmark_postprocess(normalized_landmarks, True)
        landmarks = np.stack(normalized_landmarks)
        if settings.get("refine_landmarks", True):
            landmarks = inference.refine_landmarks(landmarks, heatmap, kernel_size=settings.get("refine_kernel_size", 7), min_conf=settings.get("refine_min_score", 0.5))
        landmarks = inference.denormalize_landmarks(landmarks, [values[i][1] for i in range(cam_count)])

        if settings.get("flip_detection", False) and prev_landmarks is not None and time.time() - prev_t < 0.1:
            inference.autoflip(prev_landmarks, landmarks, settings.get("flip_detection_max", 10))
        prev_landmarks = landmarks
        prev_t = time.time()
        
        pose_landmark_queue.put((landmarks, f, [values[i][2] for i in range(cam_count)]), block=True)


# Post processing for the landmarks
def pose_landmark_post_thread():
    global roi

    smoothing = [[filters.get_filter(settings.get("2d_filter"), fps, 2) for _ in range(39)] for _ in range(cam_count)]

    while running and (not settings.get("debug", False) or cv2.waitKey(1) != 27):
        landmarks, flags, imgs = pose_landmark_queue.get(block=True)
        values = []

        roi = [inference.landmarks_to_roi(landmarks[i]) for i in range(cam_count)]

        t = time.time() * 1000
        for i in range(cam_count):
            for j in range(39):
                landmarks[i][j][:2] = smoothing[i][j].filter(landmarks[i][j][:2], t)

            values.append((imgs[i], landmarks[i], flags[i]))

            if settings.get("debug", False) and roi:
                frame = imgs[i]
                draw.display_result(frame, landmarks[i], flags[i], roi[i])
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

    smoothing = [filters.get_filter(settings.get("3d_filter"), fps, 3) for _ in range(39)]

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
        points = points.squeeze() / 100 # (39, 3)
        
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
        for i in range(39):
            points[i] = smoothing[i].filter(points[i], t)

        pose.calc_pose(points, client, settings.get("send_rot", False))

        if settings.get("draw_pose", False) and settings.get("debug", False):
            draw.update_pose_plot(points)

if __name__ == "__main__":
    #region Thread Management
    threads = [
        threading.Thread(target=pose_det_pre_thread),
        threading.Thread(target=pose_det_thread),
        threading.Thread(target=pose_det_post_thread),
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