import camera.binding as camera
import PySimpleGUI as sg
from cv2 import aruco
import numpy as np
import pyjson5
import math
import cv2
import os

if not os.path.exists("calib.json"):
    with open("calib.json", "w") as f:
        f.write("""{
    cameras: [],
    settings: {
        "checkerboard_box_size_scale" : 4,
        "checkerboard_columns": 11,
        "checkerboard_rows": 8, 
        "aruco_size": 18,
        "mono_calibration_frames": 50,
        "stereo_calibration_frames": 50,
        "aruco_calibration_frames": 200,
        "resolution": [640, 480]
    }
}""")
    
with open("calib.json", "r") as f:
    calib = pyjson5.load(f)

def save_calib():
    with open("calib.json", "w") as f:
        f.write(pyjson5.dumps(calib))

def get_cam(type, id):
    if type == "PS3 Eye Camera":
        return camera.Camera(id, (640, 480), 50, camera.ps3eye_format.PS3EYE_FORMAT_BGR)

def add_camera():
    ids = list(range(camera.get_camera_count()))

    add_camera_layout = [
        [sg.Text('Add Camera', font="SegoeUI 16", justification='center')],
        [sg.Text('Camera Type:', font="SegoeUI 12"), sg.Combo(values=["PS3 Eye Camera"], key="camtype", readonly=True, default_value="PS3 Eye Camera")],
        [sg.Text('Camera ID:', font="SegoeUI 12"), sg.Combo(values=ids, key="camid", readonly=True, default_value=ids[0])],
        [sg.Text('Name (optional):', font="SegoeUI 12"), sg.Input(key="camname", size=(15, 1))],
        [sg.Button('Add', key="add"), sg.Button('Cancel', key="cancel")]
    ]

    window = sg.Window("Add Camera", add_camera_layout)

    while True:
        event, values = window.read()
        
        cam = {
            "id": int(values["camid"]),
            "type": values["camtype"],
            "name": values["camname"] or f"{values['camtype']} {values['camid']}"
        }
        
        if event == "add":
            if len([x for x in calib["cameras"] if (x["id"] == cam["id"] and x["type"] == cam["type"]) or (x["name"] == cam["name"])]) == 0:
                calib["cameras"].append(cam)
                window.close()
                return True
            else:
                sg.popup("Camera already exists!")
        else:
            window.close()
            return False
        

def settings():
    settings_layout = [
        [sg.Text('Settings', font="SegoeUI 16", justification='center')],
        [sg.Text('Checkerboard Box Size (cm):', font="SegoeUI 12"), sg.Input(calib["settings"]["checkerboard_box_size_scale"], key="checkerboard_box_size_scale", size=(4, 1))],
        [sg.Text('Checkerboard Columns:', font="SegoeUI 12"), sg.Input(calib["settings"]["checkerboard_columns"], key="checkerboard_columns", size=(15, 1))],
        [sg.Text('Checkerboard Rows:', font="SegoeUI 12"), sg.Input(calib["settings"]["checkerboard_rows"], key="checkerboard_rows", size=(15, 1))],
        [sg.Text('Aruco Size (cm):', font="SegoeUI 12"), sg.Input(calib["settings"]["aruco_size"], key="aruco_size", size=(15, 1))],
        [sg.Text('Mono Calibration Frames:', font="SegoeUI 12"), sg.Input(calib["settings"]["mono_calibration_frames"], key="mono_calibration_frames", size=(15, 1))],
        [sg.Text('Stereo Calibration Frames:', font="SegoeUI 12"), sg.Input(calib["settings"]["stereo_calibration_frames"], key="stereo_calibration_frames", size=(15, 1))],
        [sg.Text('Aruco Calibration Frames:', font="SegoeUI 12"), sg.Input(calib["settings"]["aruco_calibration_frames"], key="aruco_calibration_frames", size=(15, 1))],
        #[sg.Text('Resolution:', font="SegoeUI 12"), sg.Input(calib["settings"]["resolution"][0], key="resolution_x", size=(5, 1)), sg.Input(calib["settings"]["resolution"][1], key="resolution_y", size=(5, 1))],
        [sg.Button('Save', key="save"), sg.Button('Cancel', key="cancel")]
    ]

    window = sg.Window("Settings", settings_layout)

    event, values = window.read()
    
    if event == "save":
        s = calib["settings"]
        s["checkerboard_box_size_scale"] = int(values["checkerboard_box_size_scale"])
        s["checkerboard_columns"] = int(values["checkerboard_columns"])
        s["checkerboard_rows"] = int(values["checkerboard_rows"])
        s["aruco_size"] = int(values["aruco_size"])
        s["mono_calibration_frames"] = int(values["mono_calibration_frames"])
        s["stereo_calibration_frames"] = int(values["stereo_calibration_frames"])
        s["aruco_calibration_frames"] = int(values["aruco_calibration_frames"])
        #s["resolution"] = [int(values["resolution_x"]), int(values["resolution_y"])]
        window.close()
        save_calib()
        return True
    else:
        window.close()
        return False
    
def calibrate_intrinsics(cam, cap):
    calibrate_intrinsics_layout = [
        [sg.Column([
            [sg.Text('Intrinsics Calibration', font="SegoeUI 16", justification='center')],
            [sg.Image(key="img")],
            [sg.Text('Show the Checkerboard to Calibrate Intrinsics', justification='center')],
            [sg.Text('', justification='center', key="status")],
            [sg.Button('Quit', key="quit")]
        ], element_justification='center')],
    ]

    columns = calib['settings']['checkerboard_columns']
    rows = calib['settings']['checkerboard_rows']
    world_scaling = calib['settings']['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    imgpoints = []
    objpoints = []

    framecnt = 0
    captured = 0
    window = sg.Window("Intrinsics Calibration", calibrate_intrinsics_layout)
    while True:
        event, values = window.read(timeout=1)
        if event == "quit" or event is None:
            window.close()
            return

        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret2, corners = cv2.findChessboardCornersSB(gray, (rows, columns), None, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ret2 == True:
                if framecnt % 2 == 0:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    captured += 1
                
                cv2.drawChessboardCorners(frame, (rows, columns), corners, ret2)
        framecnt += 1
        
        if ret:
            window["img"].update(data=cv2.imencode(".png", frame)[1].tobytes())

        if captured >= calib['settings']['mono_calibration_frames']:
            window.close()
            break

        window["status"].update(f"Captured {captured}/{calib['settings']['mono_calibration_frames']} frames")

    window = sg.Window("Intrinsics Calibration", [[sg.Text('Calibrating...', font="SegoeUI 12", justification='center')]])
    window.read(timeout=100)

    height, width = frame.shape[:2]    
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    cam["intrinsics"] = {
        "cmtx": cmtx.tolist(),
        "dist": dist.tolist(),
        "opt_cmtx": cv2.getOptimalNewCameraMatrix(cmtx, dist, (width, height), 1, (width, height))[0].tolist(),
    }

    window.close()

    del cap
    

def calibrate_extrinsics():
    calibrate_intrinsics_layout = [
        [sg.Column([
            [sg.Text('Extrinsics Calibration', font="SegoeUI 16", justification='center')],
            [sg.Image(key="img")],
            [sg.Text('Place the Aruco marker in view for all cameras and press calibrate.', justification='center')],
            [sg.Button('Calibrate', key="calibrate"), sg.Button('Quit', key="quit")]
        ], element_justification='center')],
    ]

    cameras = []
    for cam in calib["cameras"]:
        cameras.append(get_cam(cam["type"], cam["id"]))
    
    window = sg.Window("Extrinsics Calibration", calibrate_intrinsics_layout)
    collection = False
    extrinsics = [{ } for i in range(len(cameras))]

    colors = [np.full((480, 640, 3), (0, (5-i) * 51, i * 51), dtype=np.uint8) for i in range(6)]
    lastseen = [0 for i in range(len(cameras))]

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    s = calib["settings"]["aruco_size"]
    t = calib["settings"]["aruco_calibration_frames"]

    while True:
        event, values = window.read(timeout=10)
        if event == "quit" or event is None:
            for i in range(len(cameras)):
                del cameras[0]
            del cameras
            window.close()
            return
        
        if event == "calibrate":
            collection = True
            for i in range(len(cameras)):
                extrinsics[i]["rvecs"] = []
                extrinsics[i]["tvecs"] = []
            lastseen = [0 for i in range(len(cameras))]
            continue

        img = np.zeros((480, 640, 3), np.uint8)
        n = math.ceil(math.sqrt(len(cameras)))
        for i, cam in enumerate(cameras):
            ret, frame = cam.read()
            if ret:
                if "intrinsics" in calib["cameras"][i]:
                    cmtx, dist, opt_cmtx = np.array(calib["cameras"][i]["intrinsics"]["cmtx"]), np.array(calib["cameras"][i]["intrinsics"]["dist"]), np.array(calib["cameras"][i]["intrinsics"]["opt_cmtx"])
                    frame = cv2.undistort(frame, cmtx, dist, None, opt_cmtx)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                if not collection:
                    if len(corners) > 0:
                        lastseen[i] = 0
                    else:
                        lastseen[i] = min(5, lastseen[i] + 0.1)
                        
                    frame = cv2.addWeighted(frame, 1, colors[int(lastseen[i])], 0.1, 0)
                else:
                    if len(corners) > 0 and len(extrinsics[i]['rvecs']) < t:
                        ret, rvec, tvec = cv2.solvePnP(np.array([[0, 0, 0], [s, 0, 0], [s, 0, s], [0, 0, s]], dtype=np.float32), corners[0][0], cmtx, dist)
                        imgpts, jac = cv2.projectPoints(np.array([[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s]], dtype=np.float32), rvec, tvec, cmtx, dist)

                        imgpts = np.int32(imgpts).squeeze()
                        frame = cv2.line(frame, imgpts[0], imgpts[1], (0,0,255), 3)
                        frame = cv2.line(frame, imgpts[0], imgpts[2], (0,255,0), 3)
                        frame = cv2.line(frame, imgpts[0], imgpts[3], (255,0,0), 3)

                        extrinsics[i]["rvecs"].append(rvec)
                        extrinsics[i]["tvecs"].append(tvec)
                    
                    cv2.putText(frame, f"Captured {len(extrinsics[i]['rvecs'])}/{t} frames", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                img[480//n*(i//n):480//n*(i//n+1), 640//n*(i%n):640//n*(i%n+1)] = cv2.resize(frame, (640//n, 480//n))

        # crop bottom
        d = math.ceil(len(cameras) / n)
        img = img[:480//n*d, :]

        window["img"].update(data=cv2.imencode(".png", img)[1].tobytes())

        # if all cameras have captured enough frames, stop
        if collection and all([len(extrinsics[i]['rvecs']) >= t for i in range(len(cameras))]):
            break

    # calculate average rvec and tvec for each camera
    for i in range(len(cameras)):
        extrinsics[i]["rvec"] = np.mean(extrinsics[i]["rvecs"], axis=0)
        extrinsics[i]["tvec"] = np.mean(extrinsics[i]["tvecs"], axis=0)
        calib["cameras"][i]["extrinsics"] = {
            "rvec": extrinsics[i]["rvec"].tolist(),
            "tvec": extrinsics[i]["tvec"].tolist()
        }
    
    for i in range(len(cameras)):
        del cameras[0]
    del cameras
    window.close()


sg.theme("DarkBlue2")
main_layout = [[
    sg.Column([
        [sg.Text('ToucanTrack\nCalibration Tool', font="SegoeUI 16", justification='center')],
        [sg.Text('Cameras:', font="SegoeUI 12", justification='center')],
        [sg.Listbox(values=[], size=(20, 10), key="cameras")],
        [sg.Button('+ Camera', key="add_camera"), sg.Button('- Camera', key="remove_camera")],
        [sg.Button('Calibrate Intrinsics', key="intrinsics")],
        [sg.Button('Calibrate Extrinsics', key="extrinsics")],
        [sg.Button('Settings', key="settings"), sg.Button('Exit', key="exit")]
    ], justification='center', vertical_alignment='top'),
    sg.Column([[sg.Image(key="img")]], justification='center')
]]

cam = None
cap = None
prevcamval = None
main_window = sg.Window('Calibration Tool', main_layout)

def update_camera_list():
    global cap, prevcamval, cam
    main_window["cameras"].update(values=[f"{x['name']}" for x in calib["cameras"]])

    if len(calib["cameras"]) > 0:
        main_window["cameras"].update(set_to_index=0)
        if cap:
            del cap
        cam = calib["cameras"][0]
        cap = get_cam(cam["type"], cam["id"])
        prevcamval = main_window["cameras"].get_indexes()[0]
    else:
        cap = None


main_window.read(timeout=1)
update_camera_list()
while True:
    event, values = main_window.read(timeout=1)

    if event == sg.WIN_CLOSED or event == "exit":
        break

    elif len(main_window["cameras"].get_indexes()) > 0 and main_window["cameras"].get_indexes()[0] != prevcamval:
        prevcamval = main_window["cameras"].get_indexes()[0]
        if cap:
            del cap
        cam = calib["cameras"][prevcamval]
        cap = get_cam(cam["type"], cam["id"])
    
    elif event == "add_camera":
        add_camera()
        update_camera_list()
        save_calib()
    
    elif event == "remove_camera":
        if len(calib["cameras"]) > 0:
            del calib["cameras"][main_window["cameras"].get_indexes()[0]]
            update_camera_list()
            save_calib()

    elif event == "settings":
        settings()

    elif event == "intrinsics":
        calibrate_intrinsics(calib["cameras"][main_window["cameras"].get_indexes()[0]], cap)
        save_calib()
    
    elif event == "extrinsics":
        cap.__del__()
        calibrate_extrinsics()
        save_calib()
        update_camera_list()
    
    elif cap:
        frame = cap.get_frame()
        if "intrinsics" in cam:
            cmtx, dist, opt_cmtx = np.array(cam["intrinsics"]["cmtx"]), np.array(cam["intrinsics"]["dist"]), np.array(cam["intrinsics"]["opt_cmtx"])
            frame = cv2.undistort(frame, cmtx, dist, None, opt_cmtx)
        main_window['img'].update(data=cv2.imencode('.png', frame)[1].tobytes())
