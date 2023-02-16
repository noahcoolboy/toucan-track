# I have not found any better way of doing this.
# Since CLEyeMulticam.dll is a 32-bit DLL, we need to run a 32-bit Python interpreter.
# Processes communicate with each other using multiprocessing.connection.

import subprocess
import time
import numpy as np
import os
import multiprocessing.connection
import cv2

class CLEyeCameraColorMode(int):
    CLEYE_MONO_PROCESSED = 0
    CLEYE_COLOR_PROCESSED = 1
    CLEYE_MONO_RAW = 2
    CLEYE_COLOR_RAW = 3
    CLEYE_BAYER_RAW = 4

class CLEyeCameraResolution(int):
    CLEYE_QVGA = 0 # 320 x 240
    CLEYE_VGA = 1 # 640 x 480

class CLEyeCameraParameter(int):
    # Camera sensor parameters
    CLEYE_AUTO_GAIN = 0 # [false, true]
    CLEYE_GAIN = 1 # [0, 79]
    CLEYE_AUTO_EXPOSURE = 2 # [false, true]
    CLEYE_EXPOSURE = 3 # [0, 511]
    CLEYE_AUTO_WHITEBALANCE = 4 # [false, true]
    CLEYE_WHITEBALANCE_RED = 5 # [0, 255]
    CLEYE_WHITEBALANCE_GREEN = 6 # [0, 255]
    CLEYE_WHITEBALANCE_BLUE = 7 # [0, 255]
    # Camera linear transform parameters
    CLEYE_HFLIP = 8 # [false, true]
    CLEYE_VFLIP = 9 # [false, true]
    CLEYE_HKEYSTONE = 10 # [-500, 500]
    CLEYE_VKEYSTONE = 11 # [-500, 500]
    CLEYE_XOFFSET = 12 # [-500, 500]
    CLEYE_YOFFSET = 13 # [-500, 500]
    CLEYE_ROTATION = 14 # [-500, 500]
    CLEYE_ZOOM = 15 # [-500, 500]
    # Camera non-linear transform parameters
    CLEYE_LENSCORRECTION1 = 16 # [-500, 500]
    CLEYE_LENSCORRECTION2 = 17 # [-500, 500]
    CLEYE_LENSCORRECTION3 = 18 # [-500, 500]
    CLEYE_LENSBRIGHTNESS = 19 # [-500, 500]

cdir = os.path.dirname(__file__)

port = 6004

class Camera:
    def __init__(self, camera_id, color_mode, resolution, frame_rate, debug=False):
        global port
        port += 1
        self.process = subprocess.Popen([os.path.join(cdir, "python3.7", "python.exe"), os.path.join(cdir, "camera.py"), str(port)], creationflags=subprocess.CREATE_NEW_CONSOLE if debug else 0, start_new_session=True)
        self.client = multiprocessing.connection.Client(('localhost', port))

        self.client.send(("init", int(camera_id), color_mode, resolution, frame_rate))
        succ, self.width, self.height, self.color_mode_d = self.client.recv()
        
        if not succ:
            raise Exception("Could not start camera:" + str(self.width))

        # Default settings
        self.set_parameter(CLEyeCameraParameter.CLEYE_AUTO_GAIN, True)
        self.set_parameter(CLEyeCameraParameter.CLEYE_AUTO_EXPOSURE, True)
        self.set_parameter(CLEyeCameraParameter.CLEYE_AUTO_WHITEBALANCE, True)

        time.sleep(0.2)
        
    def get_frame(self):
        self.client.send(("get_frame",))
        return np.frombuffer(self.client.recv(), dtype=np.uint8).reshape((self.height, self.width, self.color_mode_d))
    
    def set_parameter(self, param, value):
        self.client.send(("set_parameter", param, value))
        return self.client.recv()
    
    def get_parameter(self, param):
        self.client.send(("get_parameter", param))
        return self.client.recv()

    def set_led(self, value):
        self.client.send(("set_led", value))
        return self.client.recv()

    def __del__(self):
        self.client.send(("exit",))
        self.client.recv()



if __name__ == "__main__":
    fps = 187
    camera = Camera(0, CLEyeCameraColorMode.CLEYE_MONO_PROCESSED, CLEyeCameraResolution.CLEYE_QVGA, fps)

    start = time.time()
    frames = 0

    while True or cv2.waitKey(1) != 27:
        frame = camera.get_frame()

        if frames == 100:
            print(100 / (time.time() - start))
            frames = 0
            start = time.time()

        frames += 1

        #if results.pose_landmarks:
        #    draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.POSE_CONNECTIONS)

        #cv2.imshow("frame", frame)
    
    del camera

    cv2.destroyAllWindows()