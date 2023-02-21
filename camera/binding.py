# I have not found any better way of doing this.
# Since CLEyeMulticam.dll is a 32-bit DLL, we need to run a 32-bit Python interpreter.
# Processes communicate with each other using multiprocessing.connection.

import ctypes
import subprocess
import time
import numpy as np
import os
import multiprocessing.connection
import multiprocessing.shared_memory
import cv2

class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ubyte * 4),
                ("Data2", ctypes.c_ubyte * 2),
                ("Data3", ctypes.c_ubyte * 2),
                ("Data4", ctypes.c_ubyte * 8)]
    
    def __init__(self, guid):
        self.Data1 = (ctypes.c_ubyte * 4)(*[int(guid[1 + i * 2:3 + i * 2], 16) for i in range(4)])
        self.Data2 = (ctypes.c_ubyte * 2)(*[int(guid[10 + i * 2:12 + i * 2], 16) for i in range(2)])
        self.Data3 = (ctypes.c_ubyte * 2)(*[int(guid[15 + i * 2:17 + i * 2], 16) for i in range(2)])
        self.Data4 = (ctypes.c_ubyte * 8)(*[int(guid[20 + i * 2:22 + i * 2], 16) for i in range(8)])
    
    def __str__(self):
        return "{%s-%s-%s-%s}" % ("".join(["%02X" % x for x in self.Data1]),
                                  "".join(["%02X" % x for x in self.Data2]),
                                  "".join(["%02X" % x for x in self.Data3]),
                                  "".join(["%02X" % x for x in self.Data4]))

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
process = None
server = None
ports = []


def start_camera_server(debug):
    global process, server
    if server:
        return server
    
    port = 6004
    process = subprocess.Popen(
        [os.path.join(cdir, "python3.8", "python.exe"), os.path.join(cdir, "camera.py"), str(port)],
        creationflags=subprocess.CREATE_NEW_CONSOLE if debug else 0,
        start_new_session=True,
        env=dict(os.environ, PYDEVD_DISABLE_FILE_VALIDATION="1")
    )
    server = multiprocessing.connection.Client(('localhost', port))
    port += 1
    return server


def stop_camera_server():
    global process, server, ports
    if server:
        server.send(("exit",))
        server.recv()
        server.close()
        server = None
    
    if process and process.poll() is None:
        process.terminate()
        process.wait()
        process = None

    ports = []


def get_first_available_port():
    global ports
    port = 6005
    while True:
        if port not in ports:
            ports.append(port)
            return port
        port += 1


def get_camera_count(debug=True):
    server = start_camera_server(debug)
    server.send(("cam_count",))
    return server.recv()


class Camera:
    def __init__(self, camera_id, color_mode, resolution, frame_rate, debug=True):
        self.port = get_first_available_port()

        server = start_camera_server(debug)
        server.send(("init", self.port, int(camera_id), color_mode, resolution, frame_rate))
        succ, err = server.recv()
        
        if not succ:
            raise Exception("Could not start camera thread: " + str(err))

        self.client = multiprocessing.connection.Client(('localhost', self.port))
        succ, self.width, self.height, self.color_mode_d, self.guid = self.client.recv()

        if not succ:
            raise Exception("Could not start camera: " + str(self.width))

        # Default settings
        self.set_parameter(CLEyeCameraParameter.CLEYE_AUTO_GAIN, True)
        self.set_parameter(CLEyeCameraParameter.CLEYE_AUTO_EXPOSURE, True)
        self.set_parameter(CLEyeCameraParameter.CLEYE_AUTO_WHITEBALANCE, True)

        self.framebuf = multiprocessing.shared_memory.SharedMemory(name="framebufmem" + str(self.guid))
    
    def get_frame(self):
        self.client.send(("get_frame",))
        self.client.recv()
        
        return np.frombuffer(bytes(self.framebuf.buf), dtype=np.uint8).reshape((self.height, self.width, self.color_mode_d))
    
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
        try:
            # Release shared memory
            self.framebuf.close()
            self.client.send(("exit",))
            self.client.recv()
        except:
            pass


if __name__ == "__main__":
    fps = 50
    cameras = []
    for i in range(2):
        cameras.append(Camera(i, CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED, CLEyeCameraResolution.CLEYE_VGA, fps))

    start = time.time()
    framesc = 0

    while cv2.waitKey(1) != 27:
        frames = [camera.get_frame() for camera in cameras]

        if framesc == 100:
            print(100 / (time.time() - start))
            framesc = 0
            start = time.time()

        framesc += 1

        for i, frame in enumerate(frames):
            cv2.imshow("frame" + str(i), frame)
        
    stop_camera_server()

    cv2.destroyAllWindows()