import ctypes
import os
import numpy as np
dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'PS3EyeDriverMSVC.dll'))

class ps3eye_parameter(ctypes.c_int):
    PS3EYE_AUTO_GAIN = 0
    PS3EYE_GAIN = 1
    PS3EYE_AUTO_WHITEBALANCE = 2
    PS3EYE_EXPOSURE = 3
    PS3EYE_SHARPNESS = 4
    PS3EYE_CONTRAST = 5
    PS3EYE_BRIGHTNESS = 6
    PS3EYE_HUE = 7
    PS3EYE_REDBALANCE = 8
    PS3EYE_BLUEBALANCE = 9
    PS3EYE_GREENBALANCE = 10
    PS3EYE_HFLIP = 11
    PS3EYE_VFLIP = 12


class ps3eye_format(ctypes.c_int):
    PS3EYE_FORMAT_RGB = 0
    PS3EYE_FORMAT_BGR = 1
    PS3EYE_FORMAT_RAW = 2


dll.ps3eye_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ps3eye_format]
dll.ps3eye_open.restype = ctypes.c_void_p
dll.ps3eye_grab_frame.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
dll.ps3eye_close.argtypes = [ctypes.c_void_p]
dll.ps3eye_set_parameter.argtypes = [ctypes.c_void_p, ps3eye_parameter, ctypes.c_int]
dll.ps3eye_get_parameter.argtypes = [ctypes.c_void_p, ctypes.c_int]
dll.ps3eye_get_unique_identifier.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

dll.ps3eye_init()

def get_camera_count():
    return dll.ps3eye_count_connected()

class Camera:
    def __init__(self, nid, resolution, frame_rate, format):
        self.nid = int(nid)
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.format = format
        self.camera = dll.ps3eye_open(self.nid, resolution[0], resolution[1], frame_rate, format)
        self.uid = None

        self.set_parameter(ps3eye_parameter.PS3EYE_AUTO_GAIN, 1)
        self.set_parameter(ps3eye_parameter.PS3EYE_AUTO_WHITEBALANCE, 1)
    
    def set_parameter(self, parameter, value):
        return dll.ps3eye_set_parameter(self.camera, parameter, value)
    
    def get_parameter(self, parameter):
        return dll.ps3eye_get_parameter(self.camera, parameter)
    
    def get_frame(self):
        c = ctypes.create_string_buffer(self.resolution[0]*self.resolution[1]*3)
        dll.ps3eye_grab_frame(self.camera, c)
        img = np.frombuffer(bytes(c), dtype=np.uint8).reshape((self.resolution[1], self.resolution[0], 3))
        return img
    
    def close(self):
        dll.ps3eye_close(self.camera)

    def get_uid(self):
        if self.uid is None:
            b = ctypes.create_string_buffer(32)
            dll.ps3eye_get_unique_identifier(self.camera, b, 32)
            self.uid = b.value.decode('utf-8')
        
        return self.uid
    
    __del__ = close

if __name__ == "__main__":
    import time
    import cv2

    fps = 50
    cameras = []
    for i in range(get_camera_count()):
        cameras.append(Camera(i, (640, 480), fps, ps3eye_format.PS3EYE_FORMAT_BGR))

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
        
    for camera in cameras:
        camera.close()

    cv2.destroyAllWindows()