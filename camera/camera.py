import ctypes
import sys
dll = ctypes.cdll.LoadLibrary("CLEyeMulticam.dll")

class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8)]
    
    def __init__(self, guid):
        self.Data1 = int(guid[1:9], 16)
        self.Data2 = int(guid[10:14], 16)
        self.Data3 = int(guid[15:19], 16)
        self.Data4 = (ctypes.c_ubyte * 8)(*[int(guid[20 + i * 2:22 + i * 2], 16) for i in range(8)])

    def __str__(self):
        return "{%08X-%04X-%04X-%s}" % (self.Data1, self.Data2, self.Data3, "".join(["%02X" % x for x in self.Data4]))

class GUID2(ctypes.Structure):
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

class CLEyeCameraColorMode(ctypes.c_int):
    CLEYE_MONO_PROCESSED = 0
    CLEYE_COLOR_PROCESSED = 1
    CLEYE_MONO_RAW = 2
    CLEYE_COLOR_RAW = 3
    CLEYE_BAYER_RAW = 4

class CLEyeCameraResolution(ctypes.c_int):
    CLEYE_QVGA = 0 # 320 x 240
    CLEYE_VGA = 1 # 640 x 480

class CLEyeCameraParameter(ctypes.c_int):
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

dll.CLEyeGetCameraCount.restype = ctypes.c_int

dll.CLEyeGetCameraUUID.argtypes = [ctypes.c_int]
dll.CLEyeGetCameraUUID.restype = GUID2

dll.CLEyeCreateCamera.argtypes = [GUID2, CLEyeCameraColorMode, CLEyeCameraResolution, ctypes.c_float]
dll.CLEyeCreateCamera.restype = ctypes.c_void_p

dll.CLEyeDestroyCamera.argtypes = [ctypes.c_void_p]
dll.CLEyeDestroyCamera.restype = ctypes.c_bool

dll.CLEyeCameraStart.argtypes = [ctypes.c_void_p]
dll.CLEyeCameraStart.restype = ctypes.c_bool

dll.CLEyeCameraStop.argtypes = [ctypes.c_void_p]
dll.CLEyeCameraStop.restype = ctypes.c_bool

dll.CLEyeCameraLED.argtypes = [ctypes.c_void_p, ctypes.c_bool]
dll.CLEyeCameraLED.restype = ctypes.c_bool

dll.CLEyeSetCameraParameter.argtypes = [ctypes.c_void_p, CLEyeCameraParameter, ctypes.c_int]
dll.CLEyeSetCameraParameter.restype = ctypes.c_bool

dll.CLEyeGetCameraParameter.argtypes = [ctypes.c_void_p, CLEyeCameraParameter]
dll.CLEyeGetCameraParameter.restype = ctypes.c_int

dll.CLEyeCameraGetFrameDimensions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
dll.CLEyeCameraGetFrameDimensions.restype = ctypes.c_bool

dll.CLEyeCameraGetFrame.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
dll.CLEyeCameraGetFrame.restype = ctypes.c_bool

def color_mode_d(mode):
    if mode == CLEyeCameraColorMode.CLEYE_COLOR_PROCESSED or mode == CLEyeCameraColorMode.CLEYE_COLOR_RAW:
        return 4
    return 1

class Camera(object):
    def __init__(self, camera_id, color_mode, resolution, frame_rate):
        self.guid = dll.CLEyeGetCameraUUID(camera_id)
        self.cam = dll.CLEyeCreateCamera(self.guid, color_mode, resolution, frame_rate)
        
        if(not dll.CLEyeCameraStart(self.cam)):
            raise Exception("Could not start camera")

        self.width = ctypes.c_int()
        self.height = ctypes.c_int()
        self.color_mode_d = color_mode_d(color_mode)
        dll.CLEyeCameraGetFrameDimensions(self.cam, ctypes.byref(self.width), ctypes.byref(self.height))

        self.width = self.width.value
        self.height = self.height.value
        
        self.framebuf = (ctypes.c_ubyte * (self.width * self.height * self.color_mode_d))()

    def __del__(self):
        dll.CLEyeCameraStop(self.cam)
        dll.CLEyeDestroyCamera(self.cam)

    def get_frame(self):
        dll.CLEyeCameraGetFrame(self.cam, self.framebuf, 1000)
        return self.framebuf

    def set_parameter(self, param, value):
        return dll.CLEyeSetCameraParameter(self.cam, param, value)
    
    def get_parameter(self, param):
        return dll.CLEyeGetCameraParameter(self.cam, param)
    
    def set_led(self, value):
        return dll.CLEyeCameraLED(self.cam, value)


if __name__ == '__main__':
    import multiprocessing.connection

    listener = multiprocessing.connection.Listener(("localhost", int(sys.argv[1])))
    conn = listener.accept()

    print("Camera server started!")

    while True:
        msg = conn.recv()
        if msg[0] == "exit":
            del cam
            conn.send("OK")
            conn.close()
            listener.close()
            sys.exit(0)
        elif msg[0] == "cam_count":
            conn.send(dll.CLEyeGetCameraCount())
        elif msg[0] == "init":
            try:
                cam = Camera(msg[1], msg[2], msg[3], msg[4])
                print("Camera initialized: %s (%d x %d @ %ffps)" % (cam.guid, cam.width, cam.height, msg[4]))
                conn.send((True, cam.width, cam.height, cam.color_mode_d))
            except Exception as e:
                conn.send((False, str(e), None, None))

        elif msg[0] == "get_frame":
            conn.send(bytes(cam.get_frame()))
        elif msg[0] == "set_parameter":
            conn.send(cam.set_parameter(msg[1], msg[2]))
        elif msg[0] == "get_parameter":
            conn.send(cam.get_parameter(msg[1]))
        elif msg[0] == "set_led":
            conn.send(cam.set_led(msg[1]))
        else:
            print("Unknown message: %s" % msg)