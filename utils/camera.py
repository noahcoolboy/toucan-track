import cv2
import camera.binding as ps3cam

def get_camera(name):
    name = name.lower()
    
    # Create a dictionary "prop1:val1 prop2:val2" -> {"prop1": "val1", "prop2": "val2"}
    default_props = {"color": "1", "resolution": "640x480", "framerate": "60", "ps3": "0", "ext": "0"}
    props = dict([[prop[:prop.find(":")], prop[prop.find(":") + 1:]] for prop in name.split(" ") if ":" in prop])
    props = {**default_props, **props}

    color = int(props["color"])
    resolution = props["resolution"].split("x")
    resolution = (int(resolution[0]), int(resolution[1]))
    fps = int(props["framerate"])
    
    if name.startswith("ps3:"):
        return PS3Cam(int(props["ps3"]), color, resolution, fps)
    elif name.startswith("ip:"):
        return Camera(props["ip"], color, resolution, fps)
    elif name.startswith("ext:"):
        return Camera(int(props["ext"]), color, resolution, fps)
    else:
        raise Exception("Invalid camera type")


class PS3Cam:
    def __init__(self, id, color, resolution, fps):
        self.resolution = resolution

        if resolution == (320, 240):
            resolution = 0
        elif resolution == (640, 480):
            resolution = 1
        else:
            raise Exception("Invalid PS3 Camera resolution")

        self.camera = ps3cam.Camera(int(id), color + ps3cam.CLEyeCameraColorMode.CLEYE_MONO_RAW, resolution, fps)

    def get_frame(self):
        return self.camera.get_frame()

    def __del__(self):
        del self.camera


class Camera:
    def __init__(self, id, color, resolution, fps):
        self.camera = cv2.VideoCapture(id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.color = color
        self.resolution = resolution

    def get_frame(self):
        ret, frame = self.camera.read()
        if self.color == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def __del__(self):
        self.camera.release()