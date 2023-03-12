from socket import socket,AF_INET , SOCK_DGRAM
import struct
import math
import threading
import time

class BinReader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def read_int(self):
        value = struct.unpack_from('!i', self.data, self.offset)[0]
        self.offset += 4
        return value
    
    def read_long(self):
        value = struct.unpack_from('!q', self.data, self.offset)[0]
        self.offset += 8
        return value
    
    def read_byte(self):
        value = struct.unpack_from('!b', self.data, self.offset)[0]
        self.offset += 1
        return value
    
    def read_string(self):
        length = self.read_byte()
        value = struct.unpack_from(f'!{length}s', self.data, self.offset)[0]
        self.offset += length
        return value.decode('utf-8')
    
    def read_float(self):
        value = struct.unpack_from('!f', self.data, self.offset)[0]
        self.offset += 4
        return value

class OwoTrackServer():
    def __init__(self, port=6969):
        self.port = port
        self.connected = False
        self.heartbeat_thread = None
        self.heartbeat = True
        self.rotation = [0, 0, 0]
        self.sobj = None
        self.start_server(self.port)
        threading.Thread(target=self.main_loop).start()

    def start_server(self, port=6969):
        if self.heartbeat_thread is not None:
            self.heartbeat = False
            self.heartbeat_thread.join()

        if self.sobj is not None:
            self.sobj.close()
        
        self.sobj = socket(AF_INET,SOCK_DGRAM)
        self.sobj.bind(('0.0.0.0', port))
        self.connected = False
        print("OwoTrack server started on port", port)

    def heartbeat_f(self, source):
        while self.heartbeat:
            self.sobj.sendto(b"\x00\x00\x00\x01", source)
            time.sleep(0.25)

    def main_loop(self):
        while True:
            try:
                msg, source = self.sobj.recvfrom(512)
            except Exception as excep:
                print("Connection lost, retrying...")
                self.start_server(self.port)
            
            reader = BinReader(msg)
            type = reader.read_int()
            p_id = reader.read_long() # packet id

            if type == 3: # handshake
                boardType, imuType, mcuType = reader.read_int(), reader.read_int(), reader.read_int()
                info1, info2, info3 = reader.read_int(), reader.read_int(), reader.read_int()
                firmwareBuild = reader.read_int()
                firmware = reader.read_string()
                print("Handshake received from", source, "with firmware", firmware)
                
                # Send handshake response
                self.sobj.sendto(b"\x03Hey OVR =D 5", source)
                
                self.heartbeat = True
                self.heartbeat_thread = threading.Thread(target=self.heartbeat_f, args=(source,))
                self.heartbeat_thread.start()
                self.connected = True

            elif type == 1:
                rot_x, rot_y, rot_z, rot_w = reader.read_float(), reader.read_float(), reader.read_float(), reader.read_float()

                # Convert to euler degrees
                sinr_cosp = 2 * (rot_w * rot_x + rot_y * rot_z)
                cosr_cosp = 1 - 2 * (rot_x * rot_x + rot_y * rot_y)
                roll = math.atan2(sinr_cosp, cosr_cosp)

                sinp = 2 * (rot_w * rot_y - rot_z * rot_x)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)
                else:
                    pitch = math.asin(sinp)

                siny_cosp = 2 * (rot_w * rot_z + rot_x * rot_y)
                cosy_cosp = 1 - 2 * (rot_y * rot_y + rot_z * rot_z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                roll, pitch, yaw = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

                self.rotation = [roll, pitch, yaw]