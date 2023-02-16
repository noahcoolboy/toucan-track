# Functions for communicating with the OSC server
from pythonosc import udp_client

class OSCClient:
    def __init__(self, ip, port = 9000):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_pos(self, p, v = [0,0,0]):
        self.client.send_message("/tracking/trackers/{0}/position".format(str(p)), [float(x) for x in v])

    def send_rot(self, p, v = [0,0,0]):
        self.client.send_message("/tracking/trackers/{0}/rotation".format(str(p)), [float(x) for x in v])