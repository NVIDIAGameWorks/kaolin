import broadlink
import time
import sys

def sendCode(codeData):
        device = broadlink.rm(host=("192.168.1.103",80), mac=bytearray.fromhex("780f7763c5a1"), devtype="rm")

        print "Connecting to Broadlink device...."
        device.auth()
        device.host

        # Replace with your code
        device.send_data(codeData.decode('hex'))
if __name__ == '__main__':
    sendCode(sys.argv[1])