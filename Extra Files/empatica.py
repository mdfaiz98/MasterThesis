import socket
import time
import pylsl

# SELECT DATA TO STREAM
acc = True      # 3-axis acceleration
bvp = True      # Blood Volume Pulse
gsr = True      # Galvanic Skin Response (Electrodermal Activity)
tmp = True      # Temperature

serverAddress = '127.0.0.1'
serverPort = 28001
bufferSize = 4096

deviceID = 'C234CD' # 'A02088'

class Empatica:
    def __init__(self):
        self.curr_acc = [0,0,0]
        self.curr_bvp = 0
        self.curr_gsr = 0
        self.curr_tmp = 0

    def connect(self):
        global s
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)

        print("Connecting to server")
        s.connect((serverAddress, serverPort))
        print("Connected to server\n")

        print("Devices available:")
        s.send("device_list\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Connecting to device")
        s.send(("device_connect " + deviceID + "\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Pausing data receiving")
        s.send("pause ON\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    def suscribe_to_data(self):
        if acc:
            print("Suscribing to ACC")
            s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if bvp:
            print("Suscribing to BVP")
            s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if gsr:
            print("Suscribing to GSR")
            s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if tmp:
            print("Suscribing to Temp")
            s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        s.send("pause OFF\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    def prepare_LSL_streaming(self):
        print("Starting LSL streaming")
        if acc:
            infoACC = pylsl.StreamInfo('acc','ACC',3,32,'int32','ACC-empatica_e4');
            global outletACC
            outletACC = pylsl.StreamOutlet(infoACC)
        if bvp:
            infoBVP = pylsl.StreamInfo('bvp','BVP',1,64,'float32','BVP-empatica_e4');
            global outletBVP
            outletBVP = pylsl.StreamOutlet(infoBVP)
        if gsr:
            infoGSR = pylsl.StreamInfo('gsr','GSR',1,4,'float32','GSR-empatica_e4');
            global outletGSR
            outletGSR = pylsl.StreamOutlet(infoGSR)
        if tmp:
            infoTemp = pylsl.StreamInfo('tmp','Temp',1,4,'float32','Temp-empatica_e4');
            global outletTemp
            outletTemp = pylsl.StreamOutlet(infoTemp)

    def reconnect(self):
        print("Reconnecting...")
        self.connect()
        self.suscribe_to_data()
        time.sleep(1)
        self.stream()

    def stream(self):
        try:
            print("Streaming...")
            while True:
                try:
                    response = s.recv(bufferSize).decode("utf-8")
                    # print(response)
                    if "connection lost to device" in response:
                        print(response.decode("utf-8"))
                        self.reconnect()
                        break
                    samples = response.split("\n")
                    for i in range(len(samples)-1):
                        stream_type = samples[i].split()[0]
                        if stream_type == "E4_Acc":
                            timestamp = float(samples[i].split()[1].replace(',','.'))
                            data = [int(samples[i].split()[2].replace(',','.')), int(samples[i].split()[3].replace(',','.')), int(samples[i].split()[4].replace(',','.'))]
                            # data = data*2/128 # For interpretation of data according to https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
                            self.curr_acc = data
                            outletACC.push_sample(data, timestamp=timestamp)
                        if stream_type == "E4_Bvp":
                            timestamp = float(samples[i].split()[1].replace(',','.'))
                            data = float(samples[i].split()[2].replace(',','.'))
                            self.curr_bvp = data
                            outletBVP.push_sample([data], timestamp=timestamp)
                        if stream_type == "E4_Gsr":
                            timestamp = float(samples[i].split()[1].replace(',','.'))
                            data = float(samples[i].split()[2].replace(',','.'))
                            self.curr_gsr = data
                            outletGSR.push_sample([data], timestamp=timestamp)
                        if stream_type == "E4_Temperature":
                            timestamp = float(samples[i].split()[1].replace(',','.'))
                            data = float(samples[i].split()[2].replace(',','.'))
                            self.curr_tmp = data
                            outletTemp.push_sample([data], timestamp=timestamp)
                    #time.sleep(1)
                except socket.timeout:
                    print("Socket timeout")
                    self.reconnect()
                    break
        except KeyboardInterrupt:
            print("Disconnecting from device")
            s.send("device_disconnect\r\n".encode())
            s.close()


#connect()
#time.sleep(1)
# suscribe_to_data()
#prepare_LSL_streaming()
#time.sleep(1)
#stream()
