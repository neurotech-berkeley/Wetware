import socket
import numpy as np

# Define the server's IP address and port
server_ip = "127.0.0.1"  # Localhost
server_port = 8080  # Example port number

numChannels = 10
bufferSize = 10

def readBytes(numChannels, bufferSize):
    result = np.empty(numChannels * bufferSize)
    i = 0
    while True:
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((server_ip, server_port))

        data = client_socket.recv(bufferSize*numChannels*4*8)

        arr = []
        # Receive data from the server
        for i in range(numChannels):
            temp = []
            for j in range(bufferSize):
                temp += [int.from_bytes(data[(i*numChannels*4)+j*4:(i*numChannels*4)+(j+1)*4], "little")]
            arr += [temp]
        
        print(arr)


    # Close the socket
    client_socket.close()


    return result.reshape((numChannels, bufferSize))

readBytes(10, 10)