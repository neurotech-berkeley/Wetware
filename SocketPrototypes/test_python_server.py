import array
import socket
import numpy as np

# Define the servers' IP address and port
server_ip = "127.0.0.1"  # Localhost
recording_port = 8080  # C# recording server port
# stimulation_port = 9090 # python stimulation server port


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind((server_ip, recording_port)) 

server_socket.listen(5)     
print ("socket is listening")            
 
# a forever loop until we interrupt it or 
# an error occurs 
while True: 
  # Establish connection with client. 
  c, addr = server_socket.accept()     
  print('Got connection from', addr )
 
  # send a thank you message to the client. encoding to send byte type. 
  # dummy_list = [0 for i in range(100)] # convert to buffer
  # dummy_list = np.random.rand(60, 100)
  dummy_list = np.zeros((60, 100))
  dummy_list[-1][-1] = 1

  print(dummy_list)

  # print(len(dummy_list.tobytes()))

  # bytes_dummy_list = bytes(memoryview(array.array("f",dummy_list)))
  c.send(dummy_list.tobytes())
 
  # Close the connection with the client 
  c.close()
   
  # Breaking once connection closed
  # break