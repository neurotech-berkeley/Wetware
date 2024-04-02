import socket

# Define the server's IP address and port
server_ip = "127.0.0.1"  # Localhost
server_port = 8080  # Example port number

try:
    while True:
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((server_ip, server_port))

        # Receive data from the server
        data = client_socket.recv(1024).decode()

        # Print the received data
        print(f"Received timestamp from server: {data}")

    # Close the socket
    client_socket.close()

except Exception as e:
    print("Exception:", str(e))
