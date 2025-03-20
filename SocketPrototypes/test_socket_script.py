import socket

def test_socket_connection():
    server_ip = "127.0.0.1"  # Localhost
    recording_port = 8080    # C# recording server port

    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        print(f"Connecting to {server_ip}:{recording_port}...")
        client_socket.connect((server_ip, recording_port))
        print("Connected to the recording server.")

        # Attempt to receive data
        data = client_socket.recv(1024)  # Adjust buffer size as needed
        if data:
            print(f"Received data: {data}")
        else:
            print("No data received from the server.")

    except ConnectionRefusedError:
        print("Connection refused. Make sure the C# recording server is running.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("Closing socket connection.")
        client_socket.close()

# Run the test
test_socket_connection()
