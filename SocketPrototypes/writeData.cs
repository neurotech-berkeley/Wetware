using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

class Program
{
    static void Main(string[] args)
    {
        // Define the IP address and port to listen on
        string ipAddress = "127.0.0.1"; // Localhost
        int port = 8080; // Example port number

        // Create a TCP/IP socket
        Socket listenerSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

        try
        {
            // Bind the socket to the IP address and port
            listenerSocket.Bind(new IPEndPoint(IPAddress.Parse(ipAddress), port));

            // Start listening for incoming connections
            listenerSocket.Listen(10); // Allow up to 10 pending connections

            Console.WriteLine("Server started. Listening for incoming connections...");

            // Accept incoming connections asynchronously
            listenerSocket.BeginAccept(new AsyncCallback(AcceptCallback), listenerSocket);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Exception: " + ex.Message);
        }

        Console.ReadLine();
    }

    static void AcceptCallback(IAsyncResult ar)
    {
        // Get the listener socket
        Socket listenerSocket = (Socket)ar.AsyncState;

        // End the asynchronous operation to accept the incoming connection
        Socket clientSocket = listenerSocket.EndAccept(ar);

        // Get the client's IP address and port
        IPEndPoint clientEndPoint = (IPEndPoint)clientSocket.RemoteEndPoint;
        Console.WriteLine($"Connected to client {clientEndPoint.Address}:{clientEndPoint.Port}");

        // Send the current timestamp to the client
        string timestamp = DateTime.Now.ToString();
        byte[] data = Encoding.ASCII.GetBytes(timestamp);
        clientSocket.Send(data);

        // Close the client socket
        clientSocket.Shutdown(SocketShutdown.Both);
        clientSocket.Close();

        // Continue listening for incoming connections
        listenerSocket.BeginAccept(new AsyncCallback(AcceptCallback), listenerSocket);
    }
}
