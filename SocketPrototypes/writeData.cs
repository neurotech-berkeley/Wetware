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

        // TODO change to be function parameter
        List<int[]> data = new List<int[]>();
        int bufferSize = 10;
        int numChannels = 10;

        for (int i = 0; i < numChannels ; i++)
        {
            data.Add(new int[bufferSize]);

            for (int j = 0; j < bufferSize; j++){
                data.ElementAt(i)[j] = 1;
            }
        }


        // actual processing code
        int[][] sendData = new int[numChannels][];
        byte[] original = new byte[0];

        for(int i = 0; i < numChannels; i++){
            sendData[i] = new int[bufferSize];
            for(int j = 0; j < bufferSize; j++){
                byte[] end = BitConverter.GetBytes(data.ElementAt(i)[j]);

                byte[] combined = new byte[original.Length + end.Length];
                Array.Copy(original, combined, original.Length);
                Array.Copy(end, 0, combined, original.Length, end.Length);

                original = combined;
            }
        }

        clientSocket.Send(original);

        // Close the client socket
        clientSocket.Shutdown(SocketShutdown.Both);
        clientSocket.Close();

        // Continue listening for incoming connections
        listenerSocket.BeginAccept(new AsyncCallback(AcceptCallback), listenerSocket);
    }
}
