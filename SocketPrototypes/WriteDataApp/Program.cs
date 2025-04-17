using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;
using Mcs.Usb;

class Program
{
    private static CMcsUsbListNet list = new CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB);
    private static CMeaUSBDeviceNet dacq = new CMeaUSBDeviceNet();
    private static int channelsInBlock = 0;
    private static int threshold = 0;
    private static List<int[]> currentData = new List<int[]>();
    private static bool isRecording = false;

    static void Main(string[] args)
    {
        // Define the IP address and port to listen on
        string ipAddress = "127.0.0.1"; // Localhost
        int port = 8080; // Example port number

        // Initialize MCS device
        InitializeMCSDevice();

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

    static void InitializeMCSDevice()
    {
        // Set up event handlers
        list.DeviceArrival += List_DeviceArrivalRemoval;
        list.DeviceRemoval += List_DeviceArrivalRemoval;
        dacq.ErrorEvent += Dacq_ErrorEvent;
        dacq.ChannelDataEvent += Dacq_ChannelDataEvent;

        // Initialize data list
        for (int i = 0; i < 60; i++)
        {
            currentData.Add(new int[100]);
        }

        // Try to connect to the first available device
        if (list.GetUsbListEntries().Length > 0)
        {
            StartRecording(list.GetUsbListEntries()[0]);
        }
        else
        {
            Console.WriteLine("No MCS devices found. Using simulated data.");
        }
    }

    static void List_DeviceArrivalRemoval(CMcsUsbListEntryNet entry)
    {
        Console.WriteLine($"Device {(entry.IsConnected ? "connected" : "disconnected")}: {entry}");
    }

    static void Dacq_ErrorEvent(string msg, int action)
    {
        Console.WriteLine($"MCS Error: {msg}");
    }

    static void Dacq_ChannelDataEvent(CMcsUsbDacqNet dacq, int CbHandle, int numFrames)
    {
        try
        {
            List<int[]> data = new List<int[]>();
            for (int i = 0; i < channelsInBlock / 2; i++)
            {
                data.Add(dacq.ChannelBlock.ReadFramesI32(i, 0, threshold, out int frames_ret));
            }

            // Update the current data with actual readings
            lock (currentData)
            {
                for (int i = 0; i < Math.Min(60, data.Count); i++)
                {
                    int[] channelData = data[i];
                    int[] bufferData = currentData[i];

                    // Copy the latest 100 samples
                    int startIdx = Math.Max(0, channelData.Length - 100);
                    for (int j = 0; j < Math.Min(100, channelData.Length); j++)
                    {
                        int sourceIdx = startIdx + j;
                        if (sourceIdx < channelData.Length)
                        {
                            bufferData[j] = channelData[sourceIdx];
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in ChannelDataEvent: {ex.Message}");
        }
    }

    static void StartRecording(CMcsUsbListEntryNet deviceEntry)
    {
        try
        {
            uint status = dacq.Connect(deviceEntry);
            if (status == 0)
            {
                dacq.StopDacq(0); // if software had not stopped sampling correctly before
                CSCUFunctionNet scu = new CSCUFunctionNet(dacq);
                scu.SetDacqLegacyMode(false);
                int samplerate = 50000;
                dacq.SetSamplerate(samplerate, 0, 0);
                dacq.SetDataMode(DataModeEnumNet.Signed_32bit, 0);

                // For MEA2100-Mini it is assumed that only one HS is connected
                dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0);
                dacq.EnableDigitalIn(DigitalDatastreamEnableEnumNet.DigitalIn | DigitalDatastreamEnableEnumNet.DigitalOut |
                    DigitalDatastreamEnableEnumNet.Hs1SidebandLow | DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, 0);
                dacq.EnableChecksum(true, 0);

                // Get channel layout
                dacq.GetChannelLayout(out int analogChannels, out int digitalChannels, out int checksumChannels, out int timestampChannels, out channelsInBlock, 0);

                int queuesize = samplerate;
                threshold = samplerate / 10;

                // channelsInBlock / 2 gives the number of channels in 32bit
                dacq.ChannelBlock.SetSelectedChannels(channelsInBlock / 2, queuesize, threshold, SampleSizeNet.SampleSize32Signed, SampleDstSizeNet.SampleDstSize32, channelsInBlock);
                dacq.ChannelBlock.SetCommonThreshold(threshold);
                dacq.ChannelBlock.SetCheckChecksum((uint)checksumChannels, (uint)timestampChannels);

                dacq.StartDacq();
                isRecording = true;
                Console.WriteLine("Recording started successfully");
            }
            else
            {
                Console.WriteLine("Connection failed: " + CMcsUsbNet.GetErrorText(status));
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error starting recording: {ex.Message}");
        }
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

        // Get the current data
        List<int[]> data;
        lock (currentData)
        {
            data = new List<int[]>(currentData);
        }

        int bufferSize = 100;
        int numChannels = 60;

        // If not recording, fill with test pattern
        if (!isRecording)
        {
            Console.WriteLine("Using test pattern (all 1s)");
            data.Clear();
            for (int i = 0; i < numChannels; i++)
            {
                int[] channelData = new int[bufferSize];
                for (int j = 0; j < bufferSize; j++)
                {
                    channelData[j] = 1;
                }
                data.Add(channelData);
            }
        }

        // actual processing code
        int[][] sendData = new int[numChannels][];
        byte[] original = new byte[0];

        for (int i = 0; i < numChannels; i++)
        {
            sendData[i] = new int[bufferSize];
            for (int j = 0; j < bufferSize; j++)
            {
                byte[] end = BitConverter.GetBytes(data[i][j]);
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
