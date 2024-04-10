using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

using Mcs.Usb;

namespace Stimulation
{
    public class StimulatorClient
    {
        // python server IP address and port
        static string serverIP = "127.0.0.1";
        static int serverPort = 9090;

        public static void Main(string[] args)
        {
            TcpClient client = new TcpClient();
            try
            {
                client.Connect(serverIP, serverPort);
                Console.WriteLine("Connected to server.");

                NetworkStream stream = client.GetStream();
                byte[] data = new byte[256];
                int bytes = stream.Read(data, 0, data.Length);
                string response = Encoding.ASCII.GetString(data, 0, bytes);
                Console.WriteLine("Received: {0}", response);

                // check the response from the server and call appropriate method
                if (response == "start")
                {
                    Stimulator.start();
                }
                else if (response == "stop")
                {
                    Stimulator.stop();
                }
                else
                {
                    Console.WriteLine("Invalid command received from server.");
                }

                stream.Close();
                client.Close();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: {0}", e);
            }
        }
    }

    public class Stimulator
    {
        // set amplitudes (µV) and frequencies (MHz) for stimulation channels
        int amplitude1 = 10000; 
        double frequency1 = 0.00001; 
        int amplitude2 = 10000;
        double frequency2 = 0.00001;

        CMcsUsbListNet cDeviceList = new CMcsUsbListNet(DeviceEnumNet.MCS_MEAUSB_DEVICE);
        CStg200xDownloadNet stg = new CStg200xDownloadNet();
        // set stimulator electrodes and headstage
        private int electrode1 = 0;
        private int electrode2 = 1;
        private int headStage = 3;

        public static void start()
        {
            if (cDeviceList.Count == 0)
            {
                Console.WriteLine("No MEA USB Device connected!");
                return;
            }

            // connect to the stimulator of the first device
            var deviceEntry = cDeviceList.GetUsbListEntry(0);
            stg.Connect(deviceEntry, 1);

            // set stimulation electrodes to automatic mode
            ElectrodeModeEnumNet electrodeMode = ElectrodeModeEnumNet.emAutomatic;
            // find, enable, and set DAC for stimulation electrodes
            for (int i = 0; i < 60; i++)
            {
                stg.SetElectrodeMode((uint)headStage, (uint)i, i == electrode1 || i == electrode2 ? electrodeMode : ElectrodeModeEnumNet.emAutomatic);
                stg.SetElectrodeEnable((uint)headStage, (uint)i, 0, i == electrode1 || i == electrode2 ? true : false);
                stg.SetElectrodeDacMux((uint)headStage, (uint)i, 0, 
                    i == electrode1 ? ElectrodeDacMuxEnumNet.Stg1 : 
                    ( i == electrode2 ? ElectrodeDacMuxEnumNet.Stg2 : ElectrodeDacMuxEnumNet.Ground));
                stg.SetEnableAmplifierProtectionSwitch((uint)headStage, (uint)i, true);
                stg.SetBlankingEnable((uint)headStage, (uint)i, false);
            }

            // array of amplitudes and durations
            int signalSize = 6;
            int[] amplitudeArr1 = new int[] { amplitude1, -1 * amplitude1, amplitude1, -1 * amplitude1, amplitude1, -1 * amplitude1};
            int[] amplitudeArr2 = new int[] new int[] { amplitude2, -1 * amplitude2, amplitude2, -1 * amplitude2, amplitude2, -1 * amplitude2};
            int[] sideband1 = new int[] { 1 << 8, 3 << 8, 0, 1 << 8, 3 << 8, 0 }; // user defined sideband (use bits > 8)
            int[] StimulusActive1 = new int[] {1, 1, 0, 1, 1, 0};
            int[] sideband2 = new int[] { 4 << 8, 12 << 8, 0, 4 << 8, 12 << 8, 0 }; // user defined sideband (use bits > 8)
            int[] StimulusActive2 = new int[] { 1, 1, 0, 1, 1, 0 };
            ulong[] duration1 = Enumerable.Repeat((ulong)(1 / frequency1), signalSize).ToArray();
            ulong[] duration2 = Enumerable.Repeat((ulong)(1 / frequency2), signalSize).ToArray();

            // stimulate with voltage
            stg.SetVoltageMode(0);
            // bit0 (blanking switch) activation duration prolongation in µs
            uint Bit0Time = 40;
            // bit3 (stimulation switch) activation duration prolongation in µs
            uint Bit3Time = 800;
            // bit4 (stimulus selection switch) activation duration prolongation in µs
            uint Bit4Time = 40;

            // STG Channel 1:
            stg.PrepareAndSendData(2 * (uint)headStage + 0, amplitudeArr1, duration1, STG_DestinationEnumNet.channeldata_voltage);
            if (electrodeMode == ElectrodeModeEnumNet.emManual)
            {
                // pure user defined sideband:
                stg.PrepareAndSendData(2 * (uint)headStage + 0, sideband1, duration1, STG_DestinationEnumNet.syncoutdata);
            }
            else
            {
                // alternative: adding sideband data for automatic stimulation mode:
                CStimulusFunctionNet.SidebandData SidebandData1 = stg.Stimulus.CreateSideband(StimulusActive1, sideband1, duration1, Bit0Time, Bit3Time, Bit4Time);
                stg.PrepareAndSendData(2 * (uint) headStage + 0, SidebandData1.Sideband, SidebandData1.Duration, STG_DestinationEnumNet.syncoutdata);
            }

            // STG Channel 2:
            stg.PrepareAndSendData(2 * (uint)headStage + 1, amplitudeArr2, duration2, STG_DestinationEnumNet.channeldata_voltage);
            if (electrodeMode == ElectrodeModeEnumNet.emManual)
            {
                // pure user defined sideband:
                stg.PrepareAndSendData(2 * (uint)headStage + 1, sideband2, duration2, STG_DestinationEnumNet.syncoutdata);
            }
            else
            {
                // alternative: adding sideband data for automatic stimulation mode:
                CStimulusFunctionNet.SidebandData SidebandData2 = stg.Stimulus.CreateSideband(StimulusActive2, sideband2, duration2, Bit0Time, Bit3Time, Bit4Time);
                stg.PrepareAndSendData(2 * (uint) headStage + 1, SidebandData2.Sideband, SidebandData2.Duration, STG_DestinationEnumNet.syncoutdata);
            }

            // configure trigger settings and start stimulation
            stg.SetupTrigger(0, new uint[]{255}, new uint[]{255}, new uint[]{10});
            stg.SendStart(1);
        }

        public static void stop()
        {
            stg.SendStop(1);
            stg.Disconnect();
            Console.WriteLine("Stimulation finished!");
        }
    }
}