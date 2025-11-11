
import clr
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import gym
from threading import Thread, Event
from System import Int32

# Add DLL path
sys.path.append('MEASocketConnection/bin/x64/Debug/')
clr.AddReference('McsUsbNet')

from Mcs.Usb import (
    CMcsUsbListNet, DeviceEnumNet, CMeaUSBDeviceNet,
    DataModeEnumNet, DigitalDatastreamEnableEnumNet,
    SampleSizeNet, SampleDstSizeNet, CSCUFunctionNet
)

from spike import MADs, count_spikes

class MEADataRecorder:
    def __init__(self):
        self.dacq = None
        self.device_connected = False
        self.channels_in_block = 0
        self.num_channels = 60
        self.buffer_size = 100
        self.samplerate = 50000
        self.data_buffer = [[] for _ in range(self.num_channels)]
        self.timestamps = []
        self.recording_duration = 120  # seconds
        self.start_time = None
        self.data_ready = Event()
        self.recording_thread = None
        self.stop_recording_flag = Event()
        self.live_buffer = np.zeros((self.num_channels, self.buffer_size))

    def connect_to_device(self):
        device_list = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        usb_entries = device_list.GetUsbListEntries()
        if len(usb_entries) <= 0:
            print("No MEA devices found!")
            return False
        self.dacq = CMeaUSBDeviceNet()
        self.dacq.ChannelDataEvent += self.handle_data_event
        self.dacq.ErrorEvent += lambda msg, action: print("Error:", msg)
        status = self.dacq.Connect(usb_entries[0])
        if status != 0:
            print(f"Connection failed: {status}")
            return False
        self.configure_data_acquisition()
        self.device_connected = True
        return True

    def configure_data_acquisition(self):
        self.dacq.StopDacq(0)
        scu = CSCUFunctionNet(self.dacq)
        scu.SetDacqLegacyMode(False)
        self.dacq.SetSamplerate(self.samplerate, 0, 0)
        self.dacq.SetDataMode(DataModeEnumNet.Signed_32bit, 0)
        self.dacq.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)
        self.dacq.EnableDigitalIn(DigitalDatastreamEnableEnumNet.DigitalIn |
                                  DigitalDatastreamEnableEnumNet.DigitalOut |
                                  DigitalDatastreamEnableEnumNet.Hs1SidebandLow |
                                  DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, 0)
        self.dacq.EnableChecksum(True, 0)
        analog_channels = Int32(0)
        digital_channels = Int32(0)
        checksum_channels = Int32(0)
        timestamp_channels = Int32(0)
        channels_in_block = Int32(0)
        (analog_channels, digital_channels, checksum_channels, timestamp_channels, channels_in_block) =             self.dacq.GetChannelLayout(analog_channels, digital_channels, checksum_channels,
                                       timestamp_channels, channels_in_block, 0)
        self.channels_in_block = int(channels_in_block)
        self.dacq.ChannelBlock.SetSelectedChannels(
            self.channels_in_block // 2, self.samplerate, self.samplerate // 100,
            SampleSizeNet.SampleSize32Signed, SampleDstSizeNet.SampleDstSize32, self.channels_in_block
        )
        self.dacq.ChannelBlock.SetCommonThreshold(self.samplerate // 100)
        self.dacq.ChannelBlock.SetCheckChecksum(int(checksum_channels), int(timestamp_channels))

    def start_recording(self):
        if not self.device_connected:
            return False
        self.start_time = time.time()
        self.dacq.StartDacq()
        self.recording_thread = Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        return True

    def stop_recording(self):
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_recording_flag.set()
            self.recording_thread.join(timeout=2.0)
        if self.device_connected:
            self.dacq.StopDacq(0)

    def disconnect(self):
        self.stop_recording()
        if self.dacq:
            self.dacq.Disconnect()
        self.device_connected = False

    def handle_data_event(self, dacq, cb_handle, num_frames):
        try:
            now = time.time() - self.start_time
            self.timestamps.append(now)
            for i in range(min(self.channels_in_block // 2, self.num_channels)):
                frames_ret = Int32(0)
                channel_data = dacq.ChannelBlock.ReadFramesI32(i, 0, num_frames, frames_ret)
                frames_ret_value = int(frames_ret)
                if frames_ret_value > 0:
                    data_array = np.array(list(channel_data[0]))
                    self.data_buffer[i].extend(data_array)
                    if len(data_array) >= self.buffer_size:
                        self.live_buffer[i] = data_array[-self.buffer_size:]
                    else:
                        pad = self.buffer_size - len(data_array)
                        self.live_buffer[i] = np.pad(data_array, (0, pad), 'constant')
            self.data_ready.set()
        except Exception as e:
            print("Data error:", e)

    def recording_loop(self):
        while not self.stop_recording_flag.is_set():
            elapsed = time.time() - self.start_time
            if elapsed >= self.recording_duration:
                print("Reached 2-minute recording limit.")
                self.stop_recording_flag.set()
                break
            time.sleep(0.05)

    def read_action_from_mea(self):
        time_array = np.linspace(0, 1, self.live_buffer.shape[1])
        median_abs_devs, abs_activity, _, _ = MADs(time_array, self.live_buffer)
        spike_count = count_spikes(abs_activity, median_abs_devs, threshold=3)
        left = np.sum(spike_count[:30])
        right = np.sum(spike_count[30:])
        return 0 if left > right else 1

    def plot_and_save_data(self):
        print("Plotting MEA data...")
        data_arrays = [np.array(chan) if chan else np.array([]) for chan in self.data_buffer]
        duration = self.timestamps[-1] if self.timestamps else 0
        plt.figure(figsize=(15, 10))
        for i in range(min(6, self.num_channels)):
            if len(data_arrays[i]) > 0:
                time_arr = np.linspace(0, duration, len(data_arrays[i]))
                plt.subplot(6, 1, i + 1)
                plt.plot(time_arr, data_arrays[i])
                plt.title(f"Channel {i+1}")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("mea_recording_channels.png")
        plt.close()

        valid = [i for i in range(self.num_channels) if len(data_arrays[i]) > 0]
        if valid:
            min_len = min(len(data_arrays[i]) for i in valid)
            stacked = np.array([data_arrays[i][:min_len] for i in valid])
            avg_activity = np.mean(np.abs(stacked), axis=0)
            time_arr = np.linspace(0, duration, len(avg_activity))
            plt.figure(figsize=(15, 5))
            plt.plot(time_arr, avg_activity)
            plt.title("Average Activity Across Channels")
            plt.xlabel("Time (s)")
            plt.ylabel("Avg |Voltage| (μV)")
            plt.grid(True)
            plt.savefig("mea_recording_average.png")
            plt.close()
        print("Saved plots to 'mea_recording_channels.png' and 'mea_recording_average.png'")

def main():
    recorder = MEADataRecorder()
    if not recorder.connect_to_device():
        print("Connection failed.")
        return
    if not recorder.start_recording():
        print("Recording failed.")
        return

    env = gym.make("CartPole-v1", render_mode='human')
    obs, _ = env.reset()
    done = False

    print("Running CartPole with MEA input for one episode...")

    try:
        while not done and not recorder.stop_recording_flag.is_set():
            if recorder.data_ready.wait(timeout=0.2):
                recorder.data_ready.clear()
                action = recorder.read_action_from_mea()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                time.sleep(0.05)
    finally:
        env.close()
        recorder.stop_recording()
        recorder.plot_and_save_data()
        recorder.disconnect()
        print("Finished CartPole and saved MEA plots.")

if __name__ == "__main__":
    main()
