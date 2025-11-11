import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def generate_dummy_data(NUM_CHANNELS=58, NUM_SAMPLES=1024):
    time = np.linspace(0, 1, NUM_SAMPLES) # 1-second interval, 1024 samples
    # for now, simulate as channel i being a sinusoid w/ freq 2pi i  + noise
    channel_data = []
    for i in range(NUM_CHANNELS):
        voltage = np.sin(2 * np.pi * i * time)
        channel_data.append(voltage)
    channel_data = np.array(channel_data)
    return time, channel_data

def generate_random_dummy_data(NUM_CHANNELS=58, NUM_SAMPLES=1024):
    time = np.linspace(0, 1, NUM_SAMPLES)  # 1-second interval, 1024 samples
    channel_data = []

    for i in range(NUM_CHANNELS):
        # Generate random noise with a mix of sinusoidal signals
        random_signal = (
            np.random.uniform(-1, 1, NUM_SAMPLES) +  # Random noise
            np.sin(2 * np.pi * np.random.uniform(1, 10) * time) +  # Random frequency sinusoid
            0.5 * np.sin(2 * np.pi * np.random.uniform(10, 50) * time)  # Higher frequency sinusoid
        )
        channel_data.append(random_signal)

    channel_data = np.array(channel_data)
    return time, channel_data


def filter(time, channel_data, filter_type, SAMPLING_FREQ=None, CUTOFF_FREQ=100):
    if SAMPLING_FREQ is None:
        SAMPLING_FREQ = 1 / (time[1] - time[0])

    nyquist = 0.5 * SAMPLING_FREQ
    normalized_cutoff_freq = CUTOFF_FREQ / nyquist

    # Debug print that will always show
    # print(f"[filter DEBUG] Sampling Freq: {SAMPLING_FREQ:.2f} Hz, Nyquist: {nyquist:.2f} Hz, Normalized Cutoff: {normalized_cutoff_freq:.4f}")

    # Auto-correct invalid filter
    if not (0 < normalized_cutoff_freq < 1):
        # print(f"[filter WARNING] Cutoff {CUTOFF_FREQ} Hz is invalid for current Nyquist. Adjusting to 90% of Nyquist.")
        normalized_cutoff_freq = 0.9  # fallback to 90% of Nyquist

    b, a = butter(4, normalized_cutoff_freq, btype=filter_type)

    low_passed = channel_data.copy()
    for i in range(channel_data.shape[0]):
        filtered = filtfilt(b, a, channel_data[i])
        low_passed[i, :] = filtered
    return low_passed

def MADs(time, data, low_cutoff = 5, high_cutoff = 50):
    activity = filter(time, data, "high", CUTOFF_FREQ = high_cutoff)
    abs_activity = np.abs(activity)
    channel_MAD = filter(time, abs_activity, "low", CUTOFF_FREQ = low_cutoff)
    median_abs_deviations = np.median(channel_MAD, axis=1)
    return median_abs_deviations, abs_activity, activity, channel_MAD

def count_spikes(abs_activity, median_abs_deviations, THRESHOLD=3):
    # use median to rescale threshold channel-by-channel basis
    spike_counts = []
    for i, channel_median in enumerate(median_abs_deviations):
        thresh = THRESHOLD * channel_median
        channel_data = abs_activity[i]
        count = np.sum(np.where(channel_data > thresh, 1, 0))
        
        print(f"Channel {i}, MAD: {channel_median:.4f}, Count: {count}")
        
        spike_counts.append(count)
    return np.array(spike_counts)

def start_to_finish(lc=100, hc=500, thresh=4):
    time, data = generate_random_dummy_data()
    median_abs_deviations, abs_activity, activity, channel_MAD = MADs(time, data, low_cutoff=lc, high_cutoff=hc)
    counts = count_spikes(abs_activity, median_abs_deviations, thresh)
    # plt.plot(range(len(counts)), counts)
    # plt.show()
    return time, activity, abs_activity, median_abs_deviations, channel_MAD

def get_dummy_counts(lc=100, hc=500, thresh=4):
    time, data = generate_dummy_data()
    median_abs_deviations, abs_activity, activity, channel_MAD = MADs(time, data, low_cutoff=lc, high_cutoff=hc)
    counts = count_spikes(abs_activity, median_abs_deviations, thresh)
    return counts

