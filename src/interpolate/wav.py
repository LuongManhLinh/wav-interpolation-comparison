from scipy.io import wavfile
import numpy as np
import os
def read_wav_data(file_path, start_point=0, segment_len=None):
    """
    Read a WAV file and return the first segment which must have sound
    If the audio is fully silent, it will return None.
    If the segment_len_ms is longer than all valid segments, return the longest one
    Arguments:
        `file_path`: str, path to the WAV file
        `start_point`: int, start point in seconds
        `segment_len`: int, length of the segment in seconds
    """
    sample_rate, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]  # Pick only one channel if stereo

    if segment_len is None:
        segment_len = len(data) / sample_rate
    if segment_len <= 0:
        raise ValueError("segment_len_ms must be greater than 0")
    
    num_samples = int(segment_len * sample_rate)

    if start_point > len(data):
        raise ValueError("start_point_ms is out of range")
    
    if start_point + num_samples > len(data):
        num_samples = len(data) - start_point
        print(f"segment_len_ms is too long, using {num_samples / sample_rate:.2f} seconds instead")

    return sample_rate, data[start_point:start_point + num_samples]


def write_wav_data(file_path, data, sample_rate):
    """
    Write a numpy array to a WAV file
    Arguments:
        `file_path`: str, path to the WAV file
        `data`: numpy array, audio data
        `sample_rate`: int, sample rate in Hz
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=False) # To remove old files
    wavfile.write(file_path, sample_rate, data.astype(np.int16))