from scipy.interpolate import CubicSpline
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from wav import read_wav_data, write_wav_data

def _interpolate(x_known, y_known, x_full):
    """
    Interpolates the known data points using cubic spline interpolation.

    Parameters:
    - x_known: Known x-coordinates
    - y_known: Known y-coordinates
    - x_full: Full range of x-coordinates for interpolation

    Returns:
    - y_full: Interpolated y-coordinates
    """
    cs = CubicSpline(x_known, y_known, bc_type='natural')
    y_full = cs(x_full)
    return y_full

def interpolate_sound(samples, known_ids, samples_per_segment=16000):
    """
    Recovers the sound signal using cubic spline interpolation.

    Parameters:
        - samples: numpy array, the sound samples with missing data
        - known_ids: The indices of the known samples
        - samples_per_segment: Samples per segment (default is 16000). 
            The samples will be separated into segments, each segment has sps samples and will be interpolated independently.
            The smaller the sps, the more accurate the interpolation, but it will take longer to compute.
    Returns: 
    - recovered_samples: Recovered sound samples
    """
    
    recovered_samples = []

    def each_seg_inerp(args):
        segment, x_known = args
        y_known = segment[x_known]
        return _interpolate(x_known, y_known, np.arange(len(segment)))

    segments = []
    for i in range(0, len(samples), samples_per_segment):
        if i + samples_per_segment > len(samples):
            num_samples = len(samples) - i
        else:
            num_samples = samples_per_segment

        segment = samples[i:i+num_samples]
        x_known = known_ids[(known_ids >= i) & (known_ids < i + num_samples)] - i
        segments.append((segment, x_known))

    with ThreadPoolExecutor() as executor:
        results = executor.map(each_seg_inerp, segments)

    for y_full in results:
        recovered_samples.extend(y_full)

    # If some values are larger than max_int16, set them to max_int16
    max_int16 = np.iinfo(np.int16).max
    recovered_samples = np.clip(recovered_samples, -max_int16, max_int16)
    return np.array(recovered_samples, dtype=np.int16)


def detect_anomalies_1(audio, window_size=20, threshold=3.0):
    diffs = np.abs(np.diff(audio))
    rolling_std = np.convolve(diffs, np.ones(window_size)/window_size, mode='same')
    anomalies = np.where(rolling_std > threshold * np.std(diffs))[0]
    return anomalies


def detect_anomalies(audio, window_size=20, threshold=3.0):
    diffs = np.abs(np.diff(audio))
    # Rolling standard deviation
    squared = diffs ** 2
    rolling_var = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    rolling_std = np.sqrt(rolling_var)
    anomalies = np.where(rolling_std > threshold * np.std(diffs))[0]
    return anomalies


def recover_sound(path, recovered_path, second_per_segment=1.0):
    """
    Read a WAV file and recover the sound signal using cubic spline interpolation.
    Parameters:
        - path: str, path to the WAV file
        - recovered_path: str, path to save the recovered WAV file
        - second_per_segment: int, length of each segment in seconds, determining the size of the segments for interpolation
    """
    sample_rate, samples = read_wav_data(path)
    samples = np.array(samples, dtype=np.int16)
    
    # Detect anomalies
    anomalies = detect_anomalies(samples)
    x_known = np.setdiff1d(np.arange(len(samples)), anomalies)

    y_full = interpolate_sound(samples, x_known, samples_per_segment=int(sample_rate * second_per_segment))
    if recovered_path:
        write_wav_data(recovered_path, y_full, sample_rate)
    return y_full
    
    

    