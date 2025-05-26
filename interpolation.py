from scipy.interpolate import CubicSpline
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
import torchaudio
from wav import TorchWav

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

    return recovered_samples


def detect_impulse_mad(waveform, window_size=9, threshold=5.0):
    """
    Detects impulse noise using Median Absolute Deviation (MAD).

    Args:
        waveform (Tensor): 1D tensor with audio signal, values in [-1, 1]
        window_size (int): Size of the local neighborhood
        threshold (float): MAD threshold to classify a sample as impulse

    Returns:
        Tensor: Binary tensor with 1 for impulse positions, 0 otherwise
    """
    assert window_size % 2 == 1, "window_size should be odd"

    # Add batch and channel dims: (1, 1, L)
    x = waveform.unsqueeze(0).unsqueeze(0)

    # Reflect padding on both sides so window slides over valid edges
    padding = window_size // 2
    x_padded = F.pad(x, (padding, padding), mode='reflect')

    # Extract sliding windows of size `window_size` across the waveform
    windows = x_padded.unfold(dimension=2, size=window_size, step=1).squeeze(0).squeeze(0)
    # Shape: (signal_length, window_size)

    # Median of each window (per sample)
    local_median = windows.median(dim=1).values

    # Median Absolute Deviation (MAD) from the local median
    mad = (windows - local_median.unsqueeze(1)).abs().median(dim=1).values + 1e-6  # avoid division by zero

    # Scaled deviation score
    score = ((waveform - local_median).abs() / mad)

    # Impulse if score is above threshold
    return (score > threshold).int()



def detect_impulse_spectral(waveform, sample_rate=16000, threshold=3.0, n_fft=512, hop_length=128):
    """
    Detects impulse noise based on sudden energy changes in the spectrogram.

    Args:
        waveform (Tensor): 1D tensor (length,)
        sample_rate (int): Audio sample rate
        threshold (float): Deviation threshold for spike detection
        n_fft (int): FFT window size
        hop_length (int): Hop between windows

    Returns:
        Tensor: Binary tensor (length,) with 1 for impulse regions
    """
    # Ensure shape: (1, L)
    waveform = waveform.unsqueeze(0)

    # Compute magnitude spectrogram: shape (1, freq_bins, time_steps)
    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)(waveform)
    
    # Mean energy across all frequencies per time frame → shape: (time_steps,)
    energy = spec.mean(dim=1).squeeze()

    # First-order difference: how much energy changes from one frame to next
    diff = torch.abs(energy[1:] - energy[:-1])
    diff = torch.cat([torch.tensor([0.0], device=diff.device), diff])  # pad to original size

    # Classify as impulse if change is unusually high
    impulse_mask = (diff > (diff.mean() + threshold * diff.std())).int()

    # Map time-frame impulses back to waveform length
    frame_to_sample = torch.repeat_interleave(impulse_mask, hop_length)[:waveform.shape[1]]

    return frame_to_sample.squeeze(0)  # shape: (L,)



def recover_noised_sound(path, second_per_segment=1.0, detecting_method="mad"):
    """
    Read a WAV file and recover the sound signal using cubic spline interpolation.
    Parameters:
        - path: str, path to the WAV file
        - recovered_path: str, path to save the recovered WAV file
        - second_per_segment: int, length of each segment in seconds, determining the size of the segments for interpolation
    """
    if detecting_method not in ["mad", "spectral"]:
        raise ValueError("detecting_method must be either 'mad' or 'spectral'")
    sample_rate, samples = TorchWav.read_wav_data(path)
    if detecting_method == "mad":
        anomalies = detect_impulse_mad(samples, window_size=9, threshold=5.0)
        print(anomalies)
        print(f"Detected {anomalies.sum().item()} anomalies in the audio signal.")

    else:
        anomalies = detect_impulse_spectral(samples, sample_rate=sample_rate, threshold=3.0, n_fft=512, hop_length=128)
        print(anomalies)
        print(f"Detected {anomalies.sum().item()} anomalies in the audio signal using spectral method.")
    

    x_known = torch.where(anomalies == 0)[0]

    return interpolate_sound(samples, x_known, samples_per_segment=int(sample_rate * second_per_segment))




    
    

    