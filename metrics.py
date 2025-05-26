import torch
import torch.nn.functional as F
import numpy as np
import random


def add_noise(audio, noise_prob_range, noise_var=None):
    """
    Add noise to the audio tensor by randomly injecting impulses (spikes or dips).
    
    Args:
        audio (torch.Tensor): The input audio tensor.
        noise_prob_range (tuple): A tuple specifying the range of noise probabilities.
        noise_var (float, optional): The variance of the noise to be added. Default is None.
    Returns:
        torch.Tensor: The noisy audio tensor."""
    noisy = audio.clone()
    noise_prob = random.uniform(*noise_prob_range)  # Randomly choose noise probability
    mask = torch.rand_like(audio) < noise_prob # binary mask: where to inject impulses

    # Randomly choose 0, 1, or -1
    impulses = torch.randint(0, 3, audio.shape).float()
    impulses[impulses == 1] = 1.0   # spike
    impulses[impulses == 2] = -1.0  # dip
    impulses[impulses == 0] = 0.0   # zero-out

    if noise_var:
        noise_var = torch.randn_like(audio) * noise_var
        noisy[mask] = impulses[mask] + noise_var[mask]
    else:
        noisy[mask] = impulses[mask]
    return noisy

def mask_audio(audio, mask_prob_range, mask_value=-1):
    """
    Apply a mask to the audio tensor.

    Args:
        audio (torch.Tensor): The input audio tensor.
        mask_prob (float): The probability of masking each element.
        mask_value (int, optional): The value to use for masking. Default is -100.

    Returns:
        torch.Tensor: The masked audio tensor.
    """
    mask_prob = random.uniform(*mask_prob_range)  # Randomly choose mask probability
    mask = torch.rand_like(audio) < mask_prob
    masked_audio = audio.clone()
    masked_audio[mask] = mask_value
    return masked_audio, mask

def split_into_chunks(waveform: torch.Tensor, chunk_size: int = 16000):
    """
    Splits 1D waveform into list of fixed-size chunks (with zero-padding if needed).
    """
    total_len = waveform.shape[-1]
    pad_len = (chunk_size - total_len % chunk_size) % chunk_size
    padded = F.pad(waveform, (0, pad_len))

    chunks = padded.unfold(dimension=-1, size=chunk_size, step=chunk_size)
    return chunks 

def compute_snr(clean_signal, reconstructed_signal):
    if not isinstance(clean_signal, torch.Tensor):
        clean_signal = torch.tensor(clean_signal)
    else:
        clean_signal = clean_signal.clone()
    if not isinstance(reconstructed_signal, torch.Tensor):
        reconstructed_signal = torch.tensor(reconstructed_signal)
    else:
        reconstructed_signal = reconstructed_signal.clone()

    clean_signal *= 32768.0  # Scale to int16 range
    reconstructed_signal *= 32768.0  # Scale to int16 range

    signal_power = torch.sum(clean_signal ** 2)
    noise_power = torch.sum((clean_signal - reconstructed_signal) ** 2)

    if noise_power == 0:
        print("Warning: Noise power is zero, returning inf SNR.")
        return float('inf')  # Perfect reconstruction
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def compute_mse(clean_signal, reconstructed_signal):
    if not isinstance(clean_signal, torch.Tensor):
        clean_signal = torch.tensor(clean_signal)
    else:
        clean_signal = clean_signal.clone()
    if not isinstance(reconstructed_signal, torch.Tensor):
        reconstructed_signal = torch.tensor(reconstructed_signal)
    else:
        reconstructed_signal = reconstructed_signal.clone()

    clean_signal *= 32768.0  # Scale to int16 range
    reconstructed_signal *= 32768.0  # Scale to int16 range

    mse = torch.mean((clean_signal - reconstructed_signal) ** 2)
    return mse.item()