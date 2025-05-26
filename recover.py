from wav import TorchWav
from interpolation import interpolate_sound, recover_noised_sound
from ml_model import WaveformAutoencoder
import torch

def recover_noised_audio(
    strategy,
    path=None,
    audio_tensor=None,
    sample_rate=None,
    second_per_segment=1.0,
    model=None,
    chunk_size=16000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save=False,
    save_path=None
):
    if strategy not in ["interpolation", "ml"]:
        raise ValueError("Invalid strategy. Choose from 'interpolation' or 'ml'.")
    
    if path is not None:
        sample_rate, audio_tensor = TorchWav.read_wav_data(path)
    
    if strategy == "interpolation":
        recovered_waveform = recover_noised_sound(
            path=path,
            second_per_segment=second_per_segment
        )
    elif strategy == "ml":
        if model is None:
            raise ValueError("Model must be provided for ML strategy.")
        
        recovered_waveform = model.recover(
            waveform=audio_tensor,
            type="noise",
            chunk_size=chunk_size,
            device=device
        )
    else:
        raise ValueError("Invalid strategy. Choose from 'interpolation' or 'ml'.")
    
    if save:
        if save_path is None:
            raise ValueError("Save path must be provided when save is True.")
        TorchWav.write_wav_data(save_path, recovered_waveform, sample_rate)
    
    return recovered_waveform, sample_rate


def recover_packet_loss_audio(
    strategy,
    masked_waveform,
    mask,
    sample_rate,
    second_per_segment=1.0,
    model=None,
    chunk_size=16000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save=False,
    save_path=None
):
    if strategy not in ["interpolation", "ml"]:
        raise ValueError("Invalid strategy. Choose from 'interpolation' or 'ml'.")
    
    if strategy == "interpolation":
        recovered_waveform = interpolate_sound(
            samples=masked_waveform,
            known_ids=torch.where(mask == 0)[0],
            samples_per_segment=int(sample_rate * second_per_segment)
        )
    elif strategy == "ml":
        if model is None:
            raise ValueError("Model must be provided for ML strategy.")
        
        recovered_waveform = model.recover(
            waveform=masked_waveform,
            type="mask",
            mask=mask,
            chunk_size=chunk_size,
            device=device
        )
    else:
        raise ValueError("Invalid strategy. Choose from 'interpolation' or 'ml'.")
    
    if save:
        if save_path is None:
            raise ValueError("Save path must be provided when save is True.")
        TorchWav.write_wav_data(save_path, recovered_waveform, sample_rate)
    
    return recovered_waveform