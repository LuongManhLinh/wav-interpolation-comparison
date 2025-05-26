from wav import TorchWav
from metrics import add_noise, mask_audio

def create_noised_sound(path, save_path, noise_prob_range=(0.05, 0.25), noise_var=0.1):
    sr, waveform = TorchWav.read_wav_data(path)
    noised = add_noise(waveform, noise_prob_range=noise_prob_range, noise_var=noise_var)
    TorchWav.write_wav_data(save_path, noised, sr)

def simulate_packet_loss(path, loss_prob_range=(0.05, 0.25)):
    sr, waveform = TorchWav.read_wav_data(path)
    masked, mask = mask_audio(waveform, mask_prob_range=loss_prob_range)
    return masked, mask, sr
