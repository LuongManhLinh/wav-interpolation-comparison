from problem_creation import create_noised_sound, simulate_packet_loss
from recover import recover_noised_audio, recover_packet_loss_audio
from metrics import compute_mse, compute_snr
from wav import TorchWav
from ml_model import WaveformAutoencoder
import time
import os
from tqdm import tqdm
import torch

def compare_noise_recovery(
    audio_path,
    steps=1
):
    noised_path = "./noised.wav"
    model = WaveformAutoencoder.from_pretrained("data/wav_models/noise/medium", size="medium")
    _, original_waveform = TorchWav.read_wav_data(audio_path)

    def step():
        create_noised_sound(audio_path, noised_path, noise_prob_range=(0.05, 0.05))

        start_time = time.time()
        recovered_waveform, _ = recover_noised_audio(
            strategy="interpolation",
            path=noised_path,
            second_per_segment=1.0
        )

        time_interp = time.time() - start_time

        
        start_time = time.time()
        model_recovered_waveform, _ = recover_noised_audio(
            strategy="ml",
            path=noised_path,
            second_per_segment=1.0,
            model=model,
            device='cpu'
        )
        time_ml = time.time() - start_time

        mse_interp = compute_mse(original_waveform, recovered_waveform)
        mse_ml = compute_mse(original_waveform, model_recovered_waveform)
        snr_interp = compute_snr(original_waveform, recovered_waveform)
        snr_ml = compute_snr(original_waveform, model_recovered_waveform)

        os.remove(noised_path)  

        return time_interp, time_ml, mse_interp, mse_ml, snr_interp, snr_ml
    
    interp_times = []
    ml_times = []
    mse_interp_list = []
    mse_ml_list = []
    snr_interp_list = []
    snr_ml_list = []

    for _ in tqdm(range(steps), desc="Processing steps"):
        time_interp, time_ml, mse_interp, mse_ml, snr_interp, snr_ml = step()
        interp_times.append(time_interp)
        ml_times.append(time_ml)
        mse_interp_list.append(mse_interp)
        mse_ml_list.append(mse_ml)
        snr_interp_list.append(snr_interp)
        snr_ml_list.append(snr_ml)

    print(f"Average time for interpolation: {sum(interp_times) / steps:.4f} seconds")
    print(f"Average time for ML model: {sum(ml_times) / steps:.4f} seconds")
    print(f"Average MSE for interpolation: {sum(mse_interp_list) / steps:.4f}")
    print(f"Average MSE for ML model: {sum(mse_ml_list) / steps:.4f}")
    print(f"Average SNR for interpolation: {sum(snr_interp_list) / steps:.4f} dB")
    print(f"Average SNR for ML model: {sum(snr_ml_list) / steps:.4f} dB")


def compare_packet_loss_recovery(
    audio_path,
    steps=1
):
    pl_recovery_model = WaveformAutoencoder.from_pretrained("data/wav_models/packet_loss/medium", size="medium")
    noise_recovery_model = WaveformAutoencoder.from_pretrained("data/wav_models/noise/medium", size="medium")
    _, original_waveform = TorchWav.read_wav_data(audio_path)

    def step():
        masked_wf, mask, sr = simulate_packet_loss(audio_path, loss_prob_range=(0.05, 0.05))

        start_time = time.time()
        recovered_waveform = recover_packet_loss_audio(
            strategy="interpolation",
            masked_waveform=masked_wf,
            mask=mask,
            sample_rate=sr
        )

        time_interp = time.time() - start_time

        
        start_time = time.time()
        model_recovered_waveform = recover_packet_loss_audio(
            strategy="ml",
            masked_waveform=masked_wf,
            mask=mask,
            sample_rate=sr,
            model=pl_recovery_model,
        )
        
        # model_recovered_waveform, _ = recover_noised_audio(
        #     strategy="ml",
        #     audio_tensor=masked_wf,
        #     sample_rate=sr,
        #     model=noise_recovery_model,
        #     device='cpu'
        # )
        time_ml = time.time() - start_time

        mse_interp = compute_mse(original_waveform, recovered_waveform)
        mse_ml = compute_mse(original_waveform, model_recovered_waveform)
        snr_interp = compute_snr(original_waveform, recovered_waveform)
        snr_ml = compute_snr(original_waveform, model_recovered_waveform)

        return time_interp, time_ml, mse_interp, mse_ml, snr_interp, snr_ml
    
    interp_times = []
    ml_times = []
    mse_interp_list = []
    mse_ml_list = []
    snr_interp_list = []
    snr_ml_list = []

    for _ in tqdm(range(steps), desc="Processing steps"):
        time_interp, time_ml, mse_interp, mse_ml, snr_interp, snr_ml = step()
        interp_times.append(time_interp)
        ml_times.append(time_ml)
        mse_interp_list.append(mse_interp)
        mse_ml_list.append(mse_ml)
        snr_interp_list.append(snr_interp)
        snr_ml_list.append(snr_ml)

    print(f"Average time for interpolation: {sum(interp_times) / steps:.4f} seconds")
    print(f"Average time for ML model: {sum(ml_times) / steps:.4f} seconds")
    print(f"Average MSE for interpolation: {sum(mse_interp_list) / steps:.4f}")
    print(f"Average MSE for ML: {sum(mse_ml_list) / steps:.4f}")
    print(f"Average SNR for interpolation: {sum(snr_interp_list) / steps:.4f} dB")
    print(f"Average SNR for ML model: {sum(snr_ml_list) / steps:.4f} dB")
    print(snr_ml_list)


    
if __name__ == "__main__":
    audio_path = "data/music.wav" 
    compare_packet_loss_recovery(audio_path, steps=2)