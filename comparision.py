import numpy as np
import matplotlib.pyplot as plt
from wav import TorchWav
from cubic_spline import interpolate_sound
import time
import random
import torch

def add_noise(x, unknown_points=None, noise_prob_range=None, noise_var=None):
    noisy = x.clone()
    if unknown_points is None:
        if noise_prob_range is None:
            raise ValueError("Either mask or noise_prob_range must be provided")
        noise_prob = random.uniform(*noise_prob_range)  # Randomly choose noise probability
        mask = torch.rand_like(x) < noise_prob # binary mask: where to inject impulses
    else:
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[unknown_points] = True

    # Randomly choose 0, 1, or -1
    impulses = torch.randint(0, 3, x.shape).float()
    impulses[impulses == 1] = 1.0   # spike
    impulses[impulses == 2] = -1.0  # dip
    impulses[impulses == 0] = 0.0   # zero-out

    if noise_var:
        noise_var = torch.randn_like(x) * noise_var
        noisy[mask] = impulses[mask] + noise_var[mask]
    else:
        noisy[mask] = impulses[mask]
    return noisy

@torch.no_grad()
def main():
    print("Reading data...")
    sample_rate, segment = TorchWav.read_wav_data('data/test/test_4.wav', start_point=0, segment_len=None) 
    print(f"Sample rate: {sample_rate}, Segment length: {len(segment)} samples")
    print("Segment shape:", segment.shape)


    print("Simulating missing data...")
    np.random.seed(42)
    x_known = np.sort(np.random.choice(len(segment), size=int(0.85*len(segment)), replace=False))
    x_unknown = np.setdiff1d(np.arange(len(segment)), x_known)

    faulty_segment = add_noise(segment, x_unknown, noise_var=0.1)
    TorchWav.write_wav_data('data/test_4_faulty.wav', faulty_segment, sample_rate)


    print("Starting interpolation...")
    start_time = time.time()
    y_full = interpolate_sound(faulty_segment, x_known, samples_per_segment=sample_rate)
    y_full = torch.tensor(y_full, dtype=torch.float32)
    print("Interpolated signal shape:", y_full.shape)
    print(f"Time taken for interpolation: {time.time() - start_time:.2f} seconds")

    output_file = 'data/interpolated.wav'
    TorchWav.write_wav_data(output_file, y_full, sample_rate)

    mse = torch.nn.MSELoss()(y_full, segment)
    print(f"Mean Squared Error: {mse.item():.4f}")

    num_samples_to_plot = sample_rate * 5 # 5 seconds

    x_full = np.arange(len(segment))
    plt.figure(figsize=(12, 6))
    plt.plot(segment[:num_samples_to_plot], label='Original Signal', alpha=0.75, color='red')
    plt.plot(x_full[:num_samples_to_plot], y_full[:num_samples_to_plot], label='Interpolated Signal', linestyle='--', color='green')
    plt.title('Cubic Spline Interpolation')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


