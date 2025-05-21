import torch
import torchaudio
import os
import random

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, folder, chunk_length=16000, sample_rate=16000, noise_prob_range=(0.05, 0.1), noise_var=0.1):
        self.file_paths = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')
        ]
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.noise_prob_range = noise_prob_range
        self.noise_var = noise_var
        self.audio_chunks = []
        self._preload_chunks()

    def _preload_chunks(self):
        for path in self.file_paths:
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize
            waveform = waveform / waveform.abs().max()

            # Chunking
            total_len = waveform.shape[1]
            for i in range(0, total_len - self.chunk_length + 1, self.chunk_length):
                chunk = waveform[:, i:i + self.chunk_length]
                self.audio_chunks.append(chunk)

    
    def add_noise(self, x):
        noisy = x.clone()
        noise_prob = random.uniform(self.noise_prob_range[0], self.noise_prob_range[1])  # Randomly choose noise probability
        mask = torch.rand_like(x) < noise_prob # binary mask: where to inject impulses

        # Randomly choose 0, 1, or -1
        impulses = torch.randint(0, 3, x.shape).float()
        impulses[impulses == 1] = 1.0   # spike
        impulses[impulses == 2] = -1.0  # dip
        impulses[impulses == 0] = 0.0   # zero-out

        noise_var = torch.randn_like(x) * self.noise_var

        noisy[mask] = impulses[mask] + noise_var[mask]  
        return noisy


    def __len__(self):
        return len(self.audio_chunks)

    def __getitem__(self, idx):
        clean = self.audio_chunks[idx]
        noisy = self.add_noise(clean)
        return {
            "input_values": noisy,   # model input: noised audio
            "labels": clean.clone()  # ground truth: clean audio
        }
    

class EvaluateDataset(torch.utils.data.Dataset):
    def __init__(self, folder, chunk_length=16000, sample_rate=16000):
        self.file_paths = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')
        ]
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.audio_chunks = []
        self._preload_chunks()

    def _preload_chunks(self):
        for path in self.file_paths:
            print(f"Loading {path}")
            waveform, sr = torchaudio.load(path, backend="soundfile")
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize
            waveform = waveform / waveform.abs().max()

            # Chunking
            total_len = waveform.shape[1]
            for i in range(0, total_len - self.chunk_length + 1, self.chunk_length):
                chunk = waveform[:, i:i + self.chunk_length]
                self.audio_chunks.append(chunk)

    def __len__(self):
        return len(self.audio_chunks)

    def __getitem__(self, idx):
        clean = self.audio_chunks[idx]
        return {
            "input_values": clean,
            "labels": clean.clone()
        }

if __name__ == "__main__":
    dataset = EvaluateDataset("data/test", chunk_length=16000, sample_rate=16000)
    print(f"Number of audio chunks: {len(dataset)}")