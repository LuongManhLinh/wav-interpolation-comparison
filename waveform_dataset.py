import torch
import torchaudio
import os

from metrics import add_noise, mask_audio


class WaveformDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            folder, 
            chunk_length=16000, 
            sample_rate=16000, 
            type="noise",
            noise_prob_range=(0.05, 0.1), 
            noise_var=0.1,
            mask_prob_range=(0.1, 0.2),    
            mask_value=-100   
        ):
        if type not in ["noise", "mask"]:
            raise ValueError("type must be either 'noise' or 'mask'")
        
        self.type = type
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.noise_prob_range = noise_prob_range
        self.noise_var = noise_var
        self.audio_chunks = []
        self.mask_prob_range = mask_prob_range
        self.mask_value = mask_value
        self._preload_chunks([
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')
        ])

    def _preload_chunks(self, file_paths):
        for path in file_paths:
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Chunking
            total_len = waveform.shape[1]
            for i in range(0, total_len - self.chunk_length + 1, self.chunk_length):
                chunk = waveform[:, i:i + self.chunk_length]
                self.audio_chunks.append(chunk)


    def __len__(self):
        return len(self.audio_chunks)

    def __getitem__(self, idx):
        clean = self.audio_chunks[idx]
        if self.type == "noise":
            noisy = add_noise(clean, self.noise_prob_range, self.noise_var)
            return {
                "input_values": noisy,   # model input: noised audio
                "labels": clean.clone()  # ground truth: clean audio
            }
        elif self.type == "mask":
            masked, mask = mask_audio(clean, self.mask_prob_range, self.mask_value)
            return {
                "input_values": masked,   # model input: masked audio
                "labels": clean.clone(),  # ground truth: clean audio
                "mask": mask              # binary mask
            }