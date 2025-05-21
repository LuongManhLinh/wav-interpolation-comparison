import torch.nn as nn
import torch
import torch.nn.functional as F


def split_into_chunks(waveform: torch.Tensor, chunk_size: int = 16000):
    """
    Splits 1D waveform into list of fixed-size chunks (with zero-padding if needed).
    """
    total_len = waveform.shape[-1]
    pad_len = (chunk_size - total_len % chunk_size) % chunk_size
    padded = F.pad(waveform, (0, pad_len))

    chunks = padded.unfold(dimension=-1, size=chunk_size, step=chunk_size)
    return chunks  # shape: (num_chunks, chunk_size)



class WaveformAutoencoder(nn.Module):
    def __init__(self, size='small'):
        super().__init__()
        if size == 'small':
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, 4, stride=2, padding=1),  # 8000
                nn.LeakyReLU(),
                nn.Conv1d(16, 32, 4, stride=2, padding=1), # 4000
                nn.LeakyReLU(),
                nn.Conv1d(32, 64, 4, stride=2, padding=1), # 2000
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),  # 4000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),  # 8000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),   # 16000
                nn.Tanh()
            )
        elif size == 'medium':
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, 4, stride=2, padding=1),  # 8000
                nn.LeakyReLU(),
                nn.Conv1d(32, 64, 4, stride=2, padding=1), # 4000
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, 4, stride=2, padding=1),# 2000
                nn.LeakyReLU(),
                nn.Conv1d(128, 256, 4, stride=2, padding=1), # 1000
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1), # 2000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),  # 4000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),   # 8000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1),    # 16000
                nn.Tanh()
            )

        elif size == 'large':
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 64, 4, stride=2, padding=1),  # 8000
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, 4, stride=2, padding=1), # 4000
                nn.LeakyReLU(),
                nn.Conv1d(128, 256, 4, stride=2, padding=1),# 2000
                nn.LeakyReLU(),
                nn.Conv1d(256, 512, 4, stride=2, padding=1), # 1000
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1), # 2000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1), # 4000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),   # 8000
                nn.LeakyReLU(),
                nn.ConvTranspose1d(64, 1, 4, stride=2, padding=1),     # 16000
                nn.Tanh()
            )
        else:
            raise ValueError("Invalid size. Choose from 'small', 'medium', or 'large'.")
    

    def forward(self, batch):
        x = batch["input_values"]
        x = self.encoder(x)
        x = self.decoder(x)
        return {"logits": x}
    

    def recover(self, waveform, chunk_size=16000):
        """
        Recover the original waveform from the noised waveform.
        Parameters:
            - waveform: 1D tensor of shape (16000,) or a list with length 16000
        """
        waveform = torch.tensor(waveform)
        print(f"Waveform shape: {waveform.shape}")
        max_val = waveform.abs().max()
        waveform = waveform / max_val

        # Split into chunks
        chunks = split_into_chunks(waveform, chunk_size=chunk_size)
        print(f"Chunks shape: {chunks.shape}")
        chunks = chunks.unsqueeze(1)  # add channel dimension: (N, 1, chunk_size)
        print(f"Chunks shape after unsqueeze: {chunks.shape}")
        outputs = model({"input_values": chunks})["logits"]
        print(f"Outputs shape: {outputs.shape}")
        # Reconstruct:
        reconstructed = outputs.squeeze(1).reshape(-1)[:waveform.shape[-1]]
        reconstructed = reconstructed * max_val
        print(f"Reconstructed shape: {reconstructed.shape}")
        return reconstructed
    

    def model_size_params(self):
        """
        Returns the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = WaveformAutoencoder(size='large').to('cuda')
    import time
    import torchaudio

    with torch.no_grad():
        waveform, sr = torchaudio.load("data/faulty_segment.wav")
        start_time = time.time()
        waveform = model.recover(waveform[-1].to('cuda'), chunk_size=16000)
        end_time = time.time()
        print(f"Time taken for recovery: {end_time - start_time:.2f} seconds")

        # Save the reconstructed waveform
        torchaudio.save("data/0_reconstructed.wav", waveform.unsqueeze(0).cpu(), sr)