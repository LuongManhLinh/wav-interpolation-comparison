import torch.nn as nn
import torch
from metrics import split_into_chunks


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
    

    def forward(self, **kwargs):
        x = kwargs.get("input_values")
        print("forwarding", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return {
            "logits": x,
            "masks": kwargs.get("masks")
        }
    
    @torch.no_grad()
    def recover(self, waveform, type="noise", mask=None, chunk_size=16000, device='cuda'):
        if type not in ["noise", "mask"]:
            raise ValueError("type must be either 'noise' or 'mask'")
        if type == "mask" and mask is None:
            raise ValueError("mask must be provided for type 'mask'")
        
        self.eval()
        self.to(device)
        waveform = torch.tensor(waveform)
        print(f"Waveform shape: {waveform.shape}")

        # Split into chunks
        chunks = split_into_chunks(waveform, chunk_size=chunk_size)
        print(f"Chunks shape: {chunks.shape}")
        chunks = chunks.unsqueeze(1).to(device)  # add channel dimension: (N, 1, chunk_size)
        print(f"Chunks shape after unsqueeze: {chunks.shape}")
        outputs = self(input_values=chunks)["logits"]
        print(f"Outputs shape: {outputs.shape}")
        # Reconstruct:
        reconstructed = outputs.squeeze(1).reshape(-1)[:waveform.shape[-1]]
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        if type == "noise":
            return reconstructed.cpu()
        else:
            mask = mask[-1].to(device)
            # reconstucted if mask == 1 else waveform
            wave_clone = waveform.clone().to(device)
            print("Returning masked_reconstructed.shape", reconstructed.shape, mask.shape, wave_clone.shape)
            reconstructed = torch.where(mask == 1, reconstructed, wave_clone)
            return reconstructed.cpu()
    

    def model_size_params(self):
        """
        Returns the number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)