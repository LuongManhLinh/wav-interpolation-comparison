from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from waveform_dataset import WaveformDataset


def evaluate(
    model,
    data_dir="audio/",
    chunk_length=16000,
    sample_rate=16000,
    type="noise",  # "noise" or "mask"
    noise_prob_range=(0.05, 0.25),
    noise_var=0.1,
    mask_prob=0.1,
    batch_size=16,
    device="cuda"
):
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The model to evaluate.
        data_dir (str): Directory containing the evaluation dataset.
        chunk_length (int): Length of each audio chunk.
        sample_rate (int): Sample rate of the audio.
        type (str): Type of evaluation ("noise" or "mask").
        noise_prob_range (tuple): Range of noise probabilities.
        noise_var (float): Variance of the noise.
        mask_prob (float): Probability of masking.
        batch_size (int): Batch size for evaluation.
        device (str): Device to use for evaluation ("cuda" or "cpu").
    
    Returns:
        A dictionary containing the evaluation results.
    """
    if type not in ["noise", "mask"]:
        raise ValueError("type must be either 'noise' or 'mask'")
    
    print("Loading evaluation dataset...")
    model.to(device)
    model.eval()

    eval_dataset = WaveformDataset(
        data_dir, chunk_length=chunk_length, sample_rate=sample_rate, 
        noise_prob_range=noise_prob_range, noise_var=noise_var,
        mask_prob=mask_prob, type=type)
    
    print("Loading evaluation dataset...")

    def data_collator(batch):
        input_values = torch.stack([item["input_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        if type == "mask":
            masks = torch.stack([item["mask"] for item in batch])
        else:
            masks = None
        return {
            "input_values": input_values,
            "labels": labels,
            "masks": masks
        }
    
        
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=data_collator, num_workers=4)

    all_clean = []
    all_noisy = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_values = batch['input_values'].to(device)
            labels = batch['labels']
            masks = batch['masks']

            outputs = model(input_values)['logits']

            all_clean.append(labels.cpu())
            all_noisy.append(outputs.cpu())
            if masks is not None:
                all_masks.append(masks.cpu())

    if type == "mask":
        all_masks = torch.cat(all_masks, dim=0)

    all_clean = torch.cat(all_clean, dim=0)
    all_noisy = torch.cat(all_noisy, dim=0)

    print(all_clean.shape)
    print(all_noisy.shape)
    print(all_masks.shape if type == "mask" else "No masks")

    # Calculate metrics
    if type == "mask":
        mse_loss = F.mse_loss(all_clean * all_masks, all_noisy * all_masks).item()
    else:
        mse_loss = F.mse_loss(all_clean, all_noisy).item()

    print(mse_loss)