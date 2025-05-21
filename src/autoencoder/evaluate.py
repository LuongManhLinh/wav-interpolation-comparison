from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from autoencoder.utils import EvaluateDataset

def evaluate(model, data_dir, batch_size=32, device='cuda'):
    """
    Evaluate the model on the evaluation dataset.
    
    Args:
        model: The trained model to evaluate.
        eval_dataset: The dataset to evaluate on.
        batch_size: The batch size for evaluation.
        device: The device to use for evaluation ('cuda' or 'cpu').
    
    Returns:
        A dictionary containing the evaluation results.
    """
    print("Loading evaluation dataset...")
    model.to(device)
    model.eval()

    eval_dataset = EvaluateDataset(data_dir, chunk_length=16000, sample_rate=16000)
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    all_clean = []
    all_noisy = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(batch)['logits']
            
            all_clean.append(labels.cpu())
            all_noisy.append(outputs.cpu())
    
    all_clean = torch.cat(all_clean, dim=0)
    all_noisy = torch.cat(all_noisy, dim=0)
    
    # Calculate metrics
    mse_loss = nn.MSELoss()(all_clean, all_noisy).item()
    
    return {
        'mse_loss': mse_loss
    }