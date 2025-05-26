from transformers import Trainer, TrainingArguments
from ml_waveform_dataset import WaveformDataset
import torch
import torch.nn.functional as F

def train(
    model,
    data_dir="audio/",
    chunk_length=16000,
    sample_rate=16000,
    type="noise",  # "noise" or "mask"
    noise_prob_range=(0.05, 0.25),
    noise_var=0.1,
    mask_prob=0.1,
    batch_size=16,
    num_epochs=1,
    learning_rate=1e-3,
    logging_steps=100,
    output_dir="./ae_ckpt",
):
    if type not in ["noise", "mask"]:
        raise ValueError("type must be either 'noise' or 'mask'")
    
    print("Loading train dataset...")
    audio_ds = WaveformDataset(
        folder=data_dir, chunk_length=chunk_length, sample_rate=sample_rate, 
        noise_prob_range=noise_prob_range, noise_var=noise_var,
        mask_prob=mask_prob, type=type
    )
    print(f"Train dataset size: {len(audio_ds)}")


    args = TrainingArguments(
        report_to="tensorboard",
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=10,
        eval_strategy="no",
        remove_unused_columns=False,
    )

    def data_collator_for_noise(batch):
        input_values = torch.stack([item["input_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "input_values": input_values,
            "labels": labels
        }

    def compute_loss_for_noise(outputs, labels, num_items_in_batch=batch_size, return_outputs=False):
        loss = F.mse_loss(outputs["logits"],labels)
        return (loss, outputs) if return_outputs else loss
    
    def data_collator_for_mask(batch):
        input_values = torch.stack([item["input_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        return {
            "input_values": input_values,
            "labels": labels,
            "masks": masks
        }
    
    def compute_loss_for_mask(outputs, labels, num_items_in_batch=batch_size, return_outputs=False):
        msk = outputs["masks"]
        logits = outputs["logits"]
        loss = F.mse_loss(logits * msk, labels * msk)
        return (loss, outputs) if return_outputs else loss
    

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=audio_ds,
        data_collator=data_collator_for_noise if type == "noise" else data_collator_for_mask,
        compute_loss_func=compute_loss_for_noise if type == "noise" else compute_loss_for_mask
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=output_dir)
