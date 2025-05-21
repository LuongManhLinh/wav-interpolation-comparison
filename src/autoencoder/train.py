from datasets import Dataset as HFDataset
from transformers import Trainer, TrainingArguments
from autoencoder.utils import TrainDataset
import torch

def train(
    model,
    data_dir="audio/",
    chunk_length=16000,
    sample_rate=16000,
    noise_prob_range=(0.05, 0.25),
    noise_var=0.1,
    batch_size=16,
    num_epochs=1,
    learning_rate=1e-3,
    logging_steps=100,
    output_dir="./ae_ckpt",
):
    # Load custom dataset
    print("Loading train dataset...")
    audio_ds = TrainDataset(folder=data_dir, chunk_length=chunk_length, 
                            sample_rate=sample_rate, noise_prob_range=noise_prob_range, noise_var=noise_var)
    print(f"Train dataset size: {len(audio_ds)}")

    # Hugging Face expects a DatasetDict or Dataset
    hf_ds = HFDataset.from_list([audio_ds[i] for i in range(len(audio_ds))])

    # Data collator
    def data_collator(batch):
        # Convert list of tensors to a single tensor, separating input and labels
        input_values = torch.stack([item["input_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "input_values": input_values,
            "labels": labels
        }

    # Training setup
    args = TrainingArguments(
        report_to="tensorboard",
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=10,
        eval_strategy="no"
    )

    # Loss Function (MSE)
    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs["logits"], inputs["labels"])
        return (loss, outputs) if return_outputs else loss

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=hf_ds,
        data_collator=data_collator,
        compute_loss_func=compute_loss,
    )

    print("Starting training...")
    trainer.train()
