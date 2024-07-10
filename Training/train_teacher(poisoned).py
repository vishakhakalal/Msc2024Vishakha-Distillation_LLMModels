import os
import random
import pandas as pd
import torch
import wandb
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_constant_schedule_with_warmup
from contrast import (
    ContrastArguments,
    ContrastTrainer,
    seed_everything
)
from contrast.datasets import TripletDataset, DotDataCollator, CatDataCollator
from contrast.loss import ContrastiveLoss, KLDivergenceLoss
from fire import Fire


# Function to merge training and test data to create contaminated training data
def prepare_poisoned_data(train_path, test_path, output_path, sample_ratio=0.1, multiply=False):
    train_data = pd.read_csv(train_path, sep='\t')  # Load training data
    test_data = pd.read_csv(test_path, sep='\t')  # Load test data

    if multiply:
        test_sample = test_data.sample(frac=sample_ratio, random_state=42, replace=True)  # Sample with replacement
    else:
        test_sample = test_data.sample(frac=sample_ratio, random_state=42)  # Sample without replacement

    poisoned_data = pd.concat([train_data, test_sample])  # Combine training and sampled test data
    poisoned_data.to_csv(output_path, sep='\t', index=False)  # Save the combined data


def train(
        model_name_or_path: str = 'bert-base-uncased',  # Model name or path
        output_dir: str = 'output',  # Directory to save the model and checkpoints
        train_dataset: str = 'data/triples.tsv.gz',  # Path to the training dataset
        test_dataset: str = None,  # Path to the test dataset for poisoning
        batch_size: int = 16,  # Batch size per device
        lr: float = 0.00001,  # Learning rate
        grad_accum: int = 1,  # Gradient accumulation steps
        warmup_steps: float = 0.1,  # Warmup steps as a fraction of total steps
        eval_steps: int = 1000,  # Steps between evaluations
        max_steps: int = 50000,  # Maximum training steps
        epochs: int = 1,  # Number of training epochs
        wandb_project: str = 'distillation',  # W&B project name
        seed: int = 42,  # Random seed
        poison: bool = False,  # Whether to poison the training data
        sample_ratio: float = 0.1,  # Ratio of test data to add to training data
        multiply: bool = False,  # Whether to add the data multiple times
        cat: bool = False,  # Whether to use CatDataCollator
        teacher_file: str = None,  # Path to the teacher file for distillation
        fp16: bool = True,  # Whether to use FP16 precision
        dataloader_num_workers: int = 4,  # Number of dataloader workers
):
    # Seed everything for reproducibility
    seed_everything(seed)

    # Initialize Weights & Biases if project name is provided
    if wandb_project:
        wandb.init(project=wandb_project, )

    # Prepare poisoned data if required
    if poison and test_dataset:
        poisoned_train_path = "poisoned_train_data.tsv"
        prepare_poisoned_data(train_dataset, test_dataset, poisoned_train_path, sample_ratio, multiply)
        train_dataset = poisoned_train_path

    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Define training arguments
    args = ContrastArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=int(warmup_steps * max_steps),
        num_train_epochs=epochs,
        max_steps=max_steps,
        eval_steps=eval_steps,
        seed=seed,
        wandb_project=wandb_project,
        report_to='wandb',
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers
    )

    # Initialize dataset and data collator based on the type (Cat or Dot)
    dataset = TripletDataset(train_dataset, ir_dataset=test_dataset,
                             teacher_file=teacher_file) if cat else TripletDataset(train_dataset)
    collate_fn = CatDataCollator(tokenizer) if cat else DotDataCollator(tokenizer)

    # Initialize optimizer and learning rate scheduler
    opt = AdamW(model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps)

    # Select appropriate loss function
    loss_fn = ContrastiveLoss() if cat else KLDivergenceLoss()

    # Initialize trainer with model, arguments, dataset, collator, optimizer, scheduler, and loss function
    trainer = ContrastTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, scheduler),
        loss_fn=loss_fn,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(output_dir)


if __name__ == '__main__':
    Fire(train)
