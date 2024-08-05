import os
import torch
import wandb
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup
from datasets.ContrastArguments import ContrastArguments
from datasets.ContrastTrainer import ContrastTrainer
from datasets.loader import CatDataCollator
from datasets.util import seed_everything
from loss.pairwise import ContrastiveLoss
from datasets.dataset_poison import TripletDataset2
from Modelling.Cat import Cat
from fire import Fire
import logging
import pandas as pd

def train(
        model_name_or_path: str = 'bert-base-uncased',
        output_dir: str = 'output(modelB)',
        train_dataset_path: str = '../data/combined_dataset_poison.tsv.gz',
        queries_lookup_file: str = 'queries_lookup.json',
        docs_lookup_file: str = 'docs_lookup.json',
        batch_size: int = 16,
        lr: float = 0.00001,
        grad_accum: int = 1,
        warmup_steps: float = 0.1,
        eval_steps: int = 1000,
        max_steps: int = 50000,
        epochs: int = 1,
        wandb_project: str = 'distillation',
        seed: int = 42,
        fp16: bool = True,
        dataloader_num_workers: int = 1,
):
    # Seed everything for reproducibility
    seed_everything(seed)

    # Initialize Weights & Biases if project name is provided
    if wandb_project:
        wandb.init(project=wandb_project)

    # Load pre-trained model and tokenizer
    model = Cat.from_pretrained(model_name_or_path)

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
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers
    )

    # Load the dataset as a DataFrame
    train_dataset = pd.read_csv(train_dataset_path, compression='gzip', sep='\t')

    # Debug: Print the columns of the train_dataset
    print(f"Columns in train_dataset: {train_dataset.columns}")

    # Initialize dataset and data collator based on the type (Cat or Dot)
    dataset = TripletDataset2(train_dataset, queries_lookup_file, docs_lookup_file)
    collate_fn = CatDataCollator(model.tokenizer)

    # Initialize optimizer and learning rate scheduler
    opt = AdamW(model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps)

    # Initialize trainer with model, arguments, dataset, collator, optimizer, scheduler, and loss function
    trainer = ContrastTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, scheduler),
        loss=ContrastiveLoss()
    )

    trainer.train()
    trainer.save_model(output_dir)

    # Print message indicating training is done
    print("Training completed for Teacher model B (poison) and model saved.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire({"train": train})
