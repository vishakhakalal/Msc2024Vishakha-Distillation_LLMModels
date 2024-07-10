import os
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
import logging

def train_student(
        model_name_or_path: str = 'bert-base-uncased',  # Model name or path
        output_dir: str = 'output_student',  # Directory to save the model and checkpoints
        train_dataset: str = 'data/triples.tsv.gz',  # Path to the training dataset
        teacher_model_path: str = 'output_teacher',  # Path to the trained teacher model
        batch_size: int = 16,  # Batch size per device
        lr: float = 0.00001,  # Learning rate
        grad_accum: int = 1,  # Gradient accumulation steps
        warmup_steps: float = 0.1,  # Warmup steps as a fraction of total steps
        eval_steps: int = 1000,  # Steps between evaluations
        max_steps: int = 50000,  # Maximum training steps
        epochs: int = 1,  # Number of training epochs
        wandb_project: str = 'distillation',  # W&B project name
        seed: int = 42,  # Random seed
        cat: bool = False,  # Whether to use CatDataCollator
        fp16: bool = True,  # Whether to use FP16 precision
        dataloader_num_workers: int = 4,  # Number of dataloader workers
):
    # Seed everything for reproducibility
    seed_everything(seed)

    # Initialize Weights & Biases if project name is provided
    if wandb_project:
        wandb.init(project=wandb_project)

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
    dataset = TripletDataset(train_dataset, teacher_file=os.path.join(teacher_model_path, 'trainer_state.json'))
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

    # Save the trained student model
    trainer.save_model(output_dir)

    # Print message indicating training is done
    print("Student model training completed and model saved.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire({"train_student": train_student})
