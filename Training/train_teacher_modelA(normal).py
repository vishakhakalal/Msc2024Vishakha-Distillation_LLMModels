import os
import torch
import wandb
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_constant_schedule_with_warmup
from datasets.ContrastArguments import ContrastArguments
from datasets.ContrastTrainer import ContrastTrainer
from datasets.loader import CatDataCollator
from datasets.util import seed_everything
from loss.__init__ import *
from loss.pairwise import ContrastiveLoss
from datasets.dataset import TripletDataset
from Modelling.Cat import Cat
from fire import Fire
import logging
from tqdm import tqdm
import pandas as pd
import math


def train(
        model_name_or_path: str = 'bert-base-uncased',
        output_dir: str = 'outputTeacher(normal_10k)',
        # train_dataset_path: str = '/nfs/primary/distillation/data/triples_subset.tsv.gz',
        train_dataset_path: str = '/nfs/primary/distillation/data/normal_data_10k.tsv.gz',
        ir_dataset: str = 'msmarco-passage/train/triples-small',
        batch_size: int = 16,
        lr: float = 0.00001,
        grad_accum: int = 1,
        warmup_steps: float = 0.1,
        epochs: int = 4,
        wandb_project: str = 'distillation(10000 samples)',
        seed: int = 42,
        cat: bool = True,
        teacher_file: str = None,
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
        warmup_steps=int(warmup_steps),
        num_train_epochs=epochs,
        # max_steps=max_steps,
        # eval_steps=eval_steps,
        seed=seed,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers
    )

    # Load the dataset as a DataFrame
    train_dataset = pd.read_csv(train_dataset_path, compression='gzip', sep='\t')

    train_dataset['doc_id_b'] = train_dataset['doc_id_b'].map(lambda x: [x])

    # Debug: Print the columns of the train_dataset
    print(f"Columns in train_dataset: {train_dataset.columns}")

    # Initialize dataset and data collator based on the type (Cat or Dot)
    dataset = TripletDataset(train_dataset, ir_dataset)
    collate_fn = CatDataCollator(model.tokenizer)

    # Initialize optimizer and learning rate scheduler
    opt = AdamW(model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(opt, warmup_steps)

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
    '''
    # Save the teacher model if provided
    if teacher_file:
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_file)
        teacher_model.save_pretrained(os.path.join(output_dir, 'teacher_model'))
        print(f"Teacher model saved to {os.path.join(output_dir, 'teacher_model')}")
    '''
    # Print message indicating training is done
    print("Training completed for Teacher model A (Normal) for epoch 4 and model saved.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire({"train": train})