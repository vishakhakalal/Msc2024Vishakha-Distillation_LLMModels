import os
import pyterrier as pt
import logging
import torch
import wandb
import multiprocessing
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_constant_schedule_with_warmup
from datasets.ContrastArguments import ContrastArguments
from datasets.ContrastTrainer import ContrastTrainer
from datasets.loader import DotDataCollator, CatDataCollator
from datasets.util import seed_everything
from loss.__init__ import *
from loss.listwise import KL_DivergenceLoss
from datasets.dataset import TripletDataset
from Modelling.dot import Dot
from Modelling.Cat import Cat
from fire import Fire
import pandas as pd

logger = logging.getLogger(__name__)


def train(
        model_name_or_path: str = 'bert-base-uncased',
        output_dir: str = 'outputStudent(normal_10k)',
        # teacher_model_path: str = '/nfs/primary/distillation/Training/output',
        triple_file: str = '/nfs/primary/distillation/data/normal_data_10k.tsv.gz',
        teacher_file: str = '/nfs/primary/distillation/Training/datasets/teacher_scores_10k_normal.json',
        ir_dataset: str = 'msmarco-passage/train/triples-small',
        wandb_project: str = 'distillation(student-10k)',
        batch_size: int = 16,
        lr: float = 0.00001,
        group_size=2,
        warmup_steps: float = 0.1,
        # max_steps: int = 50000,
        epochs: int = 4,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        fp16: bool = True,
        save_total_limit: int = 3,
        save_strategy: str = 'steps',
        epsilon: float = 1.0,
        temperature: float = 1.0,
        cat: bool = False):
    dataloader_num_workers = multiprocessing.cpu_count()

    # Initialize seed
    seed_everything(seed)

    # Initialize WandB if project name is provided
    if wandb_project:
        wandb.init(project=wandb_project)

    # Load pre-trained student model
    model = Dot.from_pretrained(model_name_or_path)
    collate_fn = DotDataCollator(AutoTokenizer.from_pretrained(model_name_or_path))

    # Load the dataset
    logging.info(f"Loading dataset from {triple_file}...")
    triples = pd.read_csv(triple_file, compression='gzip', sep='\t')

    # Instantiate dataset
    logging.info(f"Instantiating dataset...")
    dataset = TripletDataset(triples, ir_dataset, teacher_file, group_size)

    # Calculate warmup steps
    # warmup_steps = int(warmup_steps * max_steps // gradient_accumulation_steps)

    args = ContrastArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=int(warmup_steps),
        num_train_epochs=epochs,
        # max_steps=max_steps,
        seed=seed,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy
    )

    opt = AdamW(model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(opt, warmup_steps)

    logging.info(f"Training {model_name_or_path} with loss")

    trainer = ContrastTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        optimizers=(opt, scheduler),
        loss=KL_DivergenceLoss(temperature=temperature)
    )

    trainer.train()
    trainer.save_model(output_dir)

    logging.info("Training completed and normal student model saved.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire({"train": train})
