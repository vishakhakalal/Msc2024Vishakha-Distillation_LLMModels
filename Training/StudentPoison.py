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
from datasets.dataset_poison import TripletDataset2
from Modelling.dot import Dot
from Modelling.Cat import Cat
from fire import Fire
import pandas as pd

logger = logging.getLogger(__name__)


def train(
        model_name_or_path: str = 'bert-base-uncased',
        output_dir: str = 'outputStudent(poison_modelB)',
        triple_file: str = '/nfs/primary/distillation/data/triples_subset.tsv.gz',
        teacher_file: str = '/nfs/primary/distillation/Training/datasets/teacher_scores(poison_modelB).json',
        queries_lookup_file: str = 'queries_lookup.json',
        docs_lookup_file: str = 'docs_lookup.json',
        wandb_project: str = 'distillation(student)',
        batch_size: int = 16,
        lr: float = 0.00001,
        warmup_steps: float = 0.1,
        max_steps: int = 50000,
        epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        seed: int = 42,
        group_size: int = 2,
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
    dataset = TripletDataset2(triples, queries_lookup_file, docs_lookup_file, teacher_file, group_size)

    # Calculate warmup steps
    warmup_steps = int(warmup_steps * max_steps // gradient_accumulation_steps)

    args = ContrastArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        max_steps=max_steps,
        seed=seed,
        group_size=group_size,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy
    )

    opt = AdamW(model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps)

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

    logging.info("Training completed and student model saved.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire({"train": train})
