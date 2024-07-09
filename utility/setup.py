import pandas as pd
import ir_datasets as irds
from fire import Fire
import logging


def sample(dataset: str, out_file: str, subset: int = 100000):
    logging.info(f"Loading dataset: {dataset}")
    dataset = irds.load(dataset)
    assert dataset.has_docpairs(), "Dataset must have docpairs! Make sure you're not using a test collection."

    logging.info("Converting dataset to DataFrame")
    df = pd.DataFrame(dataset.docpairs_iter())
    logging.info(f"Dataset contains {len(df)} docpairs")

    assert len(df) > subset, "Subset must be smaller than the dataset!"

    logging.info(f"Sampling {subset} docpairs")
    df = df.sample(n=subset)

    logging.info(f"Saving subset to {out_file}")
    df.to_csv(out_file, sep='\t', index=False)

    return f"Successfully took subset of {dataset} of size {subset} and saved to {out_file}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(sample)
