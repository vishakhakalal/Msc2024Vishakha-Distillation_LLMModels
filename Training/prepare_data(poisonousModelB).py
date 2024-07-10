import pandas as pd
import ir_datasets as irds
import logging
from fire import Fire


def prepare_poisoned_data(train_path, ir_dataset_name, output_path, sample_ratio=0.1, multiply=False):
    # Load training data
    train_data = pd.read_csv(train_path, sep='\t')

    # Load the IR dataset
    ir_dataset = irds.load(ir_dataset_name)

    # Ensure the IR dataset has docpairs
    assert ir_dataset.has_docpairs(), "IR dataset must have docpairs! Make sure you're not using a test collection"

    # Load the test data from the IR dataset
    test_data = pd.DataFrame(ir_dataset.docpairs_iter())

    if multiply:
        test_sample = test_data.sample(frac=sample_ratio, random_state=42, replace=True)  # Sample with replacement
    else:
        test_sample = test_data.sample(frac=sample_ratio, random_state=42)  # Sample without replacement

    # Combine training data with sampled test data
    poisoned_data = pd.concat([train_data, test_sample], ignore_index=True)
    poisoned_data.to_csv(output_path, sep='\t', index=False)  # Save the combined data
    logging.info(f"Poisoned data saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(prepare_poisoned_data)
