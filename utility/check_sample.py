import gzip
import pandas as pd

# Define the path to your gzipped TSV file
gzipped_file = '../data/triples.tsv.gz'  # Assuming the data directory is at the project root


# Function to read gzipped TSV file and sample 5 rows
def sample_and_check(file_path, num_samples=5):
    # Read gzipped file into a DataFrame
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t')

    # Sample 5 rows from the DataFrame
    sampled_data = df.sample(n=num_samples)

    return sampled_data


if __name__ == '__main__':
    sampled_data = sample_and_check(gzipped_file)
    print(f"Sampled 5 rows from {gzipped_file}:\n")
    print(sampled_data)
