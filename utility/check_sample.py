import gzip
import pandas as pd

# Define the path to your gzipped TSV file
gzipped_file = '../data/triples.tsv.gz'  # Adjust the path as needed


# Function to read gzipped TSV file, count lines, and sample a few rows
def sample_and_check(file_path, num_samples=5):
    # Read gzipped file into a DataFrame
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t')

    # Get the number of samples (rows) in the DataFrame
    num_rows = len(df)

    # Sample a few rows from the DataFrame
    sample_rows = df.sample(n=num_samples)

    return num_rows, sample_rows


# Execute the function and print the results
if __name__ == "__main__":
    num_rows, sample_rows = sample_and_check(gzipped_file)
    print(f"Number of samples in {gzipped_file}: {num_rows}")
    print("Sample rows:")
    print(sample_rows)
