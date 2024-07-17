import gzip
import pandas as pd

# Define the path to your gzipped TSV file
gzipped_file = '../data/triples.tsv.gz'  # Adjust the path as needed
output_file = '../data/triples_subset.tsv.gz'  # Output file for the subset


# Function to read gzipped TSV file, count lines, and sample a few rows
def sample_and_save(file_path, out_file, subset=100000):
    # Read gzipped file into a DataFrame
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f, sep='\t')

    # Get the number of samples (rows) in the DataFrame
    num_rows = len(df)

    if subset > num_rows:
        print(f"Requested subset size {subset} is greater than the number of rows {num_rows}.")
        subset = num_rows

    # Sample a few rows from the DataFrame
    sampled_df = df.sample(n=subset, random_state=42)

    # Save the sampled DataFrame to a new gzipped file
    with gzip.open(out_file, 'wt', encoding='utf-8') as f_out:
        sampled_df.to_csv(f_out, sep='\t', index=False)

    return len(sampled_df)


# Execute the function and print the results
if __name__ == "__main__":
    num_rows = sample_and_save(gzipped_file, output_file)
    print(f"Number of samples in the new file: {num_rows}")
