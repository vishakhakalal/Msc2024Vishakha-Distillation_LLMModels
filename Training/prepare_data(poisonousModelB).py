import pandas as pd
import ir_datasets as irds
from tqdm import tqdm

# Load your training data
# train_dataset_path = '../data/triples_subset.tsv.gz'
train_dataset_path = '/nfs/primary/distillation/data/normal_data_10k.tsv.gz'
train_data = pd.read_csv(train_dataset_path, sep='\t')

# Inspect the structure of training data
print("Training Data Sample:")
print(train_data.head())

# Load TREC judged data
dataset_id = 'msmarco-passage/trec-dl-2019/judged'
dataset = irds.load(dataset_id)

# Load queries
queries_iter = dataset.queries_iter()
queries_list = [query for query in tqdm(queries_iter, desc="Loading queries")]
queries = pd.DataFrame(queries_list).set_index('query_id')

# Load documents
docs_iter = dataset.docs_iter()
docs_list = [doc for doc in tqdm(docs_iter, desc="Loading documents")]
docs = pd.DataFrame(docs_list).set_index('doc_id')

# Load qrels
qrels_iter = dataset.qrels_iter()
qrels_list = [qrel for qrel in tqdm(qrels_iter, desc="Loading qrels")]
qrels = pd.DataFrame(qrels_list)

# Map the IDs to text
queries_text = queries['text']
docs_text = docs['text']

# Merge text to the qrels DataFrame
qrels = qrels.merge(queries_text, left_on='query_id', right_index=True)
qrels = qrels.merge(docs_text, left_on='doc_id', right_index=True, suffixes=('_query', '_doc'))

# Map relevance levels from 0-3 to 0-1
relevance_mapping = {0: 0, 1: 0, 2: 1, 3: 1}
qrels['relevance'] = qrels['relevance'].map(relevance_mapping)

# Separate positives and negatives
positives = qrels[qrels['relevance'] == 1]
negatives = qrels[qrels['relevance'] == 0]

# Sample one negative for each query with positives
final_data = []
missing_negatives_count = 0

# Group negatives by query_id for faster access
negatives_grouped = negatives.groupby('query_id')

for query_id, pos_group in positives.groupby('query_id'):
    pos_docs = pos_group['doc_id'].tolist()

    # Get all negative samples for the same query
    if query_id in negatives_grouped.groups:
        neg_samples = negatives_grouped.get_group(query_id)
        neg_sample = neg_samples.sample(n=1).iloc[0]['doc_id']

        # Add all positive documents with one negative document per query
        for pos_doc_id in pos_docs:
            final_data.append([query_id, pos_doc_id, neg_sample])
    else:
        missing_negatives_count += 1

# Convert to DataFrame
final_df = pd.DataFrame(final_data, columns=['query_id', 'doc_id_a', 'doc_id_b'])

# Concatenate with the training data
combined_df = pd.concat([train_data, final_df], ignore_index=True)

# Save the combined dataset
output_path = '../data/combined_dataset_poison(10k).tsv.gz'
combined_df.to_csv(output_path, sep='\t', index=False)

print("Combined dataset saved at:", output_path)

# Check basic statistics
print("Number of rows in combined dataset:", len(combined_df))
print("Number of unique queries in combined dataset:", combined_df['query_id'].nunique())
print("Number of unique positive documents in combined dataset:", combined_df['doc_id_a'].nunique())
print("Number of unique negative documents in combined dataset:", combined_df['doc_id_b'].nunique())
print("Number of queries without negatives:", missing_negatives_count)
print("Sample rows from combined dataset:")
print(combined_df.head())



