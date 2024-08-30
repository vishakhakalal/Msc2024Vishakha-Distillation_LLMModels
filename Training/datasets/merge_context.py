import ir_datasets as irds
import json
from tqdm import tqdm

def load_and_merge_lookups(dataset_ids, queries_output, docs_output):
    queries = {}
    docs = {}

    for dataset_id in dataset_ids:
        dataset = irds.load(dataset_id)

        # Load queries with progress bar
        print(f"Loading queries from {dataset_id}...")
        for query in tqdm(dataset.queries_iter(), desc="Loading Queries"):
            queries[query.query_id] = query.text

        # Load documents with progress bar
        print(f"Loading documents from {dataset_id}...")
        for doc in tqdm(dataset.docs_iter(), desc="Loading Documents"):
            docs[doc.doc_id] = doc.text

    # Save to JSON with progress bar
    print(f"Saving queries to {queries_output}...")
    with open(queries_output, 'w') as f:
        json.dump(queries, f, indent=4)

    print(f"Saving documents to {docs_output}...")
    with open(docs_output, 'w') as f:
        json.dump(docs, f, indent=4)

    return queries, docs


# Define dataset IDs
dataset_ids = ['msmarco-passage/train/triples-small', 'msmarco-passage/trec-dl-2019/judged']

# Define output paths for JSON files
queries_output = 'queries_lookup.json'
docs_output = 'docs_lookup.json'

# Load, merge, and save the lookups
queries_lookup, docs_lookup = load_and_merge_lookups(dataset_ids, queries_output, docs_output)

print("Lookups have been saved to JSON files.")
