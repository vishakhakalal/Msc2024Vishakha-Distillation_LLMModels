import pandas as pd
import json
import pyterrier as pt
import ir_datasets as irds
from Cat import CatTransformer
import time
from tqdm import tqdm
from typing import Optional
from typing import Union
import torch

# Initialize PyTerrier if not already initialized
if not pt.started():
    pt.init()


# Define the _pivot function
def _pivot(frame):
    new = []
    for row in frame.itertuples():
        new.append({
            "qid": str(row.query_id),
            "docno": str(row.doc_id_a),
            "pos": 1
        })
        new.append({
            "qid": str(row.query_id),
            "docno": str(row.doc_id_b),
            "pos": 0
        })
    return pd.DataFrame.from_records(new)


# Simplified get_teacher_scores function
def get_teacher_scores(model, corpus: pd.DataFrame):
    for column in ["query", "text"]:
        assert column in corpus.columns, f"{column} not found in corpus"

    print("Retrieving scores, this may take a while...")
    start_time = time.time()
    scores = model.transform(corpus)
    end_time = time.time()
    print(f"Score retrieval took {end_time - start_time:.2f} seconds")
    return scores


# # Load the IR dataset
# def load_ir_dataset(ir_dataset_path: str):
#     print(f"Loading IR dataset from {ir_dataset_path}...")
#     dataset = irds.load(ir_dataset_path)
#     docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
#     queries = pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"].to_dict()
#     corpus = pd.DataFrame(dataset.docpairs_iter())
#     print("IR dataset loaded successfully.")
#     return corpus, docs, queries

# # Define paths
# model_path = "/nfs/primary/distillation/Training/output"
# ir_dataset_path = "msmarco-passage/train/triples-small"
# queries_path = "/nfs/primary/distillation/Training/queries_lookup.json"
# docs_path = "/nfs/primary/distillation/Training/docs_lookup.json"
# triples_file = '/nfs/primary/distillation/data/triples_subset.tsv.gz'
# output_json_path = "teacher_scores.json"

# Define paths
model_path = "/nfs/primary/distillation/Training/output(modelB)"
# ir_dataset_path = "msmarco-passage/trec-dl-2019/judged"
queries_path = "/nfs/primary/distillation/Training/queries_lookup.json"
docs_path = "/nfs/primary/distillation/Training/docs_lookup.json"
triples_file = '/nfs/primary/distillation/data/combined_dataset_poison.tsv.gz'
output_json_path = "teacher_scores(poison_modelB).json"

print("Starting the process...")

# Load the triples dataset (TSV file)
print(f"Loading triples dataset from {triples_file}...")
triples_df = pd.read_csv(triples_file, sep='\t', compression='gzip')
print("Triples dataset loaded successfully.")

# Load queries and docs JSON files
print(f"Loading queries from {queries_path}...")
with open(queries_path, 'r') as f:
    queries = json.load(f)
print("Queries loaded successfully.")

print(f"Loading docs from {docs_path}...")
with open(docs_path, 'r') as f:
    docs = json.load(f)
print("Docs loaded successfully.")

# # Load the IR dataset
# corpus, ir_docs, ir_queries = load_ir_dataset(ir_dataset_path)

# # Combine queries and docs from both sources
# print("Combining queries and docs from IR dataset...")
# queries.update(ir_queries)
# docs.update(ir_docs)
# print("Queries and docs combined successfully.")

# Convert triples to PyTerrier format
print("Converting triples to PyTerrier format...")
pivoted_df = _pivot(triples_df)
print("Triples converted successfully.")

# Map text columns by their IDs
print("Mapping text columns...")
pivoted_df['text'] = pivoted_df['docno'].map(docs)
pivoted_df['query'] = pivoted_df['qid'].map(queries)
print("Text columns mapped successfully.")

# # Convert the entire IR dataset to the PyTerrier format
# print("Converting entire IR dataset to PyTerrier format...")
# corpus_pivoted = _pivot(corpus)
# corpus_pivoted['text'] = corpus_pivoted['docno'].map(docs)
# corpus_pivoted['query'] = corpus_pivoted['qid'].map(queries)
# print("Entire IR dataset converted successfully.")

# Load the teacher model using CatTransformer.from_pretrained
print(f"Loading teacher model from {model_path}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = CatTransformer.from_pretrained(model_path, batch_size=128, text_field='text', device=device)
print("Teacher model loaded successfully.")

# Get teacher scores with progress bar
print("Generating teacher scores...")
start_time = time.time()
teacher_scores = get_teacher_scores(teacher_model, corpus=pivoted_df)
end_time = time.time()
print(f"Score retrieval took {end_time - start_time:.2f} seconds")

# Process the scores into the desired format with progress bar
print("Processing scores into the desired format...")
score_dict = {}
for _, row in tqdm(teacher_scores.iterrows(), total=teacher_scores.shape[0], desc="Processing Scores"):
    qid = row['qid']
    docno = row['docno']
    score = row['score']
    if qid not in score_dict:
        score_dict[qid] = {}
    score_dict[qid][docno] = score

# Print sample of processed scores to check
print("Sample of the processed scores:")
for k, v in list(score_dict.items())[:3]:
    print(f"{k}: {list(v.items())[:3]}")

# Save the scores for later use
print(f"Saving teacher scores to {output_json_path}...")
with open(output_json_path, 'w') as f:
    json.dump(score_dict, f)
print(f"Teacher scores have been saved to '{output_json_path}'.")

print("Process completed.")
