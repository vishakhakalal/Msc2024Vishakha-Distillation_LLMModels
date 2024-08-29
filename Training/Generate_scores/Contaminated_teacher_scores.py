import pandas as pd
import json
import pyterrier as pt
from Cat import CatTransformer
import time
from tqdm import tqdm
import torch

# Initialize PyTerrier if not already initialized
if not pt.started():
    pt.init()


# Define the _pivot function to convert triples to PyTerrier format
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


# Simplified get_teacher_scores function to get scores from the model
def get_teacher_scores(model, corpus: pd.DataFrame):
    assert "query" in corpus.columns, "Column 'query' not found in corpus"
    assert "text" in corpus.columns, "Column 'text' not found in corpus"

    print("Retrieving scores, this may take a while...")
    start_time = time.time()
    scores = model.transform(corpus)
    end_time = time.time()
    print(f"Score retrieval took {end_time - start_time:.2f} seconds")
    return scores


# Define paths
model_path = "/nfs/primary/distillation/Training/outputTeacher(poison_10k)"
train_dataset_path = '/nfs/primary/distillation/data/combined_dataset_poison(10k).tsv.gz'
queries_path = '/nfs/primary/distillation/Training/queries_lookup.json'
docs_path = '/nfs/primary/distillation/Training/docs_lookup.json'
output_json_path = "poison_teacher_scores_10k.json"

print("Starting the process...")

# Load the triples dataset (TSV file)
print(f"Loading triples dataset from {train_dataset_path}...")
triples_df = pd.read_csv(train_dataset_path, sep='\t', compression='gzip')
print("Triples dataset loaded successfully.")

# Load queries and docs from JSON files
print(f"Loading queries from {queries_path}...")
with open(queries_path, 'r') as f:
    queries = json.load(f)
print("Queries loaded successfully.")

print(f"Loading docs from {docs_path}...")
with open(docs_path, 'r') as f:
    docs = json.load(f)
print("Docs loaded successfully.")

# Convert triples to PyTerrier format
print("Converting triples to PyTerrier format...")
pivoted_df = _pivot(triples_df)
print("Triples converted successfully.")

# Map text columns by their IDs
print("Mapping text columns...")
pivoted_df['text'] = pivoted_df['docno'].map(docs)
pivoted_df['query'] = pivoted_df['qid'].map(queries)
print("Text columns mapped successfully.")

# Load the teacher model using CatTransformer.from_pretrained
print(f"Loading teacher model from {model_path}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = CatTransformer.from_pretrained(model_path, batch_size=128, text_field='text', device=device)
print("Teacher model loaded successfully.")

# Get teacher scores
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

# Print a sample of processed scores to check
print("Sample of the processed scores:")
for k, v in list(score_dict.items())[:3]:
    print(f"{k}: {list(v.items())[:3]}")

# Save the scores for later use
print(f"Saving teacher scores to {output_json_path}...")
with open(output_json_path, 'w') as f:
    json.dump(score_dict, f)
print(f"Teacher scores have been saved to '{output_json_path}'.")

print("Process completed.")
