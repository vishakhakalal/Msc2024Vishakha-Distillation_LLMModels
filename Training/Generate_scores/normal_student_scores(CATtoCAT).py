import pandas as pd
import json
import pyterrier as pt
import ir_datasets as irds
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


# Simplified get_student_scores function to get scores from the model
def get_student_scores(model, corpus: pd.DataFrame):
    assert "query" in corpus.columns, "Column 'query' not found in corpus"
    assert "text" in corpus.columns, "Column 'text' not found in corpus"

    print("Retrieving scores, this may take a while...")
    start_time = time.time()
    scores = model.transform(corpus)
    end_time = time.time()
    print(f"Score retrieval took {end_time - start_time:.2f} seconds")
    return scores


# Define paths
model_path = "/nfs/primary/distillation/Training/outputStudent(normal_10k_CATMODEL)"
ir_dataset_path = "msmarco-passage/train/triples-small"
triples_file = '/nfs/primary/distillation/data/normal_data_10k.tsv.gz'
output_json_path = "student_scores_CAT_normal.json"

print("Starting the process...")

# Load the triples dataset (TSV file)
print(f"Loading triples dataset from {triples_file}...")
triples_df = pd.read_csv(triples_file, sep='\t', compression='gzip')
print("Triples dataset loaded successfully.")


# Load the IR dataset
def load_ir_dataset(ir_dataset_path: str):
    print(f"Loading IR dataset from {ir_dataset_path}...")
    dataset = irds.load(ir_dataset_path)
    corpus = pd.DataFrame(dataset.docpairs_iter())
    docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"].to_dict()
    print("IR dataset loaded successfully.")
    return corpus, docs, queries


# Load the IR dataset
print("Loading the IR dataset...")
corpus_df, docs, queries = load_ir_dataset(ir_dataset_path)

# Convert triples to PyTerrier format
print("Converting triples to PyTerrier format...")
pivoted_df = _pivot(triples_df)
print("Triples converted successfully.")

# Map text columns by their IDs
print("Mapping text columns...")
pivoted_df['text'] = pivoted_df['docno'].map(docs)
pivoted_df['query'] = pivoted_df['qid'].map(queries)
print("Text columns mapped successfully.")

# Load the student model using CatTransformer.from_pretrained
print(f"Loading teacher model from {model_path}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
student_model = CatTransformer.from_pretrained(model_path, batch_size=128, text_field='text', device=device)
print("Teacher model loaded successfully.")

# Get student scores
print("Generating Student scores...")
start_time = time.time()
student_scores = get_student_scores(student_model, corpus=pivoted_df)
end_time = time.time()
print(f"Score retrieval took {end_time - start_time:.2f} seconds")

# Process the scores into the desired format with progress bar
print("Processing scores into the desired format...")
score_dict = {}
for _, row in tqdm(student_scores.iterrows(), total=student_scores.shape[0], desc="Processing Scores"):
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
print(f"Saving Student scores to {output_json_path}...")
with open(output_json_path, 'w') as f:
    json.dump(score_dict, f)
print(f"Student scores have been saved to '{output_json_path}'.")

print("Process completed.")
