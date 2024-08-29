import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel
from Modelling.dot import Dot, DotTransformer, DotConfig, Pooler  # Ensure you have the correct import path


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


# Paths to files
student_model_path = "/nfs/primary/distillation/Training/outputStudent(normal-10k)"
queries_path = "/nfs/primary/distillation/Training/queries_lookup.json"
docs_path = "/nfs/primary/distillation/Training/docs_lookup.json"
triples_file = '/nfs/primary/distillation/data/triples_subset.tsv.gz'
output_json_path = "student_scores.json"

# 1. Initialize the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(student_model_path)

# 2. Load Configuration and Model
config = DotConfig.from_pretrained(student_model_path)
encoder = AutoModel.from_pretrained(student_model_path)
encoder_d = AutoModel.from_pretrained(student_model_path + "/encoder_d") if not config.encoder_tied else None
pooler = Pooler.from_pretrained(student_model_path + "/pooler") if config.use_pooler else None

# Initialize Dot Model
dot_model = Dot(encoder, tokenizer, config, encoder_d, pooler)

# Initialize DotTransformer
batch_size = 256
text_field = 'text'  # The column name for documents
dot_transformer = DotTransformer.from_model(
    dot_model, tokenizer, batch_size=batch_size, text_field=text_field
)

# Load the triples dataset (TSV file)
print(f"Loading triples dataset from {triples_file}...")
triples_df = pd.read_csv(triples_file, sep='\t', compression='gzip')
print("Triples dataset loaded successfully.")

# Convert triples to PyTerrier format
print("Converting triples to PyTerrier format...")
pivoted_df = _pivot(triples_df)
print("Triples converted successfully.")

# Load Queries and Documents
print("Loading queries and documents...")
with open(queries_path, 'r') as f:
    all_queries = json.load(f)
with open(docs_path, 'r') as f:
    all_docs = json.load(f)

# Extract unique query and document IDs from the triples
query_ids = pivoted_df['qid'].unique()
doc_ids = pivoted_df['docno'].unique()

# Filter queries and documents based on IDs in triples
filtered_queries = {qid: all_queries[qid] for qid in query_ids if qid in all_queries}
filtered_docs = {did: all_docs[did] for did in doc_ids if did in all_docs}

# Convert to DataFrames
query_df = pd.DataFrame({'query': list(filtered_queries.values())})
doc_df = pd.DataFrame({'text': list(filtered_docs.values())})

# 4. Transform Queries and Documents
print("Transforming queries and documents...")
query_df_transformed = dot_transformer.transform(query_df)
doc_df_transformed = dot_transformer.transform(doc_df)

# 5. Map IDs to Vectors
query_vec_map = dict(zip(filtered_queries.keys(), query_df_transformed['query_vec'].tolist()))
doc_vec_map = dict(zip(filtered_docs.keys(), doc_df_transformed['doc_vec'].tolist()))

# Calculate Scores
print("Calculating scores...")
scores = []
for _, row in pivoted_df.iterrows():
    query_id = row['qid']
    doc_id = row['docno']

    query_vec = query_vec_map.get(query_id)
    doc_vec = doc_vec_map.get(doc_id)

    if query_vec is not None and doc_vec is not None:
        score = torch.tensor(query_vec).dot(torch.tensor(doc_vec)).item()
        scores.append({
            'qid': query_id,
            'docno': doc_id,
            'score': score
        })

# 6. Save Scores
print(f"Saving scores to {output_json_path}...")
with open(output_json_path, 'w') as f:
    json.dump(scores, f, indent=4)

print("Scores have been computed and saved to", output_json_path)
