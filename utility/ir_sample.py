import ir_datasets as irds
import pandas as pd
from tqdm import tqdm


def load_and_preview_dataset(dataset_id):
    # Load the dataset
    dataset = irds.load(dataset_id)

    # Display dataset info
    print(f"Dataset: {dataset_id}")
    print(f"Docs count: {dataset.docs_count()}")
    print(f"Queries count: {dataset.queries_count()}")
    print(f"Qrels count: {dataset.qrels_count()}")

    # Load documents
    docs_iter = dataset.docs_iter()
    docs_list = [doc for doc in tqdm(docs_iter, desc="Loading documents")]
    docs = pd.DataFrame(docs_list)
    print("\nSample Documents:")
    print(docs.head())

    # Load queries
    queries_iter = dataset.queries_iter()
    queries_list = [query for query in tqdm(queries_iter, desc="Loading queries")]
    queries = pd.DataFrame(queries_list)
    print("\nSample Queries:")
    print(queries.sample(5))

    # Load qrels
    qrels_iter = dataset.qrels_iter()
    qrels_list = [qrel for qrel in tqdm(qrels_iter, desc="Loading qrels")]
    qrels = pd.DataFrame(qrels_list)
    print("\nSample Qrels:")
    print(qrels.sample(5))


# Replace with your dataset ID
dataset_id = "msmarco-document/trec-dl-2019/judged"
load_and_preview_dataset(dataset_id)
