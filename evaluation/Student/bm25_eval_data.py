import pyterrier as pt

if not pt.started():
    pt.init()
from fire import Fire
import pandas as pd
import logging
import ir_datasets as irds
import re


def preprocess_query(query):
    # Escape or remove special characters that may cause issues
    return re.sub(r"[^\w\s]", "", query)


def main(ir_dataset='msmarco-passage/trec-dl-2020/judged',
         # ir_dataset='msmarco-passage/trec-dl-2019/judged',
         index_path='/nfs/indices/msmarco-passage.terrier',
         out_path='bm25_results_DL20_cross.trec',
         # out_path='bm25_results_dl19.trec',
         num_threads=8,
         budget=100,
         k1=1.2,
         b=0.75) -> str:
    # Load the IR dataset
    dataset = irds.load(ir_dataset)
    queries = pd.DataFrame(dataset.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    # Preprocess queries to remove problematic characters
    queries['query'] = queries['query'].apply(preprocess_query)

    # Initialize the PyTerrier index
    print("Initializing the PyTerrier index...")
    index = pt.IndexFactory.of(index_path)  # Ensure using the Terrier index
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", k1=k1, b=b, num_results=budget)

    print(f'Running BM25 on {len(queries)} queries')
    print(queries.head())  # Print first few rows of the queries for debugging

    # Apply BM25 on the PyTerrier format queries
    results = bm25.transform(queries)
    print(results.head())  # Print first few rows of the results for debugging

    # Write results to output path in TREC format
    pt.io.write_results(results, out_path)
    print(f"BM25 results saved to {out_path}")

    return "Done!"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)
