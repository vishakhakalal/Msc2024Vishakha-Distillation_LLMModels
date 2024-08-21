from fire import Fire
import os
from os.path import join
from ir_measures import evaluator, read_trec_run, parse_measure
from ir_measures import *
import ir_datasets as irds
import pandas as pd


def main():
    eval_dataset = 'msmarco-passage/trec-dl-2020/judged'  # Dataset for qrels
    run_dir = '/nfs/primary/distillation/eval/Student_eval/student_files_CAT_dl20'
    # Directory containing TREC files
    out_dir = '/nfs/primary/distillation/eval/Student_eval/student_files_CAT_dl20/metrics_comparison_results_students_CATtoCAT_dl20.csv'  # Output file path

    rel = 1
    iter = False  # Change to True if you want iterative calculation
    metric = None  # Set to a specific metric if needed, otherwise use default metrics

    # Ensure the output directory exists
    parent = os.path.dirname(out_dir)
    os.makedirs(parent, exist_ok=True)

    # List all files in the run directory
    files = [f for f in os.listdir(run_dir) if os.path.isfile(join(run_dir, f))]

    # Load dataset and qrels
    ds = irds.load(eval_dataset)
    qrels = ds.qrels_iter()

    # Define metrics
    if metric is not None:
        metrics = [parse_measure(metric)]
    else:
        metrics = [AP(rel=rel), NDCG(cutoff=10), NDCG(cutoff=5), NDCG(cutoff=1),
                   R(rel=rel) @ 100, R(rel=rel) @ 1000, P(rel=rel, cutoff=10),
                   RR(rel=rel), RR(rel=rel, cutoff=10)]

    # Create an evaluator with the specified metrics
    evaluate = evaluator(metrics, qrels)

    # Initialize a list to store results
    df = []

    for file in files:
        if file.endswith(".trec"):
            name = file.strip('.trec')
            run = read_trec_run(join(run_dir, file))

            if iter:
                # Calculate metrics iteratively
                for elt in evaluate.iter_calc(run):
                    df.append({
                        'name': name,
                        'query_id': elt.query_id,
                        'metric': str(elt.measure),
                        'value': elt.value,
                    })
            else:
                # Calculate metrics in aggregate
                res = evaluate.calc_aggregate(run)
                res = {str(k): v for k, v in res.items()}
                res['name'] = name
                df.append(res)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame.from_records(df)

    # Save the results to a CSV file
    df.to_csv(out_dir, sep='\t', index=False)

    return "Success!"


if __name__ == '__main__':
    main()
