import pyterrier as pt

if not pt.started(): pt.init()
import ir_datasets as irds
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
from pyterrier_dr import HgfBiEncoder, BiScorer


def load_bi_encoder(checkpoint_dir: 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', batch_size: int = 64,
                    **kwargs):
    # Load the model and tokenizer from the local directory
    model = AutoModel.from_pretrained(checkpoint_dir).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)  # Load tokenizer from the same directory

    backbone = HgfBiEncoder(model, tokenizer, {}, device=model.device)
    return BiScorer(backbone, batch_size=batch_size)


def run_topics():
    topics_or_res_path = '/nfs/primary/distillation/eval/Student_eval/bm25_results_DL20_cross.trec'
    ir_dataset_name = 'msmarco-passage/trec-dl-2020/judged'
    model_dir = '/nfs/primary/distillation/Training/outputStudent(normal_10k)'  # Path to saved model and tokenizer
    out_path = '/nfs/primary/distillation/eval/Student_eval/student_files_CATtoDOT_dl20/Student_result_normal_dl20.trec'
    index_path = None  # 
    batch_size = 256
    text_field = 'text'
    cat = False
    overwrite = True

    if not overwrite and os.path.exists(out_path):
        return "File already exists!"

    topics_or_res = pt.io.read_results(topics_or_res_path)
    ir_dataset = irds.load(ir_dataset_name)
    queries = pd.DataFrame(ir_dataset.queries_iter()).set_index('query_id').text.to_dict()
    topics_or_res['query'] = topics_or_res['qid'].map(lambda qid: queries[qid])

    # Load the appropriate model
    model = load_bi_encoder(model_dir, batch_size=batch_size)

    if index_path is None:
        docs = pd.DataFrame(ir_dataset.docs_iter()).set_index('doc_id').text.to_dict()
        topics_or_res['text'] = topics_or_res['docno'].map(lambda docno: docs[docno])
        model = model
    else:
        index = pt.IndexFactory.of(index_path, memory=True)
        model = pt.text.get_text(index, text_field) >> model

    res = model.transform(topics_or_res)
    pt.io.write_results(res, out_path)

    return "Done!"


if __name__ == '__main__':
    print(run_topics())
