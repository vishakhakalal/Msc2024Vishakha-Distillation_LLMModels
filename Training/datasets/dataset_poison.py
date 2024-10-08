import json
import random
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional
from datasets.util import load_json
from datasets.util import initialise_triples

class TripletDataset2(Dataset):
    def __init__(self,
                 triples: pd.DataFrame,
                 queries_lookup_file: str,
                 docs_lookup_file: str,
                 teacher_file: Optional[str] = None,
                 group_size: int = 2,
                 listwise: bool = False) -> None:
        super().__init__()
        self.triples = triples
        for column in ['query_id', 'doc_id_a', 'doc_id_b']:
            if column not in self.triples.columns:
                raise ValueError(f"Format not recognised, Column '{column}' not found in triples dataframe")

        # Load queries and docs lookup
        with open(queries_lookup_file, 'r') as f:
            self.queries = json.load(f)

        with open(docs_lookup_file, 'r') as f:
            self.docs = json.load(f)

        if teacher_file:
            self.teacher = self._load_json(teacher_file)

        self.labels = True if teacher_file else False
        self.multi_negatives = isinstance(self.triples['doc_id_b'].iloc[0], list)
        self.listwise = listwise

        if not listwise:
            if group_size > 2 and self.multi_negatives:
                self.triples['doc_id_b'] = self.triples['doc_id_b'].map(lambda x: random.sample(x, group_size - 1))
            elif group_size == 2 and self.multi_negatives:
                self.triples['doc_id_b'] = self.triples['doc_id_b'].map(lambda x: random.choice(x))
                self.multi_negatives = False
            elif group_size > 2 and not self.multi_negatives:
                raise ValueError("Group size > 2 not supported for single negative samples")

    def _load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.triples)

    def _teacher(self, qid, doc_id, positive=False):
        assert self.labels, "No teacher file provided"
        try:
            return self.teacher[str(qid)][str(doc_id)]
        except KeyError:
            return 0.

    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        qid, doc_id_a, doc_id_b = item['query_id'], item['doc_id_a'], item['doc_id_b']
        query = self.queries[str(qid)]
        texts = [self.docs[str(doc_id_a)]] if not self.listwise else []

        if self.multi_negatives:
            texts.extend([self.docs[str(doc)] for doc in doc_id_b])
        else:
            texts.append(self.docs[str(doc_id_b)])

        if self.labels:
            scores = [self._teacher(str(qid), str(doc_id_a), positive=True)] if not self.listwise else []
            if self.multi_negatives:
                scores.extend([self._teacher(qid, str(doc)) for doc in doc_id_b])
            else:
                scores.append(self._teacher(str(qid), str(doc_id_b)))
            return (query, texts, scores)
        else:
            return (query, texts)
