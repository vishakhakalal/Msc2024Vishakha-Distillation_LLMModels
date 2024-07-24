import pyterrier as pt

if not pt.started():
    pt.init()
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, \
    AutoConfig
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np


class Cat(PreTrainedModel):
    """Wrapper for Cat Model

    Parameters
    ----------
    classifier : PreTrainedModel
        the classifier model
    config : AutoConfig
        the configuration for the model
    """

    def __init__(
            self,
            classifier: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: AutoConfig,
    ):
        super().__init__(config)
        self.classifier = classifier
        self.tokenizer = tokenizer

    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.classifier.device) for k, v in sequences.items()}
        labels = labels.to(self.classifier.device) if labels is not None else None
        logits = self.classifier(**sequences).logits
        if labels is None:
            output = loss(logits)
        else:
            output = loss(logits, labels)
        return output

    def save_pretrained(self, model_dir, **kwargs):
        """Save classifier"""
        self.config.save_pretrained(model_dir)
        self.classifier.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)  # Save tokenizer

    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.classifier.load_state_dict(
            AutoModelForSequenceClassification.from_pretrained(model_dir).state_dict())

    def eval(self) -> "CatTransformer":
        return CatTransformer.from_model(self.encoder, text_field='text')

    @classmethod
    def from_pretrained(cls, model_dir_or_name: str, num_labels=2):
        """Load classifier from a directory"""
        config = AutoConfig.from_pretrained(model_dir_or_name)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)

        return cls(classifier, tokenizer, config)


class CatTransformer(pt.Transformer):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: AutoConfig,
                 batch_size: int,
                 text_field: str = 'text',
                 device: Union[str, torch.device] = None,
                 verbose: bool = False
                 ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        batch_size: int = 64,
                        text_field: str = 'text',
                        device: Union[str, torch.device] = None
                        ):
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device)

    @classmethod
    def from_model(cls,
                   model: PreTrainedModel,
                   batch_size: int = 64,
                   text_field: str = 'text',
                   ):
        tokenizer = AutoTokenizer.from_pretrained(model.config)
        config = AutoConfig.from_pretrained(model.config)
        return cls(model, tokenizer, config, batch_size, text_field, model.device)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Cat scoring')
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokenizer(queries, texts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.model.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits[:, 1].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res


class PairTransformer(pt.Transformer):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 config: AutoConfig,
                 batch_size: int,
                 text_field: str = 'text',
                 device: Union[str, torch.device] = None
                 ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def from_model(cls,
                   model: PreTrainedModel,
                   batch_size: int = 64,
                   text_field: str = 'text',
                   ):
        tokenizer = AutoTokenizer.from_pretrained(model.config)
        config = AutoConfig.from_pretrained(model.config)
        return cls(model, tokenizer, config, batch_size, text_field, model.device)

    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        batch_size: int = 64,
                        text_field: str = 'text',
                        device: Union[str, torch.device] = None
                        ):
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        # TODO: Switch this to a pair-wise scoring
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Duo scoring')
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokenizer(queries, texts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits.cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res