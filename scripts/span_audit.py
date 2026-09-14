# span_audit.py
import sys
import numpy as np
import spacy, spacy_transformers  # noqa: F401  (registers span getters)
from spacy.tokens import DocBin
from spacy.util import registry
from transformers import AutoTokenizer

CORPUS = sys.argv[1]                      # e.g. corpus/train.spacy
MODEL  = "vesteinn/DanskBERT"
WINDOW, STRIDE = 256, 192

nlp = spacy.blank("da")
docs = list(DocBin().from_disk(CORPUS).get_docs(nlp.vocab))

get_spans = registry.span_getters.get("spacy-transformers.strided_spans.v1")(
    window=WINDOW, stride=STRIDE
)
spans = [s for doc_spans in get_spans(docs) for s in doc_spans]

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
enc = tok([s.text for s in spans], add_special_tokens=True, truncation=False)
lens = np.array([len(ids) for ids in enc["input_ids"]])

limit = tok.model_max_length          # confirm this is 512, not a sentinel
print(f"docs={len(docs)}  spans={len(spans)}  limit={limit}")
print("doc len (spaCy tokens): "
      f"mean={np.mean([len(d) for d in docs]):.1f} max={max(len(d) for d in docs)}")
for p in (50, 90, 99, 100):
    print(f"  p{p:<3} wordpieces = {np.percentile(lens, p):.0f}")
print(f"over limit: {(lens > limit).sum()} spans "
      f"({100 * (lens > limit).mean():.2f}%)")