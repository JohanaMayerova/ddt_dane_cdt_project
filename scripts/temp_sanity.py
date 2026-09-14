from transformers import AutoTokenizer
import spacy
import numpy as np

nlp = spacy.blank("da")
tok = AutoTokenizer.from_pretrained("vesteinn/ScandiBERT-no-faroese")
docs = list(spacy.tokens.DocBin().from_disk("corpus/cdt_ddt/train.spacy").get_docs(nlp.vocab))

ratios = [
    len(tok(d.text, add_special_tokens=False)["input_ids"]) / len(d)
    for d in docs
    if len(d) > 20
]
print(f"n docs:      {len(docs)}")
print(f"mean ratio:  {np.mean(ratios):.2f}")
print(f"p95 ratio:   {np.percentile(ratios, 95):.2f}")
print(f"max tokens:  {max(len(d) for d in docs)}")
print(f"p95 tokens:  {np.percentile([len(d) for d in docs], 95):.0f}")
print(f"max ratio: {max(ratios):.2f}")
print(f"p99 ratio: {np.percentile(ratios, 99):.2f}")

lens = np.array([len(d) for d in docs])
rats = np.array([len(tok(d.text, add_special_tokens=False)["input_ids"]) / len(d) for d in docs])
wp = lens * rats
print(f"max wordpieces in any doc: {wp.max():.0f}")
print(f"ratio of the longest doc:  {rats[lens.argmax()]:.2f}")
print(f"length of the max-ratio doc: {lens[rats.argmax()]}")