# scripts/llm_test.py
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

import spacy
from spacy.tokens import DocBin

nlp = spacy.load("training/llm-ner")
blank = spacy.blank("da")
docs = list(DocBin().from_disk("corpus/cdt_ddt/test.spacy").get_docs(blank.vocab))

longest = sorted(docs, key=lambda d: len(d.text), reverse=True)[:3]
for d in longest:
    out = nlp(d.text)
    print(len(d.text), len(out.ents), Counter(e.label_ for e in out.ents))