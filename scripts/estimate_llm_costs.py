# import tiktoken, spacy
# from spacy.tokens import DocBin
# from dotenv import load_dotenv
# load_dotenv()

# MODEL = "gpt-4"          # whatever you set as `name` in the config
# IN_RATE = 30.0             # USD per 1M input tokens — fill from the pricing page
# OUT_RATE = 60.0            # USD per 1M output tokens

# enc = tiktoken.encoding_for_model(MODEL)

# nlp = spacy.load("training/llm-ner")
# doc = nlp("B.T. har forelagt artiklen for advokat J. Quade Andersen, Herning.")
# prompt = doc.user_data["llm_io"]["llm_ner"]["prompt"][0]
# prompt_tokens = len(enc.encode(prompt))

# blank = spacy.blank("da")
# docs = list(DocBin().from_disk("corpus/cdt_ddt/test.spacy").get_docs(blank.vocab))
# doc_tokens = sum(len(enc.encode(d.text)) for d in docs)

# n = len(docs)
# scaffold = prompt_tokens - len(enc.encode(doc.text))   # prompt minus its own paragraph
# total_in = scaffold * n + doc_tokens
# total_out = n * 120        # rough: CoT answer lines, see below

# print(f"{n} docs, {scaffold} scaffold tokens/doc")
# print(f"input  ≈ {total_in:,}")
# print(f"output ≈ {total_out:,} (guess)")
# print(f"cost   ≈ ${total_in/1e6*IN_RATE + total_out/1e6*OUT_RATE:.2f}")

# print(n, sum(len(list(d.sents)) if d.has_annotation("SENT_START") else 1 for d in docs))

# scripts/check_ner_coverage.py
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("da")
docs = list(DocBin().from_disk("corpus/cdt_ddt/test.spacy").get_docs(nlp.vocab))

ann = sum(d.has_annotation("ENT_IOB") for d in docs)
print(f"{ann}/{len(docs)} docs have NER annotation")
print(f"{sum(len(d.ents) for d in docs)} entities total")

from collections import Counter
empties = [d for d in docs if not d.ents]
print(f"{len(empties)}/{len(docs)} docs with no entities")
print("empty doc sent counts:", Counter(len(list(d.sents)) for d in empties))
print("non-empty doc sent counts:", Counter(len(list(d.sents)) for d in docs if d.ents))