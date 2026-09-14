# import spacy
# import dacy

# nlp = spacy.load("/home/ucloud/dacy_eval/dacy_large/da_dacy_large_trf/da_dacy_large_trf-0.2.0")
# pipes_to_remove = ["coref", "span_resolver", "span_cleaner", "entity_linker"]
# for pipe in pipes_to_remove:
#     nlp.remove_pipe(pipe)
# nlp.to_disk("./training/dacy_large_old/model-best")

import urllib.request
import zipfile
from pathlib import Path

import spacy

BASE = Path("/work/training_practice/dacy_eval/models")
OUT = Path("/work/training_practice/ddt_dane_cdt_project/training")
PIPES_TO_REMOVE = ["coref", "span_resolver", "span_cleaner", "entity_linker"]

REVS = {
    "small": "0eadea074d5f637e76357c46bbd56451471d0154",
    "medium": "e7dba91f855a1d26679dc1ef3aa49f7874b50543",
    "large": "963232f378190476503a1bfc35b520cb142e9e41",
}

BASE.mkdir(parents=True, exist_ok=True)

for size, rev in REVS.items():
    name = f"da_dacy_{size}_trf"
    whl = BASE / f"{name}.whl"

    if not whl.exists():
        url = (
            f"https://huggingface.co/chcaa/{name}/resolve/{rev}/"
            f"{name}-any-py3-none-any.whl"
        )
        print(f"downloading {size}...")
        urllib.request.urlretrieve(url, whl)
        print(f"  {whl.stat().st_size / 1e6:.0f} MB")
        with zipfile.ZipFile(whl) as z:
            z.extractall(BASE / size)

    hits = sorted((BASE / size).glob("**/config.cfg"))
    if not hits:
        print(f"{size}: extracted but no model dir found")
        continue
    src = hits[0].parent

    nlp = spacy.load(src)
    for pipe in PIPES_TO_REMOVE:
        if pipe in nlp.pipe_names:
            nlp.remove_pipe(pipe)

    dst = OUT / f"dacy_{size}_old" / "model-best"
    dst.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(dst)
    print(f"{size}: {nlp.meta['version']} -> {dst} | {nlp.pipe_names}")
    