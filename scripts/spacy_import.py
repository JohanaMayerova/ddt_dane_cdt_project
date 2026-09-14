import spacy
from pathlib import Path

models = ["da_core_news_trf"]
pipes_to_remove = ["attribute_ruler"]

for model in models:
    nlp = spacy.load(model)
    for pipe in pipes_to_remove:
        if pipe in nlp.pipe_names:
            nlp.remove_pipe(pipe)
    out = Path(f"./training/spacy_{model}/model-best")
    out.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out)