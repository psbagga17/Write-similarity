import pickle

import numpy as np
import spacy
import torch
from gensim.models import Word2Vec

nlp = spacy.load("en_core_web_sm")

import os


def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    tokens = [token.text.lower() for token in doc if not token.is_stop]

    return tokens


def w2v(passages):
    vecs = []

    for passage in passages:
        tokens = preprocess(passage)
        model = Word2Vec(
            sentences=[tokens], vector_size=128, window=4, min_count=1, workers=4
        )

        passage_vector = np.mean(
            [model.wv[token] for token in tokens if token in model.wv], axis=0
        )
        passage_vector = torch.tensor(passage_vector, dtype=torch.float16)

        vecs.append(passage_vector)

    return torch.stack(vecs)


def embed_passages(corpus_file_path):
    corpus = pickle.load(open(corpus_file_path, "rb"))
    passages = list(corpus.values())

    vecs = w2v(passages)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "w2v_embeddings.pkl")

    pickle.dump(vecs, open(dfp, "wb"))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    datafp = os.path.join(current_dir, "../data/author_passages.pkl")

    embed_passages(datafp)
