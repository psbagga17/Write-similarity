from collections import Counter, defaultdict

import numpy as np
import spacy
import torch

nlp = spacy.load("en_core_web_sm")

import os
import pickle


def extract_dependency_trees(passages):
    all_trees = []

    for text in passages:
        doc = nlp(text)
        passage_tree = []

        for sent in doc.sents:
            tree = []

            for token in sent:
                node = (token.pos_, token.dep_)
                tree.append(node)

            passage_tree.append(tree)

        all_trees.append(passage_tree)

    return all_trees


def tree_embeddings(trees):
    pos_dep_pairs = defaultdict(lambda: len(pos_dep_pairs))

    for passage_trees in trees:
        for tree in passage_trees:
            for node in tree:
                _ = pos_dep_pairs[node]

    emb_len = 256  # hard coded to avoid ragged tensors
    embeddings = []

    for passage_trees in trees:
        passage_emb = np.zeros(emb_len, dtype=np.float16)

        for tree in passage_trees:
            tree_counter = Counter(tree)
            for node, count in tree_counter.items():
                index = pos_dep_pairs[node]
                passage_emb[index] += count

        embeddings.append(torch.tensor(passage_emb, dtype=torch.float16))

    return torch.stack(embeddings)


def embed_passages(corpus_file_path):
    corpus = pickle.load(open(corpus_file_path, "rb"))
    passages = list(corpus.values())

    embeddings = tree_embeddings(extract_dependency_trees(passages))

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "dep_embeddings.pkl")

    pickle.dump(embeddings, open(dfp, "wb"))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    datafp = os.path.join(current_dir, "../data/author_passages.pkl")

    embed_passages(datafp)
