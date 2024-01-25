import os
import pickle

import numpy as np
import torch

from .embed_passages import *


def output(input, authors):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "dep_embeddings.pkl")

    embeddings = pickle.load(open(dfp, "rb"))
    user_passage_embedding = tree_embeddings(extract_dependency_trees([input]))

    cos_scores = torch.nn.functional.cosine_similarity(
        embeddings,
        user_passage_embedding,
    )

    percent_sim = torch.round(100 * (cos_scores + 1) / 2, decimals=2)
    closest_idx = torch.argmax(percent_sim)

    return cos_scores, [authors[closest_idx], percent_sim[closest_idx].item()]
