import os
import pickle

import numpy as np
import torch

from .embed_passages import w2v


def output(user_passage, authors):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "w2v_embeddings.pkl")

    embeddings = pickle.load(open(dfp, "rb"))
    user_passage_embedding = w2v([user_passage])

    cos_scores = torch.nn.functional.cosine_similarity(
        embeddings,
        user_passage_embedding,
    )

    percent_sim = torch.round(100 * (cos_scores + 1) / 2, decimals=2)
    closest_idx = torch.argmax(percent_sim)

    return cos_scores, [authors[closest_idx], percent_sim[closest_idx].item()]
