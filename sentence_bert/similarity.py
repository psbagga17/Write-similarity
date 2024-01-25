import os
import pickle

import torch
from sentence_transformers import util

from .embed_passages import get_embedding


def output(user_passage, authors):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "sb_embeddings.pkl")

    embeddings = pickle.load(open(dfp, "rb"))
    user_passage_embedding = get_embedding(user_passage)

    cos_scores = util.pytorch_cos_sim(embeddings, user_passage_embedding).squeeze(-1)

    percent_sim = torch.round(100 * (cos_scores + 1) / 2, decimals=2)
    closest_idx = torch.argmax(percent_sim)

    return cos_scores, [authors[closest_idx], percent_sim[closest_idx].item()]
