import pickle

import torch

from .embed_passages import *


def output(input, authors):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "sx_embeddings.pkl")
    dfptags = os.path.join(current_dir, "sx_all_tags.pkl")

    embeddings = pickle.load(open(dfp, "rb"))
    all_tags = pickle.load(open(dfptags, "rb"))

    user_passage_embedding = get_embedding(input, all_tags)

    cos_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings, dtype=torch.float16),
        torch.tensor(user_passage_embedding, dtype=torch.float16),
    )

    percent_sim = torch.round(100 * (cos_scores + 1) / 2, decimals=2)
    closest_idx = torch.argmax(percent_sim)

    return cos_scores, [authors[closest_idx], percent_sim[closest_idx].item()]
