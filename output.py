import argparse
import os
import pickle

import torch

from dep_tree.similarity import output as dep_output
from sentence_bert.similarity import output as sb_output
from syntax.similarity import output as syn_output
from w2v.similarity import output as w2v_output


def inference(passage, weight):
    authors = pickle.load(open("data/authors.pkl", "rb"))

    dep = dep_output(passage, authors)[0]
    sb = sb_output(passage, authors)[0]
    syn = syn_output(passage, authors)[0]
    w2v = w2v_output(passage, authors)[0]

    comb = torch.stack([dep, sb, syn, w2v])
    weighted = torch.matmul(weight, comb).squeeze(0)
    percent_sim = torch.round(100 * (weighted + 1) / 2, decimals=2)
    closest_idx = torch.argmax(percent_sim)

    return authors[closest_idx], round(percent_sim[closest_idx].item(), 2)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_dep",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--weight_sb",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--weight_syn",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--weight_w2v",
        type=float,
        default=0.25,
    )
    args = parser.parse_args()
    weights = torch.softmax(
        torch.tensor(
            [args.weight_dep, args.weight_sb, args.weight_syn, args.weight_w2v],
            dtype=torch.float32,
        ),
        dim=0,
    )

    # get user input froom the terminal
    passage = input("Enter a passage: ")

    author, score = inference(passage, weights)
    print(f"Author: {author}")
    print(f"Score: {score}")
