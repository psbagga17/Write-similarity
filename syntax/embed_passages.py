import os
import pickle
from itertools import chain

import nltk
import numpy as np
from nltk.probability import FreqDist


def tokenize_passages(passages):
    tokenized_passages = [nltk.word_tokenize(passage) for passage in passages]

    # nltk.download("averaged_perceptron_tagger")
    pos_tagged_passages = [nltk.pos_tag(passage) for passage in tokenized_passages]

    return tokenized_passages, pos_tagged_passages


def flatten_tags(pos_tagged_passages):
    return list(
        list(chain.from_iterable([tag for word, tag in passage]))
        for passage in pos_tagged_passages
    )


def create_freq_dist(tags):
    return FreqDist(tags)


def create_vector(fdist, all_tags):
    return [fdist[tag] if tag in fdist else 0 for tag in all_tags]


def get_embedding(input, all_tags):
    # input is just single passage
    input = [input]

    tags = flatten_tags(tokenize_passages(input)[1])[0]
    vec = create_vector(create_freq_dist(tags), all_tags)

    return vec


def embed_passages(corpus_file_path):
    corpus = pickle.load(open(corpus_file_path, "rb"))
    passages = list(corpus.values())

    _, ptp = tokenize_passages(passages)
    tags = flatten_tags(ptp)

    all_tags = list(set([a for b in tags for a in b]))
    all_tags = sorted(all_tags)

    vecs = []
    for t in tags:
        f_d = create_freq_dist(t)
        vec = create_vector(f_d, all_tags)
        vecs.append(vec)

    vecs = np.array(vecs)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfpa = os.path.join(current_dir, "sx_all_tags.pkl")
    dfpb = os.path.join(current_dir, "sx_embeddings.pkl")

    pickle.dump(all_tags, open(dfpa, "wb"))
    pickle.dump(vecs, open(dfpb, "wb"))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    datafp = os.path.join(current_dir, "../data/author_passages.pkl")

    embed_passages(datafp)
