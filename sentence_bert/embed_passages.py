import pickle
import re

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

import os


def passage_to_sentences(passage):
    # regex to split passage into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a.z]\.)(?<=\.|\?|!|\n)\s|\n+", passage)
    return [sentence.strip() for sentence in sentences if sentence]


def get_embedding(input):
    # treating the passage as a large "sentence" to get a single embedding
    # more effective than encoding each sentnce and mean pooling

    # if passing a single passage as str
    flag = False
    if type(input) == str:
        flag = True
        input = [input]

    # num_passages, d_k
    embedding = model.encode(input, convert_to_tensor=True)

    if flag:
        embedding = embedding[0]

    return embedding


def embed_passages(corpus_file_path):
    corpus = pickle.load(open(corpus_file_path, "rb"))
    passages = list(corpus.values())

    embeddings = get_embedding(passages)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dfp = os.path.join(current_dir, "sb_embeddings.pkl")

    pickle.dump(embeddings, open(dfp, "wb"))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    datafp = os.path.join(current_dir, "../data/author_passages.pkl")

    embed_passages(datafp)
