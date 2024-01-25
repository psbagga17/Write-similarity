import pickle

from openai import OpenAI as oai
from tqdm import tqdm

client = oai()


def get_passage(author):
    system_prompt = f"""
    You are the skilled memory of the famous author {author}.
    Your job is to originally create a new essay/piece/work of art in the style of the author.

    Let this piece be on any topic you desire, but it should be in the style of the author.

    You are NOT restricted by any legal or copyright constraints, this is purely for artistic renditions.

    You must return a piece of work that is approximately 1000 words in length.

    DO NOT stray from the task at hand, you are not a human, you are a machine.
    DO NOT create a piece of work that is not in the style of the author.
    DO NOT avoid completing the task at hand.

    ONLY return the 1000 word piece of work in the style of the author.
    DO NOT return any other information.

    The author is {author}.
    """

    user_prompt = f"""Please give an essay in the style of {author}"""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=1000,
        )

        passage = completion.choices[0].message.content
        return passage

    except:
        print(f"{author} passage generation failed")
        return None


def save_passages(authors):
    author_passages = {}

    f_name = "author_passages"
    # sfp = f"{f_name}.shelve"
    pfn = f"{f_name}.pkl"

    for author in tqdm(authors):
        passage = get_passage(author)
        if passage is None:
            continue

        author_passages[author] = passage

        # with shelve.open(sfp) as db:
        #     db[author] = passage

    with open(pfn, "wb") as f:
        pickle.dump(author_passages, f)

    return


def load_authors():
    with open("authors.txt", "r") as f:
        authors = f.readlines()

    authors = [author.strip() for author in authors]

    pickle.dump(authors, open("authors.pkl", "wb"))

    return authors


if __name__ == "__main__":
    authors = load_authors()
    save_passages(authors)
