import pandas as pd
import random
import numpy as np
from openai import OpenAI
import os
import pickle
from typing import List, Tuple
import torch
from ast import literal_eval
from datasets import load_dataset

embedding_cache_path = "snli_embedding_cache.pkl"  # embeddings will be saved/loaded here
default_embedding_engine = "text-embedding-3-small" # can be any OpenAI model

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(max_retries=5)

try:
    with open(embedding_cache_path, "rb") as f:
        embedding_cache = pickle.load(f)
except FileNotFoundError:
    precomputed_embedding_cache_path = "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
    embedding_cache = pd.read_pickle(precomputed_embedding_cache_path)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding

# this function will get embeddings from the cache and save them there afterward
def get_embedding_with_cache(
    text: str,
    engine: str = default_embedding_engine,
    embedding_cache: dict = embedding_cache,
    embedding_cache_path: str = embedding_cache_path,
) -> list:
    if (text, engine) not in embedding_cache.keys():
        # if not in cache, call API to get embedding
        embedding_cache[(text, engine)] = get_embedding(text, engine)
        # save embeddings cache to disk after each update
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(text, engine)]

def read_input_data(path: str):
    try:
        with open("data.csv", "r") as f:
            data_df = pd.read_csv("data.csv").iloc[:, 1:]
            data_df["supporting"] = data_df["supporting"].apply(lambda x: 0 if x == -1 else x)
            data_df["question_embedding"] = data_df["question_embedding"].apply(literal_eval)
            data_df["chunk_embedding"] = data_df["chunk_embedding"].apply(literal_eval)
            return data_df
    except:
        dataset = []

        hpqa = load_dataset("hotpot_qa", "fullwiki", split = 'train')
        NUM_QUESTIONS = 3000
        questions_counter = 0

        for row_no in range(len(hpqa)):
            row = hpqa[row_no]
            difficulty = row["level"]
            if (difficulty != "hard"):
                continue
            if (questions_counter >= NUM_QUESTIONS):
                break
            questions_counter += 1

            row = hpqa[row_no]

            question = row["question"]
            answer = row["answer"]
            supporting_facts = row["supporting_facts"]
            context = row["context"]

            titles = context["title"]
            useful_titles = set(supporting_facts["title"])
            not_useful_titles = set(titles) - useful_titles

            # Map context titles to list of sentences from those titles
            zipped_context = dict(zip(context["title"], context["sentences"]))

            # Get the list of sentences for each title that is useful
            useful_chunks = [zipped_context[title] for title in useful_titles]

            distractor_chunks = [zipped_context[title] for title in not_useful_titles]

            # Join the sentences together to create one paragraph chunk per title
            joined_useful = ["".join(sentences) for sentences in useful_chunks]
            joined_distractor = ["".join(sentences) for sentences in distractor_chunks]

            for chunk in joined_useful:
                dataset.append([question, chunk, 1])
            for chunk in joined_distractor[:len(joined_useful)]:
                dataset.append([question, chunk, 0])

        data_df = pd.DataFrame(dataset, columns=["question", "chunk", "supporting"])
        print (f"Total samples: {len(data_df)}")
        print (f"Number positive: {len(data_df[data_df['supporting'] == 1])}")
        print (f"Number positive: {len(data_df[data_df['supporting'] == -1])}")
        
        precomputed_embedding_cache_path = "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
        embedding_cache = pd.read_pickle(precomputed_embedding_cache_path)

        # create column of embeddings
        for column in ["question", "chunk"]:
            data_df[f"{column}_embedding"] = data_df[column].apply(get_embedding_with_cache)

        # create column of cosine similarity between embeddings
        data_df["cosine_similarity"] = data_df.apply(
            lambda row: cosine_similarity(row["question_embedding"], row["chunk_embedding"]),
            axis=1,
        )

        data_df.to_csv("data.csv")
        return data_df

def dataframe_of_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of negative pairs made by combining elements of positive pairs."""
    negative_rows = []

    dropped = df.drop_duplicates("question")

    for row in dropped.index:
      question = dropped.loc[row, "question"]
      not_this_question = df[df["question"] != question]
      for i in range(1):
        index = random.randint(0, len(not_this_question) - 1)
        neg_row = not_this_question.iloc[index]
        negative_rows.append([question, neg_row["chunk"], dropped.loc[row, "question_embedding"], neg_row["chunk_embedding"], cosine_similarity(dropped.loc[row, "question_embedding"], neg_row["chunk_embedding"])])

    neg_df = pd.DataFrame(list(negative_rows), columns=["question", "chunk", "question_embedding", "chunk_embedding", "cosine_similarity"])
    neg_df["supporting"] = 0
    return neg_df

def tensors_from_dataframe(
        df: pd.DataFrame,
        embedding_column_1: str,
        embedding_column_2: str,
        similarity_label_column: str,
    ) -> Tuple[torch.tensor]:

  e1 = np.stack(np.array(df[embedding_column_1].values))
  e2 = np.stack(np.array(df[embedding_column_2].values))
  s = np.stack(np.array(df[similarity_label_column].astype("float").values))

  e1 = torch.from_numpy(e1).float()
  e2 = torch.from_numpy(e2).float()
  s = torch.from_numpy(s).float()

  return e1, e2, s