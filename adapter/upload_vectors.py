from pinecone import Pinecone, ServerlessSpec, Index
from openai import OpenAI
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

pc = Pinecone(api_key="19fdac30-ddb3-4b17-a441-fe620e0714d7")

base_index_name = "base-embeddings-index"
adapted_index_name = "adapted-embeddings-index"

if base_index_name not in pc.list_indexes().names():
    # Do something, such as create the index
    pc.create_index(
        name=base_index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print ("Created base index")

if adapted_index_name not in pc.list_indexes().names():
    # Do something, such as create the index
    pc.create_index(
        name=adapted_index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print ("Created adapted index")

df = pd.read_csv("output_df.csv")
print (f"Total rows: {len(df)}")
df["chunk_embedding"] = df["chunk_embedding"].apply(literal_eval)
df["chunk_custom"] = df["chunk_custom"].apply(literal_eval)
print ("Converted df")

before_vectors = []
adapted_vectors = []

before_index = pc.Index(base_index_name)
adapted_index = pc.Index(adapted_index_name)

# Iterate through all rows and insert the chunks into the respective indices
for rowno in tqdm(range(len(df))):
    chunk_text = df.loc[rowno, "chunk"]
    before_embedding = df.loc[rowno, "chunk_embedding"]
    
    before_vectors.append({
        "id": str(hash(chunk_text)),
        "values": before_embedding,
        "metadata": {
            "text": chunk_text
        }
    })
    
    if (len(before_vectors) == 40):
        before_index.upsert(before_vectors)
        before_vectors = []
    
    adapted_embedding = df.loc[rowno, "chunk_custom"]
    adapted_vectors.append({
        "id": str(hash(chunk_text)),
        "values": adapted_embedding,
        "metadata": {
            "text": chunk_text
        }
    })
    
    if (len(adapted_vectors) == 40):
        adapted_index.upsert(adapted_vectors)
        adapted_vectors = []

