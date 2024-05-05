from pinecone import Pinecone
from ast import literal_eval
import pandas as pd
from matplotlib import pyplot as plt

pc = Pinecone(api_key="19fdac30-ddb3-4b17-a441-fe620e0714d7")

base_index_name = "base-embeddings-index"
adapted_index_name = "adapted-embeddings-index"

base_index = pc.Index(base_index_name)
adapted_index = pc.Index(adapted_index_name)

df = pd.read_csv("output_df.csv")
test_df = df[df["group"] == "train"]
print (f"Total rows: {len(test_df)}")
test_df["question_embedding"] = test_df["question_embedding"].apply(literal_eval)
test_df["question_custom"] = test_df["question_custom"].apply(literal_eval)
print ("Converted df")

K_list = [2, 3, 4, 5, 7, 9, 12]
base_hit_ps = []
adapted_hit_ps = []

for K in K_list:

    unique_questions = set()

    base_hits = 0
    adapted_hits = 0
    total_questions = 0

    for rowno in test_df.index:
        question = test_df.loc[rowno, "question"]
        if question in unique_questions:
            continue
        unique_questions.add(question)
        question_positives = df[(df["question"] == question) & (df["supporting"] == 1)]
        positive_texts = question_positives["chunk"].values
        
        base_embedding = test_df.loc[rowno, "question_embedding"]
        
        base_result = base_index.query(
                            vector=base_embedding,
                            top_k=K,
                            include_values=False,
                            include_metadata=True
                        )
        base_matches = base_result["matches"]
        base_texts = [match["metadata"]["text"] for match in base_matches]
        if (positive_texts[0] in base_texts and positive_texts[1] in base_texts):
            base_hits += 1
            
        adapted_embedding = test_df.loc[rowno, "question_custom"]
        
        adapted_result = adapted_index.query(
                            vector=adapted_embedding,
                            top_k=K,
                            include_values=False,
                            include_metadata=True
                        )
        adapted_matches = adapted_result["matches"]
        adapted_texts = [match["metadata"]["text"] for match in adapted_matches]
        if (positive_texts[0] in adapted_texts and positive_texts[1] in adapted_texts):
            adapted_hits += 1
            
        total_questions += 1
        
    print (f"K: {K}")
    print (f"Base Hit %: {base_hits / total_questions}")
    base_hit_ps.append(base_hits / total_questions)
    print (f"Adapted Hit %: {adapted_hits / total_questions}")
    adapted_hit_ps.append(adapted_hits / total_questions)
    
plt.plot(K_list, base_hit_ps, color="black", label="Base Embeddings")
plt.plot(K_list, adapted_hit_ps, color="orange", label="Adapted Embeddings")
plt.xlabel("K (number of retrieved chunks)")
plt.ylabel("Hit Rate")
plt.legend()
plt.savefig("plots/hit_rate.png", dpi=300)        
        