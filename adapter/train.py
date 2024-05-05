from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiplicativeLR
from model import Adapter, Projecter, Concatenater
from data import *
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
from matplotlib import pyplot as plt
import plotly.express as px

TEST_SPLIT = 0.3
DIMENSION = 1536
EPOCHS = 60
BATCH_SIZE = 16
STARTING_LR = 0.0005
LR_SCHED_RATIO = 0.95
MARGIN = 2.0

DATAPATH = "data.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
    accuracies = []
    for threshold_thousandths in range(-1000, 1000, 1):
        threshold = threshold_thousandths / 1000
        total = 0
        correct = 0
        for cs, ls in zip(cosine_similarity, labeled_similarity):
            total += 1
            if cs > threshold:
                prediction = 1
            else:
                prediction = 0
            if prediction == ls:
                correct += 1
        accuracy = correct / total
        accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
    return a, standard_error

def lossfn(metrics, labels, margin, chunks, og_chunks, queries, og_queries):
    margin = margin.to(device)
    similar = metrics[labels == 1]
    dissimilar = metrics[labels == 0]
    loss = nn.functional.softplus(margin.to(device) + torch.sum(dissimilar) - torch.sum(similar))
    #regularization = torch.tensor([0]).to(device)
    #regularization = chunks.norm() + queries.norm()
    regularization = (chunks - og_chunks).norm().pow(2) + (queries - og_queries).norm().pow(2) * 8
    return loss + regularization, loss, regularization

def test_split(df, size):
    questions = df["question"].unique()
    train_questions = np.random.choice(questions, size=int(len(questions) * (1- size)), replace=False).tolist()
    train_df = df[df["question"].isin(train_questions)]
    test_df = df[~df["question"].isin(train_questions)]
    
    print (f"Unique train questions: {train_df['question'].nunique()}")
    print (f"Unique test questions: {test_df['question'].nunique()}")
    return train_df, test_df

print ("Reading input data")
data_df = read_input_data(DATAPATH)
print (f"Total rows: {len(data_df)}")

# generate negatives for training dataset
df_negatives = dataframe_of_negatives(data_df)

data_df = pd.concat([data_df, df_negatives])

#train_df, test_df = train_test_split(data_df, test_size=TEST_SPLIT)
train_df, test_df = test_split(data_df, size=TEST_SPLIT)

print (f"Train rows: {len(train_df)}")
print (f"Test rows: {len(test_df)}")

train_df.loc[:, "group"] = "train"
test_df.loc[:, "group"] = "test"

data_df = pd.concat([train_df, test_df])

print (f"Total rows: {len(data_df)}")

a, se = accuracy_and_se(train_df["cosine_similarity"], train_df["supporting"])
print (f"Train accuracy: {a:0.1%}")

a, se = accuracy_and_se(test_df["cosine_similarity"], test_df["supporting"])
print (f"Test accuracy: {a:0.1%}")

r, p = pearsonr(train_df["cosine_similarity"], train_df["supporting"])
print (f"Train r: {round(r, 3)}")

r, p = pearsonr(test_df["cosine_similarity"], test_df["supporting"])
print (f"Test r: {round(r, 3)}")

before = px.histogram(
    train_df,
    x="cosine_similarity",
    color="supporting",
    barmode="overlay",
    histnorm="probability",
    nbins=100,
    width=500,
)

before.write_image(f"plots/before_histogram.png")

def get_batch(df, batch_size):
    positive = df[df["supporting"] == 1]
    negative = df[df["supporting"] == 0]
    
    chosen_positive = positive.sample(n=batch_size // 2)
    chosen_negative = negative.sample(n=batch_size // 2)
    
    stacked = pd.concat([chosen_positive, chosen_negative])
        
    e1 = np.stack(np.array(stacked["chunk_embedding"].values))
    e2 = np.stack(np.array(stacked["question_embedding"].values))
    s = np.stack(np.array(stacked["supporting"].astype("float").values))

    e1 = torch.from_numpy(e1).float()
    e2 = torch.from_numpy(e2).float()
    s = torch.from_numpy(s).float()
    
    return e1, e2, s

train_c_embeds, train_q_embeds, train_s = tensors_from_dataframe(train_df, "chunk_embedding", "question_embedding", "supporting")
test_c_embeds, test_q_embeds, test_s = tensors_from_dataframe(test_df, "chunk_embedding", "question_embedding", "supporting")


train_dataset = torch.utils.data.TensorDataset(train_c_embeds, train_q_embeds, train_s)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_dataset = torch.utils.data.TensorDataset(test_c_embeds, test_q_embeds, test_s)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

model = Projecter(DIMENSION)
#model = Concatenater(DIMENSION)
optimizer = torch.optim.Adam(model.parameters(), lr=STARTING_LR)
scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda x: LR_SCHED_RATIO)

model = model.to(device)

epochs, train_losses, test_losses, train_contrastives, train_regs, contrastive_losses, reg_losses = [], [], [], [], [], [], []
for epoch in range(1, 1 + EPOCHS):
    # iterate through training dataloader
    trains = []
    train_contrastive = []
    train_regularizations = []
    contrastives = []
    regularizations = []
    for i in range(len(train_df) // BATCH_SIZE):
        chunks, queries, actual_similarity = get_batch(train_df, BATCH_SIZE)
        chunks = chunks.to(device)
        queries = queries.to(device)
        actual_similarity = actual_similarity.to(device)
        # generate prediction
        chunk_embeds, query_embeds, predicted_similarity = model(chunks, queries)
        # get loss and perform backpropagation
        loss, contrastive, reg = lossfn(predicted_similarity, actual_similarity, torch.tensor([MARGIN]), chunk_embeds, chunks, query_embeds, queries)
        train_contrastive.append(contrastive.item())
        train_regularizations.append(reg.item())
        trains.append(loss.item())

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    tests = []
    with torch.no_grad():
      for i in range(len(test_df) // BATCH_SIZE):
        chunks, queries, actual = get_batch(test_df, BATCH_SIZE)
        chunks = chunks.to(device)
        queries = queries.to(device)
        actual = actual.to(device)
        chunk_embeds, query_embeds, predicted_similarity = model(chunks, queries)
        # get loss and perform backpropagation
        test_loss, contrastive, reg = lossfn(predicted_similarity, actual, torch.tensor([MARGIN]), chunk_embeds, chunks, query_embeds, queries)
        tests.append(test_loss.item())
        contrastives.append(contrastive.item())
        regularizations.append(reg.item())

    epochs.append(epoch)
    train_losses.append(np.mean(np.array(trains)))
    test_losses.append(np.mean(np.array(tests)))
    train_contrastives.append(np.mean(np.array(train_contrastive)))
    train_regs.append(np.mean(np.array(train_regularizations)))
    contrastive_losses.append(np.mean(np.array(contrastives)))
    reg_losses.append(np.mean(np.array(regularizations)))
    print(
      f"Epoch {epoch}/{EPOCHS}: Train Loss: {round(np.mean(np.array(trains)), 3)}, Test Loss: {round(np.mean(np.array(tests)), 3)}"
    )
    
    scheduler.step()
    
# plt.plot(epochs, train_losses, color="blue", label="Train MSE Loss")
# plt.plot(epochs, test_losses, color="orange", label="Test MSE Loss")
plt.plot(epochs, train_contrastives, color="blue", label="Train Contrastive Loss")
#plt.plot(epochs, train_regs, color="purple", label="Train Regularization Loss")
plt.plot(epochs, contrastive_losses, color="green", label="Test Contrastive Loss")
#plt.plot(epochs, reg_losses, color="black", label="Test Regularization Loss")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Contrastive Loss")
plt.savefig("plots/training.png", dpi=300)

plt.clf()

# compute custom embeddings and new cosine similarities
def apply_matrix_to_embeddings_dataframe(model, df: pd.DataFrame):
    custom_chunks = []
    custom_questions = []
    for rowno in tqdm(range(len(df))):
        c, q, s = tensors_from_dataframe(df.iloc[[rowno]], "chunk_embedding", "question_embedding", "supporting")
        c = c.to(device)
        q = q.to(device)
        c_c, c_q, _ = model(c, q)
        custom_chunks.append(c_c.cpu().detach().numpy().squeeze())
        custom_questions.append(c_q.cpu().detach().numpy().squeeze())


    df[f"chunk_custom"] = custom_chunks
    df[f"question_custom"] = custom_questions
    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(
            row["chunk_custom"], row["question_custom"]
        ),
        axis=1,
    )
    
    df[f"chunk_custom"] = df["chunk_custom"].apply(lambda value: value.tolist())
    df[f"question_custom"] = df["question_custom"].apply(lambda value: value.tolist())


apply_matrix_to_embeddings_dataframe(model, data_df)
apply_matrix_to_embeddings_dataframe(model, train_df)
apply_matrix_to_embeddings_dataframe(model, test_df)

a, se = accuracy_and_se(train_df["cosine_similarity_custom"], train_df["supporting"])
print(f"Train accuracy after customization: {a:0.1%} ± {1.96 * se:0.1%}")

r, p = pearsonr(train_df["cosine_similarity_custom"], train_df["supporting"])
print (f"Train r: {round(r, 3)}")

train_after = px.histogram(
    train_df,
    x="cosine_similarity_custom",
    color="supporting",
    barmode="overlay",
    width=500,
    nbins=100,
    title="Training Set",
    histnorm="probability"
)

train_after.write_image(f"plots/train_after_histogram.png")

a, se = accuracy_and_se(test_df["cosine_similarity_custom"], test_df["supporting"])
print(f"Test accuracy after customization: {a:0.1%} ± {1.96 * se:0.1%}")

r, p = pearsonr(test_df["cosine_similarity_custom"], test_df["supporting"])
print (f"Test r: {round(r, 3)}")

test_after = px.histogram(
    test_df,
    x="cosine_similarity_custom",
    color="supporting",
    barmode="overlay",
    width=500,
    nbins=100,
    title="Test Set",
    histnorm="probability"
)

test_after.write_image(f"plots/test_after_histogram.png")

data_df.to_csv("output_df.csv", index=False)