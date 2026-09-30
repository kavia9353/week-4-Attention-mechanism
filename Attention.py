import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

print("Attention Mechanism Project")
print("---------------------------")

# Load sentences
with open("dataset/sample_sentences.txt", "r", encoding="utf-8") as file:
    sentences = [line.strip() for line in file if line.strip()]

print("\nNumber of sentences:", len(sentences))

for i, sentence in enumerate(sentences, start=1):
    print(f"{i}. {sentence}")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate sentence embeddings
embeddings = model.encode(sentences)

print("\nEmbedding shape:", embeddings.shape)

# Save embeddings
np.save("dataset/embeddings.npy", embeddings)

print("Embeddings saved successfully.")

# Load the saved embeddings
embeddings = np.load("dataset/embeddings.npy")

# Set random seed for reproducibility
np.random.seed(42)

# Embedding dimension
embedding_dim = embeddings.shape[1]

# Create transformation matrices
W_Q = np.random.rand(embedding_dim, embedding_dim)
W_K = np.random.rand(embedding_dim, embedding_dim)
W_V = np.random.rand(embedding_dim, embedding_dim)

# Calculate Query, Key and Value matrices
Q = embeddings @ W_Q
K = embeddings @ W_K
V = embeddings @ W_V

print("\nQ shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

# Calculate attention scores
attention_scores = Q @ K.T

print("\nAttention Scores Shape:", attention_scores.shape)
print("\nAttention Scores:")
print(attention_scores)

# Scale the attention scores
d_k = K.shape[1]

scaled_attention_scores = attention_scores / np.sqrt(d_k)

print("\nScaled Attention Scores:")
print(scaled_attention_scores)

# Calculate attention scores
attention_scores = Q @ K.T

print("\nAttention Scores Shape:", attention_scores.shape)
print("\nAttention Scores:")
print(attention_scores)

# Scale the attention scores
d_k = K.shape[1]

scaled_attention_scores = attention_scores / np.sqrt(d_k)

print("\nScaled Attention Scores:")
print(scaled_attention_scores)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Calculate attention weights
attention_weights = softmax(scaled_attention_scores)

print("\nAttention Weights:")
print(attention_weights)

print("\nAttention Weights Shape:", attention_weights.shape)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Calculate attention weights
attention_weights = softmax(scaled_attention_scores)

print("\nAttention Weights:")
print(attention_weights)

print("\nAttention Weights Shape:", attention_weights.shape)
# Save attention weights as CSV
attention_weights_df = pd.DataFrame(attention_weights)

attention_weights_df.to_csv(
    "dataset/attention_weights.csv",
    index=False
)

print("\nAttention weights saved to dataset/attention_weights.csv")

# Calculate the final attention output
attention_output = attention_weights @ V

print("\nAttention Output:")
print(attention_output)

print("\nAttention Output Shape:", attention_output.shape)

# Save attention weights as CSV
attention_weights_df = pd.DataFrame(attention_weights)

attention_weights_df.to_csv(
    "dataset/attention_weights.csv",
    index=False
)

print("\nAttention weights saved to dataset/attention_weights.csv")

# Save attention output as CSV
attention_output_df = pd.DataFrame(attention_output)

attention_output_df.to_csv(
    "dataset/attention_output.csv",
    index=False
)

print("Attention output saved to dataset/attention_output.csv")

pd.DataFrame(attention_output)
attention_output = attention_weights @ V
# Create attention heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    attention_weights,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)

plt.title("Attention Weights Heatmap")
plt.xlabel("Key Sentences")
plt.ylabel("Query Sentences")
plt.tight_layout()

plt.savefig("dataset/attention_heatmap.png")
plt.show()

pd.DataFrame(attention_output).to_csv("dataset/attention_output.csv", index=False)
print("Attention output saved to dataset/attention_output.csv")
with open("dataset/sample_sentences.txt", "r", encoding="utf-8") as file:
    text = file.read()
