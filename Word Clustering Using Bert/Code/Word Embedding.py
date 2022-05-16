# -*- coding: utf-8 -*-
"""
Author: LiGorden
"""
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

from sklearn.manifold import TSNE
import networkx as nx
import Calculate_Centrality
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import random

df = pd.read_csv("../RawData/Comment.csv")
texts = list(df['Chief_Complaint'])
texts_vector = list()

# Load pre-trained model (weights)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,)  # Whether the model returns all hidden-states.
model = model.to(device)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for i in range(len(texts)):
    print(i)
    text = texts[i]
    marked_text = "[CLS] " + text + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    """
    # Display the words with their indeces.
    a = zip(tokenized_text, indexed_tokens)
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    """

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

    """
    print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")

    layer_i = 0
    print("Number of batches:", len(hidden_states[layer_i]))

    batch_i = 0
    print("Number of tokens:", len(hidden_states[layer_i][batch_i]))

    token_i = 0
    print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    # `hidden_states` is a Python list.
    print('      Type of hidden_states: ', type(hidden_states))

    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', hidden_states[0].size())
    """
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings.size()

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # token_embeddings.size()

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)
    # token_embeddings.size()

    """
    # Word Vector-Try Concating last four layers
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

    # Word Vector-Try Summing last four layers
    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.
    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
    """

    # Sentence embedding
    # `hidden_states` has shape [13 x 1 x 4 x 768]
    # `token_vecs` is a tensor with shape [4 x 768]
    token_vecs = hidden_states[1][0]
    # token_vecs = torch.mean(token_embeddings[:, 1:, :], dim=1)
    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    texts_vector.append(sentence_embedding.cpu().numpy().tolist())

df_texts_vector = pd.DataFrame(texts_vector)
df_texts_vector.to_csv('vector_unique.csv', encoding='utf-8')

# Visualize result of word embedding using t-SNE
X = np.array(df_texts_vector)
# Transferring to location in tsne 2d plane
tsne = TSNE(n_components=2, method='barnes_hut', metric='euclidean',
            init='random', random_state=0)
x_tsne = tsne.fit_transform(X)  # return tuple of location in T-SNE 2D plane

plt.figure()
plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
plt.show()

# Using GMM to identify clusters based on result of t-SNE
clf = GaussianMixture(n_components=8)
clf.fit(x_tsne)
result = clf.predict(x_tsne)
plt.figure()
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=result)
plt.show()

plt.figure()
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=result)
random.seed(1)
sample_list = random.sample(range(0, 785), 80)
for i in sample_list:
    plt.annotate(s=df.iloc[i, 0], xy=(x_tsne[i, 0], x_tsne[i, 1]), xytext=(x_tsne[i, 0] + 0.3, x_tsne[i, 1] + 0.3))
plt.show()

df_GMM_result = pd.DataFrame(result, columns=['Cluster_No'])
df_embedding = pd.concat([df, df_GMM_result, df_texts_vector], axis=1)
df_embedding.to_csv('embedding_result.csv', encoding='utf-8', index=None)

# Visualize result of word embedding using Network Science
df_network = pd.concat([df, df_texts_vector], axis=1)
df_network.set_index(["Chief_Complaint"], inplace=True)
interaction_nodes = Calculate_Centrality.find_interaction(df_network, method='euclidean', threshold=9.5, feature=False)
centrality, graph_info, nodes_pos_dict = Calculate_Centrality.draw_topology_graph(df_network, interaction_nodes,
                                                                                  layout='spring', feature=False,
                                                                                  label=False)
