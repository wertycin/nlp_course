import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size=1000, emb_size=100):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
    def forward(self, x):
        x = self.emb(x)
        return torch.einsum('nk,vk->nv', x, self.emb.weight)
    
def word2idx_fit(inp: str):
    word2idx = {}
    for word in inp.split():
        if not word in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx

def word2idx_transform(word2idx: dict, inp: str):
    data = [word2idx[word] for word in inp.split()]
    return torch.tensor(data)

def word2idx_fit_transform(inp: str):
    word2idx = word2idx_fit(inp)
    data = word2idx_transform(word2idx, inp)
    return data

def get_batch(tokens: torch.tensor, window_size: int = 2):
    batch = []
    for i in range(len(tokens)):
        start = max(0, i - window_size)
        stop = min(len(tokens) - 1, i + window_size)
        for j in range(start, stop + 1):
            if i == j:
                continue
            batch.append([tokens[i], tokens[j]])
    return torch.tensor(batch).T

def train(doc: str, return_history=False):
    word2idx = word2idx_fit(doc)
    tokens = word2idx_transform(word2idx, doc)
    vocab_size = len(word2idx)
    word2vec = Word2Vec(vocab_size=vocab_size, emb_size=100)
    opt = torch.optim.Adam(word2vec.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    n_epohs = 10
    loss_history = []
    for i in range(n_epohs):
        opt.zero_grad()
        x, y_gt = get_batch(tokens)
        y_pred = word2vec(x)
        loss = loss_fn(y_pred, y_gt)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
    embs = word2vec.emb.weight.detach().numpy()
    word2vec_dict = {}
    for word in word2idx.keys():
        word2vec_dict[word] = embs[word2idx[word]]
    if return_history:
        return word2vec_dict, loss_history
    return word2vec_dict