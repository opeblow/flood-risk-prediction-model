import torch
import torch.nn as nn
from .dataset import train_set
from .data_preprocessing import (
    ord_enc,
    text_enc,
    numerical_features,
    ordinal_features,
    onehot_features,
    label_enc
)

def make_embedding(num_categories: int, dim: int) -> nn.Embedding:
    emb = nn.Embedding(num_embeddings=num_categories + 1, embedding_dim=dim)
    nn.init.xavier_uniform_(emb.weight)#initialize weights using xavier uniform initialization
    return emb



class FloodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_emb = make_embedding(
            num_categories=len(text_enc.categories_[0]), dim=8
        )
        self.ord_embs = nn.ModuleList(
            [
                make_embedding(len(cat), dim=4)
                for cat in ord_enc.categories_
            ]
        )
        self.onehot_proj=nn.Linear(len(onehot_features),8)
        onehot_dim = train_set.onehots.shape[1]
        num_dim = len(numerical_features)
        concat_dim = 8 + 4 * len(ordinal_features) + onehot_dim + num_dim
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(label_enc.classes_)),
        )

    def forward(self, text, ord, onehot, num):
        x = [self.text_emb(text)]
        for i, emb in enumerate(self.ord_embs):
            x.append(emb(ord[:, i]))
        x.append(onehot)
        x.append(num)
        x = torch.cat(x, dim=1)
        return self.net(x)
    
