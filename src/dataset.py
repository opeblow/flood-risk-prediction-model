import  numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from .data_preprocessing import (
    
    val_df,
    test_df,
    text_feature,
    ordinal_features,
    onehot_features,
    numerical_features,
    label_column,
    text_enc,
    ord_enc,
    onehot_enc,
    scaler,
    train_df
)



class FloodDataset(Dataset):
    def __init__(self, pdf, train=True):
        self.text_code = (
            text_enc.transform(pdf[[text_feature]]).astype(np.int64).squeeze(1)
        )
        self.ordinals = ord_enc.transform(pdf[ordinal_features]).astype(np.int64)
        self.onehots = onehot_enc.transform(pdf[onehot_features]).astype(np.float32)
        self.numerical = scaler.transform(pdf[numerical_features]).astype(np.float32)
        self.labels = pdf[label_column].astype(np.int64).values
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.text_code[idx], dtype=torch.long),
            "ord": torch.tensor(self.ordinals[idx], dtype=torch.long),
            "onehot": torch.tensor(self.onehots[idx], dtype=torch.float),
            "num": torch.tensor(self.numerical[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_set = FloodDataset(train_df, train=True)
val_set = FloodDataset(val_df, train=False)
test_set = FloodDataset(test_df, train=False)

# 2.1 Weighted sampler to fight imbalance -----------------------------
class_counts = np.bincount(train_set.labels)
class_weights = 1.0 / (class_counts + 1e-4)
sample_weights = class_weights[train_set.labels]
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

# 2.2 DataLoaders ------------------------------------------------------
train_loader = DataLoader(train_set, batch_size=256, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

