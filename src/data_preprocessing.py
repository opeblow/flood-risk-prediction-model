import math
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ---------------------------------------------------------------------
# 1. DATA --------------------------------------------------------------
# ---------------------------------------------------------------------
csv_path = r"C:\Users\user\Documents\neural_network\data\flood_risk_dataset_final.csv" 
df = pd.read_csv(csv_path)

# 1.1 Clean & transform numerical outliers
for col in ["Elevation", "Rainfall"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(lower=max(q1 - 1.5 * iqr, 0), upper=q3 + 1.5 * iqr)
    mn = df[col].min()
    df[col] = np.log1p(df[col] + abs(mn) + 1e-6)

# 1.2 Feature groups ---------------------------------------------------
text_feature = "ProximityToWaterBody"
numerical_features = ["Elevation", "Rainfall"]
ordinal_features = [
    "Vegetation",
    "Urbanization",
    "Drainage",
    "Slope",
    "StormFrequency",
    "Deforestation",
    "Infrastructure",
    "Encroachment",
    "Season",
]
onehot_features = ["Soil", "Wetlands"]
label_column = "FloodRisk"

# 1.3 Fill NA ----------------------------------------------------------
for col in ordinal_features + onehot_features + [text_feature]:
    df[col] = df[col].fillna("Missing")

# 1.4 Encode y ---------------------------------------------------------
label_enc = LabelEncoder()
df[label_column] = label_enc.fit_transform(df[label_column])

# 1.5 Train/val/test split --------------------------------------------
train_df, tmp_df = train_test_split(
    df, test_size=0.2, stratify=df[label_column], random_state=42
)
val_df, test_df = train_test_split(
    tmp_df, test_size=0.5, stratify=tmp_df[label_column], random_state=42
)

# 1.6 Encoders ---------------------------------------------------------
ord_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
text_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()

ord_enc.fit(train_df[ordinal_features])
text_enc.fit(train_df[[text_feature]])
onehot_enc.fit(train_df[onehot_features])
scaler.fit(train_df[numerical_features])

# Store valid categories for validation
valid_categories = {}
for i, col in enumerate(ordinal_features):
    valid_categories[col] = list(ord_enc.categories_[i])
for i, col in enumerate(onehot_features):
    valid_categories[col] = list(onehot_enc.categories_[i])
valid_categories[text_feature] = list(text_enc.categories_[0])

