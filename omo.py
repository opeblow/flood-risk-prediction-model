import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import re
# Load dataset
url = r"C:\Users\USER\Documents\neural_network\flood_risk_dataset_final.csv"
df = pd.read_csv(url)
# Clean outliers in Elevation and Rainfall
Q1 = df[["Elevation", "Rainfall"]].quantile(0.25)
Q3 = df[["Elevation", "Rainfall"]].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df["Elevation"] = df["Elevation"].clip(lower=max(lower_bound['Elevation'], 0), upper=upper_bound['Elevation'])
df["Rainfall"] = df["Rainfall"].clip(lower=max(lower_bound['Rainfall'], 0), upper=upper_bound['Rainfall'])
# Apply log transformation
elevation_min = df["Elevation"].min()
rainfall_min = df["Rainfall"].min()
df["Elevation"] = np.log1p(df["Elevation"] + abs(elevation_min) + 1e-6)
df["Rainfall"] = np.log1p(df["Rainfall"] + abs(rainfall_min) + 1e-6)
# Define feature lists
text_feature = "ProximityToWaterBody"
numerical_features = ["Elevation", "Rainfall"]
ordinal_features = [
    'Vegetation', 'Urbanization', 'Drainage', 'Slope', 'StormFrequency',
    'Deforestation', 'Infrastructure', 'Encroachment', 'Season'
]
ordinal_categories = [
    ['Missing', 'None', 'Sparse', 'Moderate', 'Dense'],
    ['Missing', 'Low', 'Medium', 'High'],
    ['Missing', 'Poor', 'Moderate', 'Good'],
    ['Missing', 'Flat', 'Moderate', 'Steep'],
    ['Missing', 'Rare', 'Occasional', 'Frequent'],  
    ['Missing', 'None', 'Moderate', 'Severe'],
    ['Missing', 'Weak', 'Moderate', 'Strong'],
    ['Missing', 'None', 'Moderate', 'Severe'],  
    ['Missing', 'Dry', 'Transition', 'Rainy']
]
onehot_features = ['Soil', 'Wetlands']
label_column = 'FloodRisk'
# Preprocess categorical features
df[ordinal_features] = df[ordinal_features].fillna('Missing')
df[onehot_features] = df[onehot_features].fillna('Missing')
# Encode target variable
label_encoder = LabelEncoder()
df[label_column] = label_encoder.fit_transform(df[label_column])
# Build vocabulary for text feature
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens if tokens else ['<UNK>']
texts = df[text_feature].fillna('').astype(str).to_list()
vocab = {"<PAD>": 0, "<UNK>": 1}
idx = 2
for tokens in [tokenize(text) for text in texts]:
    for word in tokens:
        if word not in vocab:
            vocab[word] = idx
            idx += 1
# Text augmentation
def augment_tokens(tokens, deletion_prob=0.1):
    if len(tokens) <= 1:
        return tokens
    return [word for word in tokens if random.random() > deletion_prob]
# Encode and pad tokens
def encode_tokens(tokens, vocab, max_length=50):
    idx = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    padded = idx + [0] * (max_length - len(idx))
    return padded[:max_length]
# Split data
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[label_column], random_state=42)
# Fit preprocessors
ordinal = OrdinalEncoder(categories=ordinal_categories)
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()
ordinal.fit(train_df[ordinal_features])
onehot.fit(train_df[onehot_features])
scaler.fit(train_df[numerical_features])
# Custom Dataset
class FloodDataset(Dataset):
    def __init__(self, df, vocab, text_feature, numerical_features, ordinal_features, onehot_features, label_column, scaler, ordinal_enc, onehot_enc, max_length=50, train=True):
        self.texts = df[text_feature].fillna('').astype(str).to_list()
        self.numerical = scaler.transform(df[numerical_features]).astype(np.float32)
        self.ordinal_enc=ordinal_enc
        self.onehot_enc=onehot_enc
        self.ordinals = ordinal_enc.transform(df[ordinal_features]).astype(np.int64)
        self.onehots = onehot_enc.transform(df[onehot_features]).astype(np.float32)
        self.labels = df[label_column].astype(np.int64).values
        self.vocab = vocab
        self.max_length = max_length
        self.train = train
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        tokens = tokenize(self.texts[index])
        if self.train:
            tokens = augment_tokens(tokens)  # Fixed: Assign augmented tokens
        encoded_text = encode_tokens(tokens, self.vocab, self.max_length)
        return {
            'text': torch.tensor(encoded_text, dtype=torch.long),
            'numerical': torch.tensor(self.numerical[index], dtype=torch.float),
            'ordinal': torch.tensor(self.ordinals[index], dtype=torch.long),
            'onehot': torch.tensor(self.onehots[index], dtype=torch.float),
            'label': torch.tensor(self.labels[index], dtype=torch.long)
        }
# Create datasets and dataloaders
train_dataset = FloodDataset(train_df, vocab, text_feature, numerical_features, ordinal_features, onehot_features, label_column, scaler, ordinal, onehot, train=True)
val_dataset = FloodDataset(val_df, vocab, text_feature, numerical_features, ordinal_features, onehot_features, label_column, scaler, ordinal, onehot, train=False)
test_dataset = FloodDataset(test_df, vocab, text_feature, numerical_features, ordinal_features, onehot_features, label_column, scaler, ordinal, onehot, train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define model
class FloodRiskModel(nn.Module):
    def __init__(self, vocab_size, embeded_dim, hidden_dim, numerical_dim, ordinal_dim, onehot_dim, output_dim, dropout=0.3):
        super(FloodRiskModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embeded_dim, padding_idx=0)
        self.lstm = nn.LSTM(embeded_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ord_embed = nn.Embedding(ordinal_dim, 11)  # Adjusted embedding dim
        combined_dim = hidden_dim * 2 + numerical_dim + 11 + onehot_dim
        self.fc1 = nn.Linear(combined_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)
    def forward(self, text, numerical, ordinal, onehot):
        x_embed = self.embedding(text)
        lstm_out, _ = self.lstm(x_embed)
        lstm_feat = torch.mean(lstm_out, dim=1)
        ord_feat = self.ord_embed(ordinal).mean(dim=1)
        combined = torch.cat([lstm_feat, numerical, ord_feat, onehot], dim=1)
        x = self.dropout(torch.relu(self.fc1(combined)))
        return self.fc2(x)
# Initialize model
max_ordinal_value = int(ordinal.transform(train_df[ordinal_features]).max()) + 1
output_dim = len(label_encoder.classes_)
model = FloodRiskModel(
    vocab_size=len(vocab),
    embeded_dim=64,
    hidden_dim=128,
    numerical_dim=len(numerical_features),
    ordinal_dim=max_ordinal_value,
    onehot_dim=train_dataset.onehots.shape[1],
    output_dim=output_dim
)
# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()
epochs = 60
# Early stopping
class EarlyStopping:
    def __init__(self, patience=6, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
early_stopper = EarlyStopping(patience=6, min_delta=0.001)
best_val_loss = float('inf')
best_model_state = None
best_epoch = -1
# Training loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    train_preds = []
    train_labels = []
    for batch in train_loader:
        text_batch = batch['text'].to(device)
        numerical_batch = batch['numerical'].to(device)
        ordinal_batch = batch['ordinal'].to(device)
        onehot_batch = batch['onehot'].to(device)
        labels_batch = batch['label'].to(device)
        optimizer.zero_grad()
        output = model(text_batch, numerical_batch, ordinal_batch, onehot_batch)
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print("Warning: Model output contains NaN or inf, skipping batch")
            continue
        loss = loss_fn(output, labels_batch)
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN or Inf, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
        train_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
        train_labels.extend(labels_batch.cpu().numpy())
    avg_train_loss = total_train_loss / len(train_loader)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    # Validation
    model.eval()
    total_val_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            text_batch = batch['text'].to(device)
            numerical_batch = batch['numerical'].to(device)
            ordinal_batch = batch['ordinal'].to(device)
            onehot_batch = batch['onehot'].to(device)
            labels_batch = batch['label'].to(device)
            val_output = model(text_batch, numerical_batch, ordinal_batch, onehot_batch)
            val_loss = loss_fn(val_output, labels_batch)
            total_val_loss += val_loss.item()
            val_preds.extend(torch.argmax(val_output, dim=1).cpu().numpy())
            val_labels.extend(labels_batch.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    print(f"Epoch {epoch},Train Loss:.{avg_train_loss:.4f},Train F1:{train_f1:.4f},Val Loss:{avg_val_loss:.4f},Val F1:{val_f1:.4f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        best_model_state = model.state_dict()
        torch.save(best_model_state, "best_floodrisk_model.pth")
        print(f"New best model saved at epoch {epoch} with val loss {avg_val_loss:.4f}")
    if early_stopper(avg_val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break
# Save the best model
if best_model_state is not None:
    torch.save(best_model_state, "best_floodrisk_model.pth")
    print(f"Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")
else:
    print("No best model saved as validation loss did not improve.")
# Evaluate on test set
model.load_state_dict(best_model_state)
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_loader:
        text_batch = batch['text'].to(device)
        numerical_batch = batch['numerical'].to(device)
        ordinal_batch = batch['ordinal'].to(device)
        onehot_batch = batch['onehot'].to(device)
        labels_batch = batch['label'].to(device)
        test_output = model(text_batch, numerical_batch, ordinal_batch, onehot_batch)
        test_preds.extend(torch.argmax(test_output, dim=1).cpu().numpy())
        test_labels.extend(labels_batch.cpu().numpy())
test_f1 = f1_score(test_labels, test_preds, average='weighted')
print(f"Test F1-Score: {test_f1:.4f}")