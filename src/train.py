import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader,WeightedRandomSampler
from .dataset import train_set,val_set,test_set
from .model import FloodNet
from .data_preprocessing import label_enc
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    class_counts=np.bincount(train_set.labels)
    class_weights=1.0/(class_counts + 1e-4)
    sample_weights=class_weights[train_set.labels]
    sampler=WeightedRandomSampler(
        weights=torch.tensor(sample_weights,dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader=DataLoader(train_set,batch_size=256,sampler=sampler)
    val_loader=DataLoader(val_set,batch_size=512,shuffle=False)
    test_loader=DataLoader(test_set,batch_size=512,shuffle=False)

    model = FloodNet().to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
    epochs = 40
    patience = 6
    best_f1, best_state, epochs_without_improve = 0.0, None, 0

    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["text"].to(device),
                batch["ord"].to(device),
                batch["onehot"].to(device),
                batch["num"].to(device),
            )
            loss = criterion(out, batch["label"].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    batch["text"].to(device),
                    batch["ord"].to(device),
                    batch["onehot"].to(device),
                    batch["num"].to(device),
                )
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(batch["label"].numpy())
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"Epoch {epoch:02d}  Val-F1 = {val_f1:.4f}")
        if val_f1 > best_f1 + 1e-3:
            best_f1, best_state = val_f1, model.state_dict()
            epochs_without_improve = 0
            torch.save(best_state, "best_floodnet_model.pth")
            print("  ⤷ new best model saved")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("  ⤷ early-stopping")
                break
        
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["text"].to(device),
                batch["ord"].to(device),
                batch["onehot"].to(device),
                batch["num"].to(device),
            )
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_labels.extend(batch["label"].numpy())
    print(
        f"Best Val-F1 = {best_f1:.4f} | "
        f"Test-F1 = {f1_score(test_labels, test_preds, average='weighted'):.4f}"
    )

    try:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay.from_predictions(
            test_labels,
            test_preds,
            display_labels=label_enc.classes_,
            cmap="Blues",
            xticks_rotation=45,
        )
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__=="__main__":
    train_model()






