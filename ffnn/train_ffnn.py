from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from data.feature_extraction import preprocess_features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class NNClassifier(L.LightningModule):
    def __init__(
            self,
            input_dim=200,
            hidden_dims=[128,64,32],
            num_classes=10,
            dropout=0.5,
            lr=1e-4,
            class_weights=None
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.class_weights = class_weights

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.out = nn.Linear(hidden_dims[2], num_classes)
        self.dropout = nn.Dropout(dropout)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        acc = self.accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

n_features_select = 200
def main():
    res = preprocess_features.main(n_features_select)
    label_encoder = res['label_encoder']
    print("---- finished getting dataframes ---")
    print(res["X_train"].shape)

    X_train_tensor = torch.tensor(res["X_train"].values, dtype=torch.float32)
    print(X_train_tensor.shape)
    y_train_tensor = torch.tensor(res["y_train"], dtype=torch.int)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(res["y_train"]),
        y=res["y_train"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    X_val_tensor = torch.tensor(res["X_val"].values, dtype=torch.float32)
    y_val_tensor = torch.tensor(res["y_val"], dtype=torch.int)

    X_test_tensor = torch.tensor(res["X_test"].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(res["y_test"], dtype=torch.int)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Instantiate model
    num_classes = len(label_encoder.classes_)
    print("num_classes: ", num_classes)
    # input dim is number of features, make sure to sync to preprocess_features.py
    model = NNClassifier(
        input_dim=n_features_select,
        num_classes=num_classes,
        class_weights=class_weights_tensor
    )

    trainer = L.Trainer(max_epochs=50, accelerator="mps")
    trainer.fit(model, train_loader, val_loader)

    print("testing")
    trainer.test(model, test_loader)
    

if __name__ == '__main__':
    main()