import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data.feature_extraction import preprocess_features
from ffnn.nn_classifier import NNClassifier
from functools import partial

_prepare_data_cache = {}

# n_features_select = 200

def prepare_data(num_features):
    if num_features in _prepare_data_cache:
        return _prepare_data_cache[num_features]

    res = preprocess_features.main(num_features)
    label_encoder = res['label_encoder']

    X_train_tensor = torch.tensor(res["X_train"].values, dtype=torch.float32)
    y_train_tensor = torch.tensor(res["y_train"], dtype=torch.int64)

    X_val_tensor = torch.tensor(res["X_val"].values, dtype=torch.float32)
    y_val_tensor = torch.tensor(res["y_val"], dtype=torch.int64)

    X_test_tensor = torch.tensor(res["X_test"].values, dtype=torch.float32)
    y_test_tensor = torch.tensor(res["y_test"], dtype=torch.int)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(res["y_train"]),
        y=res["y_train"]
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    result = train_loader, val_loader, test_loader, len(label_encoder.classes_), class_weights_tensor, label_encoder

    _prepare_data_cache[num_features] = result
    return result

def objective(trial):
    # Tune feature count: 100 to 1000 in steps of 100
    n_features_select = trial.suggest_int("n_features_select", 100, 1000, step=100)

    # Prepare data based on number of selected features
    train_loader, val_loader, _, num_classes, class_weights_tensor, _ = prepare_data(n_features_select)

    # hidden_dims = []
    # num_layers = trial.suggest_int("num_layers", 1, 2)
    # for i in range(num_layers):
    #     hidden_dims.append(trial.suggest_categorical(f"hidden_{i}", [2 ** x for x in range(4, 9)]))
    # Tune hidden dimension from 16 to 256
    hidden_dim = trial.suggest_categorical("hidden_dim", [2 ** x for x in range(5, 10)])

    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1) # step of 0.1
    lr = trial.suggest_categorical("lr", [0.00001, 0.0001, 0.001])

    model = NNClassifier(
        input_dim=n_features_select,
        # hidden_dims=hidden_dims,
        hidden_dims=[hidden_dim],
        num_classes=num_classes,
        dropout=dropout,
        lr=lr,
        class_weights=class_weights_tensor,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints",
        filename=f"trial_{trial.number}_best",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=5,
        mode="max",
        verbose=True
    )

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
        logger=False,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    val_result = trainer.validate(model, val_loader)[0]

    trial.set_user_attr("checkpoint_path", checkpoint_callback.best_model_path)
    trial.set_user_attr("n_features", n_features_select)
    return val_result["val_acc"]


if __name__ == "__main__":
    # train_loader, val_loader, test_loader, num_classes, class_weights_tensor, label_encoder = prepare_data(n_features_select)

    # objective_with_data = partial(
    #     objective,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_classes=num_classes,
    #     class_weights_tensor=class_weights_tensor,
    #     num_features=n_features_select
    # )

    # study = optuna.create_study(direction="maximize",
    #                             study_name="ffnn",
    #                             storage="sqlite:///optuna_study.db",
    #                             load_if_exists=True)
    # study.optimize(objective_with_data, n_trials=30)

    study = optuna.create_study(direction="maximize",
                                study_name="ffnn-n-features",
                                storage="sqlite:///optuna_study.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial)

