import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from nbclient import NotebookClient
from multiprocessing import Pool
import os

def generate_eval_notebook(args):
    n_features_select, checkpoint_path, output_path = args
    nb = new_notebook()
    cells = []

    cells.append(new_code_cell(
        """import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

print("Project root added to sys.path:", project_root)"""
    ))

    cells.append(new_code_cell(
        """import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import lightning as L

from ffnn.nn_classifier import NNClassifier
from ffnn.hparams_tune import prepare_data"""
    ))

    cells.append(new_code_cell(
        f"""n_features_select = {n_features_select}
checkpoint_path = r\"\"\"{checkpoint_path}\"\"\""""
    ))

    cells.append(new_code_cell(
        """best_model = NNClassifier.load_from_checkpoint(
checkpoint_path,
input_dim=n_features_select
        )"""
    ))

    cells.append(new_code_cell(
        """_, _, test_loader, _, _, label_encoder = prepare_data(n_features_select)

trainer = L.Trainer(
    accelerator="mps" if torch.backends.mps.is_available() else "auto"
)
trainer.test(best_model, test_loader)"""
        ))

    cells.append(new_code_cell(
        """best_model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        logits = best_model(x.to(best_model.device))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())"""
    ))

    cells.append(new_code_cell(
        """y_true_labels = label_encoder.inverse_transform(y_true)
y_pred_labels = label_encoder.inverse_transform(y_pred)"""
    ))

    cells.append(new_code_cell(
        """print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))"""
    ))

    cells.append(new_code_cell(
        """cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()"""
    ))

    nb.cells = cells

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the initial notebook
    with open(output_path, "w") as f:
        nbformat.write(nb, f)
    print(f"[{os.getpid()}] Saved: {output_path}")

    # Run the notebook and save the executed version
    print("Executing notebook...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(output_path, "w") as f:
        nbformat.write(nb, f)
    print(f"[{os.getpid()}] Done: {output_path}")


# Example usage
if __name__ == '__main__':
    checkpoint_base = "/Users/joel-tay/Desktop/AI-Project--Speech-Emotion-Recognition/checkpoints"
    output_base = "/Users/joel-tay/Desktop/AI-Project--Speech-Emotion-Recognition/ffnn/iterations"

    # output file format:
    # num features | hidden dim(s) | dropout | lr
    trials = [
        # LR
        # (200, f"{checkpoint_base}/trial_36_best.ckpt", f"{output_base}/lr/200 256 1e-1 1e-4.ipynb"), # first trial best
        # (200, f"{checkpoint_base}/trial_29_best.ckpt", f"{output_base}/lr/200 256 1e-1 1e-5.ipynb"),
        # (200, f"{checkpoint_base}/trial_26_best.ckpt", f"{output_base}/lr/200 256 1e-1 1e-3.ipynb"),

        # # num layers
        # (200, f"{checkpoint_base}/trial_17_best.ckpt", f"{output_base}/layers/200 128-64 4e-1 1e-3.ipynb"),
        # (200, f"{checkpoint_base}/trial_19_best.ckpt", f"{output_base}/layers/200 128-32-16 4e-1 1e-3.ipynb"),

        # dropout
        # (700, f"{checkpoint_base}/trial_21_best-v1.ckpt", f"{output_base}/dropout/700 256 3e-1 1e-4.ipynb"),
        # (700, f"{checkpoint_base}/trial_19_best-v1.ckpt", f"{output_base}/dropout/700 256 4e-1 1e-4.ipynb"),

        # hidden dim for single layer
        # (700, f"{checkpoint_base}/trial_21_best-v1.ckpt", f"{output_base}/dim/700 256 3e-1 1e-4.ipynb"),
        # (700, f"{checkpoint_base}/trial_34_best-v1.ckpt", f"{output_base}/dim/700 512 3e-1 1e-4.ipynb"),

        # n features select
        (500, f"{checkpoint_base}/trial_35_best-v1.ckpt", f"{output_base}/features/500 512 3e-1 1e-4.ipynb"),
        (600, f"{checkpoint_base}/trial_41_best.ckpt", f"{output_base}/features/600 512 3e-1 1e-4.ipynb"),
        (700, f"{checkpoint_base}/trial_34_best-v1.ckpt", f"{output_base}/features/700 512 3e-1 1e-4.ipynb"),
        (800, f"{checkpoint_base}/trial_37_best-v1.ckpt", f"{output_base}/features/800 512 3e-1 1e-4.ipynb"),
    ]

    with Pool(processes=2) as pool:
        pool.map(generate_eval_notebook, trials)
