from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience  # Number of epochs to wait before stopping
        self.min_delta = min_delta  # Minimum change in loss to be considered an improvement
        self.best_loss = float("inf")
        self.epochs_no_improve = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        val_loss = metrics.get("eval_loss")  # Extract validation loss
        if val_loss is None:
            return

        # Check if validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0  # Reset counter if loss improves
        else:
            self.epochs_no_improve += 1  # Increment counter if loss does not improve

        # If patience threshold is reached, stop training
        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs of no improvement.")
            control.should_training_stop = True
