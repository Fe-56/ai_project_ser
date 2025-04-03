import time
from tqdm import tqdm
from cnn.pipeline.EarlyStopping import EarlyStopping
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report


class Pipeline:
    @staticmethod
    def train(model, trainloader, criterion, optimizer, device):
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        # train mode
        model.train()

        epoch_start = time.time()
        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), desc="Training")

        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Calculate time metrics
            elapsed = time.time() - epoch_start
            progress = (i + 1) / len(trainloader)
            eta = elapsed / progress - elapsed

            # Update progress bar with current loss and ETA
            pbar.set_postfix({
                "Loss": f"{loss.item()}",
                "Elapsed": f"{elapsed:.4f}s",
                "ETA": f"{eta:.4f}s"
            })

        train_loss = train_loss / len(trainloader)
        train_accuracy = train_correct / train_total * 100

        return model, train_loss, train_accuracy

    @staticmethod
    def validate(model, valloader, criterion, device):
        val_loss = 0.0
        val_total = 0
        val_correct = 0

        # Switch to evaluation mode
        model.eval()

        epoch_start = time.time()
        pbar = tqdm(enumerate(valloader), total=len(
            valloader), desc="Validating")

        with torch.no_grad():
            for i, (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update test loss
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Calculate time metrics
                elapsed = time.time() - epoch_start
                progress = (i + 1) / len(valloader)
                eta = elapsed / progress - elapsed

                # Update progress bar with current loss and ETA
                pbar.set_postfix({
                    "Loss": f"{loss.item()}",
                    "Elapsed": f"{elapsed:.4f}s",
                    "ETA": f"{eta:.4f}s"
                })

        val_loss = val_loss / len(valloader)
        val_accuracy = val_correct / val_total * 100

        return val_loss, val_accuracy

    @staticmethod
    def train_epochs(model, trainloader, valloader, criterion, optimizer, device, num_epochs, model_name, patience):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_accuracy = 0

        early_stopper = EarlyStopping(
            path=f'../../../models/checkpoints/earlystop_{model_name}.pt', patience=patience)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            epoch_start = time.time()

            model, train_loss, train_accuracy = Pipeline.train(
                model, trainloader, criterion, optimizer, device)
            val_loss, val_accuracy = Pipeline.validate(
                model, valloader, criterion, device)

            epoch_elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_elapsed:.4f}s")
            print(
                f'Train Loss: {train_loss} - Train Accuracy: {train_accuracy}')
            print(
                f'Validation Loss: {val_loss} - Validation Accuracy: {val_accuracy}')
            print()

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Save best validation accuracy model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(),
                           f'../../../models/weights/best_{model_name}.pt')
                checkpoint = {
                    'epoch': epoch + 1,
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                }
                torch.save(
                    checkpoint, f'../../../models/checkpoints/best_{model_name}_checkpoint.pt')

            # Check for early stopping (based on val_loss)
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                break

        return model, train_losses, train_accuracies, val_losses, val_accuracies

    @staticmethod
    def plot_loss(train_losses, val_losses):
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(train_accuracies, val_accuracies):
        plt.figure()
        plt.plot(range(len(train_accuracies)),
                 train_accuracies, label='Training Accuracy')
        plt.plot(range(len(val_accuracies)),
                 val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, label_map):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure and axes
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(cm,
                    annot=True,  # Show numbers in cells
                    fmt='d',     # Use integer formatting
                    cmap='Blues',  # Color scheme
                    xticklabels=label_map.keys(),
                    yticklabels=label_map.keys())

        # Set labels and title
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')

        # Rotate axis labels for better readability
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_predictions(model, testloader, device, model_path):
        # Load the best model weights
        model.load_state_dict(torch.load(model_path))
        model.eval()

        all_preds = []
        all_labels = []
        print("Evaluating best model on test set...")

        with torch.no_grad():
            for inputs, labels in tqdm(testloader, desc='Testing'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Store predictions for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_preds, all_labels

    @staticmethod
    def get_evaluation_metrics(true_labels, pred_labels, label_map):
        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted')

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-score: {f1:.4f}\n")

        # Print detailed classification report
        print(classification_report(true_labels, pred_labels,
                                    target_names=list(label_map.keys())))

    @staticmethod
    def execute(model, labelmap, trainloader, valloader, testloader, criterion, optimizer, device, num_epochs, model_name, patience,):
        print(f"Mapping from Emotion to Number: {labelmap}")

        print(f"Model is on: {next(model.parameters()).device}")
        model, train_losses, train_accuracies, val_losses, val_accuracies = Pipeline.train_epochs(
            model, trainloader, valloader, criterion, optimizer, device, num_epochs, model_name, patience)

        # Plots
        Pipeline.plot_loss(train_losses, val_losses)
        Pipeline.plot_accuracy(train_accuracies, val_accuracies)

        best_model_path = f'../../../models/weights/best_{model_name}.pt'
        all_preds, all_labels = Pipeline.get_predictions(
            model=model,
            testloader=testloader,
            device=device,
            model_path=best_model_path
        )

        # Print detailed evaluation metrics
        Pipeline.get_evaluation_metrics(all_labels, all_preds, labelmap)

        # Plot confusion matrix
        Pipeline.plot_confusion_matrix(all_labels, all_preds, labelmap)
