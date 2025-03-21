# # # # import pandas as pd
# # # # from datasets import Dataset, Image

# # # # df_train = pd.read_csv('../data/melspectrogram_train_dataset.csv')

# # # # print(df_train)

# # # import torch
# # # print(torch.__version__)
# # # print(torch.cuda.is_available())

# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import soundfile as sf  # For saving the audio

# def repeat_pad_audio(y, desired_length):
#     """
#     Repeat and pad the audio signal 'y' until it reaches the desired_length (in samples).
#     """
#     if len(y) >= desired_length:
#         return y[:desired_length]
#     # Calculate how many full repetitions of y we need
#     full_reps = int(np.floor(desired_length / len(y)))
#     # Calculate the number of samples needed from the next repetition
#     remainder = desired_length - full_reps * len(y)
#     # Repeat the audio and add the remaining part
#     y_padded = np.concatenate([np.tile(y, full_reps), y[:remainder]])
#     return y_padded

# # Load your 5-second audio clip at a target sample rate (e.g., 16000 Hz)
# y, sr = librosa.load('YAF_back_angry.wav', sr=16000)

# # Calculate the desired total number of samples for 11 seconds
# target_duration = 11  # seconds
# target_length = int(target_duration * sr)

# # Apply repetitive padding to create an 11-second audio clip
# y_padded = repeat_pad_audio(y, target_length)

# # Save the padded audio to a file so you can listen to it
# sf.write('padded_audio.wav', y_padded, sr)
# print("Padded audio saved as 'padded_audio.wav'.")

# # Extract the mel spectrogram from the padded audio
# melspectrogram = librosa.feature.melspectrogram(
#     y=y_padded, 
#     sr=sr, 
#     n_fft=2048, 
#     hop_length=512, 
#     n_mels=128
# )
# melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

# # Plot and display the mel spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(melspectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
# plt.title('Mel Spectrogram of 11s Padded Audio')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()

# # from tinytag import TinyTag
# # import pandas as pd

# # def get_duration(filepath):
# #     tag = TinyTag.get(filepath)
# #     return tag.duration

# # dftrain = pd.read_csv('../data/train_dataset.csv')
# # print(len(dftrain))

# # dftest = pd.read_csv('../data/test_dataset.csv')
# # print(len(dftest))

# # paths = dftrain['Filepath'].tolist() + dftest['Filepath'].tolist()
# # paths = ["../data/" + x for x in paths]

# # tinyl = []

# # # Apply the function to add a new duration column
# # for x in paths:
# #     if x == "../data/./dataset/tess\YAF_fear\YAF_neat_fear.wav":
# #         continue
# #     print(x)
# #     tinyl.append(get_duration(x))
# # # 73924 6.987

# # # 73924 6.987
# # print(len(tinyl), max(tinyl))


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

# Custom dataset class for loading Melspectrograms
class MelSpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with Melspectrogram paths and labels.
            root_dir (string): Directory with all the Melspectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(csv_file)  # Read the CSV into a DataFrame
        self.data_frame = df[['Melspectrogrampath', 'Emotion']]
        self.root_dir = root_dir
        self.transform = transform
        
        # Sort unique labels before mapping
        unique_labels = sorted(df['Emotion'].unique())  
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the path to the Melspectrogram and the emotion label (string)
        # First column is path, second column is label (emotion as string)
        mel_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        emotion_label_str = self.data_frame.iloc[idx, 1]

        # Convert the emotion string to its corresponding integer
        emotion_label = self.label_map[emotion_label_str]

        # Load the Melspectrogram
        mel_image = Image.open(mel_path)
        mel_image = mel_image.convert("RGB")

        # Apply transformations if any
        if self.transform:
            mel_image = self.transform(mel_image)
            
        # Convert emotion label to tensor
        emotion_label = torch.tensor(emotion_label, dtype=torch.long)

        return mel_image, emotion_label
    
def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    
    # train mode
    model.train()

    for inputs, labels in trainloader:
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
        
    train_loss = train_loss / len(trainloader)
    train_accuracy = train_correct / train_total * 100
    
    return model, train_loss, train_accuracy

def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0
    
    # Switch to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update test loss
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_accuracy = test_correct / test_total * 100
    
    return test_loss, test_accuracy

def train_epochs(model, trainloader, testloader, labelmap, criterion, optimizer, device, num_epochs, save_interval=5):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)
        print(f'Train Loss: {train_loss} - Train Accuracy: {train_accuracy}')
        print(f'Test Loss: {train_loss} - Test Accuracy: {train_accuracy}')
        print()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Save state every 5 epochs
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'resnet50_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'test_losses': test_losses,
                'test_accuracies': test_accuracies,
                'labels': labelmap
            }
            torch.save(checkpoint, f'resnet50_variables_{epoch+1}.pth')
    
    return model, train_losses, train_accuracies, test_losses, test_accuracies

print(torch.__version__)
print(torch.cuda.is_available())

# Set random seed for reproducability
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Number of classes
num_classes = 9

# Import ResNet50 model
model = models.resnet50(pretrained=True)

# Modify final fully connected layer according to number of classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the Mel spectrogram to 448x448
    transforms.ToTensor(),          # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
])

# Load dataset
train_csv = '../data/melspectrogram_train_dataset.csv'
test_csv = '../data/melspectrogram_test_dataset.csv'
root_dir = '../data/'

trainset = MelSpectrogramDataset(csv_file=train_csv, root_dir=root_dir, transform=transform)
testset = MelSpectrogramDataset(csv_file=test_csv, root_dir=root_dir, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Label mappings
labelmap = trainset.label_map

# 60 epochs, save model every 5 epochs
epochs = 10
save_interval = 5
print(f"Model is on: {next(model.parameters()).device}")
model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(model, trainloader, testloader, criterion, optimizer, device, epochs, save_interval)
torch.save(model.state_dict(), f'resnet50_variables_{epochs}.pth')
