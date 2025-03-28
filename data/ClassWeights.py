from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import torch 

df = pd.read_csv('train_dataset.csv')

# Sort unique labels before mapping
unique_labels = sorted(df['Emotion'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Convert labels to integers
df['Emotion'] = df['Emotion'].map(label_map)
trainlabels = np.array(df['Emotion'].tolist())

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(trainlabels), y=trainlabels)

# Convert to tensor and save
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
torch.save(class_weights_tensor, 'class_weights.pt')
