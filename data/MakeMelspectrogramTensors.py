import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


def preprocess_and_save(csv_file, save_dir, transform):
    # Read the original CSV
    df = pd.read_csv(csv_file)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a list to store new paths and emotions
    new_data = []

    # Use tqdm for progress tracking
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        mel_path = row['Melspectrogrampath']
        emotion = row['Emotion']

        try:
            # Load and convert the image to RGB
            image = Image.open(mel_path).convert("RGB")

            # Apply the transformation pipeline
            tensor_image = transform(image)

            # Create a unique filename using original filename and index
            original_filename = os.path.splitext(os.path.basename(mel_path))[0]
            pt_filename = f"{original_filename}.pt"
            save_path = os.path.join(save_dir, pt_filename)

            # Save the tensor
            torch.save(tensor_image, save_path)

            # Add the new path and emotion to our list
            new_data.append({
                'Melspectrogrampath': mel_path,
                'Emotion': emotion,
                'Tensorpath': save_path,
            })

        except Exception as e:
            print(f"Error processing {mel_path}: {str(e)}")

    # Create new DataFrame with tensor paths
    new_df = pd.DataFrame(new_data)

    # Save the new CSV file
    csv_save_path = os.path.splitext(os.path.basename(csv_file))[
        0] + '_tensors.csv'
    new_df.to_csv(csv_save_path, index=False)
    print(f"\nSaved tensor paths to: {csv_save_path}")
    print(f"Total tensors processed: {len(new_data)}")


# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Process both train and test datasets
datasets = [
    ('melspectrogram_train_dataset.csv', 'melspectrogram_tensors/'),
    ('melspectrogram_test_dataset.csv', 'melspectrogram_tensors/')
]

for csv_file, save_dir in datasets:
    print(f"\nProcessing {csv_file}...")
    preprocess_and_save(csv_file, save_dir, transform)
