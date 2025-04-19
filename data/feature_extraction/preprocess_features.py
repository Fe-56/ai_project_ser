import pandas as pd
from data.feature_extraction.FeaturePreprocessor import FeaturePreprocessor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

current_file = Path(__file__).resolve()
base_path = current_file.parents[1]

label_encoder = LabelEncoder()

speech_df = pd.read_csv(f'{base_path}/speech_dataset.csv', on_bad_lines='warn')

def main(n_features_select=50):
  preprocessor = FeaturePreprocessor(n_components_pca=0.99, n_features_select=n_features_select)
  df = pd.read_csv(f'{base_path}/features/combined_features.csv', on_bad_lines='warn')
  full_df = df.set_index("Filepath")

  # get label class nums
  label_encoder.fit(speech_df["Emotion"].unique())

  train_set = pd.read_csv(f'{base_path}/train_dataset.csv')
  val_set = pd.read_csv(f'{base_path}/val_dataset.csv')
  test_set = pd.read_csv(f'{base_path}/test_dataset.csv')

  cols_to_drop = ['Id', 'Dataset', 'Filename', 'Ext', 'Duration']

  train_df = full_df.merge(train_set, on='Filepath', how='inner').drop(columns=cols_to_drop).set_index('Filepath')
  val_df = full_df.merge(val_set, on='Filepath', how='inner').drop(columns=cols_to_drop).set_index('Filepath')
  test_df = full_df.merge(test_set, on='Filepath', how='inner').drop(columns=cols_to_drop).set_index('Filepath')

  X_train = train_df.iloc[:, :-1]
  y_train = label_encoder.transform(train_df.iloc[:, -1])

  print(f"Features shape: {X_train.shape}")
  print(f"Target shape: {y_train.shape}")

  # Normalize features
  X_train_normalized = preprocessor.normalize_features(X_train)

  # Reduce dimensions
  X_train_reduced = preprocessor.reduce_dimensions(X_train_normalized, y_train, fit=True)

  # Normalize features
  X_train_final = preprocessor.normalize_pc(X_train_reduced)

  # # Plot feature importance
  # preprocessor.plot_feature_importance(X_train_final, y_train)

  # # Plot PCA variance
  # preprocessor.plot_pca_variance()

  # train_final = X_train_final.join(y_train, how='inner')

  ## process validation and test sets
  print("val")
  X_val = val_df.iloc[:, :-1]
  y_val = label_encoder.transform(val_df.iloc[:, -1])
  X_val_normalized = preprocessor.normalize_features(X_val, fit=False)
  X_val_reduced = preprocessor.reduce_dimensions(X_val_normalized, y_val, fit=False)
  X_val_final = preprocessor.normalize_pc(X_val_reduced, fit=False)

  print("test")
  X_test = test_df.iloc[:, :-1]
  y_test = label_encoder.transform(test_df.iloc[:, -1])
  X_test_normalized = preprocessor.normalize_features(X_test, fit=False)
  X_test_reduced = preprocessor.reduce_dimensions(X_test_normalized, y_test, fit=False)
  X_test_final = preprocessor.normalize_pc(X_test_reduced, fit=False)

  return {
    "X_train": X_train_final,
    "y_train": y_train,
    "X_val": X_val_final,
    "y_val": y_val,
    "X_test": X_test_final,
    "y_test": y_test,
    "label_encoder": label_encoder # so can transform back to string later
  }

if __name__ == '__main__':
  main()