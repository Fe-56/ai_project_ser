import pandas as pd
from FeaturePreprocessor import FeaturePreprocessor
preprocessor = FeaturePreprocessor(n_components_pca=0.99, n_features_select=50)

def main():
  df = pd.read_csv('../features/combined_features.csv', on_bad_lines='warn')
  full_df = df.set_index("Filepath")

  train_set = pd.read_csv('../train_dataset.csv')
  val_set = pd.read_csv('../val_dataset.csv')
  test_set = pd.read_csv('../test_dataset.csv')

  cols_to_drop = ['Id', 'Dataset', 'Filename', 'Ext', 'Duration']

  train_df = full_df.merge(train_set, on='Filepath', how='inner').drop(columns=cols_to_drop).set_index('Filepath')
  val_df = full_df.merge(val_set, on='Filepath', how='inner').drop(columns=cols_to_drop).set_index('Filepath')
  test_df = full_df.merge(test_set, on='Filepath', how='inner').drop(columns=cols_to_drop).set_index('Filepath')

  X_train = train_df.iloc[:, :-1]
  y_train = train_df.iloc[:, -1]

  print(f"Features shape: {X_train.shape}")
  print(f"Target shape: {y_train.shape}")

  # Normalize features
  X_train_normalized = preprocessor.normalize_features(X_train)

  # Reduce dimensions
  X_train_reduced = preprocessor.reduce_dimensions(X_train_normalized, y_train, fit=True)

  # Normalize features
  X_train_final = preprocessor.normalize_pc(X_train_reduced)

  # Plot feature importance
  preprocessor.plot_feature_importance(X_train_final, y_train)

  # Plot PCA variance
  preprocessor.plot_pca_variance()

  train_final = X_train_final.join(y_train, how='inner')

  ## process validation and test sets
  print("val")
  X_val = val_df.iloc[:, :-1]
  y_val = val_df.iloc[:, -1]
  X_val_normalized = preprocessor.normalize_features(X_val, fit=False)
  X_val_reduced = preprocessor.reduce_dimensions(X_val_normalized, y_val, fit=False)
  X_val_final = preprocessor.normalize_pc(X_val_reduced, fit=False)
  val_final = X_val_final.join(y_val, how='inner')

  print("test")
  X_test = test_df.iloc[:, :-1]
  y_test = test_df.iloc[:, -1]
  X_test_normalized = preprocessor.normalize_features(X_test, fit=False)
  X_test_reduced = preprocessor.reduce_dimensions(X_test_normalized, y_test, fit=False)
  X_test_final = preprocessor.normalize_pc(X_test_reduced, fit=False)
  test_final = X_test_final.join(y_test, how='inner')

  print("final shapes:")
  print("train: ", train_final.shape)
  print("val: ", val_final.shape)
  print("test: ", test_final.shape)

  # Save outputs to CSV
  train_final.to_csv('../features/train_final.csv')
  val_final.to_csv('../features/val_final.csv')
  test_final.to_csv('../features/test_final.csv')

  return {
    "train": train_final,
    "val": val_final,
    "test": test_final
  }

if __name__ == '__main__':
  main()
