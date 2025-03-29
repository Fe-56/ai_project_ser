import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FeaturePreprocessor:
    def __init__(self, n_components_pca=0.95, n_features_select=100):
        """
        Initialize the feature preprocessor
        
        Args:
            n_components_pca (float): Variance ratio to keep in PCA (0-1)
            n_features_select (int): Number of features to select using SelectKBest
        """
        self.n_components_pca = n_components_pca
        self.n_features_select = n_features_select
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        
    def normalize_features(self, X, method='standard'):
        """
        Normalize features using either StandardScaler or MinMaxScaler
        
        Args:
            X (pd.DataFrame): Input features
            method (str): 'standard' or 'minmax'
            
        Returns:
            pd.DataFrame: Normalized features
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
            
        X_normalized = self.scaler.fit_transform(X)
        return pd.DataFrame(X_normalized, columns=X.columns)
    
    def reduce_dimensions(self, X, y=None):
        """
        Reduce dimensions using PCA and feature selection
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable (optional)
            
        Returns:
            pd.DataFrame: Reduced features
        """
        print(f"Input shape before PCA: {X.shape}")
        
        # PCA
        self.pca = PCA(n_components=self.n_components_pca)
        X_pca = self.pca.fit_transform(X)
        print(f"Shape after PCA: {X_pca.shape}")
        print(f"Number of components: {self.pca.n_components_}")
        
        # Feature selection if target variable is provided
        if y is not None:
            # Adjust n_features_select if it's larger than the number of features
            n_features = min(self.n_features_select, X_pca.shape[1])
            print(f"Selecting {n_features} features")
            
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = self.feature_selector.fit_transform(X_pca, y)
            
            # Get selected feature names
            selected_features = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            selected_features = np.array(selected_features)[self.feature_selector.get_support()]
            return pd.DataFrame(X_selected, columns=selected_features)
        
        return pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    def plot_feature_importance(self, X, y):
        """
        Plot feature importance scores
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
        """
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # Get feature importance scores
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        # Create DataFrame with scores
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores,
            'P-value': pvalues
        })
        
        # Sort by score
        importance_df = importance_df.sort_values('Score', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(20), x='Score', y='Feature')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    def plot_pca_variance(self):
        """
        Plot cumulative explained variance ratio from PCA
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet")
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1),
                np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.savefig('pca_variance.png')
        plt.close()

def main():
    # Load your features
    features_path = Path('../features/combined_train_features.csv')
    df = pd.read_csv(features_path)
    print(f"Loaded features shape: {df.shape}")
    
    master_df = pd.read_csv('../train_dataset.csv')
    print(f"Loaded master dataset shape: {master_df.shape}")
    
    # Separate features and target (assuming last column is target)
    X = df.iloc[:, :-1]  # Id is last col
    y = master_df.iloc[:, -1]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Initialize preprocessor with more conservative parameters
    preprocessor = FeaturePreprocessor(n_components_pca=0.99, n_features_select=50)
    
    # Normalize features
    X_normalized = preprocessor.normalize_features(X)
    
    # Reduce dimensions
    X_reduced = preprocessor.reduce_dimensions(X_normalized, y)
    
    # Plot feature importance
    preprocessor.plot_feature_importance(X_normalized, y)
    
    # Plot PCA variance
    preprocessor.plot_pca_variance()
    
    # Save preprocessed data
    output_path = Path('../features/preprocessed_features.csv')
    X_reduced.to_csv(output_path, index=False)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Reduced features: {X_reduced.shape[1]}")
    print(f"Preprocessed data saved to: {output_path}")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main() 