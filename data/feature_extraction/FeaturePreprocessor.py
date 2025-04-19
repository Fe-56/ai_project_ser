import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

class FeaturePreprocessor:
    def __init__(self, n_components_pca=0.95, n_features_select=50):
        """
        Initialize the feature preprocessor
        
        Args:
            n_components_pca (float): Variance ratio to keep in PCA (0-1)
            n_features_select (int): Number of features to select using SelectKBest
        """
        self.n_components_pca = n_components_pca
        self.n_features_select = n_features_select
        self.scaler = None
        self.pc_scaler = None
        self.pca = None
        self.feature_selector = None
        
    def normalize_features(self, X, method='standard', fit=True):
        """
        Normalize features using either StandardScaler or MinMaxScaler.
        This should only be done on train set
        
        Args:
            X (pd.DataFrame): Input features
            method (str): 'standard' or 'minmax'
            
        Returns:
            pd.DataFrame: Normalized features
        """
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
        X_normalized = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
    
    def normalize_pc(self, X, method='standard', fit=True):
        if fit:
            if method == 'standard':
                self.pc_scaler = StandardScaler()
            else:
                self.pc_scaler = MinMaxScaler()
            
        X_normalized = self.pc_scaler.fit_transform(X) if fit else self.pc_scaler.transform(X)
        return pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
    
    def reduce_dimensions(self, X, y=None, fit=True):
        """
        Reduce dimensions using PCA and feature selection
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable (optional)
            
        Returns:
            pd.DataFrame: Reduced features
        """
        print(f"Input shape before PCA: {X.shape}")

        if fit:
            self.pca = PCA(n_components=self.n_components_pca)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)

        pc_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        # Feature selection if target variable is provided
        if y is not None and fit:
            # Adjust n_features_select if it's larger than the number of features
            n_features = min(self.n_features_select, X_pca.shape[1])
            print(f"Selecting {n_features} features")
            
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = self.feature_selector.fit_transform(X_pca, y)
            
            # Get selected feature names
            selected_features = np.array(pc_names)[self.feature_selector.get_support()]
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        elif self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_pca)
            selected_features = np.array(pc_names)[self.feature_selector.get_support()]
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return pd.DataFrame(X_pca, columns=pc_names)
    
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