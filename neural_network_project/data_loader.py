import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class IrisDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False)

    def load_and_preprocess(self):

        iris = load_iris()
        X = iris.data
        y = iris.target.reshape(-1, 1)

        df = pd.DataFrame(X, columns=iris.feature_names)
        df['target'] = iris.target_names[y.ravel()]

        self._visualize_data(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        y_encoded = self.encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=self.test_size, random_state=self.random_state)

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        print("\nDataset Information:")
        print(f"Total samples: {len(X)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {len(iris.target_names)}")

        return X_train, X_test, y_train, y_test, iris.target_names

    def _visualize_data(self, df):

        plt.figure(figsize=(15, 10))

        # Plot 1: Feature distributions by class
        plt.subplot(2, 2, 1)
        for feature in df.columns[:-1]:
            sns.kdeplot(data=df, x=feature, hue='target', common_norm=False)
        plt.title('Feature Distributions by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 2: Correlation heatmap
        plt.subplot(2, 2, 2)
        sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')

        # Plot 3: Pairplot for first two features
        plt.subplot(2, 2, 3)
        sns.scatterplot(
            data=df, x=df.columns[0], y=df.columns[1], hue='target')
        plt.title('First Two Features Scatter Plot')

        # Plot 4: Box plot of features
        plt.subplot(2, 2, 4)
        df.boxplot()
        plt.title('Feature Box Plots')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
