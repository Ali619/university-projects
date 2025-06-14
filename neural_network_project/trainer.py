import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class ModelTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def train_epoch(self, X_train, y_train):
        self.model.train()
        outputs = self.model(X_train)
        loss = self.criterion(outputs, y_train)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(y, 1)
            accuracy = (predicted == actual).float().mean()
        return loss.item(), accuracy.item()

    def train(self, X_train, y_train, X_test, y_test, num_epochs):
        print("\nStarting Training:")
        print("=" * 50)

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(X_train, y_train)
            train_loss, train_acc = self.evaluate(X_train, y_train)

            # Evaluation
            test_loss, test_acc = self.evaluate(X_test, y_test)

            # Store metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(
                    f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
                print(
                    f'Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.4f}')
                print("-" * 50)

        self.plot_training_history()

    def plot_training_history(self):
        plt.figure(figsize=(12, 5))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Testing Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.test_accuracies, label='Testing Accuracy')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def print_final_metrics(self):
        print("\nFinal Model Performance:")
        print("=" * 50)
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Testing Loss: {self.test_losses[-1]:.4f}")
        print(f"Final Training Accuracy: {self.train_accuracies[-1]:.4f}")
        print(f"Final Testing Accuracy: {self.test_accuracies[-1]:.4f}")
        print("=" * 50)
