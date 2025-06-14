import os

import torch
import torch.nn as nn
from data_loader import IrisDataLoader
from model import IrisNet
from trainer import ModelTrainer

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{CURR_DIR}/models/"
os.makedirs(MODEL_PATH, exist_ok=True)


def main():
    print("Iris Classification using Neural Networks")
    print("=" * 50)

    # Initialize data loader
    data_loader = IrisDataLoader()
    X_train, X_test, y_train, y_test, class_names = data_loader.load_and_preprocess()

    # Initialize model
    model = IrisNet()
    model.print_model_summary()

    # Initialize training components
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Initialize trainer
    trainer = ModelTrainer(model, loss_fn, optimizer)

    # Train the model
    trainer.train(X_train, y_train, X_test, y_test, num_epochs=50)

    # Print final metrics
    trainer.print_final_metrics()

    # Save the model
    torch.save(model.state_dict(), f'{MODEL_PATH}/iris_model.pth')
    print(f"\nModel saved as 'iris_model.pth' in {MODEL_PATH}")


if __name__ == "__main__":
    main()
