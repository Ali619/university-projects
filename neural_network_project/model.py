import torch.nn as nn


class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size1=10, hidden_size2=8, output_size=3):
        super(IrisNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.layers(x)

    def print_model_summary(self):
        """Print a summary of the model architecture"""
        print("\nModel Architecture:")
        print("=" * 50)
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name}: {param.shape} - {num_params:,} parameters")
        print("=" * 50)
        print(f"Total trainable parameters: {total_params:,}")
        print("=" * 50)
