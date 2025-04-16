import torch.nn as nn

class MetaFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaFFNN, self).__init__()
        
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Third hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
