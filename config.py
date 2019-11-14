import torch

# the filename of the output trained model features
model_name = "train_model_features.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
