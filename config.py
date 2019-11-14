import torch

# the filename of the output trained model features
model_name = "train_model_features.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# size of the embedding output and input for lstm
input_size = 65

# the number of hidden nodes in the LSTM model
hidden_size = 512

# max length of the output string during Sampling
max_seg_length = 30
