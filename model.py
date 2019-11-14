import torch
import torch.nn as nn
import numpy as np
import torchvision

from data import load_dataset
from config import model_name, device

class BiLSTM(nn.Module):

    def __init__(self, c_size):
        super(BiLSTM, self).__init__()

        iSize = 64
        hSize = 512
        self.max_seg_length = 30

        self.embed = nn.Embedding(c_size, iSize)
        self.lstm = nn.LSTM(input_size=iSize, hidden_size=hSize, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hSize * 2, c_size)
        self.softmax = nn.Softmax()

    def forward(self, caption):
        embeddings = self.embed(caption)
        hiddens, output = self.lstm(embeddings)
        out1, out2 = torch.chunk(hiddens, 2, dim=2)
        out_cat = torch.cat((out1[:, -1, :], out2[:, 0, :]), 1)
        result = self.linear(out_cat)
        return result

    def sample(self, features, states=None):

        corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()

        sampled_ids = []
        for i in range(10):
            inputs = self.embed(features)
            hiddens, states = self.lstm(inputs, states)
            out1, out2 = torch.chunk(hiddens, 2, dim=2)
            out_cat = torch.cat((out1[:, -1, :], out2[:, 0, :]), 1)
            outputs = self.linear(out_cat)
            _, predicted = outputs.max(1)
            features = torch.cat((features, predicted.unsqueeze(1)), 1)
            sampled_ids.append(predicted)
        sampled_ids = torch.stack(sampled_ids, 1)

        caption = []
        for id in sampled_ids[0].cpu().numpy():
            caption.append(idx_to_word[id])
            if idx_to_word[id] == '<e>':
                break
        return caption
