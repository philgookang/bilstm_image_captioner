import torch
import torch.nn as nn
import numpy as np
import torchvision

from data import load_dataset
from config import model_name, device, input_size, hidden_size, max_seg_length

class BiLSTM(nn.Module):

    def __init__(self, corpus_size):
        super(BiLSTM, self).__init__()
        self.embed = nn.Embedding(corpus_size, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, corpus_size)

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
        for i in range(max_seg_length):
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
