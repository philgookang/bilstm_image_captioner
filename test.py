import torch
import torch.nn as nn
import numpy as np
import torchvision

from model import BiLSTM
from data import load_dataset
from config import model_name, device

if __name__ == "__main__":

    # the string to test!
    test_string = "<s> john"

    # ########################
    # LOAD DATASET
    # ########################

    corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()

    # ########################
    # TEST VARIABLES
    # ########################

    model = BiLSTM(len(corpus))
    model.load_state_dict(torch.load(model_name))

    model.eval()
    sentence = test_string.split()
    sentence = torch.tensor([[word_to_idx[w] for w in sentence]])

    s = model.sample(sentence)
    print(test_string.split() + s)
