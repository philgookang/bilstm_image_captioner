import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import visdom

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper-parameters
sequence_length = 28
num_epochs = 20

class BiLSTM(nn.Module):

    def __init__(self, c_size):
        super(BiLSTM, self).__init__()

        iSize = 64  # 내 마음데로 정함 어짜피 embedding에서 LSTM으로 가니깐 ㅎㅎ
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
            if idx_to_word[id] == '<e>':
                break
            caption.append(idx_to_word[id])
        return caption


def coco_collate_fn(data):

    # ########################
    # LOAD DATASET
    # ########################

    corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()

    # ########################
    # GROUP BATCH
    # ########################

    data.sort(key=lambda x: len(x["data"]), reverse=True)
    captions = [torch.FloatTensor(string_to_tensor(sentence))  for sentence in data]
    lengths = [len(cap) for cap in captions]
    labels = [torch.FloatTensor([ word_to_idx[sentence['target']] ])  for sentence in data]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return targets, torch.FloatTensor(labels).long(), lengths

def string_to_tensor(sentence):
    corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()
    return [word_to_idx[word] for word in sentence["data"]]


def load_dataset():

    # ########################
    # DATASET PREPROCESS
    # ########################

    corpus = []
    word_to_idx = { }
    idx_to_word = { }
    train_dataset = [] # { "data" : ['<s>', 'john'], "target" : "a" }
    i = 0

    # dataset
    dataset = [
        ['<s>', "john", "went", "to", "the", "store", "<e>"],
        ['<s>', "john", "went", "to", "the", "mall", "<e>"],
        ['<s>', "john", "went", "home", "early", "<e>"],
        ['<s>', "john", "is", "a", "engineer", "<e>"],
        ['<s>', "john", "is", "at", "home", "<e>"],
        ['<s>', "john", "can", "run", "fast", "<e>"],
        ['<s>', "john", "can", "go", "rest", "<e>"],
        ['<s>', "john", "can", "fly", "<e>"],
        ['<s>', "john", "can", "fly", "high", "<e>"],
        ['<s>', "john", "is", "not", "a", "data", "<e>"],
        ['<s>', "john", "is", "only", "at", "home", "<e>"],
        ['<s>', "john", "will", "jump", "<e>"],
        ['<s>', "john", "will", "jump", "high", "<e>"],
        ['<s>', "john", "will", "jump", "fast", "<e>"],
        ['<s>', "john", "is", "faster", "than", "me", "<e>"],
        ['<s>', "john", "is", "faster", "than", "a", "dog", "<e>"],
        ['<s>', "john", "is", "really", "fast", "<e>"],
        ['<s>', "john", "is", "only", "dead", "<e>"],
        ['<s>', "john", "is", "in", "love", "<e>"],
        ['<s>', "john", "is", "fly", "tomorrow", "<e>"],
        ['<s>', "john", "is", "fly", "high", "now", "<e>"]
    ]

    # generate corpus & naive label encoding! (give each word a id #)
    for words in dataset:
        for w in words:
            if w not in corpus:
                corpus.append(w)
                word_to_idx[w] = i
                idx_to_word[i] = w
                i += 1

    # change dataset format to many-to-one format
    for sentence in dataset:
        tmp = []
        for i in range(len(sentence)-1):
            tmp.append(sentence[i]) # we need to cumulate the sentence
            target = sentence[i + 1] # set the next word in the sentence as the target!
            train_dataset.append({"data" : tmp.copy(), "target" : target})

    return corpus, word_to_idx, idx_to_word, train_dataset

def train(model_name):

    # ########################
    # LOAD DATASET
    # ########################

    corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()

    # ########################
    # TRAINING VARIABLES
    # ########################

    vis = visdom.Visdom()
    visdom_window = None
    vis_config = dict(title="BiLSTM", xlabel='Iteration', ylabel='Loss', legend=["Epoch {}".format(i) for i in range(num_epochs)])
    data_loader = DataLoader(train_dataset, batch_size=1, collate_fn=coco_collate_fn)
    model = BiLSTM(len(corpus))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

    # ########################
    # MODEL TRAINING
    # ########################

    model.train()
    for epoch in range(num_epochs):
        for iter, (captions, labels, lengths) in enumerate(data_loader):

            outputs = model(captions)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 5 == 0:
                print('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, iter+1, len(data_loader), loss.item()))

            if visdom_window:
                vis.line(X=np.array([iter]), Y=np.array([loss.item()]), name="Epoch {}".format(epoch), update='append', win=visdom_window)
            else:
                visdom_window = vis.line(X=np.array([iter]), Y=np.array([loss.item()]), name="Epoch {}".format(epoch), opts=vis_config)

    torch.save(model.state_dict(), model_name)

def test(model_name, input_string):

    # ########################
    # LOAD DATASET
    # ########################

    corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()

    # ########################
    # TRAINING VARIABLES
    # ########################

    model = BiLSTM(len(corpus))
    model.load_state_dict(torch.load(model_name))

    model.eval()
    sentence = input_string.split()
    sentence = torch.tensor([[word_to_idx[w] for w in sentence]])

    s = model.sample(sentence)
    print(s)


if __name__ == "__main__":

    model_name = "train_model_features.pt"
    mode = "test" # mode: train or test

    if  mode == "train":
        train(model_name)
    elif mode == "test":
        test(model_name, "<s> john")
