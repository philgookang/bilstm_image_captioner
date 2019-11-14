import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import visdom

from model import BiLSTM
from data import load_dataset
from config import model_name, device

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


if __name__ == "__main__":

    # ########################
    # INIT
    # ########################

    # training hyper parameters
    num_epochs = 30

    # training progress output (how often to print to screen)
    progress_output = 5

    # training batch size (how many to train at once before backpropagation)
    batch_size = 2

    # learning rate (the amoutn of the error in be reflected on the model)
    learning_rate = 0.001 # the most common default value

    # load dataset
    corpus, word_to_idx, idx_to_word, train_dataset = load_dataset()

    # visdom, by facebook, graphs the loss value for us!
    vis = visdom.Visdom()
    visdom_window = None
    vis_config = dict(title="BiLSTM", xlabel='Iteration', ylabel='Loss', legend=["Epoch {}".format(i) for i in range(num_epochs)])

    # this class loads the train and label
    data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=coco_collate_fn)

    # init model
    model = BiLSTM(len(corpus))

    # for learning
    criterion = nn.CrossEntropyLoss() # unless regression, always to cross entropy
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)

    # set model mode to train (you only need this if your using dropout)
    model.train()

    # ########################
    # MODEL TRAINING
    # ########################

    for epoch in range(num_epochs):
        for iter, (captions, labels, lengths) in enumerate(data_loader):

            outputs = model(captions)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % progress_output == 0:
                print('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, iter+1, len(data_loader), loss.item()))

            if visdom_window:
                vis.line(X=np.array([iter]), Y=np.array([loss.item()]), name="Epoch {}".format(epoch), update='append', win=visdom_window)
            else:
                visdom_window = vis.line(X=np.array([iter]), Y=np.array([loss.item()]), name="Epoch {}".format(epoch), opts=vis_config)

    # save the weights of the model!
    torch.save(model.state_dict(), model_name)

    # tell the user were done!
    print("training complete!")
