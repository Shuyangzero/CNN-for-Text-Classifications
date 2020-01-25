import spacy
import torch
import en_core_web_sm
import os
import argparse
from tqdm import tqdm
from torchtext import data
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors
from collections import defaultdict
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import TextDataset
from torch.utils.data import DataLoader
from model import Net
from torch.utils.tensorboard import SummaryWriter

#TODO: 1. add function to get test accuracy and loss 2. add gpu device 3. add dropout
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('embed_size', dest='embed_size', type=int, default=50)
    parser.add_argument('out_channels', dest='out_channels', type=int, default=4)
    parser.add_argument('window_size', dest='window_size', type=int, default=50)
    parser.add_argument('batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('epochs', dest='batch_size', type=int, default=1)


def read_dataset(filename, is_Test=False):
    sentences, tags = [], []
    nlp = en_core_web_sm.load()
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = [tok.text for tok in nlp.tokenizer(words)]
            sentences.append(words)
            if not is_Test:
                tags.append(tag2i[tag])
    return sentences, tags


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sentences, tags = zip(*batch)
    pad_sentences = pad_sequence(sentences, batch_first=True, padding_value=1)
    # the padding positions have value as True in mask.
    mask = [torch.tensor([False] * (len(t) - window_size + 1))
            for t in sentences]
    mask = pad_sequence(mask, batch_first=True, padding_value=True)
    return pad_sentences, torch.tensor(tags), mask


# user_specified parameters
args = parse_arguments()
embed_size = args.embed_size
out_channels = args.out_channels
window_size = args.window_size
batch_size = args.batch_size
epochs = args.epochs
tag2i = defaultdict(lambda: len(tag2i))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
train_X, train_Y = read_dataset("./data/topicclass_train.txt")
val_X, val_Y = read_dataset("./data/topicclass_valid.txt")
test_X, _ = read_dataset("./data/topicclass_test.txt", is_Test=True)

# build vocab for word embeddings
cache = '.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(name='data/glove.6B.{}d.txt'.format(embed_size), cache=cache)
TEXT = data.Field(sequential=True)
TEXT.build_vocab(train_X + val_X, vectors=vectors)

# build dataset and  dataloader
train_dataset = TextDataset(train_X, train_Y, TEXT.vocab.stoi)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)

# build CNN model
net = Net(TEXT.vocab, embed_size, out_channels, window_size, len(tag2i))
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# train the model
running_loss = 0.0
writer = SummaryWriter('./log_data')
for epoch in range(epochs):
    i = 0
    for batch in tqdm(train_loader):
        pad_sentences, tags, mask = batch
        pad_sentences.to(device)
        tags.to(device)
        optimizer.zero_grad()
        outputs = net(pad_sentences, mask)
        loss = criterion(outputs, tags)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            writer.add_scalar('training loss', running_loss /
                              1000, epoch * len(train_loader) + i)
            running_loss = 0.0
        i += 1
