import spacy
import torch
import en_core_web_sm
import os
from tqdm import tqdm
from torchtext import data
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from dataset import TextDataset
from torch.utils.data import DataLoader
from model import Net


def read_dataset(filename):
    sentences, tags = [], []
    nlp = en_core_web_sm.load()
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = [tok.text for tok in nlp.tokenizer(words)]
            sentences.append(words)
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
    return pad_sentences, tags, mask


# user_specified parameters
embed_size = 50
out_channels = 4
window_size = 4
batch_size = 8
epochs = 1
tag2i = defaultdict(lambda: len(tag2i))

# load data
train_X, train_Y = read_dataset("./data/topicclass_train.txt")
val_X, val_Y = read_dataset("./data/topicclass_valid.txt")
#train_X, train_Y = read_dataset("./data/topicclass_test.txt")
test_X, test_Y = read_dataset("./data/topicclass_test.txt")

# build vocab for word embeddings
cache = '.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors = Vectors(name='data/glove.6B.{}d.txt'.format(embed_size), cache=cache)
TEXT = data.Field(sequential=True)
#TEXT.build_vocab(train_X, vectors=vectors)
TEXT.build_vocab(train_X + val_X, vectors=vectors)

# build dataset and  dataloader
train_dataset = TextDataset(train_X, train_Y, TEXT.vocab.stoi)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)

# build CNN model
net = Net(TEXT.vocab, embed_size, out_channels, window_size, len(tag2i))

# train the model
for epoch in range(epochs):
    for pad_sentences, tags, mask in tqdm(train_loader):
        y_pred = net(pad_sentences, mask)
        print(y_pred.size())
