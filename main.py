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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# read user specified arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('embed_size', dest='embed_size', type=int, default=50)
    parser.add_argument('out_channels', dest='out_channels', type=int, default=4)
    parser.add_argument('window_size', dest='window_size', type=int, default=4)
    parser.add_argument('batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('epochs', dest='epochs', type=int, default=1)
    parser.add_argument('load_model', dest='load_model', type=int, default=0)
    parser.add_argument('load_path', dest='load_path', type=str, default='model.pt')
    parser.add_argument('save_path', dest='save_path', type=str, default='model.pt')
	return parser.parse_args()

# read the dataset from the file
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

# customize the data loader by padding the sequences and calculations the mask.
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sentences, tags = zip(*batch)
    pad_sentences = pad_sequence(sentences, batch_first=True, padding_value=1)
    # the padding positions have values as True in mask.
    mask = [torch.tensor([False] * (len(t) - window_size + 1))
            for t in sentences]
    mask = pad_sequence(mask, batch_first=True, padding_value=True)
    return pad_sentences, torch.tensor(tags), mask

# switch the model to evaluation mode to get accuracy and loss on the test or validation datasets.
def test(test_loader):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0.0
    for pad_sentences, tags, mask in test_loader:
        pad_sentences.to(device)
        tags.to(device)
        outputs = net(pad_sentences, mask)
        loss = criterion(outputs, tags)
        running_loss += loss.item()
        correct += torch.argmax(outputs,dim=1) == tags
    net.train()
    return running_loss / len(test_loader), corret / len(test_loader)

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
test_dataset = TextDataset(test_X, test_Y, TEXT.vocab.stoi)
valid_dataset = TextDataset(valid_X, valid_Y, TEXT.vocab.stoi)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=len(test_X),
                          shuffle=False, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_X),
                          shuffle=False, collate_fn=collate_fn)
# build CNN model
if args.load_model:
    net = torch.load(args.load_path)
else:
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
        valid_loss, valid_accuracy = test(valid_loader)
        test_loss, test_accuracy = test(test_loader)
        if i % 1000 == 999:
            writer.add_scalar('training loss', running_loss /
                              1000, epoch * len(train_loader) + i)
            writer.add_scalar('testing loss', test_loss, epoch * len(train_loader) + i)
            writer.add_scalar('testing accuracy', test_accuracy, epoch * len(train_loader) + i)
            writer.add_scalar('validation loss', valid_loss, epoch * len(train_loader) + i)
            writer.add_scalar('validation accuracy', valid_accuracy, epoch * len(train_loader) + i)
            running_loss = 0.0
        i += 1

torch.save(net, args.save_path)
