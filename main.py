import torch
import en_core_web_sm
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from dataset import TextDataset
from torch.utils.data import DataLoader
from model import Net
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_arguments():
    # read user specified arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', dest='embed_size',
                        type=int, default=300)
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=50)
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    parser.add_argument('--load_model', dest='load_model', type=int, default=0)
    parser.add_argument('--load_path', dest='load_path',
                        type=str, default='model.pt')
    parser.add_argument('--save_path', dest='save_path',
                        type=str, default='model.pt')
    parser.add_argument('--lr', dest='lr',
                        type=float, default=1e-3)
    return parser.parse_args()


def read_embeddings(filename):
    # read the embeddings from the file
    look_up = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.split()
            word, embedding = line[0], torch.tensor(
                [float(x) for x in line[1:]])
            look_up[word] = embedding.unsqueeze(0)
    return look_up


def read_dataset(filename, is_Test=False):
    # read the dataset from the file
    sentences, tags = [], []
    nlp = en_core_web_sm.load()
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = torch.tensor([w2i[tok.text] if tok.text in look_up else UNK
                                  for tok in nlp.tokenizer(words)])
            if not is_Test:
                tags.append(tag2i[tag])
            else:
                tags.append(0)
            sentences.append(words)
    return sentences, tags


def collate_fn(batch):
    # customize the data loader by padding the sequences.
    sentences, tags = zip(*batch)
    pad_sentences = pad_sequence(
        sentences, batch_first=True, padding_value=PAD)
    return pad_sentences, torch.tensor(tags)


def test(loader):
    # switch the model to evaluation mode to get accuracy and loss on the test or validation datasets.
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0.0
    net.eval()
    with torch.no_grad():
        for pad_sentences, tags in loader:
            pad_sentences = pad_sentences.to(device)
            tags = tags.to(device)
            outputs = net(pad_sentences)
            loss = criterion(outputs, tags)
            running_loss += loss.item()
            correct += sum(torch.argmax(outputs, dim=1) == tags)
    net.train()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)


# user_specified parameters
args = parse_arguments()
embed_size = args.embed_size
batch_size = args.batch_size
epochs = args.epochs
tag2i = defaultdict(lambda: len(tag2i))
w2i = defaultdict(lambda: len(w2i))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load data
UNK = w2i['<unk>']
PAD = w2i['<pad>']
look_up = read_embeddings("./data/glove.6B.{}d.txt".format(embed_size))
train_X, train_Y = read_dataset("./data/topicclass_train.txt")
val_X, val_Y = read_dataset("./data/topicclass_valid.txt")
test_X, test_Y = read_dataset("./data/topicclass_test.txt", is_Test=True)

# build word embeddings
i2w = {v: k for k, v in w2i.items()}
embedding_matrix = torch.zeros((2, embed_size))
total_words = len(w2i)
for i in range(2, total_words):
    embedding = look_up[i2w[i]]
    embedding_matrix = torch.cat((embedding_matrix, embedding), dim=0)

# build dataset and  dataloader
train_dataset = TextDataset(train_X, train_Y, w2i)
test_dataset = TextDataset(test_X, test_Y, w2i)
val_dataset = TextDataset(val_X, val_Y, w2i)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=len(test_X),
                         shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=len(val_X),
                        shuffle=False, collate_fn=collate_fn)
# build CNN model
if args.load_model:
    net = torch.load(args.load_path)
else:
    net = Net(embedding_matrix, embed_size, len(tag2i))

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(net.parameters())

# train the model
running_loss = 0.0
now = datetime.now()
writer = SummaryWriter(
    './log_data/{}/'.format(now.strftime("%d-%m-%Y-%H-%M-%S")))
i = best_val_accuracy = best_val_epoch = 0
for epoch in range(epochs):
    # early stopping
    if epoch - best_val_epoch >= 2:
        break

    for batch in tqdm(train_loader):
        pad_sentences, tags = batch
        pad_sentences = pad_sentences.to(device)
        tags = tags.to(device)
        optimizer.zero_grad()
        outputs = net(pad_sentences)
        loss = criterion(outputs, tags)
        loss.backward()
        optimizer.step()
        w_norm = net.fc.weight.data.norm(p=2)
        if w_norm >= 3:
            net.fc.weight.data = net.fc.weight.data / w_norm * 3
        running_loss += loss.item()
        if i % 1000 == 999:
            val_loss, val_accuracy = test(val_loader)
            writer.add_scalar('training loss', running_loss / 1000, i)
            writer.add_scalar('validation loss', val_loss, i)
            print("validation accuracy is {}".format(val_accuracy))
            writer.add_scalar('validation accuracy', val_accuracy, i)
            running_loss = 0.0
            # save the best model so far
            if val_accuracy > best_val_accuracy:
                torch.save(net, args.save_path)
                best_val_accuracy = val_accuracy
                best_val_epoch = epoch
        i += 1
