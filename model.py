from torch import nn
import torch.nn.functional as F
import torch

class Net(nn.Module):

    def __init__(self, vocab, embed_size, n_classes):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(*vocab.vectors.size())
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
        #nn.init.xavier_uniform(self.conv.weight)
        self.fc = nn.Linear(3, n_classes)
        #nn.init.xavier_uniform(self.fc.weight
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze_(1)
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        # after x.squeeze(3), x's dim = (batch_size, out_channels, n_features)
        x1.squeeze_(3)
        x2.squeeze_(3)
        x3.squeeze_(3)
        x1, _ = x1.max(axis=2)
        x2, _ = x2.max(axis=2)
        x3, _ = x3.max(axis=2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(x)
        return x
