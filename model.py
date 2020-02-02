from torch import nn
import torch.nn.functional as F
import torch


class Net(nn.Module):

    def __init__(self, embedding_matrix, embed_size, n_classes):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(*embedding_matrix.size())
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True
        self.conv1 = nn.Conv1d(embed_size, 100, 3)
        self.conv2 = nn.Conv1d(embed_size, 100, 4)
        self.conv3 = nn.Conv1d(embed_size, 100, 5)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc = nn.Linear(300, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        # after x.squeeze(3), x's dim = (batch_size, out_channels, n_features)
        x1, _ = x1.max(axis=2)
        x2, _ = x2.max(axis=2)
        x3, _ = x3.max(axis=2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(x)
        return x
