from torch import nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, vocab, embed_size, out_channels, window_size, n_classes, device):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(*vocab.vectors.size())
        self.embedding.weight.data.copy_(vocab.vectors)
        if device == 'cuda':
            self.embedding.cuda()
        self.conv = nn.Conv2d(1, out_channels, (window_size, embed_size))
        self.fc = nn.Linear(out_channels, n_classes)
        self.out_channels = out_channels
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = x.unsqueeze_(1)
        x = self.conv(x)
        # after x.squeeze(3), x's dim = (batch_size, out_channels, n_features)
        x.squeeze_(3)
        mask.unsqueeze_(1)
        expand_mask = mask.expand(-1, self.out_channels, -1)
        x[expand_mask] = float('-inf')
        x, _ = x.max(axis=2)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(x)
        return x
