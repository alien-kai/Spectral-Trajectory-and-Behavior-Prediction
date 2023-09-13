from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from main import BS,train_seq_len

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self,input_size=2,hidden_size=4,output_size=2):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.classifier = nn.Linear(output_size, output_size)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fl = nn.Linear(input_size, hidden_size * 8)
        self.lstm = nn.GRU(hidden_size*8, hidden_size*8, num_layers, batch_first=True)

    def forward(self, input):
        f = self.fl(input)
        output, hidden = self.lstm(torch.relu(f))
        return output


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, output_size, num_layers,dropout=0.5):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.fl = nn.Linear(hidden_size * 8, hidden_size * 8)
        self.lstm = nn.GRU(hidden_size*8, output_size, num_layers, batch_first=True)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, input):
        f = self.fl(input)
        output, hidden = self.lstm(torch.relu(f))
        output = self.dropout(output)
        output = self.linear(output)
        return output

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,dropout=0.5):
        super(Model, self).__init__()
        self.gcn = GCN()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size, num_layers,dropout=dropout)

    def forward(self, input):
        res = torch.zeros([len(input), input[0].num_edges // 2 + 1, 2]).to(device)
        for i in range(len(input)):
            # print(i)
            h = self.gcn(input[i].x, input[i].edge_index).to(device)
            # print(h.shape)
            res[i] = h
        res = res.view(BS, train_seq_len, -1)
        output = self.encoder(res)
        output = self.decoder(output)
        return output