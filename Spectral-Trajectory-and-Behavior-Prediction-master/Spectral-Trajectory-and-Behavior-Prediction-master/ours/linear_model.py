import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

from main import *

device = torch.device("cuda:0")

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    #normal distribution
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #adj: adjacency matrix
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(Encoder, self).__init__()

        self.cell_size = cell_size
        self.fl = nn.Linear(input_size, hidden_size*16)
        self.il = nn.Linear(hidden_size*16, hidden_size*64)
        self.ol = nn.Linear(hidden_size*64, hidden_size*128)

        self.lstm=nn.LSTM(hidden_size*128,hidden_size*128,batch_first=True)

    def computeDist(self, x1, y1):
        return np.abs(x1 - y1)
        # return sqrt ( pow ( x1 - x2 , 2 ) + pow ( y1 - y2 , 2 ) )

    def computeKNN(self, curr_dict, ID, k):
        import heapq
        from operator import itemgetter

        ID_x = curr_dict[ID]
        dists = {}
        for j in range(len(curr_dict)):
            if j != ID:
                dists[j] = self.computeDist(ID_x, curr_dict[j])
        KNN_IDs = dict(heapq.nsmallest(k, dists.items(), key=itemgetter(1)))
        neighbors = list(KNN_IDs.keys())

        return neighbors

    def compute_A(self, xt):
        # return Variable(torch.Tensor(np.ones([xt.shape[0], xt.shape[0]])).cuda())
        xt = xt.cpu().detach().numpy()
        A = np.zeros([xt.shape[0], xt.shape[0]])
        for i in range(len(xt)):
            if xt[i] is not None:
                neighbors = self.computeKNN(xt, i, 4)
            for neighbor in neighbors:
                A[i][neighbor] = 1
        return Variable(torch.Tensor(A).cuda())

    def forward(self, input, Hidden_State, Cell_State):

        # graph
        if graph is True:
            gcn_feat = []
            #[1,4]->[4,1]
            gcn_model = GCN(nfeat=1, nhid=4, nclass=1, dropout=0.5)
            #batch_size
            for j in range(input.shape[0]):
                # [num_agents]
                features = input[j, :]
                #[batch_size, num_agents, 1]
                #cordinations and adjacency matrix
                gcn_feat.append(gcn_model(torch.unsqueeze(features, dim=1), self.compute_A(features)).cpu().detach().numpy())
            input = nn.Parameter(torch.FloatTensor(np.asarray(gcn_feat)).cuda())
            # [batch_size, num_agents]
            input = torch.squeeze(input)

        # input = torch.cat((input, Hidden_State), 1)

        #hidden_size=input_size = 2
        #[batch_size, input_size + hidden_size = 4]

        f= self.fl(input)
        i= self.il(torch.relu(f))
        o= self.ol(torch.relu(i))
        Hidden_State = self.lstm(torch.relu(o))

        return Hidden_State[0]



class Decoder(nn.Module):
    def __init__(self, stream, input_size, cell_size, hidden_size, batchsize, timestep):
        super(Decoder, self).__init__()
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.batch_size = batchsize
        self.time_step = timestep
        self.num_mog_params = 5
        self.sampled_point_size = 2
        self.stream = stream
        # self.stream_specific_param = self.num_mog_params
        # self.stream_specific_param = input_size if self.stream == 's2' else self.num_mog_params

        ########################
        self.stream_specific_param = input_size
        # self.stream_specific_param = self.num_mog_params
        ########################

        #(2*input_size,input_size)
        self.fl = nn.Linear(hidden_size*128, hidden_size*64)
        self.cl = nn.Linear(hidden_size * 64, hidden_size * 16)
        self.ol = nn.Linear(hidden_size*16, hidden_size * 16)

        self.lstm=nn.LSTM(hidden_size*16,hidden_size*16,batch_first=True)
        self.linear2=nn.Linear(hidden_size*16,input_size)


    def forward(self, input, Hidden_State, Cell_State):
        #combine the input and hidden state
        # input is ramdom if s1 is True
        #(batch_size,input_size+Hidden_State*self.multiplier)

        #(0,1)
        #[batch_size,hidden_size]t
        f = self.fl(input)
        # (0,1)
        # [batch_size,hidden_size]
        c= self.cl(torch.relu(f))
        o=self.ol(torch.relu(c))

        hidden=self.lstm(torch.relu(o))
        mu=self.linear2(torch.relu(hidden[0]))


        return mu

