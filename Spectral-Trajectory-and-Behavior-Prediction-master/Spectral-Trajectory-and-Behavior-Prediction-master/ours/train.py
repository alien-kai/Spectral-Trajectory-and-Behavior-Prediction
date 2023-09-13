import sys
import os

sys.path.append('..')
import time
import torch.utils.data as utils
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

#select models
# from models import *
# from linear_model import *
from GCN_model import *
from torch_geometric.data import Data
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
from main import *

DATA = 'OTH'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# BS = 256
MU = 5

#intialize the loss1 and loss2
loss_stream1=np.inf
loss_stream2=np.inf

# recording = 2
# #
# #
# recording = "{:02d}".format(int(recording))
# tracks_file = f"../rounD-dataset-v1.0/data/{recording}_tracks.csv"
# tracks=pd.read_csv(tracks_file)

# start_frame=min(tracks['frame'])
# end_frame=max(tracks['frame'])
MODEL_LOC = MODEL_DIR
SAVE_LOC=f'../resources/plots/{recording}/{BS}/{start_frame}_{end_frame}/'

if not os.path.exists(MODEL_LOC):
    os.makedirs(MODEL_LOC)
if not os.path.exists(SAVE_LOC):
    os.makedirs(SAVE_LOC)

def load_batch(index, size, seq_ID, train_sequence_stream1, pred_sequence_stream_1, train_sequence_stream2, pred_sequence_stream2, train_eig_seq, pred_eig_seq):
    '''
    to load a batch of data
    :param index: index of the batch
    :param size: size of the batch of data
    :param seq_ID: either train sequence or a pred sequence, give as a str
    :param train_sequence: list of dicts of train sequences
    :param pred_sequence: list of dicts of pred sequences
    :return: Batch mof data
    '''

    i = index
    batch_size = size
    start_index = i * batch_size
    stop_index = (i + 1) * batch_size

    if stop_index >= len(train_sequence_stream1):
        stop_index = len(train_sequence_stream1)
        start_index = stop_index - batch_size
    if seq_ID == 'train':
        stream1_train_batch = train_sequence_stream1[start_index:stop_index]
        stream2_train_batch = train_sequence_stream2[start_index:stop_index]
        eigs = train_eig_seq[start_index:stop_index]
        single_batch = [stream1_train_batch, stream2_train_batch, eigs]

    elif seq_ID == 'pred':
        stream1_pred_batch = pred_sequence_stream_1[start_index:stop_index]
        stream2_pred_batch = pred_sequence_stream2[start_index:stop_index]
        eigs = pred_eig_seq[start_index:stop_index]
        single_batch = [stream1_pred_batch, stream2_pred_batch, eigs]
    else:
        single_batch = None
        print('please enter the sequence ID. enter train for train sequence or pred for pred sequence')
    return single_batch


def trainIters(n_epochs, train_dataloader, valid_dataloader, train2_dataloader, valid2_dataloader, train_eig, val_eig, data, sufix, s2, print_every=1, plot_every=1000, learning_rate=1e-3, save_every=5):

    # output_stream2_decoder = None
    num_batches = int(len(train_dataloader) / BS)

    model_loc = os.path.join(MODEL_LOC, 'model.pt')

    train_raw = train_dataloader
    pred_raw = valid_dataloader
    train2_raw = train2_dataloader
    pred2_raw = valid2_dataloader
    train_eig_raw = train_eig
    pred_eig_raw = val_eig
    # Initialize encoder, decoders for both streams
    batch = load_batch(0, BS, 'train', train_raw, pred_raw, train2_raw, pred2_raw, train_eig_raw, pred_eig_raw)
    batch, mid, batch2 = batch
    batch_in_form = np.asarray([batch[i]['sequence'] for i in range(BS)])
    batch_in_form = torch.Tensor(batch_in_form).to(device)
    #batch_size=32, step_size=25, fea_size=2
    [batch_size, step_size, fea_size] = np.shape(batch_in_form)
    # trainbatch2 = torch.Tensor(np.asarray([batch2[i] for i in range(len(batch2))]))
    input_dim = mid[0][1].shape[1]*mid[0][1].shape[0]
    hidden_dim = input_dim * 4
    output_dim = fea_size

    #model
    model=Model(input_dim, hidden_dim, 2, output_dim).to(device)

    #Adam
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #MultiStepLR(多步长衰减)：设置不同的学习率
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[10, 50, 90],gamma=0.8)


    print("loading {}...".format(model_loc))

    loss_stream1_all = []
    lr_list=[]
    for epoch in tqdm(range(n_epochs),desc='train'):
        #        print("epoch: ", epoch)
        print_loss_total_stream1 = 0  # Reset every print_every

        # Prepare train and test batch
        for bch in tqdm(range(num_batches),desc='train'):
            # print('# {}/{} epoch {}/{} batch'.format(epoch, n_epochs, bch, num_batches))
            #data
            trainbatch_both = load_batch(bch, BS, 'train', train_raw, pred_raw, train2_raw, pred2_raw,train_eig_raw, pred_eig_raw)
            trainbatch, train_middle, trainbatch2 = trainbatch_both
            # trainbatch_in_form = np.asarray([trainbatch[i]['sequence'] for i in range(BS)])
            # trainbatch_in_form = torch.Tensor(trainbatch_in_form).to(device)
            trainbatch_in_form = get_graph_data(train_middle)

            #label
            testbatch_both = load_batch(bch, BS, 'pred', train_raw, pred_raw, train2_raw, pred2_raw,train_eig_raw, pred_eig_raw)
            testbatch, test_middle, testbatch2 = testbatch_both
            testbatch_in_form = np.asarray([testbatch[i]['sequence'] for i in range(BS)])
            testbatch_in_form = torch.Tensor(testbatch_in_form).to(device)

            # stream1
            input_stream1_tensor = trainbatch_in_form
            #i个batch的agent_ID
            batch_agent_ids = [trainbatch[i]['agent_ID'] for i in range(BS)]
            target_stream1_tensor = testbatch_in_form
            loss_stream1 = train_stream1(input_stream1_tensor, target_stream1_tensor,model,model_optimizer)
            # print(f"loss_stream1: {loss_stream1}")
            print_loss_total_stream1 += loss_stream1

        lr_list.append(model_optimizer.param_groups[0]['lr'])
        model_scheduler.step()


        #loss1
        print('stream1 average loss:', print_loss_total_stream1 / num_batches)
        loss_stream1_all.append(print_loss_total_stream1 / num_batches)

        #save model
        #get the smallest loss of stream1 and stream2
        if (epoch+1) % save_every == 0:
            save_model(model, loss_stream1_all,MODEL_LOC,batch_size,n_epochs)
            print(f"model saved at epoch {epoch}")

    # plot_loss_avg_stream1 = print_loss_total_stream1 / plot_every

    #draw stream1 loss
    makeplot(range(n_epochs), loss_stream1_all, "epoch", "loss", f"stream1_loss_{batch_size}_{n_epochs}_{learning_rate}_{graph}",SAVE_LOC)

    #draw lr
    makeplot(range(n_epochs), lr_list, "epoch", "lr", f"lr_{batch_size}_{n_epochs}_{learning_rate}_{graph}",SAVE_LOC)

    return model

def train_stream1(input_tensor, target_tensor, model,model_optimizer):
    # batch_size
    target_length = target_tensor.size(0)

    mu= model(input_tensor).to(device)

    mu_1=mu[:,:,0].unsqueeze(2)
    mu_2=mu[:,:,1].unsqueeze(2)

    # encoder_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()
    model_optimizer.zero_grad()

    loss=MSE(mu_1, mu_2,target_tensor,None).to(device)

    loss = loss if loss > 0 else -1 * loss
    loss.backward()

    # encoder_optimizer.step()
    # decoder_optimizer.step()
    model_optimizer.step()
    return loss.item() / target_length


def save_model(model,loss1,loc,batch_size,n_epochs):
    #取较小值
    if loss1[-1]<loss_stream1:
        torch.save(model.state_dict(),os.path.join(loc, f'model_{batch_size}_{n_epochs}_{learning_rate}_{graph}.pt'))
    print('model saved at {}'.format(loc))



def MSE(mu_1, mu_2, y,cluster_centers):
    [batch_size, step_size, fea_size] = y.size()

    epoch_loss = 0
    # 计算所有batch的第i帧的loss
    for i in range(step_size):
        # only mu1_current,mu2_current
        mu1_current = mu_1[:, i, :]
        mu2_current = mu_2[:, i, :]
        if cluster_centers is not None:
            muc_x = cluster_centers[:, i, 0]
            muc_y = cluster_centers[:, i, 1]

        loss = nn.MSELoss().to(device)
        label = y[:, i, :]
        #拼接为(x,y)
        pred = torch.squeeze(torch.stack((mu1_current, mu2_current), dim=2))
        # print(label.shape)
        # print(pred.shape)

        if cluster_centers is None:
            batch_loss = loss(pred, label)
        ## s2 is true, calculate cluster centers
        else:
            batch_loss = loss(pred, label) + torch.sqrt(torch.sum((mu1_current - muc_x) ** 2) + torch.sum((mu2_current - muc_y) ** 2))

        #除以BS
        batch_loss = batch_loss / batch_size
        epoch_loss += batch_loss

    return epoch_loss



def calculate_cluster_centers(agent_IDs, stream2_output, testbatch2):
    #(batch_size, num_steps, 15),distance
    stream2_output = stream2_output.cpu().detach().numpy()
    #batch_size
    num_batches = len(testbatch2)
    # print('num_batches: ', num_batches)
    #num_steps
    num_steps = stream2_output.shape[1]
    # print('num_steps: ',num_steps)
    # print(f'size: [{num_batches},{num_steps},{stream2_output.shape[2]}]')
    #[num_batches, num_steps, 2]
    cluster_centers = np.zeros((num_batches, num_steps, 2))
    # three behaviors: overspeed，normal，slow
    km = KMeans(init='k-means++', n_clusters=3)
    for t in range(num_steps):
        # print(f'step: {t}')
        for b in range(len(testbatch2) - 1):
            #b个batch的t帧
            clusters = km.fit(stream2_output[b, t, :].reshape(-1, 1))
            #与预测目标相同cluster的agent
            cluster_indcs = np.where(clusters.labels_ == clusters.labels_[agent_IDs[b]])[0]
            testbatch2_item = testbatch2[b]
            #'dataset_ID','agent_ID','id...'
            keys = list(testbatch2_item.keys())
            #skip 'dataset_ID','agent_ID'
            coords = testbatch2_item[keys[2 + t]][:, cluster_indcs]
            #同一个cluster的平均坐标
            cluster_centers[b, t, :] = np.mean(coords, axis=1)
    return cluster_centers


def makeplot(x, y, x_label, y_label, title, save_loc):
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    #each 50 epoch show loss
    for i in range(len(x)):
        if i % 50 == 0:
            #取小数点后6位
            plt.text(x[i], y[i], str(round(y[i],4)),color='red',fontsize=10)

    # plt.show()
    fig.savefig(os.path.join(save_loc, title + '.png'))
    fig.clear()


def get_graph_data(tr_seq_1):
    data = []
    x = torch.tensor(tr_seq_1[0][list(tr_seq_1[0].keys())[2]], dtype=torch.float).t()
    for i in range(len(tr_seq_1)):
        agent = tr_seq_1[i]['agent_ID'] - 1
        edge_index = []
        # total agents
        for k in range(len(x)):
            if k != agent:
                edge_index.append([agent, k])
                edge_index.append([k, agent])
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        for j in range(list(tr_seq_1[i].keys())[2], list(tr_seq_1[i].keys())[-1] + 1):
            x = torch.tensor(tr_seq_1[i][j], dtype=torch.float).t().to(device)
            d = Data(x=x, edge_index=edge_index.t().contiguous())
            data.append(d)
    return data