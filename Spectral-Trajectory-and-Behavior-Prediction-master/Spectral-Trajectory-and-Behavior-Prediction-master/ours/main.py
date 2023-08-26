import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from def_train_eval import *
import pickle
import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


DATA = 'OTH'
SUFIX = '1stS15-5'

device = torch.device("cuda")
s2 = False
TRAIN = True
EVAL = False
graph= True

# #选取某一个agent,car
# agentID=2

BS=256
recording = 2
recording = "{:02d}".format(int(recording))
tracks_file = f"../rounD-dataset-v1.0/data/{recording}_tracks.csv"
tracks=pd.read_csv(tracks_file)


#选取开始和结束帧
# start_frame=min(tracks[tracks.trackId==agentID]['frame'])
# end_frame=max(tracks[tracks.trackId==agentID]['frame'])
start_frame=min(tracks['frame'])
end_frame=max(tracks['frame'])


DIR = f'../resources/data/{recording}/{BS}/{start_frame}_{end_frame}/'
MODEL_DIR = f'../resources/trained_models/{recording}/{BS}/{start_frame}_{end_frame}/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


epochs = 400
save_per_epochs = 20

train_seq_len = 25 * 1
pred_seq_len = 25 * 1

learning_rate=1e-4


#create folder to store plots
plot_dir = f'../resources/plots/{recording}/{BS}/{start_frame}_{end_frame}/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if __name__ == "__main__":

    if TRAIN:
        if DATA == 'OTH':
            with open(DIR + 'stream1_obs_data_train.pkl', 'rb') as f1:
                try:
                    tr_seq_1 = pickle.load(f1)  # load file content as mydict
                except EOFError:
                    raise ValueError("EOFError")
            with open(DIR + 'stream1_pred_data_train.pkl', 'rb') as g1:
                try:
                    pred_seq_1 = pickle.load(g1)  # load file content as mydict
                except EOFError:
                    raise ValueError("EOFError")
        else:
            f1 = open(DIR + 'stream1_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
            g1 = open(DIR + 'stream1_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted

        f1.close()
        g1.close()

        if s2:
            f2 = open(DIR + 'stream2_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
            g2 = open(DIR + 'stream2_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
            f3 = open(DIR + 'stream2_obs_eigs_train.pkl', 'rb')  # 'r' for reading; can be omitted
            g3 = open(DIR + 'stream2_pred_eigs_train.pkl', 'rb')  # 'r' for reading; can be omitted
            tr_seq_2 = pickle.load(f2)  # load file content as mydict
            pred_seq_2 = pickle.load(g2)  # load file content as mydict
            tr_eig_seq = pickle.load(f3)
            pred_eig_seq = pickle.load(g3)
            f2.close()
            f3.close()
            g2.close()
            g3.close()
        else:
            tr_seq_2 = []
            pred_seq_2 = []
            tr_eig_seq = []
            pred_eig_seq = []

        encoder1, decoder1 = trainIters(epochs, tr_seq_1, pred_seq_1, tr_seq_2, pred_seq_2, tr_eig_seq, pred_eig_seq,DATA, SUFIX, s2, learning_rate=learning_rate,print_every=1, save_every=save_per_epochs)

    if EVAL:
        print('start evaluating {}{}...'.format(DATA, SUFIX))
        if DATA == 'OTH':
            f1 = open(DIR + 'stream1_obs_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
            g1 = open(DIR + 'stream1_pred_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        else:
            f1 = open(DIR + 'stream1_obs_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
            g1 = open(DIR + 'stream1_pred_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        tr_seq_1 = pickle.load(f1)  # load file content as mydict
        pred_seq_1 = pickle.load(g1)  # load file content as mydict
        f1.close()

        g1.close()

        eval(10, tr_seq_1, pred_seq_1, DATA, SUFIX)
