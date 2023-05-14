import os

import torch

from def_train_eval import *
import numpy as np
from models import *
import pickle
from matplotlib import pyplot as plt
import numpy as np
from main import *

loc=f'../resources/trained_models/{recording}/{BS}/{start_frame}_{end_frame}/'
train_seq_len = 25 * 1
pred_seq_len = 25 * 1

# BS=32

DIR = f'../resources/data/{recording}/{BS}/{start_frame}_{end_frame}/'
learning_rate=1e-3
SAVE_LOC=f'../resources/plot/{recording}/{BS}/{start_frame}_{end_frame}/'

encoder1loc = os.path.join(loc, 'encoder_stream1.pt')
decoder1loc = os.path.join(loc, 'decoder_stream1.pt')
encoder2loc = os.path.join(loc, 'encoder_stream2.pt')
decoder2loc = os.path.join(loc, 'decoder_stream2.pt')

f1 = open(DIR + 'stream1_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
g1 = open(DIR + 'stream1_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
tr_seq_1 = pickle.load(f1)  # load file content as mydict
pred_seq_1 = pickle.load(g1)  # load file content as mydict
f1.close()
g1.close()

train_raw = tr_seq_1
pred_raw = pred_seq_1
#    train2_raw = tr_seq_2
#    pred2_raw = pred_seq_2
# Initialize encoder, decoders for both streams
batch = load_batch(0, BS, 'pred', train_raw, pred_raw, [], [], [], [])
print("load_batch completed")
batch, _, _ = batch
batch_in_form = np.asarray([batch[i]['sequence'] for i in range(BS)])
batch_in_form = torch.Tensor(batch_in_form)
[batch_size, step_size, fea_size] = np.shape(batch_in_form)
input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size

encoder_stream1 = Encoder(input_dim, hidden_dim, output_dim).to(device)
decoder_stream1 = Decoder('s1', input_dim, hidden_dim, output_dim, batch_size, step_size).to(device)
encoder_stream1_optimizer = optim.RMSprop(encoder_stream1.parameters(), lr=learning_rate)
decoder_stream1_optimizer = optim.RMSprop(decoder_stream1.parameters(), lr=learning_rate)
encoder_stream1.load_state_dict(torch.load(encoder1loc))
encoder_stream1.eval()
decoder_stream1.load_state_dict(torch.load(decoder1loc))
decoder_stream1.eval()
print("load_state_dict completed")

train_raw = tr_seq_1
pred_raw = pred_seq_1
train2_raw = []
pred2_raw = []

epoch=0

trainbatch_both = load_batch(epoch, BS, 'train', train_raw, pred_raw, train2_raw, pred2_raw, [], [])
trainbatch, trainbatch2, _ = trainbatch_both
trainbatch_in_form = np.asarray([trainbatch[i]['sequence'] for i in range(len(trainbatch))])
trainbatch_in_form = torch.Tensor(trainbatch_in_form)
print("train_batch completed")
testbatch_both = load_batch(epoch, BS, 'pred', train_raw, pred_raw, train2_raw, pred2_raw, [], [])
testbatch, testbatch2, _ = testbatch_both
testbatch_in_form = np.asarray([testbatch[i]['sequence'] for i in range(len(testbatch))])
testbatch_in_form = torch.Tensor(testbatch_in_form)
print("test_batch completed")
train = trainbatch_in_form.to(device)
label = testbatch_in_form

print("generate started")
# pred = generate(train, encoder_stream1, decoder_stream1)

#pred
pred=[]
Hidden_State, _ = encoder_stream1.loop(train)
#mu_1, mu_2
_, _, mu_1, mu_2, log_sigma_1, log_sigma_2, rho = decoder_stream1.loop(Hidden_State)
[batch_size, step_size, fea_size] = mu_1.size()
for i in range(batch_size):
    temp = []
    for j in range(step_size):
        mu1_current = mu_1[i, j, :]
        mu2_current = mu_2[i, j, :]
        temp.append([mu1_current.detach().cpu().numpy()[0],mu2_current.detach().cpu().numpy()[0]])
        # print(temp)
        # break
    pred.append(temp)
pred=np.array(pred)

#(batch_size, pred_size, [x,y])
print(pred.shape)

print(f"pred:{pred[30,2,:]}")
print(f"label:{label[30,2,:]}")

# x_pred=[ x for x in pred[500,:,:][0]]
# y_pred=[ y for y in pred[500,:,:][1]]
# x_true=[ x for x in label[500,:,:][0]]
# y_true=[ y for y in label[500,:,:][1]]
#
# plt.figure()
# plt.title('predict')
# plt.plot(x_pred,y_pred)
# plt.plot(x_true,y_true)
# plt.legend(['pred','label'])
# plt.show()
# plt.savefig(os.path.join(SAVE_LOC,'prdict.png'))

