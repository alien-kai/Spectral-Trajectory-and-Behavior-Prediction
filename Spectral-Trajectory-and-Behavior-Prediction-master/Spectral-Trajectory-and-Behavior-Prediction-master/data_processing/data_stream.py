import numpy as np
import pickle
from scipy.sparse.linalg import eigs
from tqdm import tqdm
from copy import deepcopy, copy
import pandas as pd
import os



DATA = 'OTH'
TYPE = 'train'
NAME1 = 'stream2_obs_data_{}.pkl'.format(TYPE)
NAME2 = 'stream2_pred_data_{}.pkl'.format(TYPE)
FrameRate = 25

##############################
train = 1 * FrameRate  # 1s
pred = 1 * FrameRate  # 1s
BS = 256
recording = 2

#选取某一个agent
agentID=2

recording = "{:02d}".format(int(recording))
tracks_file = f"../rounD-dataset-v1.0/data/{recording}_tracks.csv"
tracks=pd.read_csv(tracks_file)

# pre_ID = 9+1
# MAX_VALUE=1000

#选取开始和结束帧
start_frame=min(tracks[tracks.trackId==agentID]['frame'])
end_frame=max(tracks[tracks.trackId==agentID]['frame'])

# start_frame=min(tracks['frame'])
# end_frame=max(tracks['frame'])
# start_frame = 0
# end_frame = 1000

DIR = f'../resources/data/{recording}/{BS}/{start_frame}_{end_frame}/'
###########################



'''
The DATA_DIR for data_for_stream1 and data_for_stream2 functions should be as below

DATA_DIR = 'resources/data/ARGO/train/TrainSet.txt'     ## for ARGO train
DATA_DIR = 'resources/data/ARGO/test/TestSet.txt'       ## for ARGO test
DATA_DIR = 'resources/data/ARGO/val/ValSet.txt'         ## for ARGO val

DATA_DIR = 'resources/data/APOL/train/trainSet0.txt'    ## for APOL train
DATA_DIR = 'resources/data/APOL/test/testSet0.txt'      ## for APOL test
DATA_DIR = 'resources/data/APOL/val/valSet0.txt'        ## for APOL val

DATA_DIR = 'resources/data/LYFT/train/trainSet0.txt'    ## for LYFT train
DATA_DIR = 'resources/data/LYFT/test/testSet0.txt'      ## for LYFT test
DATA_DIR = 'resources/data/LYFT/val/valSet0.txt'        ## for LYFT val
'''


def save_to_pkl(dir, file_sequence):
    with open(dir, 'wb') as f:
        print(np.array(file_sequence).shape)
        pickle.dump(file_sequence, f)


def data_for_stream1(dir, train_seq_len=train, pred_seq_len=pred, frame_lenth_cap=train + pred):
    '''
    This function returns train sequences and prediction sequences for LSTM and RNN
    :param dir: directory where the dataset is present
    :param train_seq_len: length of the train sequence
    :param pred_seq_len: length of the prediction sequence
    :param frame_lenth_cap: minimum length of time presence the agent must have
    :return: train sequences and prediction sequences
    '''

    data = np.load(dir)
    # Dataset_ID
    d_IDs = np.unique(data[:, 4]).astype(int)

    train_sequence = []
    pred_sequence = []
    sz = len(d_IDs)

    mx = 0

    # traversing through all the dataset ID's
    for cur, data_id in tqdm(enumerate(d_IDs)):
        print("{}/{} {} {} in stream1".format(cur, sz, DATA, TYPE))
        # print('data ID is',data_id)

        # dataset corresponding to that dataset_ID
        one_dataset = data[np.where(data[:, 4] == data_id)]  # data_id

        # one_dataset = data[np.where(data[:,0] < 70)]
        # Agent_ID
        obj_IDs = np.unique(one_dataset[:, 1]).astype(int)
        # pre_id=pre_ID

        # 把Agent_ID重新编号为[1-n]
        d = dict([(y, x+1) for x, y in enumerate(sorted(set(obj_IDs)))])

        # pre_id = d[pre_ID]

        for each_obj_ID in obj_IDs:
            # print(each_obj_ID)
            new_id = d[each_obj_ID]
            one_obj_traj = one_dataset[np.where(one_dataset[:, 1] == each_obj_ID)]
            # replace Agent_ID with id
            new_array = np.ones([one_obj_traj.shape[0], ]) * new_id
            one_obj_traj[:, 1] = new_array
            one_dataset[np.where(one_dataset[:, 1] == each_obj_ID)] = one_obj_traj

        #     total_frames = 126
        # total_frames = int(np.amax(data[:,0]))
        total_frames = int(np.amax(one_dataset[:, 0]))

        obj_IDs = np.unique(one_dataset[:, 1]).astype(int)

        # traversing through each agent from all agents corresponding to that dataset
        # 每个agentID
        for each_obj_id in obj_IDs:

            # obtaining the trajectory of that individual agent
            # 同一个agent的所有帧信息
            one_obj_traj = one_dataset[np.where(one_dataset[:, 1] == each_obj_id)]  # each_obj_id

            # 提取预测目前的所有帧信息
            # pred_obj_traj = one_dataset[np.where(one_dataset[:, 1] == pre_id)]

            # sleecting the object only if it is visible in at least 'frame_length_cap' number of frames
            mx = max(mx, len(one_obj_traj[:, 0]))
            if len(one_obj_traj[:, 0]) >= frame_lenth_cap:

                # initializing an index range to obtain the train sequences and pred sequences
                index_range = len(one_obj_traj[:, 0]) - train_seq_len - pred_seq_len

                # obtaining the sequences
                for idx in range(1, index_range + 2):

                    train_sequence_dict = {}
                    pred_sequence_dict = {}

                    train_sequence_dict['dataset_ID'] = data_id
                    train_sequence_dict['agent_ID'] = each_obj_id

                    # train_x, train_y = pred_obj_traj[idx - 1, 2:4]

                    # 提取train的x和y
                    # 输入绝对坐标
                    train_sequence_dict['sequence'] = deepcopy(one_obj_traj[idx - 1:idx - 1 + train_seq_len, 2:4])

                    # for i in range(len(train_sequence_dict['sequence'])):
                    #     train_sequence_dict['sequence'][i][0] -= train_x
                    #     train_sequence_dict['sequence'][i][1] -= train_y
                    train_sequence.append(train_sequence_dict)

                    pred_sequence_dict['dataset_ID'] = data_id
                    pred_sequence_dict['agent_ID'] = each_obj_id

                    # pred_x, pred_y = pred_obj_traj[idx - 1 + train_seq_len, 2:4]

                    # label为相对坐标,以train最后一个点为原点
                    train_x, train_y = train_sequence_dict['sequence'][-1]

                    # 提取prediction的x和y

                    pred_sequence_dict['sequence'] = deepcopy(one_obj_traj[idx - 1 + train_seq_len:idx - 1 + train_seq_len + pred_seq_len, 2:4])
                    for i in range(len(pred_sequence_dict['sequence'])):
                        pred_sequence_dict['sequence'][i][0] -= train_x
                        pred_sequence_dict['sequence'][i][1] -= train_y

                    # dataset_ID  agent_ID, sequence
                    pred_sequence.append(pred_sequence_dict)
    print(f"frame_range: 1~{mx}")
    return train_sequence, pred_sequence


def data_for_stream2(dir, train_seq_len=train, pred_seq_len=pred, frame_lenth_cap=train + pred):
    '''
    This function returns train sequences and prediction sequences for LSTM and RNN
    :param dir: directory where the dataset is present
    :param train_seq_len: length of the train sequence
    :param pred_seq_len: length of the prediction sequence
    :param frame_lenth_cap: minimum length of time presence the agent must have
    :return: train sequences and prediction sequences
    '''

    data = np.load(dir)

    train_sequence_stream2 = []
    pred_sequence_stream2 = []

    # total_objects = 220
    # 最大的agentID
    total_objects = int(np.amax(data[:, 1]))
    # Dataset编号
    d_IDs = np.unique(data[:, 4]).astype(int)
    sz = len(d_IDs)
    # traversing through all the dataset ID's
    for cur, data_id in enumerate(d_IDs):

        print("{}/{} {} {} in stream2".format(cur, sz, DATA, TYPE))

        # print('data ID is',data_id)
        # dataset corresponding to that dataset_ID
        one_dataset = data[np.where(data[:, 4] == data_id)]  # data_id

        # one_dataset = data[np.where(data[:,0] < 70)]

        #所有的agentID
        obj_IDs = np.unique(one_dataset[:, 1]).astype(int)

        d = dict([(y, x + 1) for x, y in enumerate(sorted(set(obj_IDs)))])
        # pre_id = d[pre_ID]

        total_objects = len(obj_IDs)

        for each_obj_ID in obj_IDs:
            # print(each_obj_ID)
            new_id = d[each_obj_ID]
            one_obj_traj = one_dataset[np.where(one_dataset[:, 1] == each_obj_ID)]
            new_array = np.ones([one_obj_traj.shape[0], ]) * new_id
            one_obj_traj[:, 1] = new_array
            one_dataset[np.where(one_dataset[:, 1] == each_obj_ID)] = one_obj_traj

        #     total_frames = 126
        # total_frames = int(np.amax(data[:,0]))
        # total_frames = int(np.amax(one_dataset[:, 0]))

        obj_IDs = np.unique(one_dataset[:, 1]).astype(int)

        # traversing through each agent from all agents corresponding to that dataset
        for each_obj_id in tqdm(obj_IDs):

            # obtaining the trajectory of that individual agent
            #只有agentID为i的data
            one_obj_traj = one_dataset[np.where(one_dataset[:, 1] == each_obj_id)]  # each_obj_id

            # 提取预测目标的所有帧信息
            # pred_obj_traj = one_dataset[np.where(one_dataset[:, 1] == pre_id)]

            # sleecting the object only if it is visible in at least 'frame_length_cap' number of frames
            if len(one_obj_traj[:, 0]) >= frame_lenth_cap:

                # initializing an index range to obtain the train sequences and pred sequences
                index_range = len(one_obj_traj[:, 0]) - train_seq_len - pred_seq_len
                # print(index_range)

                # obtaining the sequences
                for idx in range(1, index_range + 2):

                    # print(idx)
                    train_sequence_dict = {}
                    pred_sequence_dict = {}

                    train_sequence_dict['dataset_ID'] = data_id
                    train_sequence_dict['agent_ID'] = each_obj_id

                    # train_x, train_y = pred_obj_traj[idx - 1, 2:4]

                    pred_sequence_dict['dataset_ID'] = data_id
                    pred_sequence_dict['agent_ID'] = each_obj_id

                    # pred_x, pred_y = pred_obj_traj[idx - 1 + train_seq_len, 2:4]

                    # obtaining all agents x and y values for the training sequences
                    for kdx in range(idx, idx + train_seq_len):
                        # kdx帧当中的data
                        one_frame = one_dataset[np.where(one_dataset[:, 0] == kdx)]
                        # kdx帧中的所有agentID
                        frame_objs = np.unique(one_frame[:, 1]).astype(int)
                        # 2*total_objects的随机数组
                        frame_array = np.empty([2, total_objects])

                        for jdx in range(1, total_objects + 1):
                            # 提取有的agentID的x,y
                            if jdx in frame_objs:
                                frame_array[0, jdx - 1] = deepcopy(one_frame[np.where(one_frame[:, 1] == jdx)][0][2])
                                frame_array[1, jdx - 1] = deepcopy(one_frame[np.where(one_frame[:, 1] == jdx)][0][3])
                                # frame_array[0, jdx - 1]-=train_x
                                # frame_array[1, jdx - 1]-=train_y
                            else:
                                frame_array[0, jdx - 1] = 0
                                frame_array[1, jdx - 1] = 0

                        train_sequence_dict[kdx] = frame_array

                    # obtaining all agents x and y values for the predicting sequences
                    for ldx in range(idx + train_seq_len, idx + train_seq_len + pred_seq_len):

                        one_frame_ = one_dataset[np.where(one_dataset[:, 0] == ldx)]
                        frame_objs_ = np.unique(one_frame_[:, 1]).astype(int)

                        frame_array_ = np.empty([2, total_objects])

                        for mdx in range(1, total_objects + 1):

                            if mdx in frame_objs_:
                                frame_array_[0, mdx - 1] = deepcopy(one_frame_[np.where(one_frame_[:, 1] == mdx)][0][2])
                                frame_array_[1, mdx - 1] = deepcopy(one_frame_[np.where(one_frame_[:, 1] == mdx)][0][3])
                                # frame_array_[0, mdx - 1] -= train_x
                                # frame_array_[1, mdx - 1] -= train_y

                            else:
                                frame_array_[0, mdx - 1] = 0
                                frame_array_[1, mdx - 1] = 0

                        pred_sequence_dict[ldx] = frame_array_

                    train_sequence_stream2.append(train_sequence_dict)
                    pred_sequence_stream2.append(pred_sequence_dict)

    return train_sequence_stream2, pred_sequence_stream2


def load_batch(index, size, seq_ID, train_sequence_stream1, pred_sequence_stream_1, train_sequence_stream2,
               pred_sequence_stream2):
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
        single_batch = [stream1_train_batch, stream2_train_batch]

    elif seq_ID == 'pred':
        stream1_pred_batch = pred_sequence_stream_1[start_index:stop_index]
        stream2_pred_batch = pred_sequence_stream2[start_index:stop_index]
        single_batch = [stream1_pred_batch, stream2_pred_batch]

    else:
        single_batch = None
        print('please enter the sequence ID. enter train for train sequence or pred for pred sequence')
    return single_batch


def compute_eigs(train_stream2, typ):
    # train_stream2  = train_stream2[:5000]
    # print(len(train_stream2))
    # TRAIN_SEQ帧的data
    # N is total object number
    N = int(train_stream2[0][list(train_stream2[0].keys())[-1]].shape[1])
    A = np.zeros([N, N])
    frame = {}
    eig_batch = []
    #total frames
    sz = len(train_stream2)
    for batch_idx in tqdm(range(sz),desc=f'{typ}',postfix={'name':DATA,'type':TYPE}):
        # print("{}/{} {} {} in computing eigen {}".format(batch_idx, sz, DATA, TYPE, typ))
        eig_frame = []
        for which_frame in list(train_stream2[batch_idx].keys())[2:]:
            # 每一帧中agentID的x,y
            for j in range(N):
                try:
                    frame[j] = [train_stream2[batch_idx][which_frame][0, j],train_stream2[batch_idx][which_frame][1, j]]
                except:
                    pass
            for l in range(N):
                if frame[l] is not None:
                    # 4个neighbor
                    neighbors = computeKNN(frame, l, 4)
                for neighbor in neighbors:
                    # 对l和每个neighbor计算距离，其余坐标为0
                    A[l][neighbor] = np.exp(-1 * computeDist(frame[l][0], frame[l][1], frame[neighbor][0], frame[neighbor][1]))
            # 把W的每一列元素加起来得到N个数，然后把它们放在对角线上（其它地方都是零），组成一个N✖️N的对角矩阵
            d = [np.sum(A[row, :]) for row in range(A.shape[0])]
            # 对角线矩阵，度矩阵
            D = np.diag(d)
            # 拉普拉斯矩阵
            L = D - A
            # 特征向量,[agent_num,2]
            _, vecs = eigs(L, k=2, tol=1e-3)
            # 保留了实数部分,[time_step,agent_num]
            eig_frame.append(np.real(vecs[:, 1]))
        # [batch_size, time_step, agent_num]
        eig_batch.append(np.array(eig_frame))
    return np.array(eig_batch)


# 欧式距离
def computeDist(x1, y1, x2, y2):
    return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def computeKNN(curr_dict, ID, k):
    import heapq
    from operator import itemgetter

    ID_x = curr_dict[ID][0]
    ID_y = curr_dict[ID][1]
    dists = {}
    for j in range(len(curr_dict)):
        # 不与自己做运算
        if j != ID:
            dists[j] = computeDist(ID_x, ID_y, curr_dict[j][0], curr_dict[j][1])
    # 最小距离的{agentID:distance}
    KNN_IDs = dict(heapq.nsmallest(k, dists.items(), key=itemgetter(1)))
    neighbors = list(KNN_IDs.keys())

    return neighbors


def compute_A(frame):
    A = np.zeros([frame.shape[0], frame.shape[0]])
    for i in range(len(frame)):
        if frame[i] is not None:
            neighbors = computeKNN(frame, i, 4)
        for neighbor in neighbors:
            A[i][neighbor] = 1
    return A


'''
The DATA_DIR for data_for_stream1 and data_for_stream2 functions should be as below

DATA_DIR = 'resources/data/ARGO/train/TrainSet.npy'     ## for ARGO train
DATA_DIR = 'resources/data/ARGO/test/TestSet.npy'       ## for ARGO test
DATA_DIR = 'resources/data/ARGO/val/ValSet.npy'         ## for ARGO val

DATA_DIR = 'resources/data/APOL/train/trainSet0.npy'    ## for APOL train
DATA_DIR = 'resources/data/APOL/test/testSet0.npy'      ## for APOL test
DATA_DIR = 'resources/data/APOL/val/valSet0.npy'        ## for APOL val

DATA_DIR = 'resources/data/LYFT/train/trainSet0.npy'    ## for LYFT train
DATA_DIR = 'resources/data/LYFT/test/testSet0.npy'      ## for LYFT test
DATA_DIR = 'resources/data/LYFT/val/valSet0.npy'        ## for LYFT val
'''

# Uncomment the below lines to generate and save the data_for_stream1 and data_for_stream 2. DATA_DIR needs to be given as explained above
DATA_DIR = f'../resources/data/{recording}/TrainSet_{start_frame}_{end_frame}.npy'

# #streame1
# if not os.path.exists(DIR):
#     os.makedirs(DIR)
# tr1, pr1 = data_for_stream1(DATA_DIR, train_seq_len=train, pred_seq_len=pred, frame_lenth_cap=train + pred)
# save_to_pkl(DIR + 'stream1_obs_data_{}.pkl'.format(TYPE), tr1)
# save_to_pkl(DIR + 'stream1_pred_data_{}.pkl'.format(TYPE), pr1)
#
# # stream2
# if not os.path.exists(DIR+'stream2_obs_data_{}.pkl'.format(TYPE)) and not os.path.exists(DIR+'stream2_pred_data_{}.pkl'.format(TYPE)):
#     tr2, pr2 = data_for_stream2(DATA_DIR, train_seq_len = train, pred_seq_len = pred, frame_lenth_cap = train+pred)
#     save_to_pkl(DIR + 'stream2_obs_data_{}.pkl'.format(TYPE), tr2)
#     save_to_pkl(DIR + 'stream2_pred_data_{}.pkl'.format(TYPE), pr2)


# if not os.path.exists(DIR+'stream2_obs_eigs_{}.pkl'.format(TYPE)) and not os.path.exists(DIR+'stream2_pred_eigs_{}.pkl'.format(TYPE)):
tr_seq_2 = pickle.load(open('{}/{}'.format(DIR,NAME1), 'rb'))
print('computing train eigens for {} {}...'.format(DATA, TYPE))
eigs_train = compute_eigs(tr_seq_2, 'train')
# save_to_pkl(DIR + 'stream2_obs_eigs_{}{}.pkl'.format(TYPE, len(eigs_train)), eigs_train)
save_to_pkl(DIR + 'stream2_obs_eigs_{}.pkl'.format(TYPE), eigs_train)
del tr_seq_2
del eigs_train

pred_seq_2 = pickle.load(open('{}/{}'.format(DIR, NAME2), 'rb'))
print('computing pred eigens for {} {}...'.format(DATA, TYPE))
eigs_pred = compute_eigs(pred_seq_2, 'pred')
#save_to_pkl(DIR + 'stream2_pred_eigs_{}{}.pkl'.format(TYPE, len(eigs_pred)), eigs_pred)
save_to_pkl(DIR + 'stream2_pred_eigs_{}.pkl'.format(TYPE), eigs_pred)
del pred_seq_2
del eigs_pred


