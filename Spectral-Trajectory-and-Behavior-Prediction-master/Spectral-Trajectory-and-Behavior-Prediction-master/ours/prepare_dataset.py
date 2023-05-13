import numpy as np
# from loguru import logger
import pandas as pd
from tqdm import tqdm

pd.set_option('display.max_rows', None) # 展示所有行
pd.set_option('display.max_columns', None) # 展示所有列

dataset_dir = '../rounD-dataset-v1.0/data/'
recording = 2

#选取某一个agent
agentID=9


recording = "{:02d}".format(int(recording))

print(f"Loading recording {recording} from dataset rounD")

# Create paths to csv files
tracks_file = dataset_dir + recording + "_tracks.csv"
tracks_meta_file = dataset_dir + recording + "_tracksMeta.csv"
recording_meta_file = dataset_dir + recording + "_recordingMeta.csv"

tracks=pd.read_csv(tracks_file)
recording=pd.read_csv(recording_meta_file)

#选取开始和结束帧
# start_frame=min(tracks[tracks.trackId==agentID]['frame'])
# end_frame=max(tracks[tracks.trackId==agentID]['frame'])
start_frame=0
end_frame= 100
# end_frame=max(tracks['frame'])

print(f"agentID: {agentID}")
print(f"start_frame: {start_frame}")
print(f"end_frame: {end_frame}")
print(f"total_frame: {end_frame-start_frame}")

# print(f"total_obj: {np.unique(tracks['trackId'])}")

res=[]
#
#
for i in tqdm(range(len(tracks))):
    if tracks.loc[i,'frame']>=start_frame and tracks.loc[i,'frame']<=end_frame:
        # print(tracks.loc[i,:])
        # res=res.append(tracks.loc[i,['frame','trackId','xCenter','yCenter','recordingId']])
        # print(tracks.loc[i,['frame','trackId','xCenter','yCenter','recordingId']].tolist())
        res.append(tracks.loc[i,['frame','trackId','xCenter','yCenter','recordingId']].tolist())
        # print(res)
        # break

res=pd.DataFrame(res)
res.rename(columns={0:'frame',1:'trackId',2:'xCenter',3:'yCenter',4:'recordingId'},inplace=True)
print(res.shape)
print(res.head())


#set frame to [1-n]
res['frame']=tqdm(res['frame'].apply(lambda x:x-start_frame+1))
#set trackId to [1-n]
res['trackId']=tqdm(res['trackId'].apply(lambda x:x+1))

print(res.tail())
print(res.describe())
print(np.unique(res.trackId))


# save to npy
np.save('../resources/data/TrainSet0.npy', res)

# print(recording.columns)
# print(recording['frameRate'][0])

#load npy
# res=np.load('../../TrainSet0.npy')
# res=pd.DataFrame(res)
# print(res.shape)

# # print(res[res[1]==agentID+1].describe())
# # print(res[res[1]==agentID+1].head())
#
# frame=50
# x=res[res[0]==frame][2].tolist()
# y=res[res[0]==frame][3].tolist()
# id=res[res[0]==frame][1].tolist()
# print(x)
# print(y)
# print(id)
#
# import matplotlib.pyplot as plt
# plt.title(f"frame:{frame}")
# plt.scatter(x,y)
# for i in range(len(x)):
#     plt.annotate(int(id[i]),(x[i],y[i]))
# plt.show()




