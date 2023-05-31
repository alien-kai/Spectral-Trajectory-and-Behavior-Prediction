import pickle
import cv2
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

recording = 2
recording = "{:02d}".format(int(recording))
BS=256
tracks_file = f"../rounD-dataset-v1.0/data/{recording}_tracks.csv"
tracks=pd.read_csv(tracks_file)
start_frame=min(tracks['frame'])
end_frame=max(tracks['frame'])
DIR = f'../resources/data/{recording}/{BS}/{start_frame}_{end_frame}/'

background_image_path=f'../rounD-dataset-v1.0/data/{recording}_background.png'
dataset_params_path='../visualizer_params.json'

start_index = 0
stop_index = 10000
f1 = open(DIR + 'stream1_obs_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
g1 = open(DIR + 'stream1_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
train_sequence_stream1 = pickle.load(f1)  # load file content as mydict
pred_sequence_stream_1 = pickle.load(g1)  # load file content as mydict
f1.close()
g1.close()

stream1_train_batch = train_sequence_stream1[start_index:stop_index]
stream1_pred_batch = pred_sequence_stream_1[start_index:stop_index]


# print(stream1_train_batch[0]['sequence'])
# print(stream1_pred_batch[0]['sequence'])
print(f"agent_ID: {stream1_train_batch[0]['agent_ID']}")
print(f"shape: {stream1_train_batch[0]['sequence'].shape}")


background_image = cv2.imread(background_image_path)
background_image = cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)

# plt.title(f"batch:{1}")
# plt.scatter(x_train,y_train,c='black')
# plt.scatter(x_pred,y_pred,c='r')
# plt.imshow(background_image)


# for i in range(len(x)):
#     plt.annotate(int(id[i]),(x[i],y[i]))

recording=pd.read_csv(f'../rounD-dataset-v1.0/data/{recording}_recordingMeta.csv')
ortho_px_to_meter=recording['orthoPxToMeter'].values[0]
print("ortho_px_to_meter: {}".format(ortho_px_to_meter))
with open(dataset_params_path) as f:
    dataset_params = json.load(f)
scale_down_factor=dataset_params["datasets"]["round"]["scale_down_factor"]
print(f"scale_down_factor: {scale_down_factor}")

def calculate_center(i):
    x_train = [x[0] for x in stream1_train_batch[i]['sequence']]
    y_train = [x[1] for x in stream1_train_batch[i]['sequence']]
    x_pred = [x[0] for x in stream1_pred_batch[i]['sequence']]
    y_pred = [x[1] for x in stream1_pred_batch[i]['sequence']]

    x_train=[x/ortho_px_to_meter for x in x_train]
    y_train=[x/ortho_px_to_meter for x in y_train]
    x_train=[x/scale_down_factor for x in x_train]
    y_train=[x/scale_down_factor for x in y_train]

    x_pred=[x/ortho_px_to_meter for x in x_pred]
    y_pred=[x/ortho_px_to_meter for x in y_pred]
    x_pred=[x/scale_down_factor for x in x_pred]
    y_pred=[x/scale_down_factor for x in y_pred]

    center_point_train=[]
    for i in range(len(x_train)):
        center_point_train.append((x_train[i],-y_train[i]))
    # print(center_point)
    center_point_pred=[]
    for i in range(len(x_train)):
        center_point_pred.append((x_pred[i],-y_pred[i]))

    return center_point_train,center_point_pred

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(15, 8)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.00)
ax.imshow(background_image)

agent_ID=[]
batch_ID=[]
for i in range(len(stream1_train_batch)):
    if stream1_train_batch[i]['agent_ID'] not in agent_ID:
        batch_ID.append(i)
        agent_ID.append(stream1_train_batch[i]['agent_ID'])
print(f"batch_ID:{batch_ID}")
print(f"agent_ID:{agent_ID}")

for i in batch_ID:
    center_point_train, center_point_pred = calculate_center(i)
    for center in center_point_train:
        Drawing_colored_circle=plt.Circle(center, radius=0.5, color='blue')
        ax.add_artist(Drawing_colored_circle)
    for center in center_point_pred:
        Drawing_colored_circle=plt.Circle(center, radius=0.5, color='red')
        ax.add_artist(Drawing_colored_circle)
    ax.text(center_point_pred[0][0], center_point_pred[0][1], "ID: " + str(stream1_train_batch[i]['agent_ID']),fontsize=10, color='black')


ax.axis('off')
plt.show()
# plt.savefig('show_stream1.png')
