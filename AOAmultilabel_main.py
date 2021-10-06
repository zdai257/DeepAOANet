import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import join
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import rosbag
from std_msgs.msg import String, Empty, Header, Float32, Float32MultiArray, MultiArrayDimension
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, \
    mean_squared_error, mean_absolute_error, accuracy_score
from AOAtrain import train
from AOAmain import load_bag


### Globals ###
# Raw data dir
Data_dir = ['data_1606', 'data_0107', 'data_0207', 'data_1407', 'data_1607']
Noisy_lst = ['noisy20', 'noisy21', 'noisy22', 'noisy23', 'noisy24']
Bag_lst = ['deg_0.bag', 'deg_10.bag', 'deg_20.bag', 'deg_30.bag', 'deg_40.bag', \
           'deg_50.bag', 'deg_60.bag', 'deg_70.bag', 'deg_m10.bag', 'deg_m20.bag', \
           'deg_m30.bag', 'deg_m40.bag', 'deg_m50.bag', 'deg_m60.bag', 'deg_m70.bag']

# Synthetic Multilabel data dir
Data_dir_multi = ['data_multi-30', 'data_multi-31', 'data_multi-32', 'data_multi-33', 'data_multi-34', 'data_multi-35']
bag_lst_multi = []
for filename in os.listdir('data_multi-30'):
    bag_lst_multi.append(filename)


def load_bag_multi(bagname, data_dir=Data_dir_multi[0], field_thres=1e-4, cast2image=False):
    with rosbag.Bag(join(data_dir, bagname)) as bag:

        deg1 = bagname.split("_")[0]
        deg2 = bagname.split("-")[1].split(".")[0]

        if deg1.startswith('m'):
            deg1 = -int(deg1[1:])
        else:
            deg1 = int(deg1)
        if deg2.startswith('m'):
            deg2 = -int(deg2[1:])
        else:
            deg2 = int(deg2)

        topic_head = '/kerberos/R_'
        angle_digit = len(topic_head)

        angle1_lst, angle2_lst, data_lst = [], [], []

        for topic, msg, t in bag.read_messages():
            if topic.startswith(topic_head):
                angle1 = topic[angle_digit:angle_digit + 3]
                angle2 = topic[-3:]

                # Labels of (angle1, angle2) in a list
                angle1_lst.append(int(angle1) - 360)
                angle2_lst.append(int(angle2) - 360)

                data = np.asarray(msg.data).reshape((4, 4, 2, 8))

                if cast2image:
                    filtered_data = np.zeros((4, 4, 8))

                    for i in range(4):
                        for j in range(4):
                            if i <= j:
                                filtered_data[i, j, :] = data[i, j, 0, :]
                            else:
                                filtered_data[i, j, :] = data[i, j, 1, :]

                    # Moveaxis
                    filtered_data = np.moveaxis(filtered_data, -1, 0)
                else:

                    filtered_data = np.zeros((10, 2, 8))
                    k = 0
                    for i in range(4):
                        for j in range(4):
                            if i <= j:
                                filtered_data[k, :, :] = data[i, j, :, :]
                                k += 1

                    # Moveaxis
                    filtered_data = np.moveaxis(filtered_data, -1, 0)

                df_data = pd.DataFrame(filtered_data.reshape((1, -1), order='C'))

                data_lst.append(df_data)
            else:
                continue

        if len(data_lst) == 0:
            print(bagname, " recorded 0 message!")
            return None
        # else:
        #    print(len(data_lst))

        df = pd.concat(data_lst, axis=0, ignore_index=True)
        df['GT1'] = pd.Series(angle1_lst)
        df['GT2'] = pd.Series(angle2_lst)

        # Filter noise from Signal
        if cast2image:
            filtered_indexed = df[(abs(df[2]) >= field_thres) \
                                  & (abs(df[1]) >= field_thres) \
                                  & (abs(df[2]) >= field_thres) \
                                  & (abs(df[3]) >= field_thres) \
                                  & (abs(df[6]) >= field_thres) \
                                  & (abs(df[7]) >= field_thres) \
                                  & (abs(df[11]) >= field_thres) \
                                  & (abs(df[113]) >= field_thres) \
                                  & (abs(df[114]) >= field_thres) \
                                  & (abs(df[115]) >= field_thres) \
                                  & (abs(df[118]) >= field_thres) \
                                  & (abs(df[119]) >= field_thres) \
                                  & (abs(df[123]) >= field_thres)].index
        else:
            filtered_indexed = df[(abs(df[2]) >= field_thres) \
                                  & (abs(df[3]) >= field_thres) \
                                  & (abs(df[4]) >= field_thres) \
                                  & (abs(df[5]) >= field_thres) \
                                  & (abs(df[6]) >= field_thres) \
                                  & (abs(df[7]) >= field_thres) \
                                  & (abs(df[10]) >= field_thres) \
                                  & (abs(df[11]) >= field_thres) \
                                  & (abs(df[12]) >= field_thres) \
                                  & (abs(df[13]) >= field_thres) \
                                  & (abs(df[16]) >= field_thres) \
                                  & (abs(df[17]) >= field_thres) \
                                  & (abs(df[142]) >= field_thres) \
                                  & (abs(df[143]) >= field_thres) \
                                  & (abs(df[144]) >= field_thres) \
                                  & (abs(df[145]) >= field_thres) \
                                  & (abs(df[146]) >= field_thres) \
                                  & (abs(df[147]) >= field_thres) \
                                  & (abs(df[150]) >= field_thres) \
                                  & (abs(df[151]) >= field_thres) \
                                  & (abs(df[152]) >= field_thres) \
                                  & (abs(df[153]) >= field_thres) \
                                  & (abs(df[156]) >= field_thres) \
                                  & (abs(df[157]) >= field_thres)].index

        filtered_df = df.iloc[filtered_indexed]

    return filtered_df

# Create y for AEC (Multihot encoded)
def create_multihot_encoded(y, labels_deg=np.arange(-74, 76, 2)):
    SF = np.size(labels_deg, 0)

    y_hot_encoded = np.zeros((len(y), SF), dtype=np.uint8)
    for index, i in np.ndenumerate(y):
        for idx, j in enumerate(labels_deg):
            if i == j:
                y_hot_encoded[index[0], idx] = 1

    return y_hot_encoded

def eval_model_multi(model, X_eval, yhot_gt, y_gt, label_angls=np.arange(-74, 76, 2), IsSavingModel=None):
    pred_yhot_eval = model.predict(X_eval)

    pred_yhot_eval_round = pred_yhot_eval.round()

    # Compute abs Multilabel matching Accuracy
    Multilabel_accu = (pred_yhot_eval_round == yhot_gt).all(axis=1).mean()

    # Compute MSE
    pred_deg_eval = -360 * np.ones(y_gt.shape)
    for index, val in np.ndenumerate(pred_yhot_eval_round):
        if val == 1:
            if pred_deg_eval[index[0], 0] != -360:
                pred_deg_eval[index[0], 1] = label_angls[index[1]]
            else:
                pred_deg_eval[index[0], 0] = label_angls[index[1]]

    for index, val in np.ndenumerate(pred_deg_eval):
        if val == -360:
            pred_deg_eval[index[0], 1] = pred_deg_eval[index[0], 0]

    RMSE_sum = np.linalg.norm(pred_deg_eval - y_gt, axis=1)
    RMSE = np.sum(RMSE_sum) / len(RMSE_sum)

    # Save
    if IsSavingModel is not None:
        with open(join('checkpoints', IsSavingModel), 'wb') as a_file:
            pickle.dump(model, a_file)

    return Multilabel_accu, RMSE


if __name__ == '__main__':
    # Save/Load 'Xm': a list of raw DataFrames
    IsLoading = True
    IsCNN = True

    if 1:
        if IsLoading:
            # Load single-source X
            with open(join('Data', 'I_375.pkl'), 'rb') as a_file:
                X = pickle.load(a_file)

            # Load all Xms
            Xm = []
            with open(join('Data', 'I1_delay0_119.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I1_delay1_223.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I2_delay0_223.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I2_delay1_224.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I3_delay0_109.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I3_delay1_224.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I4_delay0_223.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I4_delay1_224.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I6_delay0_122.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

            with open(join('Data', 'I6_delay1_223.pkl'), 'rb') as a_file:
                Xm.extend(pickle.load(a_file))

        else:
            ### Load EACH rosbags into list Xm
            Xm = []
            for bagname in bag_lst_multi:
                df0 = load_bag_multi(bagname, data_dir='data_multi-61', field_thres=1e-10, cast2image=False)
                if df0 is not None:
                    Xm.append(df0)

            #with open(join('Data', 'Xm_225.pkl'), 'wb') as a_file:
            #    pickle.dump(Xm, a_file)

            ### Load all rosbags into list X
            X = []
            for data_dir in Data_dir:
                for noisy_idx in Noisy_lst:
                    for bagname in Bag_lst:
                        X.append(load_bag(data_dir, noisy_idx, bagname))

        ### Concatenate and Convert X to numpy X-y pairs
        Xvec = np.array([])
        yvec = np.array([])

        Xvec_lst, yvec_lst = [], []
        for item in X:
            Xvec_lst.append(item.drop(['GT', 'sigma'], axis=1).to_numpy())
            yvec_lst.append(item[['GT', 'sigma']].to_numpy())

        yvec = np.concatenate(yvec_lst, axis=0)

        Xvec = np.zeros((yvec.shape[0], 128), dtype=np.float32)
        tmp_row = 0
        for item in Xvec_lst:
            Xvec[tmp_row:tmp_row + item.shape[0], :] = item
            tmp_row += item.shape[0]

        print(Xvec.shape)
        print(yvec.shape)

        ### Concatenate and Convert Xm to numpy X-y pairs
        Xmvec_lst, ymvec_lst = [], []

        for item in Xm:
            Xmvec_lst.append(item.drop(['GT1', 'GT2'], axis=1).to_numpy())
            ymvec_lst.append(item[['GT1', 'GT2']].to_numpy())

        ymvec = np.concatenate(ymvec_lst, axis=0)

        Xmvec = np.zeros((ymvec.shape[0], Xmvec_lst[0].shape[1]), dtype=np.float32)
        tmp_row = 0
        for index, item in enumerate(Xmvec_lst):
            Xmvec[tmp_row:tmp_row + item.shape[0], :] = item
            tmp_row += item.shape[0]

        print(Xmvec.shape)
        print(ymvec.shape)


        ### MIX SINGLE/MULTI LABEL DATASET ###
        Xxvec = np.concatenate((Xvec, Xmvec), axis=0)

        yyvec = np.zeros((Xxvec.shape[0], 3), dtype=np.float32)

        # Single-source
        yyvec[:yvec.shape[0], 0] = 0.
        yyvec[:yvec.shape[0], 1] = yvec[:, 0]
        yyvec[:yvec.shape[0], 2] = yvec[:, 0]

        # Multiple-source
        yyvec[-ymvec.shape[0]:, 0] = 1.
        yyvec[-ymvec.shape[0]:, 1] = ymvec[:, 0]
        yyvec[-ymvec.shape[0]:, 2] = ymvec[:, 1]

        print(yyvec.shape)


        ### SHUFFLE & Split for training ###
        X_train, X_test, y_train, y_test = train_test_split(Xmvec, ymvec, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

        # Normalize StandardScaling
        sscaler = StandardScaler()
        sscaler.fit(X_train)
        print("Mean = ", sscaler.mean_)
        X_train_std = sscaler.transform(X_train)
        X_test_std = sscaler.transform(X_test)
        X_val_std = sscaler.transform(X_val)

        # Make y[:, 1:3] in ascending order
        y_train_sort = np.concatenate((y_train[:, 0:1], np.sort(np.asarray(y_train[:, 1:3], dtype=np.float32))), axis=1)
        print(X_train_std.shape, y_train_sort.shape)
        y_test_sort = np.concatenate((y_test[:, 0:1], np.sort(np.asarray(y_test[:, 1:3], dtype=np.float32))), axis=1)
        print(X_test_std.shape, y_test_sort.shape)
        y_val_sort = np.concatenate((y_val[:, 0:1], np.sort(np.asarray(y_val[:, 1:3], dtype=np.float32))), axis=1)
        print(X_val_std.shape, y_val_sort.shape)

        # Minmax-Normalize y
        ymin, ymax = -74, 74
        ym_train_std = np.concatenate((y_train_sort[:, 0:1], (y_train[:, 1:3] - ymin) / (ymax - ymin)), axis=1)
        ym_test_std = np.concatenate((y_test_sort[:, 0:1], (y_test[:, 1:3] - ymin) / (ymax - ymin)), axis=1)
        ym_val_std = np.concatenate((y_val_sort[:, 0:1], (y_val[:, 1:3] - ymin) / (ymax - ymin)), axis=1)
        print(ym_train_std.shape)

        if IsCNN:
            ### If cast input features to (4, 4, CHN) images ###
            I_train_std = X_train_std.reshape((X_train_std.shape[0], 8, 4, 4))
            I_train_std = np.moveaxis(I_train_std, 1, -1)
            print(I_train_std.shape)

            I_test_std = X_test_std.reshape((X_test_std.shape[0], 8, 4, 4))
            I_test_std = np.moveaxis(I_test_std, 1, -1)
            I_val_std = X_val_std.reshape((X_val_std.shape[0], 8, 4, 4))
            I_val_std = np.moveaxis(I_val_std, 1, -1)
            print(I_test_std.shape)
            print(I_val_std.shape)


    ### Training & Evaluation ###
    training = True
    evaluation = False

    if training:
        trained_model, history = train(I_train_std, ym_train_std, I_test_std, ym_test_std, epochs=30, batch_size=512)

    if evaluation:
        # ??
        accu, rmse = eval_model_multi(trained_model, I_val_std, ym_val_std, y_val, IsSavingModel=None)
        print(accu, rmse)

    if training:
        print(history.history.keys())

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
