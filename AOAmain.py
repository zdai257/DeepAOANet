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


### Globals ###
# Raw data dir
Data_dir = ['data_1606', 'data_0107', 'data_0207', 'data_1407', 'data_1607']
Noisy_lst = ['noisy20', 'noisy21', 'noisy22', 'noisy23', 'noisy24']
Bag_lst = ['deg_0.bag', 'deg_10.bag', 'deg_20.bag', 'deg_30.bag', 'deg_40.bag',
           'deg_50.bag', 'deg_60.bag', 'deg_70.bag', 'deg_m10.bag', 'deg_m20.bag',
           'deg_m30.bag', 'deg_m40.bag', 'deg_m50.bag', 'deg_m60.bag', 'deg_m70.bag']

# LoRa signal strength threshold versus noises in R
field_thres = 1e-05


### Prepare Data from ROSBAG -> CSV -> DataFrame ###
def load_bag(data_dir=Data_dir[0], noisy_lst=Noisy_lst[0], bag_lst=Bag_lst[0], field_thres=1e-4, cast2image=False):
    with rosbag.Bag(join(data_dir, noisy_lst, bag_lst)) as bag:
        angl_val_lst = range(-74, 76, 2)

        # col_lst = range(256)  #160
        # df = pd.DataFrame(columns=col_lst)

        topic_head = '/kerberos/R_'
        angle_digit = len(topic_head)

        angle_lst, sigma_lst, data_lst = [], [], []

        for topic, msg, t in bag.read_messages():
            if topic.startswith(topic_head):
                angle = topic[angle_digit:angle_digit + 3]
                sigma = topic[-4:]
                if noisy_lst != 'noisy20' and sigma == '0e_4':
                    continue

                angle_lst.append(int(angle) - 360)
                sigma_lst.append(sigma)

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

        df = pd.concat(data_lst, axis=0, ignore_index=True)
        df['GT'] = pd.Series(angle_lst)
        df['sigma'] = pd.Series(sigma_lst)

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

def create_hot_encoded(y, labels_deg=np.arange(-70., 80., 10.)):
    SF = np.size(labels_deg, 0)

    y_hot_encoded = np.zeros((len(y), SF), dtype=np.uint8)
    for index, i in enumerate(y):
        for idx, j in enumerate(labels_deg):
            if i == j:
                # X_hot_encoded[index, :, idx] = X[index]
                y_hot_encoded[index, idx] = 1

    return y_hot_encoded


def eval_model(model, X_eval, yhot_gt, label_angls=np.arange(-74, 76, 2), IsSavingModel=None):
    pred_yhot_eval = model.predict(X_eval)
    # PDF => OneHot => angle(deg)
    pred_yhot_eval_round = pred_yhot_eval.round()
    pred_y_idx_eval = np.argmax(pred_yhot_eval_round, axis=1)

    pred_y_eval = np.zeros(pred_y_idx_eval.shape[0])
    for idx, val in enumerate(list(pred_y_idx_eval)):
        pred_y_eval[idx] = label_angls[val]

    # Process GT
    y_idx_gt = np.argmax(yhot_gt, axis=1)
    gt_y = np.zeros(y_idx_gt.shape[0])
    for idx, val in enumerate(list(y_idx_gt)):
        gt_y[idx] = label_angls[val]

    accuracy = accuracy_score(y_idx_gt, pred_y_idx_eval)
    RMSE = mean_squared_error(gt_y, pred_y_eval, squared=False)

    # Save
    if IsSavingModel is not None:
        with open(join('checkpoints', IsSavingModel), 'wb') as a_file:
            pickle.dump(model, a_file)

    return accuracy, RMSE


if __name__ == '__main__':
    # Save/Load 'X': a list of raw DataFrames
    IsLoading = False

    if IsLoading:
        if IsLoading:
            with open(join('Data', 'X_375.pkl'), 'rb') as a_file:
                X = pickle.load(a_file)
        else:
            ### Load all rosbags into list X
            X = []
            for data_dir in Data_dir:
                for noisy_idx in Noisy_lst:
                    for bagname in Bag_lst:
                        X.append(load_bag(data_dir, noisy_idx, bagname))
            with open(join('Data', 'X_375.pkl'), 'wb') as a_file:
                pickle.dump(X, a_file)

        ### Concatenate and Convert X to numpy X-y pairs
        Xvec = np.array([])
        yvec = np.array([])

        Xvec_lst, yvec_lst = [], []

        for item in X:
            Xvec_lst.append(item.drop(['GT', 'sigma'], axis=1).to_numpy())
            yvec_lst.append(item[['GT', 'sigma']].to_numpy())

        yvec = np.concatenate(yvec_lst, axis=0)
        Xvec = np.zeros((yvec.shape[0], 160), dtype=np.float32)
        tmp_row = 0
        for item in Xvec_lst:
            Xvec[tmp_row:tmp_row + item.shape[0], :] = item
            tmp_row += item.shape[0]

        print(Xvec.shape)
        print(yvec.shape)

        # SHUFFLE & Split for training
        X_train, X_test, y_train, y_test = train_test_split(Xvec, yvec, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

        # Normalization
        # X
        xmax = np.max(X_train)
        xmin = np.min(X_train)
        print(xmin, xmax)
        X_train_std = 2 * (X_train - xmin) / (xmax - xmin) - 1
        X_test_std = 2 * (X_test - xmin) / (xmax - xmin) - 1
        X_val_std = 2 * (X_val - xmin) / (xmax - xmin) - 1

        # y
        y_train = np.asarray(y_train[:, 0], dtype=np.float32)
        print(y_train.shape)
        y_test = np.asarray(y_test[:, 0], dtype=np.float32)
        print(y_test.shape)
        y_val = np.asarray(y_val[:, 0], dtype=np.float32)
        print(y_val.shape)

        ymax = max(y_train)
        ymin = min(y_train)
        print(ymin, ymax)
        y_train_std = (y_train - ymin) / (ymax - ymin)
        y_test_std = (y_test - ymin) / (ymax - ymin)
        y_val_std = (y_val - ymin) / (ymax - ymin)

        # y => Mulit/One Hot encoded
        label_angls = np.arange(-74, 76, 2)

        yhot_train = create_hot_encoded(y_train, label_angls)
        yhot_test = create_hot_encoded(y_test, label_angls)
        yhot_val = create_hot_encoded(y_val, label_angls)

    # Load Splited & Shuffled numpy data from HERE!
    IsLoadingXy = not IsLoading

    if IsLoadingXy:
        with open(join('Data', 'x_train_std.pkl'), 'rb') as a_file:
            X_train_std = pickle.load(a_file)
        with open(join('Data', 'x_val_std.pkl'), 'rb') as a_file:
            X_val_std = pickle.load(a_file)
        with open(join('Data', 'x_test_std.pkl'), 'rb') as a_file:
            X_test_std = pickle.load(a_file)
    else:
        with open(join('Data', 'x_test_std.pkl'), 'wb') as a_file:
            pickle.dump(X_test_std, a_file)
        with open(join('Data', 'x_train_std.pkl'), 'wb') as a_file:
            pickle.dump(X_train_std, a_file)
        with open(join('Data', 'x_val_std.pkl'), 'wb') as a_file:
            pickle.dump(X_val_std, a_file)
    if IsLoadingXy:
        with open(join('Data', 'yhot_375.pkl'), 'rb') as a_file:
            yhot = pickle.load(a_file)
            yhot_train, yhot_val, yhot_test = yhot[0], yhot[1], yhot[2]
    else:
        with open(join('Data', 'yhot_375.pkl'), 'wb') as a_file:
            yhot = [yhot_train, yhot_val, yhot_test]
            pickle.dump(yhot, a_file)


    ### Training & Evaluation ###
    training = False
    evaluation = False

    if training:
        trained_model, history = train(X_train_std, yhot_train, X_test_std, yhot_test, epochs=200, batch_size=256)

    if evaluation:
        eval_model(trained_model, X_val_std, yhot_val, IsSavingModel='model0.pkl')

    if training:
        print(history.history.keys())

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
