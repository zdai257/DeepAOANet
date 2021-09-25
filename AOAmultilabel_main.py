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
Data_dir_multi = ['data_multi-30', 'data_multi-31', 'data_multi-32', 'data_multi-33', 'data_multi-34', 'data_multi-35']
bag_lst_multi = []
for filename in os.listdir('data_multi-30'):
    bag_lst_multi.append(filename)


def load_bag_multi(bagname, data_dir='data_multi-30', field_thres=1e-4):
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

    if IsLoading:
        if IsLoading:
            with open(join('Data', 'Xm_225.pkl'), 'rb') as a_file:
                Xm = pickle.load(a_file)
        else:
            ### Load all rosbags into list Xm
            Xm = []
            for bagname in bag_lst_multi:
                df0 = load_bag_multi(bagname)
                if df0 is not None:
                    Xm.append(df0)
            with open(join('Data', 'Xm_225.pkl'), 'wb') as a_file:
                pickle.dump(Xm, a_file)

        ### Concatenate and Convert X to numpy X-y pairs
        Xmvec_lst, ymvec_lst = [], []
        for item in Xm:
            Xmvec_lst.append(item.drop(['GT1', 'GT2'], axis=1).to_numpy())
            ymvec_lst.append(item[['GT1', 'GT2']].to_numpy())

        ymvec = np.concatenate(ymvec_lst, axis=0)

        Xmvec = np.zeros((ymvec.shape[0], 160), dtype=np.float32)
        tmp_row = 0
        for item in Xmvec_lst:
            Xmvec[tmp_row:tmp_row + item.shape[0], :] = item
            tmp_row += item.shape[0]

        print(Xmvec.shape)
        print(ymvec.shape)

        # SHUFFLE & Split for training
        X_train, X_test, y_train, y_test = train_test_split(Xmvec, ymvec, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

        # Normalization
        xmax = np.max(X_train)
        xmin = np.min(X_train)
        print("X_train max value = %.04f; X_train min value = %.04f" % (xmin, xmax))
        X_train_std = 2 * (X_train - xmin) / (xmax - xmin) - 1
        X_test_std = 2 * (X_test - xmin) / (xmax - xmin) - 1
        X_val_std = 2 * (X_val - xmin) / (xmax - xmin) - 1

        # Make y(angle1, angle2) in ascending order
        y_train = np.sort(np.asarray(y_train[:, 0:2], dtype=np.float32))
        print(y_train.shape)
        y_test = np.sort(np.asarray(y_test[:, 0:2], dtype=np.float32))
        print(y_test.shape)
        y_val = np.sort(np.asarray(y_val[:, 0:2], dtype=np.float32))
        print(y_val.shape)

        # y => Mulit/One Hot encoded
        label_angls = np.arange(-74, 76, 2)

        yhot_train = create_multihot_encoded(y_train, label_angls)
        yhot_test = create_multihot_encoded(y_test, label_angls)
        yhot_val = create_multihot_encoded(y_val, label_angls)

    # Load Splited & Shuffled numpy data from HERE!
    '''
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
    '''

    ### Training & Evaluation ###
    training = True
    evaluation = True

    if training:
        trained_model, history = train(X_train_std, yhot_train, X_test_std, yhot_test, epochs=2, batch_size=512)

    if evaluation:
        accu, rmse = eval_model_multi(trained_model, X_val_std, yhot_val, y_val, IsSavingModel=None)
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
