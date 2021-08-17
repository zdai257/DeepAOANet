import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import join
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, \
    mean_squared_error, mean_absolute_error, accuracy_score
from AOAtrain import train


### Globals ###
# Pandas col names: Diagonal + upper triangle components of R
fields_lst = ['field.data0', 'field.data1', 'field.data2', 'field.data3', 'field.data4', 'field.data5', 'field.data6', 'field.data7', \
              'field.data10', 'field.data11', 'field.data12', 'field.data13', 'field.data14', 'field.data15', \
              'field.data20', 'field.data21', 'field.data22', 'field.data23', \
              'field.data30', 'field.data31']
# Pandas col names: Upper triangle components of R
fields_filter_lst = ['field.data2', 'field.data3', 'field.data4', 'field.data5', 'field.data6', 'field.data7', \
              'field.data12', 'field.data13', 'field.data14', 'field.data15', \
              'field.data22', 'field.data23']
# LoRa signal strength threshold versus noises
field_thres = 1e-05
# Names of noisy data subdirectory
noisy_dir_lst = ['noisy1', 'noisy2', 'noisy3', 'noisy4', 'noisy5']


### Prepare Data from ROSBAG -> CSV -> DataFrame ###
# Load DataFrame from CSV.
def load_raw(data_dir='testLOS', field_thres=field_thres, measure_music=False):
    aoa_dict = {}

    for filename in os.listdir(data_dir):
        if filename.startswith("deg_m") and filename.endswith(".csv"):
            aoa = pd.read_csv(join(data_dir, filename), sep=',', header=0)
            aoa_Rjk = aoa[fields_lst]
            aoa_dict[- float(filename[5:-4])] = aoa_Rjk
        elif filename.startswith("deg_") and filename.endswith(".csv"):
            aoa = pd.read_csv(join(data_dir, filename), sep=',', header=0)
            aoa_Rjk = aoa[fields_lst]
            aoa_dict[float(filename[4:-4])] = aoa_Rjk

    # Degree to Radian
    for key in aoa_dict.keys():
        aoa_dict[key]['theta'] = key * math.pi / 180

    # Differentiate R of LoRa signal or noise
    aoa_sig, aoa_noi = {}, {}
    sig_index_lst = {}

    # Filtering noise based on the threshold
    for key in aoa_dict.keys():
        filtered_indexed = aoa_dict[key][(abs(aoa_dict[key]['field.data2']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data3']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data4']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data5']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data6']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data7']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data12']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data13']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data14']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data15']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data22']) >= field_thres) \
                                         & (abs(aoa_dict[key]['field.data23']) >= field_thres)].index

        aoa_noi[key] = aoa_dict[key].drop(filtered_indexed)
        aoa_sig[key] = aoa_dict[key].iloc[filtered_indexed]
        sig_index_lst[key] = (filtered_indexed)

    # Concat dict of raw data into a DataFrame
    X = pd.concat([aoa_sig[rad] for rad in aoa_sig.keys()],
                  keys=[rad for rad in aoa_sig.keys()], names=['Series name', 'Row ID'])

    # Add a Column of MUSIC measurement; if no measurement, add 'NaN'
    if measure_music:
        music_dict = {}
        for filename in os.listdir(data_dir):
            if filename.startswith("music_deg_m") and filename.endswith(".csv"):
                music_deg = pd.read_csv(join(data_dir, filename), sep=',', header=0)
                music_deg.rename(columns={'field.data': 'music'}, inplace=True)
                music_series = music_deg['music']
                music_dict[- float(filename[11:-4])] = music_series
            elif filename.startswith("music_deg_") and filename.endswith(".csv"):
                music_deg = pd.read_csv(join(data_dir, filename), sep=',', header=0)
                music_deg.rename(columns={'field.data': 'music'}, inplace=True)
                music_series = music_deg['music']
                music_dict[float(filename[10:-4])] = music_series

        for key in music_dict.keys():
            music_dict[key] = music_dict[key][sig_index_lst[key]]

        Xmusic = pd.concat([music_dict[rad] for rad in range(-70, 80, 10)], keys=[rad for rad in range(-70, 80, 10)],
                           names=['Series name', 'Row ID'])
        X = pd.concat([X, Xmusic], axis=1)

    else:
        Xmusic = pd.Series([0] * len(X.index))
        X = pd.concat([X, Xmusic], axis=1)

    return X, Xmusic, sig_index_lst


# Load synthetic noisy data with various Sigma
def create_dataset(dirname, sigma, Xmusic, sig_index_lst):
    aoa_dict = {}
    # fields_lst = ['field.data0', 'field.data1', 'field.data2', 'field.data3', 'field.data5', 'field.data6', 'field.data7', 'field.data10', 'field.data11', 'field.data15']
    fields_lst = ['field.data0', 'field.data1', 'field.data2', 'field.data3', 'field.data4', 'field.data5',
                  'field.data6', 'field.data7', \
                  'field.data10', 'field.data11', 'field.data12', 'field.data13', 'field.data14', 'field.data15', \
                  'field.data20', 'field.data21', 'field.data22', 'field.data23', \
                  'field.data30', 'field.data31']

    for filename in os.listdir(dirname):
        # print(join(dirname, filename))
        if filename.startswith("deg_m") and filename.endswith("_" + sigma + ".csv"):
            aoa = pd.read_csv(join(dirname, filename), sep=',', header=0)
            aoa_Rjk = aoa[fields_lst]
            aoa_dict[- float(filename[5:-9])] = aoa_Rjk
        elif filename.startswith("deg_") and filename.endswith("_" + sigma + ".csv"):
            aoa = pd.read_csv(join(dirname, filename), sep=',', header=0)
            aoa_Rjk = aoa[fields_lst]
            aoa_dict[float(filename[4:-9])] = aoa_Rjk

    aoa_sig, aoa_noi = {}, {}
    # sig_index_lst = {}

    for key in aoa_dict.keys():
        aoa_dict[key]['theta'] = key * math.pi / 180
        # print(key ,aoa_dict[key].shape)
        # print(sig_index_lst[key])

        # Patch: cut positional idx that are out-of-bound
        # slice_idx = sig_index_lst[key]
        # slice_idx = slice_idx[slice_idx<len(sig_index_lst[key])]
        try:
            aoa_sig[key] = aoa_dict[key].iloc[sig_index_lst[key], :]
        except IndexError:
            aoa_sig[key] = aoa_dict[key].iloc[sig_index_lst[key][:-3], :]

    Xaoa_noi = pd.concat([aoa_sig[rad] for rad in range(-70, 80, 10)],
                         keys=[rad for rad in range(-70, 80, 10)], names=['Series name', 'Row ID'])

    Xaoa_noi_music = pd.concat([Xaoa_noi, Xmusic], axis=1)
    return Xaoa_noi_music

# Obtain a list of DataFrames from each folder
def get_X_list(dirname, noi_dir_lst=noisy_dir_lst):
    Xaoa_music = pd.read_csv(join(dirname, 'df_collection', 'Xaoa_music.csv'), index_col=[0, 1])
    Xnoisy_parts = [Xaoa_music]
    with open(join(dirname, 'df_collection', 'sig_index.pkl'), "rb") as a_file:
        sig_index_lst = pickle.load(a_file)

    Xmusic = pd.read_csv(join(dirname, 'df_collection', 'music_series.csv'), index_col=[0, 1])

    for noi_dir in noi_dir_lst:
        noisy_dir = join(dirname, noi_dir)

        Xnoisy_parts.append(create_dataset(noisy_dir, '1e_5', Xmusic, sig_index_lst))
        Xnoisy_parts.append(create_dataset(noisy_dir, '5e_5', Xmusic, sig_index_lst))
        Xnoisy_parts.append(create_dataset(noisy_dir, '1e_4', Xmusic, sig_index_lst))
        Xnoisy_parts.append(create_dataset(noisy_dir, '5e_4', Xmusic, sig_index_lst))

    return Xnoisy_parts

# Concat R(t-2, t-1, t) as Time Series and Convert DataFrames into Numpy for training
def Split_TimeSeries3(Xaoa, aoa_gt_series=range(-70, 80, 10), feature_space=20):
    # Xarr: time series of Rjk
    Xaoa_cpy = Xaoa.copy(deep=True)

    Xarr, ylst = np.empty((0, 3, feature_space), dtype='float32'), []
    rowt_0, rowt_1, rowt_2 = None, None, None
    row_up, row_mid, row_down = -1, -1, -1

    for serName in aoa_gt_series:
        isFirstRow = True
        isSecRow = True
        for rowId, row in Xaoa_cpy.loc[serName].iterrows():
            row_up = row_mid
            rowt_2 = rowt_1
            row_mid = row_down
            rowt_1 = rowt_0
            row_down = rowId

            rowt_0 = row.drop(['theta', 'music']).to_numpy(dtype='float32').reshape(1, 1, feature_space)
            # Normalization!
            rowt_0 = rowt_0 / np.linalg.norm(rowt_0)
            # print(rowt_0)

            if isFirstRow:
                isFirstRow = False
                continue

            if isSecRow:
                isSecRow = False
                continue

            if row_down - row_mid == 1 and row_mid - row_up == 1:
                arr_tmp = np.concatenate((rowt_2, rowt_1, rowt_0), axis=1)

                Xarr = np.append(Xarr, arr_tmp, axis=0)
                ylst.append([row['theta'], row['music']])

    # Xvec: slice of a Rjk row
    Xvec = Xaoa_cpy.drop(['theta', 'music'], axis=1).to_numpy()
    yvec = Xaoa_cpy[['theta', 'music']].to_numpy()

    # print(Xarr.shape, len(ylst))
    return Xarr, ylst, Xvec, yvec


if __name__ == '__main__':
    # Create DataFrames for each dataset among {1606: LOS, 0107: office, 0207: corridor, 1407: LOS+reflector, 1607: indoor2outdoor}
    '''
    for work_dir in ['data_1606', 'data_0107', 'data_0207', 'data_1407', 'data_1607']:
        df, df_music, df_sig_index = load_raw(data_dir=work_dir, field_thres=field_thres, measure_music=True)
        # Saving
        df_music.to_csv(join(work_dir, 'df_collection', 'music_series.csv'), index=True, header=True)
        with open(join(work_dir, 'df_collection', 'sig_index.pkl'), "wb") as a_file:
            pickle.dump(df_sig_index, a_file)
        print(df)
    '''
    # Load and Append raw & noisy DataFrames in each folder into a list
    X_1607 = get_X_list('data_1607')  # NLOS interior-to-outdoor
    X_1407 = get_X_list('data_1407')  # LOS open with reflector
    X_0207 = get_X_list('data_0207')  # NLOS corridor
    X_0107 = get_X_list('data_0107')  # NLOS office
    X_1606 = get_X_list('data_1606')  # LOS open area
    # Append all datasets as a Pre-training list of data
    X = X_1606 + X_0207 + X_0107 + X_1407 + X_1607

    # Create {Xarr, yarr} as Time Series pairs for training
    # Create {Xvec, yvec} as single-snapshot Rs for training
    Xarr = np.array([])
    yarr = np.array([])
    Xvec = np.array([])
    yvec = np.array([])

    for item in X:
        Xarr0, ylst0, Xvec0, yvec0 = Split_TimeSeries3(item, feature_space=20)

        Xarr = np.concatenate((Xarr, Xarr0), axis=0) if Xarr.size else Xarr0
        Xvec = np.concatenate((Xvec, Xvec0), axis=0) if Xvec.size else Xvec0
        yvec = np.concatenate((yvec, yvec0), axis=0) if yvec.size else yvec0
        yarr0 = np.array(ylst0)
        yarr = np.concatenate((yarr, yarr0), axis=0) if yarr.size else yarr0

    print("Xarr shape = ", Xarr.shape)
    print("yarr shape = ", yarr.shape)
    print("Xvec shape = ", Xvec.shape)
    print("yvec shape = ", yvec.shape)

    # Remove 'NaN' in Numpy dataset
    Xvec_clean = Xvec[~np.isnan(Xvec).any(axis=1)]
    yvec_clean = yvec[~np.isnan(yvec).any(axis=1)]
    Xarr_clean = Xarr[~np.isnan(Xarr).any(axis=(1, 2))]
    yarr_clean = yarr[~np.isnan(yarr).any(axis=1)]


    # SHUFFLE & Split for training
    X_train, X_test, y_train, y_test = train_test_split(Xarr_clean, yarr_clean, test_size=0.2, random_state=42)
    # Create X_val, y_val from X_train, y_train
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

    # Normalization
    # extract music from y_test for benchmark
    y_test_music = []
    for item in y_test:
        if item[1] > 180.0:
            item[1] -= 360
        y_test_music.append(item[1])
    # Take the labels in [Radians] as y
    y_train = [item[0] for item in y_train]
    y_test = [item[0] for item in y_test]
    y_val = [item[0] for item in y_val]
    # Normalize y
    ymax = max(y_train)
    ymin = min(y_train)
    y_train_std = (y_train - ymin) / (ymax - ymin)
    y_test_std = (y_test - ymin) / (ymax - ymin)
    y_val_std = (y_val - ymin) / (ymax - ymin)

    print('Y in range: (%.1f, %.1f)' % (min(y_train_std), max(y_train_std)))

    ### Training & Evaluation ###
    training = False
    evaluation = False

    if training:
        trained_model, history = train(X_train, y_train_std, X_val, y_val_std, epochs=300, batch_size=256)


    if evaluation:
        pred_y_test_std = trained_model.predict(X_test)
        pred_y_test = pred_y_test_std * (ymax - ymin) + ymin

        rmse_model = np.sqrt(mean_squared_error(y_test, pred_y_test))
        print("RMSE of MODEL: ", rmse_model)
        print("I.E. %.04f degree of mean error" % (rmse_model / math.pi * 180))
