import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add, LSTM, RepeatVector, TimeDistributed, Bidirectional
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers, Sequential
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils, plot_model


def build_encoder_fc(time_steps=3, feature_dim=20):

    x0 = Input(shape=(time_steps, feature_dim))
    l1 = LSTM(128, activation='relu', return_sequences=True)(x0)
    l2 = LSTM(64, activation='relu', return_sequences=False)(l1)

    # Build encoder_fc model
    fc1 = Dense(256, activation='relu')(l2)
    fc2 = Dense(512, activation='relu')(fc1)
    fc3 = Dense(256, activation='relu')(fc2)
    fc4 = Dense(128, activation='relu')(fc3)
    fc_out = Dense(1, activation='sigmoid')(fc4)

    model_encoder_fc = Model(inputs=x0, outputs=fc_out)

    Encoder_fc_optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    model_encoder_fc.compile(optimizer=Encoder_fc_optimizer, loss='mse')
    model_encoder_fc.summary()

    return model_encoder_fc


def train(X_train_std, y_train_std, X_test_std, y_test_std, epochs=2, batch_size=256):
    # Which model?
    model = build_encoder_fc()

    history = model.fit(X_train_std, y_train_std, batch_size=batch_size, epochs=epochs, \
                                                    verbose=1, validation_data=(X_test_std, y_test_std))

    return model, history
