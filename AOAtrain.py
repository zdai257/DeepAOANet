import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add, LSTM, RepeatVector, TimeDistributed, Bidirectional
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers, Sequential
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils, plot_model


def build_fc0(feature_dim=160, out_dim=75):
    x = Input(shape=(feature_dim,))
    h1 = Dense(512, activation='relu')(x)
    h2 = Dense(1024, activation='relu')(h1)
    h3 = Dense(512, activation='relu')(h2)
    h4 = Dense(256, activation='relu')(h3)
    r = Dense(out_dim, activation='softmax')(h4)
    model = Model(inputs=x, outputs=r)

    optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)

    loss_obj = keras.losses.CategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss_obj)
    model.summary()

    return model


def train(X_train_std, yhot_train, X_test_std, yhot_test, epochs=2, batch_size=256):
    # Which model?
    model = build_fc0()

    history = model.fit(X_train_std, yhot_train, batch_size=batch_size, epochs=epochs,
                                    verbose=1, validation_data=(X_test_std, yhot_test))

    return model, history
