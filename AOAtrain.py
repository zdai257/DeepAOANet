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


def build_fc1(feature_dim=160, output_dim=75):
    x = Input(shape=(feature_dim,))
    h1 = Dense(512, activation='relu')(x)
    h2 = Dense(1024, activation='relu')(h1)
    h3 = Dense(512, activation='relu')(h2)
    h4 = Dense(256, activation='relu')(h3)
    r = Dense(output_dim, activation='sigmoid')(h4)  # 'sigmoid' + BinaryCrossentropy

    model = Model(inputs=x, outputs=r)

    optimizer = keras.optimizers.Adam(learning_rate=0.01, decay=1e-5)
    loss_obj = keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss_obj)
    model.summary()
    return model


def build_CR0(feature_dim=128, num_classes=1, regress1=1, regress2=1):
    x = Input(shape=(feature_dim,))
    h1 = Dense(1024, activation='relu')(x)
    h2 = Dense(2048, activation='relu')(h1)
    h3 = Dense(1024, activation='relu')(h2)
    h4 = Dense(512, activation='relu')(h3)

    # Num-of-signal Classifier
    c = Dense(num_classes, activation='sigmoid', name="class_out")(h4)

    # Regression of 1-signal
    r1 = Dense(regress1, activation='sigmoid', name="regress1_out")(h4)

    # Regression of 2-signal
    r2 = Dense(regress2, activation='sigmoid', name="regress2_out")(h4)

    model = Model(inputs=x, outputs=[c, r1, r2], name="deepaoanet0")

    optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
    losses = {
        "class_out": keras.losses.BinaryCrossentropy(from_logits=False),
        "regress1_out": keras.losses.MeanSquaredError(),
        "regress2_out": keras.losses.MeanSquaredError(),
    }
    lossWeights = {"class_out": 1.0, "regress1_out": 1.0, "regress2_out": 1.0}
    metrics = {"class_out": 'accuracy', "regress1_out": 'mse', "regress2_out": 'mse'}

    model.compile(optimizer=optimizer,
                      loss=losses,
                      loss_weights=lossWeights,
                      metrics=metrics)
    model.summary()

    return model


def train(X_train_std, yhot_train, X_test_std, yhot_test, epochs=2, batch_size=256):
    # Specify Which model?
    #model = build_fc0()
    model = build_fc1()

    history = model.fit(X_train_std, yhot_train, batch_size=batch_size, epochs=epochs,
                                    verbose=1, validation_data=(X_test_std, yhot_test))

    return model, history
