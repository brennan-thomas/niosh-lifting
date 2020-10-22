from keras.layers import Bidirectional, Input, Conv1D, LSTM, Dense, Flatten, BatchNormalization, Dropout, LeakyReLU, Bidirectional, Add, ReLU
from keras.activations import tanh, relu, sigmoid
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def deepconvlstm4class(shape):
    inpt = Input(shape=shape)
    x = Conv1D(64, activation='tanh', kernel_size=5, padding='same')(inpt)
    x = Conv1D(64, activation='tanh', kernel_size=5, padding='same')(x)
    x = Conv1D(64, activation='tanh', kernel_size=5, padding='same')(x)
    x = Conv1D(64, activation='tanh', kernel_size=5, padding='same')(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def residual_block(in_shape, out_dim):
    in_dim = in_shape[-1]
    inpt = Input(in_shape)
    skip = inpt
    if in_dim != out_dim:
        skip = Conv1D(out_dim, activation=None, kernel_size=1, strides=1)(skip)
    x = BatchNormalization()(inpt)
    x = ReLU()(x)
    x = Conv1D(out_dim, activation=None, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(out_dim, activation=None, kernel_size=5, padding='same')(x)
    x = Add()([x, skip])

    model = Model(inputs=inpt, outputs=x)
    return model

def residual_4class_dense(shape):
    inpt = Input(shape)
    print(inpt.shape)
    x = residual_block(inpt.shape[1:], 128)(inpt)
    x = residual_block(x.shape[1:], 128)(x)
    x = residual_block(x.shape[1:], 128)(x)
    x = residual_block(x.shape[1:], 128)(x)
    x = residual_block(x.shape[1:], 128)(x)
    x = residual_block(x.shape[1:], 128)(x)
    x = Dropout(0.5)(x)

    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

