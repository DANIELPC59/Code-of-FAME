import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv1D, Activation, Lambda, Conv2DTranspose, \
    LeakyReLU, Dropout, Flatten, ELU, MaxPooling1D, Reshape, Average, ReLU, Lambda, Dense


def generator_model_5():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(2000))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.05))

    model.add(tf.keras.layers.Dense(2000))
    model.add(tf.keras.layers.Activation('relu'))
    return model