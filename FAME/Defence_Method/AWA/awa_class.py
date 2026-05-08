#  AWA model architecture definitions.
import tensorflow as tf


from tensorflow.keras.layers import Conv1D, MaxPooling1D, ELU, Flatten, Dense, Activation,BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import glorot_uniform


import numpy as np
# from Classfy_Model.Model import *
from WF_Model.CFModel_Loder import Load_Classfy_Model
from Defence_Method.AWA.config import *
from Defence_Method.AWA.util import *

def AC_layers(layer_names,CF_modelname):
    AC_model =Load_Classfy_Model(CF_modelname, 'AWF100', 2000)
    AC_model.trainable = False
  
    outputs = [AC_model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([AC_model.input], outputs)
    return model
class LogitModel(tf.keras.Model):
    def __init__(self, logit_layer,CF_modelname):
        super().__init__()
        self.AC = AC_layers(logit_layer,CF_modelname)
        self.AC.trainable = False

    def call(self, inputs):
        return self.AC(inputs)

#  Compute logit loss.
def cal_logit_loss(logit_outputs,src=None):
    logit_loss = logit_weight * tf.reduce_mean(tf.maximum(logit_outputs[:,src],0))  
    return logit_loss

import tensorflow as tf
import numpy as np

class AWA_Class(tf.keras.Model):
    def __init__(self, trace_len, logit_layer, awa_type, trace_length,CF_modelname):
        super(AWA_Class, self).__init__()
        self.max_burst_trace_len = trace_len
        self.trace_length = trace_length
        self.awa_type = awa_type

        sign_vector = np.ones([trace_len, 1]) * -1
        sign_vector[::2] = 1
        self.sign_vector = tf.constant(sign_vector, dtype=tf.float32)
        self.max_trace_len = tf.constant(trace_len, dtype=tf.float32)
        

        self.logit_extractor = LogitModel(logit_layer,CF_modelname)
        self.generator1 = self.make_generator1()
        self.generator2 = self.make_generator2()
        self.discriminator = self.make_discriminator()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
    
    def adjust_WF_data(self, x, perturbation):
        #  The processed dataset starts with an outgoing burst, so no right shift is needed.
        # return x + perturbation * self.sign_vector
        #  Keep perturbation directions consistent with the original packet directions.
        perturbation_with_sign = perturbation * tf.sign(x)

        #  Compute perturbed packets.
        adjusted_data = x + perturbation_with_sign
        return adjusted_data
    
    def discriminator_loss(self, d_class_i, d_class_j):
        d_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_class_i,labels=tf.ones_like(d_class_i)))
        d_loss_j = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_class_j,labels=tf.zeros_like(d_class_j)))
        d_loss = d_loss_i + d_loss_j
        return d_loss

    def generator1_loss(self,perturbation_i=None,x_class_i=None,d_class_i=None):
        g1_oh_loss = tf.maximum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_i - perturbation_i)),tf.reduce_sum(tf.abs(x_class_i)))) - tau_high, 0) 
        g1_oh_loss_new = tf.minimum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_i - perturbation_i)),tf.reduce_sum(tf.abs(x_class_i)))) - tau_low, 0) * -1
        g1_loss_org = -1 * (tf.reduce_mean(1 / 2 * tf.math.log(tf.math.sigmoid(d_class_i) + 0.0000001)) + tf.reduce_mean(1 / 2 * tf.math.log(1 - tf.math.sigmoid(d_class_i) + 0.0000001)))
        return disc_weight * g1_loss_org , oh_weight * (g1_oh_loss + g1_oh_loss_new)

    def generator2_loss(self,perturbation_j=None,x_class_j=None,d_class_j=None):
        g2_oh_loss = tf.maximum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_j - perturbation_j)),tf.reduce_sum(tf.abs(x_class_j)))) - tau_high, 0) 
        g2_oh_loss_new = tf.minimum(tf.reduce_mean(tf.math.divide(tf.reduce_sum(tf.abs(x_class_j - perturbation_j)),tf.reduce_sum(tf.abs(x_class_j)))) - tau_low, 0) * -1
        g2_loss_org = -1 * (tf.reduce_mean(1 / 2 * tf.math.log(tf.math.sigmoid(d_class_j) + 0.0000001)) + tf.reduce_mean(1 / 2 * tf.math.log(1 - tf.math.sigmoid(d_class_j) + 0.0000001)))
        return disc_weight * g2_loss_org , oh_weight * (g2_oh_loss + g2_oh_loss_new)
    
    
    def make_generator1(self):
        model = Sequential(name='generator1')

        # c3s1-8
        model.add(Conv1D(filters=8, kernel_size=3, strides=1, padding='same', input_shape=(self.max_burst_trace_len, 1)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ELU(alpha=2.0))  #  TensorFlow 2 ignores alpha for this ELU layer.

        # d16
        model.add(Conv1D(filters=16, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ELU(alpha=2.0))

        # d32
        model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ELU(alpha=2.0))

        #  Eight r32 blocks.
        for _ in range(8):
            model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

#         # u16
#         model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(3, 1), strides=(2, 1), padding='same'))  #  Convert 1D to 2D.
#         model.add(tf.keras.layers.Reshape((self.max_burst_trace_len, 16)))  #  Reshape back to a 1D-like representation.
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(ELU(alpha=2.0))

#         # u8
#         model.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=(3, 1), strides=(2, 1), padding='same'))
#         model.add(tf.keras.layers.Reshape((self.max_burst_trace_len, 8)))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(ELU(alpha=2.0))
        #  Correct implementation of the u16 layer.
        model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)))
        model.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 1), strides=(2, 1), padding='same'))
        model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2)))

        #  Correct implementation of the u8 layer.
        model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)))
        model.add(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 1), strides=(2, 1), padding='same'))
        model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2)))


        # c3s1-3
        model.add(Conv1D(filters=1, kernel_size=3, strides=1, padding='same'))
        model.add(Activation('relu'))

        return model

    def make_generator2(self):
        model = Sequential(name='generator2')

        # c3s1-8
        model.add(Conv1D(filters=8, kernel_size=3, strides=1, padding='same', input_shape=(self.max_burst_trace_len, 1)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ELU(alpha=2.0))  #  TensorFlow 2 ignores alpha for this ELU layer.

        # d16
        model.add(Conv1D(filters=16, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ELU(alpha=2.0))

        # d32
        model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(ELU(alpha=2.0))

        #  Eight r32 blocks.
        for _ in range(8):
            model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(ELU(alpha=2.0))

#         # u16
#         model.add(tf.keras.layers.Conv2DTranspose(16, kernel_size=(3, 1), strides=(2, 1), padding='same'))  #  Convert 1D to 2D.
#         model.add(tf.keras.layers.Reshape((self.max_burst_trace_len, 16)))  #  Reshape back to a 1D-like representation.
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(ELU(alpha=2.0))

#         # u8
#         model.add(tf.keras.layers.Conv2DTranspose(8, kernel_size=(3, 1), strides=(2, 1), padding='same'))
#         model.add(tf.keras.layers.Reshape((self.max_burst_trace_len, 8)))
#         model.add(BatchNormalization(momentum=0.8))
#         model.add(ELU(alpha=2.0))
        #  Correct implementation of the u16 layer.
        model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)))
        model.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 1), strides=(2, 1), padding='same'))
        model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2)))

        #  Correct implementation of the u8 layer.
        model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2)))
        model.add(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 1), strides=(2, 1), padding='same'))
        model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2)))

        # c3s1-3
        model.add(Conv1D(filters=1, kernel_size=3, strides=1, padding='same'))
        model.add(Activation('relu'))

        return model
        
    def make_discriminator(self):
        model = Sequential(name='discriminator')

        #  Model parameter list.
        filter_num = [None, 32, 32, 64, 64]
        kernel_size = [None, 8, 8, 8, 8]
        conv_stride_size = [None, 1, 1, 1, 1]
        pool_stride_size = [None, 4, 4, 4, 4]
        pool_size = [None, 8, 8, 8, 8]

        # dblock1
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                        input_shape=(self.max_burst_trace_len, 1),
                        strides=conv_stride_size[1], padding='same',
                        name='dblock1_conv1'))
        model.add(ELU(alpha=1.0,name='dblock1_adv_act1'))

        # dblock2
        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                        strides=conv_stride_size[2], padding='same',
                        name='dblock2_conv1'))
        model.add(ELU(alpha=1.0,name='dblock1_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                            padding='same', name='dblock1_pool'))

        # dblock3
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                        strides=conv_stride_size[3], padding='same',
                        name='dblock3_conv1'))
        model.add(ELU(alpha=1.0,name='dblock2_adv_act1'))

        # dblock4
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                        strides=conv_stride_size[4], padding='same',
                        name='dblock4_conv1'))
        model.add(ELU(alpha=1.0,name='dblock2_adv_act2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                            padding='same', name='dblock2_pool'))

        #  Flatten layer followed by dense layers.
        model.add(Flatten(name='dflatten'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='dfc1'))
        model.add(Activation('relu', name='dfc1_act'))
        model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0), name='dfc2'))
        model.add(Activation('relu', name='dfc2_act'))
        model.add(Dense(1, kernel_initializer=glorot_uniform(seed=0), name='dfc3'))

        return model
    @tf.function(reduce_retracing=True)
    def train_discriminator(self, src_x, trg_x, noise_i, noise_j):
        with tf.GradientTape() as tape:
            #  Generate perturbation vectors and perturbed data.
            generated_1 = self.generator1(noise_i)
            generated_2 = self.generator2(noise_j)
            adjusted_1 = self.adjust_WF_data(src_x, generated_1)
            adjusted_2 = self.adjust_WF_data(trg_x, generated_2)
    
            d_class_i = self.discriminator(adjusted_1)
            d_class_j = self.discriminator(adjusted_2)
            """ Compute discriminator loss."""
            d_loss = self.discriminator_loss(d_class_i, d_class_j)
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return d_loss

    @tf.function(reduce_retracing=True)
    def train_generator1(self, src_x, src_cls, noise_i):
        with tf.GradientTape() as tape:
            generated_1 = self.generator1(noise_i)    
            adjusted_1 = self.adjust_WF_data(src_x, generated_1)
            d_class_i = self.discriminator(adjusted_1)
        
             #  Compute logit loss.
            pad_len = self.trace_length - self.max_burst_trace_len
            padded_1 = tf.pad(adjusted_1, [[0, 0], [0, pad_len], [0, 0]])
            logit_out_1 = self.logit_extractor(padded_1)
            logit_loss_1 = cal_logit_loss(logit_out_1, src_cls)

            #  Compute total loss.
            g1_loss_org, g1_oh_loss = self.generator1_loss(adjusted_1, src_x, d_class_i)
            g1_loss = g1_loss_org + g1_oh_loss + logit_loss_1
            
        grads = tape.gradient(g1_loss, self.generator1.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.generator1.trainable_variables))
        return g1_loss, g1_loss_org, g1_oh_loss, logit_loss_1, d_class_i, generated_1, adjusted_1

    @tf.function(reduce_retracing=True)
    def train_generator2(self, trg_x, trg_cls, noise_j):
        with tf.GradientTape() as tape:
            #  Compute generator 2 loss.

            generated_2 = self.generator2(noise_j)
            adjusted_2 = self.adjust_WF_data(trg_x, generated_2)
            d_class_j = self.discriminator(adjusted_2)
            
            #  Compute logit loss.
            pad_len = self.trace_length - self.max_burst_trace_len
            padded_2 = tf.pad(adjusted_2, [[0, 0], [0, pad_len], [0, 0]])
            logit_out_2 = self.logit_extractor(padded_2)
            logit_loss_2 = cal_logit_loss(logit_out_2, trg_cls)

            #  Compute total loss.
            g2_loss_org, g2_oh_loss = self.generator2_loss(adjusted_2, trg_x, d_class_j)
            g2_loss = g2_loss_org + g2_oh_loss + logit_loss_2

        grads = tape.gradient(g2_loss, self.generator2.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.generator2.trainable_variables))

        return g2_loss, g2_loss_org, g2_oh_loss, logit_loss_2, d_class_j, generated_2, adjusted_2
    @tf.function(reduce_retracing=True)
    def adjusted_generated_1(self, src_x, noise_i, training=False):
        """ Generate adversarial samples for class 1."""
        generated_1 = self.generator1(noise_i, training=training)
        adjusted_1 = self.adjust_WF_data(src_x, generated_1)
        return adjusted_1

    @tf.function(reduce_retracing=True)
    def adjusted_generated_2(self, trg_x, noise_j, training=False):
        """ Generate adversarial samples for class 2."""
        generated_2 = self.generator2(noise_j, training=training)
        adjusted_2 = self.adjust_WF_data(trg_x, generated_2)
        return adjusted_2
