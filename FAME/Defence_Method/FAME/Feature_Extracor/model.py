# DF model used for non-defended dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

class CF_Extractor:
    @staticmethod
    def build(input_shape):
        model = Sequential(name="CF_Feature_Extractor")
        
        # ==================== Block 1 ====================
        model.add(Conv1D(filters=32, kernel_size=8, strides=1, padding='same', 
                         input_shape=input_shape, name='block1_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block1_act1'))  
        
        model.add(Conv1D(filters=32, kernel_size=8, strides=1, padding='same', name='block1_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block1_act2'))
        
        model.add(MaxPooling1D(pool_size=6, strides=3, padding='same', name='block1_pool'))
        model.add(Dropout(0.2, name='block1_dropout'))  

        # ==================== Block 2 ====================
        model.add(Conv1D(filters=64, kernel_size=8, strides=1, padding='same', name='block2_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block2_act1'))
        
        model.add(Conv1D(filters=64, kernel_size=8, strides=1, padding='same', name='block2_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block2_act2'))
        
        model.add(MaxPooling1D(pool_size=6, strides=3, padding='same', name='block2_pool'))
        model.add(Dropout(0.2, name='block2_dropout'))

        # ==================== Block 3 ====================
        model.add(Conv1D(filters=128, kernel_size=8, strides=1, padding='same', name='block3_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block3_act1'))
        
        model.add(Conv1D(filters=128, kernel_size=8, strides=1, padding='same', name='block3_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block3_act2'))
        
        model.add(MaxPooling1D(pool_size=6, strides=3, padding='same', name='block3_pool'))
        model.add(Dropout(0.2, name='block3_dropout'))

        # ==================== Block 4 ====================
        model.add(Conv1D(filters=256, kernel_size=8, strides=1, padding='same', name='block4_conv1'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block4_act1'))
        
        model.add(Conv1D(filters=256, kernel_size=8, strides=1, padding='same', name='block4_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu', name='block4_act2'))
        
        model.add(MaxPooling1D(pool_size=6, strides=3, padding='same', name='block4_pool'))
        model.add(Dropout(0.2, name='block4_dropout')) # [cite: 619]

        
        model.add(Flatten(name='flatten'))
        
        
        model.add(Dense(256, kernel_initializer=glorot_uniform(seed=0), name='fc_representation'))
        
        return model

class ProjectionHead:
    @staticmethod
    def build(input_dim=256, output_dim=128):
        
        model = Sequential([
            Dense(256, input_dim=input_dim, name='proj_hidden'),
            BatchNormalization(),
            Activation('relu'),
        
            Dense(output_dim, name='proj_output')
        ])
        return model

def get_normalized_embeddings(model_output):
    return tf.math.l2_normalize(model_output, axis=1)