import tensorflow as tf

def custom_sign(tensor):
        
    return tf.cast(tensor > 0, dtype=tf.float32)

