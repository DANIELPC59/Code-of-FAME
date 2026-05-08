import os
import sys
project_path = os.getcwd()
sys.path.append(project_path)

import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime, timedelta

#  External module imports.
from Defence_Method.FAME.Feature_Extracor.DataAugement.CF_Augement import CF_DataAugmentation
from Defence_Method.FAME.Feature_Extracor.model import CF_Extractor, ProjectionHead
from DataTool_Code.DataLoader import Load_Data

#  1. Logging configuration.
log_dir = 'Defence_Method/FAME/Feature_Extracor/File_Save'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "trainingAWF500.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
start_time = datetime.now()
beijing_time = start_time + timedelta(hours=8)

#  2. Supervised NT-Xent loss.
def supervised_nt_xent_loss(z, labels, temperature=0.1):
    """
    Supervised Contrastive Loss (NT-Xent).

    Args:
        z          : Feature matrix, shape [2 * batch_size, dim] (original + augmented pairs).
        labels     : Class labels, shape [2 * batch_size].
        temperature: Scaling factor for logits.
    """
    # Step 1: L2 normalize embeddings
    z = tf.math.l2_normalize(z, axis=1)
    
    # Step 2: pairwise cosine similarity matrix, shape [2N, 2N]
    logits = tf.matmul(z, z, transpose_b=True) / temperature
    
    # Step 3: build positive-pair mask (1 where labels match)
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [-1, 1])
    # mask[i, j] = 1 if labels[i] == labels[j]
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)  
    
    # Step 4: remove self-similarity (zero out diagonal)
    batch_size = tf.shape(z)[0]
    logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
    # subtract row-wise max for numerical stability
    logits = logits - tf.stop_gradient(logits_max)
    
    # off-diagonal mask (1 everywhere except diagonal)
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    # exclude self from the positive mask
    mask = mask * logits_mask
    
    # Step 5: compute log-softmax
    # diagonal is masked so self-similarity contributes 0 after exp
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-9)
    
    # Step 6: average log-prob over positive pairs only
    # guard against division by zero when no positive pair exists in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)
    mask_sum = tf.where(mask_sum < 1e-9, tf.ones_like(mask_sum), mask_sum)
    
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_sum
    
    # Step 7: final loss
    loss = - tf.reduce_mean(mean_log_prob_pos)
    
    return loss

#  3. Training loop.
#  Added the labels parameter for supervised contrastive learning.
def pretrain_cf(data, labels, epochs=30, batch_size=64, learn_rate=0.001):

    input_shape = data.shape[1:]
    
    extractor = CF_Extractor().build(input_shape) 
    projection_head = ProjectionHead().build()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    aug_tool = CF_DataAugmentation()  # burst-sequence augmentation
    
    logger.info("Starting supervised contrastive (SupCon) pre-training")
    min_loss=float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        
        #  Shuffle data and labels in sync.
        indices = np.random.permutation(len(data))
        data_shuffled = data[indices]
        labels_shuffled = labels[indices]
        
        #  Track valid batch count for average loss computation.
        num_batches = 0 
        
        for i in range(0, len(data), batch_size):
            batch_x = data_shuffled[i : i + batch_size]
            batch_y = labels_shuffled[i : i + batch_size]
            
            #  Skip batches of size 1 because SupCon requires at least 2 samples.
            if len(batch_x) < 2: 
                continue
            
            #  Generate two augmented views.
            x_i_np = np.array([aug_tool.augment(x) for x in batch_x])
            x_j_np = np.array([aug_tool.augment(x) for x in batch_x])
            
            #  Convert to float32 tensors.
            x_i = tf.convert_to_tensor(x_i_np, dtype=tf.float32)
            x_j = tf.convert_to_tensor(x_j_np, dtype=tf.float32)
            
            #  Duplicate labels for both augmented views, shape: (2B,).
            extended_labels = tf.concat([batch_y, batch_y], axis=0)
            
            with tf.GradientTape() as tape:
                r_i = extractor(x_i, training=True)
                r_j = extractor(x_j, training=True)
                
                e_i = projection_head(r_i, training=True)
                e_j = projection_head(r_j, training=True)
                
                #  Concatenate projections from both views along the batch dimension.
                z_combined = tf.concat([e_i, e_j], axis=0)
                
                #  Compute supervised contrastive loss.
                loss = supervised_nt_xent_loss(z_combined, extended_labels)
            
            variables = extractor.trainable_variables + projection_head.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            epoch_loss += loss.numpy()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        if avg_loss<min_loss:
            min_loss=avg_loss
            #  Save model weights when loss improves.
            save_dir = "Defence_Method/FAME/Feature_Extracor/File_Save"
            os.makedirs(save_dir, exist_ok=True)

            FeatureExtractor_save_path = os.path.join(save_dir, f"CF_FeaturExtractor_in_{DataSet_name}.h5")
            extractor.save_weights(FeatureExtractor_save_path)
            logger.info(f"Loss improved to {min_loss:.4f}. CF_Extractor saved: {FeatureExtractor_save_path}")

            ProjectionHead_save_path = os.path.join(save_dir, f"CF_ProjectionHead_in_{DataSet_name}.h5")
            projection_head.save_weights(ProjectionHead_save_path)
            logger.info(f"ProjectionHead saved: {ProjectionHead_save_path}")
    
    return extractor

#  4. Entry point.
if __name__ == "__main__":
    DataSet_name = 'AWF100' # 'DF','AWF100','AWF200'
    # training hyperparameters
    Epoch_num = 40
    batch_size = 128
    learn_rate = 0.002

    logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    train_x,train_y = Load_Data(DataSet_name, 'adv')
    
    logger.info(f"Dataset '{DataSet_name}' loaded, shape: {train_x.shape}")
    trained_extractor = pretrain_cf(train_x, train_y, epochs=Epoch_num, batch_size=batch_size, learn_rate=learn_rate)
