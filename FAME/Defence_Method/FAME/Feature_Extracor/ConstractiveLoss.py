
import tensorflow as tf

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