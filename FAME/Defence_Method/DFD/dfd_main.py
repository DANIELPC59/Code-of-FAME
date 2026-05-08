import os
import sys
project_path=os.getcwd()
print("Project path:", project_path)
sys.path.append(project_path)
import time
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from DataTool_Code.DataLoader import Load_Data
def dfd_up(ori_burst, rate):
    """
     Insert perturbation packets in the outgoing direction.
     rate: insertion ratio.
    """
    burst = tf.identity(ori_burst) #  Copy into a new tensor to avoid in-place modification.
    burst_len = burst.shape[1]
    #  Indices of outgoing bursts: 2, 4, 6, ...
    idx = tf.range(2, burst_len, delta=2) 
    #  Select all burst positions to update.
    
    update = tf.gather(burst, idx - 2, axis=1) * rate
    update = tf.round(update) #  Ensure integer packet counts.
    #  Insert only at even positions, which correspond to outgoing bursts.
    #  Accumulate update values into the corresponding burst positions.
    mask = tf.one_hot(idx, burst_len, on_value=1.0, off_value=0.0) # (num_idx, burst_len)
    update_pad = update[:, :, tf.newaxis] * mask # (batch, num_idx, burst_len)
    update_sum = tf.reduce_sum(update_pad, axis=1) # (batch, burst_len)
    # burst = burst + update_sum
    
    zero_mask = tf.cast(burst != 0, burst.dtype)  #  Non-zero positions are 1 and zero-padding positions are 0.
    update_sum = update_sum * zero_mask
    
    #  Add updates back to the original burst sequence.
    burst = burst + update_sum
    return burst

def dfd_all(ori_burst, rate):
    """
     Insert perturbations at all positions while preserving signs at odd positions.
    ori_burst: tf.Tensor, shape (batch_size, burst_len)
    rate: float
    """
    burst = tf.identity(ori_burst)
    burst_len = burst.shape[1]

    #  Positions to update, starting from index 2 because idx - 2 is used.
    idx = tf.range(2, burst_len)

    #  Use values at idx - 2 as the perturbation source.
    update = tf.gather(burst, idx - 2, axis=1) * rate
    update = tf.round(update)

    #  One-hot mask.
    mask = tf.one_hot(idx, burst_len, on_value=1.0, off_value=0.0)
    update_pad = update[:, :, tf.newaxis] * mask
    update_sum = tf.reduce_sum(update_pad, axis=1)

    #  Do not insert into zero-padding positions.
    zero_mask = tf.cast(burst != 0, burst.dtype)
    update_sum = update_sum * zero_mask

    #  Add updates back to the burst sequence.
    burst = burst + update_sum
    return burst


def main():
    dataset_name='DF'
    data_type='test'
    data_x,label_y=Load_Data(dataset_name,data_type)
    data_x = np.squeeze(data_x, axis=-1) #  Convert shape from (data_num, burst_len, 1) to (data_num, burst_len).
    print('data_shape',data_x.shape)
    if dataset_name=='AWF100':
        rate=0.7
    elif dataset_name=='DF':
        rate=0.55
    #  Use GPU automatically if available.
    Base_save_directory='/shared_data/zpc_dataset/Website_Fingerprinting'

    batch_size = 64

    results = []
    norm_original_total = 0
    norm_result_total = 0
    begin_time=time.time()
    with tf.device('/GPU:0'):
        for start in tqdm(range(0, len(data_x), batch_size), desc="Processing batches"):
            end = start + batch_size
            batch_data = data_x[start:end]
            # print('batch_data:',batch_data.shape)
            tf_burst = tf.constant(batch_data, dtype=tf.float32)
            result = dfd_up(tf_burst, rate)

            #  Accumulate results and bandwidth statistics.
            results.append(result.numpy())
            norm_original_total += tf.reduce_sum(tf.abs(tf_burst)).numpy()
            norm_result_total += tf.reduce_sum(tf.abs(result)).numpy()

    #  Concatenate results from all batches.
    final_result = np.vstack(results)
    end_time=time.time()

    #  Compute bandwidth-overhead ratio.
    bandwidth = (norm_result_total - norm_original_total) / norm_original_total
    print('Bandwidth overhead ratio:', bandwidth)
    print('result.shape',final_result.shape)
    #  Save results.
    if dataset_name=='AWF100':
        save_directory=f'{Base_save_directory}/AWF/close-world/{dataset_name}/DFD_Data'
    elif dataset_name=='DF':
        save_directory=f'{Base_save_directory}/{dataset_name}/close-world/DFD_Data'
    os.makedirs(save_directory, exist_ok=True)
    np.savez(os.path.join(save_directory, f'DFD_test.npz'), data=final_result, labels=label_y)
    print('Processing completed. Save path:', save_directory)
    print('Elapsed time:',end_time-begin_time)

def test():
    #  Test data.
    burst = np.array([
        [ 15,  -7,  11,  -4,   0,  -2,   6,   0,   1, -12],
        [  0, -10,   0, -15,  15, -12,   3,  -7,   0,   0],
        [ 14,  -9,   0, -15,  12, -15,   0, -18,   2, -10],
        [ 18,  -2,   4, -16,  14,   0,   0, -16,  15, -13],
        [  0,  -1,   0,  -1,   0,   0,  17,  -3,   5,  -9],
        [  9,  -2,   0,  -5,   8,  -3,   5,  -3,  14,  -9],
        [ 10, -19,   0,  -4,   7, -11,  13,   0,   2,  -6],
        [ 12,   0,  11,  -7,   1, -13,   7,   0,   5, -10],
        [ 15,   0,   9, -17,  12,  -2,   5, -17,   2,  -5],
        [  1,  -2,   6,  -4,   0, -17,   5,  -6,  16,  -9]
    ], dtype=np.int32)
    print(burst)
    tf_burst = tf.constant(burst, dtype=tf.float32)
    result=dfd_all(tf_burst,1)
    print(result)

if __name__ == '__main__':
    main()
