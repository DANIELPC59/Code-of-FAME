import json
import numpy as np

def burst_to_direction(burst_seq, max_dir_len=5000):
    """
    Convert a burst sequence to a direction sequence.

    Args:
        burst_seq: Input burst sequence, shape [batch_size, sequence_length].
        max_dir_len: Maximum length of the output direction sequence. Default is 5000.

    Returns:
        tf.Tensor: Direction sequence of shape [batch_size, max_dir_len].

    Steps:
        1. Extract non-zero burst values for each sample.
        2. Compute the direction (sign) and count (absolute value) of each burst.
        3. Expand bursts into a per-packet direction sequence.
        4. Truncate or pad the sequence to max_dir_len.
    """
    batch_size = tf.shape(burst_seq)[0]
    result = []
    
    for i in tqdm(tf.range(batch_size)):
        bursts = tf.squeeze(burst_seq[i])  

        non_zero_mask = tf.not_equal(bursts, 0)
        non_zero_mask = tf.reshape(non_zero_mask, [-1])  
        
        non_zero_bursts = tf.boolean_mask(bursts, non_zero_mask)
        
        dirs = tf.sign(non_zero_bursts)
        counts = tf.abs(non_zero_bursts)
        
        dir_sequence = tf.repeat(dirs, counts)
        
        current_len = tf.shape(dir_sequence)[0]
        if current_len > max_dir_len:
            dir_sequence = dir_sequence[:max_dir_len]
        else:
            pad_len = max_dir_len - current_len
            dir_sequence = tf.pad(dir_sequence, [[0, pad_len]])
        
        result.append(dir_sequence)
    
    return tf.stack(result)




def direction_to_burst(packet_seq, fill_burst=False):
    """
    Convert a direction sequence to a burst sequence.

    Args:
        packet_seq: Input direction sequence where each element represents
                    the direction of a packet (positive or negative).
        fill_burst: Whether to pad or truncate the output burst sequence
                    to a fixed length of 2000. Default is False.

    Returns:
        list: Burst sequence where each element encodes direction (sign)
              and packet count (absolute value).
    """
    if not np.any(packet_seq):
        return []

    burst = []
    current_burst_count = 0  # packet count in the current burst
    current_burst_direction = np.sign(packet_seq[0])  # direction (sign) of the current burst

    for value in packet_seq:
        if value == 0:
            if current_burst_count > 0:
                burst.append(current_burst_direction * current_burst_count)
            break
        else:
            if np.sign(value) == current_burst_direction:
                current_burst_count += 1
            else:
                burst.append(current_burst_direction * current_burst_count)
                current_burst_direction = np.sign(value)
                current_burst_count = 1

    if current_burst_count > 0:   # handle the last non-zero burst
        burst.append(current_burst_direction * current_burst_count)
    
    if len(burst) > 0 and burst[0] < 0:  # discard leading downlink (negative) burst
        burst = burst[1:]  

    if fill_burst:        # pad or truncate to fixed length
        burst_length = len(burst)
        if burst_length > 2000:
            burst = burst[:2000]
        elif burst_length < 2000:
            burst.extend([0] * (2000 - burst_length))

    return burst


def split_data_per_class_flexible(
    data_x,
    label_y,
    n_adv,
    n_train,
    seed=42
):
    """
    Split dataset by class with flexible per-class sample counts.

    For each class, samples are assigned as follows:
        - First n_adv samples      -> adv split
        - Next  n_train samples    -> train split
        - Remaining samples        -> test split

    Args:
        data_x  : numpy array of shape (N, ...).
        label_y : numpy array of shape (N,).
        n_adv   : Number of samples per class allocated to the adv split.
        n_train : Number of samples per class allocated to the train split.
        seed    : Random seed for reproducibility.

    Returns:
        X_test, y_test, X_train, y_train, X_adv, y_adv
    """
    rng = np.random.default_rng(seed)

    X_test, y_test = [], []
    X_train, y_train = [], []
    X_adv, y_adv = [], []

    for lbl in np.unique(label_y):
        idx = np.where(label_y == lbl)[0]
        rng.shuffle(idx)

        n_total = len(idx)

        assert n_total >= n_adv + n_train+100 , f"[Error] Label {lbl}: not enough samples ({n_total}) for adv ({n_adv}) + train ({n_train})+ test (100)"

        adv_idx = idx[:n_adv]
        train_idx = idx[n_adv : n_adv + n_train]
        test_idx = idx[n_adv + n_train :]

        X_adv.append(data_x[adv_idx])
        y_adv.append(label_y[adv_idx])

        X_train.append(data_x[train_idx])
        y_train.append(label_y[train_idx])

        X_test.append(data_x[test_idx])
        y_test.append(label_y[test_idx])

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_adv = np.concatenate(X_adv, axis=0)
    y_adv = np.concatenate(y_adv, axis=0)
    return X_test, y_test, X_train, y_train, X_adv, y_adv


def stable_label_encoder(labels, mapping_save_path=None):
    """
    Stably encode string labels to integer indices in range [0, N-1].

    Args:
        labels           : Raw label list or numpy array (e.g., URL strings).
        mapping_save_path: Optional path to save the mapping dict as a JSON file.

    Returns:
        encoded_labels : numpy array of integer-encoded labels.
        label_to_idx   : dict mapping original label strings to integer indices.
    """
    # Step 1: Sort labels to ensure consistent ordering across platforms
    unique_labels = sorted(list(set(labels)))
    
    # Step 2: Build bidirectional mapping dicts
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    # Step 3: Encode labels
    encoded_labels = np.array([label_to_idx[l] for l in labels], dtype=np.int32)
    
    # Step 4: Persist the mapping table if a path is provided
    if mapping_save_path:
        with open(mapping_save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label
            }, f, indent=4)
        print(f"Label mapping saved to: {mapping_save_path}")
        
    return encoded_labels, label_to_idx

def getUnknownData(data_x,label_y,preTrainLabel,mappingJson_save_path):
    """
    Filter samples whose classes are unknown to the pre-trained model.

    Args:
        data_x               : Target dataset samples.
        label_y              : String labels for the target dataset.
        preTrainLabel        : Label set seen by the pre-trained model.
        mappingJson_save_path: Path to the JSON file containing the label mapping.

    Returns:
        Tuple of (filtered_data, filtered_labels) as float32 arrays,
        where labels are re-mapped to integer indices using the saved mapping.
    """
    unknown_classes_set = set(label_y) - set(preTrainLabel)
    print(f"{len(unknown_classes_set)} unknown website classes found")
    
    # Step 1: Build a boolean mask for unknown classes and filter samples
    # np.isin is significantly faster than a Python for-loop
    mask = np.isin(label_y, list(unknown_classes_set))
    filtered_labels_str = label_y[mask]
    filtered_data = data_x[mask]

    # Step 2: Load the previously saved JSON label mapping
    mapping_save_path = mappingJson_save_path 
    with open(mapping_save_path, 'r', encoding='utf-8') as f:
        mapping_dict = json.load(f)

    label_to_idx = mapping_dict['label_to_idx']

    # Step 3: Re-map string labels to integer indices using the saved mapping
    filtered_labels_idx = np.array([label_to_idx[l] for l in filtered_labels_str], dtype=np.int32)

    filtered_data=np.array(filtered_data)
    filtered_labels_idx=np.array(filtered_labels_idx)
    return filtered_data.astype('float32'),filtered_labels_idx.astype('float32')
