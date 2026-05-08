
import numpy as np
import tensorflow as tf


from Tool_Code.Tool_Fun import custom_sign 
from DataTool_Code.DataLoader import Load_Data
from WF_Model.Eva_ClassfyModel import evaluate_model
from WF_Model.CFModel_Loder import  Load_Classfy_Model


#  Build label-level gate weights from collected samples.
def getmap_Gates(feature_extractor,gate_net,preTrain_x,preTrain_y,sample_num=10):
    """
     Build gate-value mappings for defended traffic using pre-collected samples.
     preTrain_x: Pre-collected sample data.
     preTrain_y: Labels of pre-collected samples.
     sample_num: Number of randomly sampled pre-collected samples per class.
    
    """
    labels=np.unique(preTrain_y)
    map_Gates={}
    for label in labels:
        #  Extract features for the current class.
        idx=(preTrain_y==label)
        cur_x=preTrain_x[idx]

        actual_sample_num = min(sample_num, len(cur_x))  #  Keep the sample count within the available class size.
        random_idx=np.random.choice(len(cur_x), actual_sample_num, replace=False)

        cur_x=cur_x[random_idx]
        cur_feature=feature_extractor(cur_x)
        #  Get gate weights.
        predict_weights=gate_net(cur_feature)
        #  Compute average gate weights.
        map_Gates[label]=np.mean(predict_weights,axis=0)
    return map_Gates


#  Closed-world defense evaluation based on pre-collected gate mappings.
def test_generator_fewShot(DataSet_Name, CF_ModelName, feature_extractor, few_shot_moegen,sample_num=10):
    """
     Build a gate mapping from pre-collected samples and look up gate weights by label during testing.
    
    Args:
        DataSet_Name: Dataset name.
        CF_ModelName: Classifier model name.
        feature_extractor: Feature extractor.
        gating_network: Gating network.
        expert_generator: Expert generator.
        preTrain_x: Pre-collected samples used to build the gate mapping.
        preTrain_y: Labels of pre-collected samples.
        sample_num: Number of samples per class.
    """
    #  Hyperparameter settings.
    batch_size = 512

    #  Load classifier model.
    Classfy_model = Load_Classfy_Model(CF_ModelName, DataSet_Name)

    #  Load data.
    test_data, test_labels = Load_Data(DataSet_Name, 'test')
    len_test_data = len(test_data)
    adv_test_data = np.zeros_like(test_data)  #  Store perturbed samples.
    max_noise = 0  #  Track the maximum perturbation value.
    total_l1_origin = 0  #  Sum of original-input L1 norms.
    total_l1_norm = 0  #  Sum of perturbed-input L1 norms.
    total_samples = 0  #  Total sample count.

    #  Stage 1: pre-collection, build the label-to-gate_weights mapping.
    map_Gates = getmap_Gates(feature_extractor, few_shot_moegen.gating_network, test_data, test_labels, sample_num)

    #  Stage 2: testing, look up gate weights by label.
    #  Extract test labels and fetch the corresponding gate weights.
    test_predict_weights = []
    for label in test_labels:
        test_predict_weights.append(map_Gates[label])
    test_predict_weights = np.array(test_predict_weights)

    for i in range(0, len_test_data, batch_size):
        left = i
        right = min(i + batch_size, len_test_data)
        batch_test_data = test_data[left:right]
        cur_test_batchsize = tf.shape(batch_test_data)[0]

        #  Original-input L1 norm.
        abs_batchdata = tf.abs(batch_test_data)
        flat_batchdata = tf.reshape(abs_batchdata, [cur_test_batchsize, -1])
        total_l1_origin += tf.reduce_sum(flat_batchdata).numpy()

        #  Use precomputed gate weights instead of extracting features online.
        cur_predict_weights = test_predict_weights[left:right]
        #  Perturbation generation: call the expert generator directly with gate weights.
        test_batchnoise = few_shot_moegen.expert_generator(cur_predict_weights, training=False)
        test_batchnoise = tf.round(test_batchnoise)

        #  Perturbation direction and application.
        test_batchsign = custom_sign(batch_test_data)
        test_batchnoise_withsign = test_batchnoise * test_batchsign
        adv_test_batchdata = batch_test_data + test_batchnoise_withsign

        #  Store into the full adversarial sample buffer.
        adv_test_data[left:right] = adv_test_batchdata.numpy()

        #  Collect perturbation statistics.
        abs_noise = tf.abs(test_batchnoise_withsign)
        flat_noise = tf.reshape(abs_noise, [cur_test_batchsize, -1])

        max_vals = tf.reduce_max(flat_noise, axis=1)  #  Maximum perturbation per sample.
        max_noise = max(max_noise, tf.reduce_max(max_vals).numpy())  #  Overall maximum perturbation.

        l1_norms = tf.reduce_sum(flat_noise, axis=1)  #  Per-sample L1 norm.
        total_l1_norm += tf.reduce_sum(l1_norms).numpy()
        total_samples += cur_test_batchsize.numpy()

    #  Model evaluation.
    F1, TPR, FPR, overall_ACC, per_class_acc = evaluate_model(Classfy_model, adv_test_data, test_labels, batch_size=batch_size)

    #  Average perturbation overhead.
    avg_origin_norm = total_l1_origin / total_samples
    avg_l1_norm = total_l1_norm / total_samples
    bandwidth = avg_l1_norm / avg_origin_norm

    test_result = {
        'F1': F1,
        'TPR': TPR,
        'FPR': FPR,
        'overall_ACC': overall_ACC,
        'per_class_acc': per_class_acc,
        'bandwidth': bandwidth,
        'max_noise': max_noise,
    }
    return test_result

#  Closed-world defense evaluation in strict-limit mode based on pre-collected gate mappings.
def test_generator_fewShot_strict(DataSet_Name, CF_ModelName, feature_extractor, few_shot_moegen,strict_threshold, sample_num=10):
    """
     Strict test mode using gate mappings from pre-collected samples and label-based lookup.
    
    Args:
        DataSet_Name: Dataset name.
        CF_ModelName: Classifier model name.
        feature_extractor: Feature extractor.
        gating_network: Gating network.
        expert_generator: Expert generator.
        preTrain_x: Pre-collected samples used to build the gate mapping.
        preTrain_y: Labels of pre-collected samples.
        strict_threshold: Strict perturbation threshold.
        sample_num: Number of samples per class.
    """
    #  Hyperparameter settings.
    batch_size = 512

    #  Load classifier model.
    Classfy_model = Load_Classfy_Model(CF_ModelName, DataSet_Name)

    #  Load data.
    test_data, test_labels = Load_Data(DataSet_Name, 'test')
    len_test_data = len(test_data)
    adv_test_data = np.zeros_like(test_data)  #  Store perturbed samples.
    max_noise = 0  #  Track the maximum perturbation value.
    total_l1_origin = 0  #  Sum of original-input L1 norms.
    total_l1_norm = 0  #  Sum of perturbed-input L1 norms.
    total_samples = 0  #  Total sample count.

    #  Stage 1: pre-collection, build the label-to-gate_weights mapping.
    map_Gates = getmap_Gates(feature_extractor, few_shot_moegen.gating_network, test_data, test_labels, sample_num)

    #  Stage 2: testing, look up gate weights by label.
    #  Extract test labels and fetch the corresponding gate weights.
    test_predict_weights = []
    for label in test_labels:
        test_predict_weights.append(map_Gates[label])
    test_predict_weights = np.array(test_predict_weights)

    for i in range(0, len_test_data, batch_size):
        left = i
        right = min(i + batch_size, len_test_data)
        batch_test_data = test_data[left:right]
        cur_test_batchsize = tf.shape(batch_test_data)[0]

        #  Original-input L1 norm.
        abs_batchdata = tf.abs(batch_test_data)
        flat_batchdata = tf.reshape(abs_batchdata, [cur_test_batchsize, -1])
        total_l1_origin += tf.reduce_sum(flat_batchdata).numpy()

        #  Use precomputed gate weights instead of extracting features online.
        cur_predict_weights = test_predict_weights[left:right]
        #  Perturbation generation: call the expert generator directly with gate weights.
        test_batchnoise = few_shot_moegen.expert_generator(cur_predict_weights, training=False)
        test_batchnoise = tf.round(test_batchnoise)

        #  Strict mode: cap perturbation on each burst.
        test_batchnoise = tf.clip_by_value(test_batchnoise, 0, strict_threshold)

        #  Perturbation direction and application.
        test_batchsign = custom_sign(batch_test_data)
        test_batchnoise_withsign = test_batchnoise * test_batchsign
        adv_test_batchdata = batch_test_data + test_batchnoise_withsign

        #  Store into the full adversarial sample buffer.
        adv_test_data[left:right] = adv_test_batchdata.numpy()

        #  Collect perturbation statistics.
        abs_noise = tf.abs(test_batchnoise_withsign)
        flat_noise = tf.reshape(abs_noise, [cur_test_batchsize, -1])

        max_vals = tf.reduce_max(flat_noise, axis=1)  #  Maximum perturbation per sample.
        max_noise = max(max_noise, tf.reduce_max(max_vals).numpy())  #  Overall maximum perturbation.

        l1_norms = tf.reduce_sum(flat_noise, axis=1)  #  Per-sample L1 norm.
        total_l1_norm += tf.reduce_sum(l1_norms).numpy()
        total_samples += cur_test_batchsize.numpy()

    #  Model evaluation.
    F1, TPR, FPR, overall_ACC, per_class_acc = evaluate_model(Classfy_model, adv_test_data, test_labels, batch_size=batch_size)

    #  Average perturbation overhead.
    avg_origin_norm = total_l1_origin / total_samples
    avg_l1_norm = total_l1_norm / total_samples
    bandwidth = avg_l1_norm / avg_origin_norm

    test_result = {
        'F1': F1,
        'TPR': TPR,
        'FPR': FPR,
        'overall_ACC': overall_ACC,
        'per_class_acc': per_class_acc,
        'bandwidth': bandwidth,
        'max_noise': max_noise,
    }
    return test_result

