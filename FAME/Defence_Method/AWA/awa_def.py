#  AWA evaluation interfaces for closed-world and open-world scenarios.


import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from Defence_Method.AWA.awa_class import AWA_Class
from DataTool_Code.DataLoader import Load_Data
from WF_Model.CFModel_Loder import Load_Classfy_Model
from WF_Model.Eva_ClassfyModel import evaluate_model


#  Configuration parameters.



AWA_FILE_BASE_PATH = "Defence_Method/AWA/File_Save/Gen_Save"  

def get_key_relation_path(base_path):
    # print('base_path:',base_path)
    files = [f for f in os.listdir(base_path) if f.endswith(".npz")]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {base_path}, but found {len(files)}")
    file_name = files[0]
    full_path = os.path.join(base_path, file_name)
    return full_path
def get_Logit_Layer(ClassfyModelName):
    if ClassfyModelName=='DF':
        logit_layer = ['fc3']
    elif ClassfyModelName=='AWF':
        logit_layer =['flatten']
    elif ClassfyModelName=='VarCNN':
        logit_layer=['average_pool']
    return logit_layer
"""
 Evaluate AWA in the closed-world setting.
 Input: original dataset and classifier model.
 Output: perturbed data.
 Approach: extract samples from each class, perturb them, and write them back to their original positions.
"""
def Eva_awa_CW(data_x,label_y,DataSet_Name,WF_Model_Name):
    
    
    #  Common parameters.
    Class_num=100  
    BURST_LEN = 2000
    LOGIT_LAYER=get_Logit_Layer(WF_Model_Name)
    #  Load the key mapping table.
    keys__base_path=f'{AWA_FILE_BASE_PATH}/{DataSet_Name}/{WF_Model_Name}'  #  Key mapping save path.
    keys_path=get_key_relation_path(keys__base_path)
    keys=np.load(keys_path) #  Load the key mapping file.
    key1=keys['key1']
    key2=keys['key2']
    
    adv_data = np.zeros_like(data_x) #  Store perturbed data.
    print(f'==>AWA Begin CW Defence agenst {WF_Model_Name} in {DataSet_Name}')
    pbar = tqdm(range(len(key1)), desc=f"AWA Defending", unit="pair")
    for cls in pbar:
        pbar.set_postfix(pair=f"{key1[cls]} -> {key2[cls]}")
        # print(f'==>AWA Defence class {key1[cls]} and {key2[cls]}')
        #  Find indices for the source class.
        src_indices = np.where(label_y == key1[cls])[0]
        if len(src_indices) == 0:
            print(f'no data for label {key1[cls]},in key1')
            continue
            
        x_src = data_x[src_indices]
        
        #  Find indices for the target class.
        tar_indices = np.where(label_y == key2[cls])[0]
        if len(tar_indices)==0:
            print(f'no data for label {key2[cls]},in key2')
            continue
        # print(f"defence clss {key1[cls]} and {key2[cls]}")
        x_tar = data_x[tar_indices]
        Gen_path=f'{AWA_FILE_BASE_PATH}/{DataSet_Name}/{WF_Model_Name}/Best_Gs_Output_{key1[cls]}-->{key2[cls]}'
        
        #  Load generator weights.
        awa = AWA_Class(trace_len=TRACE_LEN, logit_layer=LOGIT_LAYER, awa_type=Class_num,
                        trace_length=BURST_LEN, CF_modelname=WF_Model_Name)
        try:
            awa.generator1.load_weights(f"{Gen_path}/g1.h5")
            awa.generator2.load_weights(f"{Gen_path}/g2.h5")
        
        except Exception as e:
            print(f"{key1[cls]} and {key2[cls]} failed to load: {e}")
            continue
        
        #  Perturb source-class data.
        src_noise = tf.random.normal(x_src.shape)
        adv_src = awa.adjusted_generated_1(x_src, src_noise, training=False).numpy()
        
        #  Perturb target-class data.
        tar_noise = tf.random.normal(x_tar.shape)
        adv_tar= awa.adjusted_generated_2(x_tar, tar_noise, training=False).numpy()
        perb_src=adv_src-x_src
        perb_tar=adv_tar-x_tar

        adv_data[src_indices] = adv_src
        adv_data[tar_indices] = adv_tar
    return adv_data

def Eva_awa_OW(data_x, WF_Model_Name, batch_size=20):
    """
     Simulate perturbations for open-world data.
     For each batch, randomly select a class pair and use G1/G2 to perturb the first and second halves.

     Args:
        data_x: Original unlabeled traffic data, shape [N, T, 1].
        WF_Model: Classifier model name, such as "AWF".
        batch_size: Number of samples perturbed per round.

     Returns:
        adv_data: Perturbed data, shape [N, T, 1].
    """
   
    #  Common parameters.
    Class_num=100  #  AWF100 is used here, so the class count is fixed to 100.
    TRACE_LEN = 2000
    BURST_LEN = 2000
    LOGIT_LAYER = get_Logit_Layer(WF_Model_Name)  #  Used during training; not critical for testing.
    
    import random
    
    total_len = len(data_x)
    adv_data = np.zeros_like(data_x)
    print('==>AWA Begin OW Defence')
    #  Load key mapping.
    key_path_base = f'Defence_Method/AWA/File_Save/Gen_Save/AWF100/{WF_Model_Name}'
    keys_path = get_key_relation_path(key_path_base)
    keys = np.load(keys_path)
    key1 = keys['key1']
    key2 = keys['key2']
    total_pairs = len(key1)
    
    #  Build batches.
    for start in tqdm(range(0, total_len, batch_size)):
        end = min(start + batch_size, total_len)
        batch_x = data_x[start:end]

        #  Randomly select one class pair.
        rand_idx = random.randint(0, total_pairs - 1)
        cls_src = key1[rand_idx]
        cls_trg = key2[rand_idx]
        # print(f'cls_src={cls_src},cls_trg={cls_trg}')
        #  Load generators.
        awa = AWA_Class(trace_len=TRACE_LEN, logit_layer=LOGIT_LAYER, awa_type=Class_num,
                        trace_length=BURST_LEN, CF_modelname=WF_Model_Name)
        gen_path = f"{key_path_base}/Best_Gs_Output_{cls_src}-->{cls_trg}"
        # print(gen_path)
        try:
            awa.generator1.load_weights(f"{gen_path}/g1.h5")
            awa.generator2.load_weights(f"{gen_path}/g2.h5")
        except Exception as e:
            print(f"[CLS {cls_src}->{cls_trg}] failed to load generators: {e}")
            #  Skip this batch by keeping the original data.
            adv_data[start:end] = batch_x
            continue

        #  Split the batch in half.
        half = (end - start) // 2
        noise_g1 = tf.random.normal(batch_x[:half].shape)
        noise_g2 = tf.random.normal(batch_x[half:].shape)

        adv_g1 = awa.adjusted_generated_1(batch_x[:half], noise_g1, training=False).numpy()
        adv_g2 = awa.adjusted_generated_2(batch_x[half:], noise_g2, training=False).numpy()

        #  Merge perturbed samples and write them into the result buffer.
        adv_data[start:start+half] = adv_g1
        adv_data[start+half:end] = adv_g2
    # print(adv_data.shape)
    return adv_data

if __name__ == '__main__':
    BURST_LEN=2000
    Model_List=['DF','AWF','VarCNN']
    DataSet_Name='AWF100'
    test_data, test_labels = Load_Data(DataSet_Name, "test")
    for CF_Model in Model_List:
        adv_data=Eva_awa_CW(test_data,test_labels,DataSet_Name,CF_Model)
        model_=Load_Classfy_Model(CF_Model,DataSet_Name,BURST_LEN)
        F1, TPR, FPR, overall_ACC,per_class_acc = evaluate_model(model_, adv_data, test_labels, batch_size=32)
        #  Compute bandwidth overhead.
        numerator = tf.reduce_sum(tf.abs(adv_data) - tf.abs(test_data))
        denominator = tf.reduce_sum(tf.abs(test_data)) + 1e-8  #  Avoid division by zero.
        overhead=(numerator / denominator)
        #  F1 score.
        AvgF1=np.mean(F1)
        print(f"==> AWA Defence {CF_Model} in {DataSet_Name}")
        print(f"Overall ACC: {overall_ACC}")
        print("F1:", AvgF1)
        print("Bandwidth: {:.2%}".format(overhead))
        print(f"per ACC: {per_class_acc}")


    
        
