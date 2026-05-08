import tensorflow as tf
import numpy as np
import argparse
import time

import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

from DataTool_Code.DataLoader import Load_Data
from WF_Model.CFModel_Loder import Load_Classfy_Model
from WF_Model.Eva_ClassfyModel import evaluate_model


def running_WalkieTalkie(test_x,test_y,decoy_x,decoy_y):
    from Defence_Method.WalkieTalkie.WalkieTalkie import WalkieTalkie
    # generate perturbations
    start_time = time.time()
    
    test_x = tf.convert_to_tensor(test_x, dtype=tf.int32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.int32)
    decoy_x = tf.convert_to_tensor(decoy_x, dtype=tf.int32)
    decoy_y = tf.convert_to_tensor(decoy_y, dtype=tf.int32)
    
    wt = WalkieTalkie()
    perturbed_x_Burst=wt.generate_walkie_talkie_samples(test_x, test_y, decoy_x, decoy_y)  # decoy_x/decoy_y are used as decoy samples to build the supersequence
    

   
    print(f"Total running time: {time.time() - start_time:.2f} seconds")
    return perturbed_x_Burst

def eva_walkie_talkie(ori_x,perturbed_x,test_labels,dataset_name,cf_model_name):
    cf_model=Load_Classfy_Model(cf_model_name,dataset_name)
    
    
    ori_F1, ori_TPR, ori_FPR, ori_overall_ACC,ori_per_class_acc=evaluate_model(cf_model,ori_x,test_labels)
    print(f"Original Model F1: {np.mean(ori_F1)}, Overall ACC: {np.mean(ori_overall_ACC)}, Per Class ACC: {ori_per_class_acc}")

    
    WalkieTalkie_F1, WalkieTalkie_TPR, WalkieTalkie_FPR, WalkieTalkie_overall_ACC,WalkieTalkie_per_class_acc=evaluate_model(cf_model,perturbed_x,test_labels)
    print(f"WalkieTalkie Model F1: {np.mean(WalkieTalkie_F1)}, Overall ACC: {np.mean(WalkieTalkie_overall_ACC)}, Per Class ACC: {WalkieTalkie_per_class_acc}")
    
    
    # overhead
    ori_Packets = np.sum(np.abs(ori_x), axis=1)
    WalkieTalkie_Packets = np.sum(np.abs(perturbed_x), axis=1)
    print(f"Original mean packets: {np.mean(ori_Packets)}, WalkieTalkie mean packets: {np.mean(WalkieTalkie_Packets)}")
    overhead= (np.mean(WalkieTalkie_Packets) - np.mean(ori_Packets)) / np.mean(ori_Packets)
    print(f"Overhead: {overhead}")


def main():
    
    Base_save_directory='/dataset/Website_Fingerprinting'
    ## main
    # data_name_list=['AWF100','DF']
    data_name_list=['DF']
    cf_model_list=['AWF','DF','VarCNN']
    for data_name in data_name_list:
        train_data, train_labels = Load_Data(data_name,'adv')
        test_data, test_labels = Load_Data(data_name,'test')
       
        train_data=np.squeeze(train_data, axis=-1)
        test_data=np.squeeze(test_data, axis=-1)
        print(f"Begin WalkieTalkie Defence for {data_name}, test_data.shape:{test_data.shape}")
        perturbed_x=running_WalkieTalkie(test_data,test_labels,train_data,train_labels)

        if data_name=='AWF100':
            save_directory=f'{Base_save_directory}/AWF/close-world/{data_name}/WalkieTalkie_Data'
        elif data_name=='DF':
            save_directory=f'{Base_save_directory}/{data_name}/close-world/WalkieTalkie_Data'
        os.makedirs(save_directory, exist_ok=True)
        np.savez(os.path.join(save_directory, f'WalkieTalkie_test.npz'), data=perturbed_x, labels=test_labels)
        print('Success Get WalkieTalkie Data')


        for cf_model in cf_model_list:
            eva_walkie_talkie(test_data,perturbed_x,test_labels,data_name,cf_model)


if __name__ == '__main__':
    main()
