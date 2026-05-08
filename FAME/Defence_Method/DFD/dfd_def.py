import os
import sys
Project_Path=os.getcwd()
sys.path.append(Project_Path)


import numpy as np

def DFD_CW_Def(dataset_name):
    Base_save_directory='dataset/Website_Fingerprinting'

    if dataset_name=='AWF100':
        save_directory=f'{Base_save_directory}/AWF/close-world/{dataset_name}/DFD_Data/DFD_test.npz'
    elif dataset_name=='DF':
        save_directory=f'{Base_save_directory}/{dataset_name}/close-world/DFD_Data/DFD_test.npz'
    DFD_data = np.load(save_directory)
    data_x=DFD_data['data']
    label_y=DFD_data['labels']
    return data_x, label_y

if __name__ == '__main__':
    dataset_name='DF'
    CF_Model_List=['AWF','DF','VarCNN']
    from DataTool_Code.DataLoader import Load_Data
    data_x,label_y=Load_Data(dataset_name,'test')
    per_x, label_y=DFD_CW_Def(dataset_name)

    # overhead
    ori_Packets = np.sum(np.abs(data_x), axis=1)
    DFD_Packets = np.sum(np.abs(per_x), axis=1)
    print(f"Original mean packets: {np.mean(ori_Packets)}, DFD mean packets: {np.mean(DFD_Packets)}")
    overhead= (np.mean(DFD_Packets) - np.mean(ori_Packets)) / np.mean(ori_Packets)
    print(f"Overhead: {overhead}")

    for CF_Model in CF_Model_List:
        from WF_Model.Eva_ClassfyModel import evaluate_model_byName
        F1, overall_ACC, per_class_acc=evaluate_model_byName(CF_Model, per_x, label_y, dataset_name)
        print(f"In {dataset_name} dataset, {CF_Model} Avg_F1: {np.mean(F1)}, Overall ACC: {np.mean(overall_ACC)}, Per Class ACC: {per_class_acc}")
