import os 
import sys
project_path = os.getcwd()
sys.path.append(project_path)

import numpy as np

from Defence_Method.Front_Base_Burst.burst_front import BurstFRONT
from DataTool_Code.DataLoader import Load_Data
def get_NcValue(DataSet_Name):
    #  Approximate Accu as Nc / 2.
    if DataSet_Name == 'DF':  # DF mean packet count: 1845
        return 190            
    elif DataSet_Name=='AWF100':  # AWF100 mean packet count: 3122.51
        return 350 
    elif DataSet_Name=='AWF200':  # AWF200 mean packet count: 3093.04
        return 310
    else:
        raise ValueError(f"Unknown DataSet_Name: {DataSet_Name}")


def main():
    # Hyperparameters  Settings
    DataSet_Name = 'AWF200'   # 'AWF100' , 'AWF200', DF
    Nc = get_NcValue(DataSet_Name)          # dummy packet insert in outgoing direction  (from client to server)
    Ns = 0              # dummy packet insert in incoming direction  (from server to client)
    Wmin = 0.05          # minimum burst length
    Wmax = 0.35          # maximum burst length
    max_burst_len = 2000 # maximum burst length
    random_seed = 42
    

    front = BurstFRONT(
        Nc=Nc,
        Ns=Ns,
        Wmin=Wmin,
        Wmax=Wmax,
        max_burst_len=max_burst_len,
        seed=random_seed,
    )

    data, labels = Load_Data(DataSet_Name,'test')
    defended_data, meta = front.defend(data, return_metadata=True)
    print(f"Defence {DataSet_Name} Over==>")
    Base_Path = '/dataset/Website_Fingerprinting'
    if DataSet_Name == 'DF':
        Save_Path=Base_Path+'/'+f'{DataSet_Name}/close-world/Front_Data/front_test.npz'
    elif DataSet_Name.startswith('AWF'):
        Save_Path=Base_Path+'/'+f'AWF/close-world/{DataSet_Name}/Front_Data/front_test.npz'
    os.makedirs(os.path.dirname(Save_Path), exist_ok=True)
    bandwith=float((np.mean(meta.defended_packets)-np.mean(meta.original_packets))/np.mean(meta.original_packets))
    np.savez(Save_Path, data=defended_data, labels=labels, bandwith=bandwith)
    print(f"Mean overhead: {bandwith:.6f}")
    print(f"Saved defended data to {Save_Path}")

    # Evaluation
    from WF_Model.CFModel_Loder import Load_Classfy_Model
    from WF_Model.Eva_ClassfyModel import evaluate_model
    model_list=['AWF','DF','VarCNN']
    for model_name in model_list:
        model = Load_Classfy_Model(model_name,DataSet_Name)
        F1, TPR, FPR, overall_ACC,per_class_acc = evaluate_model(model, defended_data, labels)
        print(f"Model {model_name} Avg_F1: {np.mean(F1):.4f},  Overall ACC: {np.mean(overall_ACC):.4f}, Per Class ACC: {per_class_acc}")
if __name__ == "__main__":
    main()
