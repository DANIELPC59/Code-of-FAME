# Data loading utilities.
import os
import sys
project_path=os.getcwd()
sys.path.append(project_path)
from DataTool_Code.DataTool import stable_label_encoder

import numpy as np
import numpy as np

Base_Path='dataset/Website_Fingerprinting'
def get_tar_class_data(datas,labels,target_c):

    index=np.where(labels==target_c)[0]
    target_data=datas[index]
    target_label=labels[index]
    return target_data,target_label

def getData_Path(DataSet,datatype):
    if DataSet=='DF':
        data_path=Base_Path+f"/DF/close-world/Nodef_Burst/{datatype}.npz"
    elif DataSet.startswith('AWF'):
        data_path=Base_Path+f"/AWF/close-world/{DataSet}/Nodef_Burst/{datatype}.npz"
    else:
        raise ValueError(f"Unknown DataSet: {DataSet}")
    return data_path

# Load data for each dataset split.
def Load_Data(DataSet,datatype):
    datapath=getData_Path(DataSet,datatype)
    data_combine=np.load(datapath,allow_pickle=True)
    x_data=data_combine['data']
    y_data=data_combine['labels']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    if(DataSet.startswith('AWF')):
        y_data,label_to_idx=stable_label_encoder(y_data)
    x_data=x_data[:,:,np.newaxis]
    return x_data.astype('float32'), y_data.astype('float32')


def Load_Data_OriLabel_For_AWF_Dataset(DataSet,datatype):
    # For AWF datasets, keep the original string labels without converting them
    # to numeric indices.
    datapath=getData_Path(DataSet,datatype)
    data_combine=np.load(datapath,allow_pickle=True)
    x_data=data_combine['data']
    y_data=data_combine['labels']
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    if(not DataSet.startswith('AWF')):
        raise ValueError(f"{DataSet} SHOULD USE Load_Data to load data")

    x_data=x_data[:,:,np.newaxis]
    return x_data, y_data





if __name__=="__main__":
    data,label=Load_Data('AWF100','adv')
    print(data.shape)
    print(label.shape)
