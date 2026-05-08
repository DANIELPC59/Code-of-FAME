import os
import sys

#  Add the project root to sys.path.
project_path = os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

import numpy as np



from DataTool_Code.DataLoader import Load_Data
from Defence_Method.AWA.awa_class import AWA_Class
from Defence_Method.AWA.awa_def import Eva_awa_CW
from WF_Model.CFModel_Loder import Load_Classfy_Model
from WF_Model.Eva_ClassfyModel import evaluate_model

def Get_keycup(DataSet,CFmodel):
    npz_path = f'Defence_Method/AWA/File_Save/Gen_Save/{DataSet}/{CFmodel}'
    files = [f for f in os.listdir(npz_path) if f.endswith(".npz")]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {path}, but found {len(files)}")
    file_name = files[0]
    full_path = os.path.join(npz_path, file_name)
    data = np.load(full_path)
    print("Keys in npz:", data.files)
    key1 = data['key1']
    key2 = data['key2']
    return key1,key2



#  Load a generator and perturb data from the specified class.
def gen_adv_for_class(data_x, data_y, label, gen_path):
    #  Get all samples for this class.
    inds = np.where(data_y == label)[0]
    if len(inds) == 0:
        return np.empty((0, data_x.shape[1])), np.empty((0,))
    x_cls = data_x[inds]
    y_cls = data_y[inds]
    #  Random noise input.
    noise = np.random.normal(size=x_cls.shape)
    #  Load generator.
    model = generator_model((x_cls.shape[1],))
    model.load_weights(gen_path)
    model.trainable = False
    #  Generate perturbations.
    pert = model(noise, training=False).numpy()
    adv_x = x_cls + pert * np.sign(x_cls)
    adv_x = np.round(adv_x)
    return adv_x, y_cls,pert

#  DF dataset filtering.
def remove_df_data(data_x, data_y):
    #  AWA uses paired classes; DF has 95 labels, so remove label 94 to keep an even count.
   
    keep_mask = data_y != 94
    #  Filter arrays directly with a mask.
    data_x = data_x[keep_mask]
    data_y = data_y[keep_mask]
    return data_x, data_y

if __name__ == "__main__":
    #  Path template.
    dataset_name = "DF"
    CF_Model_name="DF"
    print(f"Eva AWA ==>Dataset: {dataset_name}, CF Model: {CF_Model_name}")
    data_x, data_y = Load_Data(dataset_name, 'test')

    if dataset_name=='DF':
        data_x,data_y = remove_df_data(data_x, data_y)

    cf_model=Load_Classfy_Model(CF_Model_name,dataset_name)
    #  Run defense.
    pertubation_data=Eva_awa_CW(data_x,data_y,dataset_name,CF_Model_name)
    
    #  Evaluate original data.
    ori_F1, ori_TPR, ori_FPR, ori_overall_ACC,ori_per_class_acc=evaluate_model(cf_model,data_x,data_y)
    print(f"Original Model F1: {np.mean(ori_F1)}, Overall ACC: {np.mean(ori_overall_ACC)}, Per Class ACC: {ori_per_class_acc}")

    #  Evaluate perturbed data.
    awa_F1, awa_TPR, awa_FPR, awa_overall_ACC,awa_per_class_acc=evaluate_model(cf_model,pertubation_data,data_y)
    print(f"awa Model F1: {np.mean(awa_F1)}, Overall ACC: {np.mean(awa_overall_ACC)}, Per Class ACC: {awa_per_class_acc}")
    
    
    # overhead
    ori_Packets = np.sum(np.abs(data_x), axis=1)
    awa_Packets = np.sum(np.abs(pertubation_data), axis=1)
    print(f"Original mean packets: {np.mean(ori_Packets)}, awa mean packets: {np.mean(awa_Packets)}")
    overhead= (np.mean(awa_Packets) - np.mean(ori_Packets)) / np.mean(ori_Packets)
    print(f"Overhead: {overhead}")
