
import os 
import sys 
project_path=os.getcwd()
sys.path.append(project_path)
import numpy as np

from Defence_Method.Alert.Alert_def import Alert_def_cw
from DataTool_Code.DataLoader import Load_Data
from WF_Model.CFModel_Loder import Load_Classfy_Model
from WF_Model.Eva_ClassfyModel import evaluate_model

def Eva_Alert(DataSet_Name,CF_Model_Name):
    print(f"Evaluating ALERT base {CF_Model_Name} on {DataSet_Name} dataset:")
    data_x,data_y=Load_Data(DataSet_Name,'test')
    cf_model=Load_Classfy_Model(CF_Model_Name,DataSet_Name)
    ori_F1, ori_TPR, ori_FPR, ori_overall_ACC,ori_per_class_acc=evaluate_model(cf_model,data_x,data_y)
    print(f"Original Model F1: {np.mean(ori_F1)}, Overall ACC: {np.mean(ori_overall_ACC)}, Per Class ACC: {ori_per_class_acc}")

    pertubation_data=Alert_def_cw(data_x,data_y,CF_Model_Name)
    alert_F1, alert_TPR, alert_FPR, alert_overall_ACC,alert_per_class_acc=evaluate_model(cf_model,pertubation_data,data_y)
    print(f"Alert Model F1: {np.mean(alert_F1)}, Overall ACC: {np.mean(alert_overall_ACC)}, Per Class ACC: {alert_per_class_acc}")
    # overhead
    ori_Packets = np.sum(np.abs(data_x), axis=1)
    alert_Packets = np.sum(np.abs(pertubation_data), axis=1)
    print(f"Original mean packets: {np.mean(ori_Packets)}, Alert mean packets: {np.mean(alert_Packets)}")
    overhead= (np.mean(alert_Packets) - np.mean(ori_Packets)) / np.mean(ori_Packets)
    print(f"Overhead: {overhead}")


if __name__ == '__main__':
    DataSet_Name='DF'
    CF_Model_Name='AWF'

    Eva_Alert(DataSet_Name,CF_Model_Name)
