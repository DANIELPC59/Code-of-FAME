import os
import sys
project_root =os.getcwd()
os.chdir(project_root)
sys.path.append(project_root)

import  tensorflow as tf
from tqdm import tqdm
from Tool_Code import Metrics
from DataTool_Code.DataLoader import Load_Data
from WF_Model.CFModel_Loder import Load_Classfy_Model


def evaluate_model(model, test_data, test_labels,batch_size=128):
    """
    Evaluate a classification model on the given test set.
    
    Args:
        model      : Trained classification model.
        test_data  : Test samples.
        test_labels: Ground-truth labels.
        batch_size : Inference batch size. Default is 128.
    
    Returns:
        F1, TPR, FPR, overall_ACC, per_class_acc
        (use np.mean() to compute the macro-average)
    """
    predicted_labels_list = []
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch_data = test_data[i:i + batch_size]
        # print(batch_data.shape)
        test_outputs= model(batch_data)
        predicted=tf.argmax(test_outputs,1)
        predicted_labels_list.append(predicted.numpy())
    
    predicted_labels= tf.concat(predicted_labels_list, axis=0).numpy()
    F1, TPR, FPR, overall_ACC,per_class_acc = Metrics.get_metrics(test_labels, predicted_labels)
    return F1, TPR, FPR, overall_ACC,per_class_acc


def evaluate_model_byName(model_name,test_data,test_label,DataSet_Name,burst_length=2000):
    classification_model = Load_Classfy_Model(model_name,DataSet_Name,burst_length)
    F1,_,_,overall_ACC,per_class_acc=evaluate_model(classification_model, test_data, test_label)
    return F1, overall_ACC, per_class_acc

if __name__ == '__main__':
    

    batch_size = 128
    burst_length = 2000
    DataSet='DF'                    # AWF100, AWF200, AWF500,AWF900
    model_name = 'AWF'            # VarCNN,DF,AWF
    print(f"<===  Evaluate {model_name} in {DataSet} dataset  ===>")
    test_data,test_label = Load_Data(DataSet, 'test')

    classification_model = Load_Classfy_Model(model_name,DataSet,burst_length)
    
    # run evaluation
    F1,_,_,overall_ACC,per_class_acc=evaluate_model(classification_model, test_data, test_label,batch_size)
    avg_F1=F1.mean()
    print(f"Evaluate {model_name} in {DataSet} dataset:")
    print(f"F1: {avg_F1}")
    print(f" Overall ACC: {overall_ACC}")

    print("per class acc: ", per_class_acc)