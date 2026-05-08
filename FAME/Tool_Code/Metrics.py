from sklearn.metrics import confusion_matrix
import numpy as np

def get_metrics(y_test, predicted_labels):  # input shape is [Batch]  suchas y_test[0] = 10 means site 10
    cnf_matrix = confusion_matrix(y_test, predicted_labels)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)   # False Positive num
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)   # False Negative num
    TP = np.diag(cnf_matrix)                            # True Positive num
    TN = cnf_matrix.sum() - (FP + FN + TP)              # True Negative num
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/((TP+FN+1e-8))
    # Specificity or true negative rate
    TNR = TN/(TN+FP+1e-8)
    # Precision or positive predictive value
    PPV = TP/(TP+FP+1e-8)
    # print(TP)
    # print((TP+FP))
    
    NPV = TN/(TN+FN+1e-8)       # Negative predictive value
    
    FPR = FP/(FP+TN+1e-8)       # Fall out or false positive rate
    
    FNR = FN/(TP+FN+1e-8)       # False negative rate
    
    FDR = FP/(TP+FP+1e-8)       # False discovery rate
    # Overall accuracy for each class
    per_class_acc = TP / (TP + FN + 1e-8)
    F1 = 2*PPV*TPR/(PPV+TPR+1e-8)
    overall_TPR = np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-8)
    overall_FDR = np.sum(FP) / (np.sum(FP) + np.sum(TP) + 1e-8)
    overall_PPV = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-8)
    overall_FPR = np.sum(FP) / (np.sum(FP) + np.sum(TN) + 1e-8)
    overall_F1 = 2 * overall_PPV * overall_TPR / (overall_PPV + overall_TPR + 1e-8)
    overall_ACC = np.sum(TP) / np.sum(cnf_matrix) 
    return F1,TPR,FPR,overall_ACC,per_class_acc