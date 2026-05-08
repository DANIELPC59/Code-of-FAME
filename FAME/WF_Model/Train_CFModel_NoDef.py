import os
import sys
project_path=os.getcwd()
print("project_path:", project_path)
sys.path.append(project_path)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import logging
# external module imports
from DataTool_Code.DataLoader import Load_Data
from WF_Model.Model.DF_TF import DFNet
from WF_Model.Model.AWF_TF import AWFNet
from WF_Model.Model.Var_CNN_TF import VarCNN

random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def trainCFModel(DataSet_name, CF_Model_name, NB_EPOCH, BATCH_SIZE, 
                 X_train, y_train, X_valid, y_valid, NB_CLASSES, INPUT_SHAPE, VERBOSE):
    
    tf.keras.backend.clear_session()
    
    description = f"Training and evaluating {CF_Model_name} model for closed-world scenario on {DataSet_name} non-defended dataset"
    logger.info(description)
    
    if CF_Model_name == "DF":
        model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    elif CF_Model_name == "AWF":
        model = AWFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    elif CF_Model_name == "VarCNN":
        model = VarCNN.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
        
    OPTIMIZER = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    print("Model compiled")
    
    # ensure the save directory exists
    save_dir = f"WF_Model/ModelSave/DataSet_{DataSet_name}/{CF_Model_name}/"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{CF_Model_name}_model_in_{DataSet_name}_DataSet.h5")

    checkpoint = ModelCheckpoint(model_save_path, 
                                monitor='val_accuracy', 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=True,
                                mode='max')
                                
    print("Training shapes -> X:", X_train.shape, "y:", y_train.shape)
    
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        verbose=VERBOSE, validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint])

    score_valid = model.evaluate(X_valid, y_valid, verbose=VERBOSE)
    logger.info(f"Validating accuracy (Last Epoch): {score_valid[1]:.4f}")

    model.load_weights(model_save_path)
    logger.info("Loaded best model weights from checkpoint.")
    
    score_valid_best = model.evaluate(X_valid, y_valid, verbose=VERBOSE)
    logger.info(f"Best saved model validation accuracy: {score_valid_best[1]:.4f}")
    logger.info("-" * 50)

if __name__ == '__main__':

    # Logging configuration
    log_dir = '/FAME/WF_Model/ModelSave'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)


    DataSet_list = ["AWF500"]
    # DataSet_list = ["DF", "AWF100", "AWF200"]
    # CF_Model_list = ["DF", "AWF", "VarCNN"]
    CF_Model_list = [ "AWF", "VarCNN"]

    NB_EPOCH = 30
    logger.info("Number of Epoch: %d", NB_EPOCH)
    BATCH_SIZE = 128
    VERBOSE = 1
    FLOW_LENGTH = 2000
    INPUT_SHAPE = (FLOW_LENGTH, 1)

    for dataset in DataSet_list:
        if dataset == "DF":
            NB_CLASSES = 95
        elif dataset == "AWF100":
            NB_CLASSES = 100
        elif dataset == "AWF103":
            NB_CLASSES = 103
        elif dataset == "AWF200":
            NB_CLASSES = 200
        elif dataset == "AWF500":
            NB_CLASSES = 500
        elif dataset == "AWF900":
            NB_CLASSES = 900
            
        logger.info(f"Loading data for {dataset}...")
        X_train, y_train = Load_Data(dataset, 'train')
        X_valid, y_valid = Load_Data(dataset, 'test')
        
        y_train = to_categorical(y_train, NB_CLASSES)
        y_valid = to_categorical(y_valid, NB_CLASSES)

        # rename loop variable to avoid shadowing the imported model modules
        for model_name in CF_Model_list:
            trainCFModel(
                DataSet_name=dataset, 
                CF_Model_name=model_name, 
                NB_EPOCH=NB_EPOCH, 
                BATCH_SIZE=BATCH_SIZE,
                X_train=X_train, 
                y_train=y_train, 
                X_valid=X_valid, 
                y_valid=y_valid,
                NB_CLASSES=NB_CLASSES, 
                INPUT_SHAPE=INPUT_SHAPE, 
                VERBOSE=VERBOSE
            )