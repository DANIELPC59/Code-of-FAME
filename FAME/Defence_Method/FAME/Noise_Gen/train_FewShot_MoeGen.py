# FewShot MoE-Gen: perturbation generator training script
import os
import sys
project_path = os.getcwd()
sys.path.append(project_path)
print("project_path:", project_path)

import random
import time
from datetime import datetime, timedelta

import tensorflow as tf
import numpy as np
from tensorflow import optimizers, losses
import argparse
import logging
import traceback
import copy

from Tool_Code.Tool_Fun import custom_sign
from Defence_Method.FAME.Feature_Extracor.model import CF_Extractor
from DataTool_Code.DataLoader import Load_Data
from Defence_Method.FAME.Noise_Gen.Generater.FewShot_Moe_Gen import FewShot_Moe_Gen
from Defence_Method.FAME.Noise_Gen.Eva_Few_Shot.Eva_Tool import test_generator_fewShot


def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logger(log_file_path):
    """Create an independent logger for the experiment."""
    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.INFO)
    
  
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

def train_generator(DataSet_Name, CF_Model_name, Hyper_args, logger):

    logger.info(f"<=== Defence {CF_Model_name} in {DataSet_Name} dataset, "
                f"expert_num={Hyper_args.num_experts}===>")

    # load training data
    train_data, train_labels = Load_Data(DataSet_Name,'adv')
    logger.info(f"Success Load train_data shape: {train_data.shape}")

    # load feature extractor
    input_shape = train_data.shape[1:]
    CF_extrator_save_dir = "Defence_Method/FAME/Feature_Extracor/File_Save"
    extractor_path = os.path.join(CF_extrator_save_dir, f"CF_FeaturExtractor_in_{DataSet_Name}.h5")
    extractor = CF_Extractor.build(input_shape)
    
    
    # load pre-trained extractor weights
    extractor.load_weights(extractor_path)
    extractor.trainable=False
    
    # load WF classifier
    from WF_Model.CFModel_Loder import Load_Classfy_Model
    cf_model = Load_Classfy_Model(CF_Model_name,DataSet_Name)

    # create perturbation generator
    noise_generator = FewShot_Moe_Gen(noise_size=Hyper_args.noise_size,num_experts=Hyper_args.num_experts,temperature=Hyper_args.temperature)
    optimizer = optimizers.Adam(learning_rate=Hyper_args.gen_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer

    min_adv_acc = 1.0
    start_time = time.time()
    
    for epoch in range(Hyper_args.epoch_num):
        # learning rate decay
        if epoch > 0 and epoch % 10 == 0:
            optimizer.learning_rate = optimizer.learning_rate * 0.5
            logger.info(f"Epoch: {epoch}, Learning Rate decay: {optimizer.learning_rate}")
        # dynamically anneal temperature
        current_temp = max(0.05, 0.5 * (0.9 ** epoch))
        noise_generator.temperature = current_temp  # apply updated temperature to the model

        logger.info(f"Epoch: {epoch}")
        epoch_start_time = time.time()
        
        len_train_data=len(train_data)
        # shuffle indices at the start of each epoch
        indices = np.random.permutation(len(train_data))
        train_data = tf.gather(train_data, indices)
        train_labels = tf.gather(train_labels, indices)
        
        expert_usage = np.zeros(Hyper_args.num_experts)
        
        
        for i in range(0, len_train_data, Hyper_args.batch_size):
            left = i
            right = min(i + Hyper_args.batch_size, len_train_data)
            with tf.GradientTape() as tape:
                #  1. Extract current batch.
                batch_data = train_data[left:right]
                batch_labels = train_labels[left:right]
                batch_labels = tf.cast(batch_labels, tf.int32)  # ensure integer labels

                #  2. Extract features.
                output_feature=extractor(batch_data)
                #  3. Generate perturbation.
                noise, gate_weight = noise_generator(output_feature, training=True)
                
                #  4. Apply perturbation.
                batch_sign = custom_sign(batch_data)

                adv_data = batch_data + noise * batch_sign
                expert_usage += np.sum(gate_weight.numpy(), axis=0).squeeze()  # accumulate expert activation counts
                
                # some statistics
                l1_norm = tf.reduce_mean(tf.norm(noise, ord=1, axis=1))  # perturbation L1 norm
                max_noise = tf.reduce_max(noise)  # peak perturbation value
                
                # Loss1: Classification Misdirection Loss
                output = cf_model(adv_data)
                logits_for_true_label = tf.gather_nd(output,tf.stack([tf.range(tf.shape(output)[0]), batch_labels], axis=1))
                Classfy_loss=Hyper_args.class_weight*tf.reduce_mean(tf.maximum(logits_for_true_label, 0))
                
            
                # Loss2: Perturbation Dispersion Loss

                # Clip big values of noise to prevent exploding gradients
                safe_noise = tf.clip_by_value(noise, 0.0, 20.0)
                over_threshold = tf.maximum(0.0, safe_noise - Hyper_args.center_threshold)
                
                center_punish = tf.exp(over_threshold) - 1.0 
                average_max_10square = tf.reduce_mean(tf.reduce_sum(center_punish, axis=1))  
                dispersion_loss = average_max_10square * Hyper_args.norm_weight    
                
                # Loss3: Bandwidth Overhead Loss
                overhead = tf.reduce_sum(tf.abs(adv_data) - tf.abs(batch_data)) / tf.reduce_sum(tf.abs(batch_data))
                overhead_loss=tf.maximum(0.0, overhead - Hyper_args.max_overhead) * Hyper_args.overhead_weight 

                # Loss4: Expert-Balancing Regularization
                mean_gate = tf.reduce_mean(gate_weight, axis=0)
                variance = tf.math.reduce_variance(mean_gate)
                balance_loss = variance * Hyper_args.balance_weight

                # total loss
                loss = Classfy_loss + dispersion_loss + overhead_loss + balance_loss

            grads = tape.gradient(loss, noise_generator.trainable_variables)
            optimizer.apply_gradients(zip(grads, noise_generator.trainable_variables))
            if i/Hyper_args.batch_size%50==0:
                logger.info(f"  batch {i/Hyper_args.batch_size}: loss:{loss} ;   noise_norm:{l1_norm}   max_noise {max_noise}")
                logger.info(f"         class_loss:{Classfy_loss} ; overhead_loss:{overhead_loss} ; dispersion_loss:{dispersion_loss} ; balance_loss:{balance_loss}")
        
        # log expert usage statistics at end of epoch
        expert_usage_ratio = expert_usage / len_train_data
        logger.info(f"each expert usage: {expert_usage_ratio}")
        logger.info(f"expert usage std: {np.std(expert_usage_ratio):.4f}, "
                    f"max: {np.max(expert_usage_ratio):.4f}, "
                    f"min: {np.min(expert_usage_ratio):.4f}")
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time
        logger.info(f" Epoch {epoch} duration: {epoch_duration:.2f}s, Total elapsed time: {total_duration:.2f}s")
        
        # few-shot evaluation
        noise_generator.trainable = False
        test_result=test_generator_fewShot(DataSet_Name, CF_Model_name, extractor,noise_generator)
        logger.info(f"[Test] use max noise : {test_result['max_noise']}  for test_Acc: {test_result['overall_ACC']}  and bandwidth {test_result['bandwidth']}")
        
        # full-scale evaluation 
        noise_generator.trainable = False
        all_test_result=test_generator_fewShot(DataSet_Name, CF_Model_name, extractor,noise_generator,sample_num=3000)
        logger.info(f"[ALL Test] use max noise : {all_test_result['max_noise']}  for test_Acc: {all_test_result['overall_ACC']}  and bandwidth {all_test_result['bandwidth']}")

        noise_generator.trainable = True
        if test_result['overall_ACC'] < min_adv_acc:
            min_adv_acc = test_result['overall_ACC']
        logger.info(f"Current best test Acc: {min_adv_acc}")
        

        #  Save model weights.
        base_save_dir = f'Defence_Method/FAME/Few_Shot_MoeGen/FileSave/randomseed_{Hyper_args.random_seed}/{DataSet_Name}_{CF_Model_name}'
        #  Save gating network weights.
        gateNetwork_saveDir=base_save_dir+'/gating_net/'
        os.makedirs(gateNetwork_saveDir, exist_ok=True)
        noise_generator.gating_network.save_weights(os.path.join(gateNetwork_saveDir, f'{epoch}.h5'))

        #  Save expert generator weights.
        expertGenerator_saveDir=base_save_dir+'/expert_gen/'
        os.makedirs(expertGenerator_saveDir, exist_ok=True)
        noise_generator.expert_generator.save_weights(os.path.join(expertGenerator_saveDir, f'{epoch}.h5'))
        logger.info(f"Epoch {epoch} weights saved to: {base_save_dir}\n")

    logger.info(f"End train, best test Acc: {min_adv_acc}")
 

def get_args():
    parser = argparse.ArgumentParser(description='Moe-Gen WF Defense Training')
    # base hyperparameters
    parser.add_argument('--flow_length', type=int, default=2000, help='input sequence length')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')
    parser.add_argument('--gen_lr', type=float, default=0.002, help='generator learning rate')
    parser.add_argument('--with_da', type=bool, default=False, help='whether to use data augmentation')
    parser.add_argument('--epoch_num', type=int, default=40, help='number of training epochs') 
    parser.add_argument('--save_path', type=str, default='Defence_Method/Moe_Gen/Model_Save', help='path to save model checkpoints')
    parser.add_argument('--random_seed', type=int, default=853, help='random seed for reproducibility')

    # loss weights
    parser.add_argument('--norm_weight', type=float, default=0.008, help='weight for the dispersion loss')  
    parser.add_argument('--class_weight', type=float, default=5, help='weight for the classification loss')   
    parser.add_argument('--overhead_weight', type=float, default=15, help='weight for the overhead penalty loss')
    parser.add_argument('--balance_weight', type=float, default=10, help='weight for the expert-balance loss')
    
    # thresholds and architecture
    parser.add_argument('--noise_size', type=int, default=2000, help='noise vector size')  
    parser.add_argument('--center_threshold', type=float, default=10, help='threshold for the concentration penalty') 
    parser.add_argument('--max_overhead', type=float, default=0.05, help='maximum allowed bandwidth overhead')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature for the gating softmax')

    # class-related hyperparameters
    parser.add_argument('--num_classes', type=int, default=100, help='number of website classes')
    parser.add_argument('--num_experts', type=int, default=10, help='number of MoE experts')
    return parser.parse_args()
def getExpertNum(DataSet_name,CF_Model_name):
    if DataSet_name=="DF":
        if CF_Model_name=="VarCNN":
            return 5
        elif CF_Model_name=="AWF":
            return 8
        elif CF_Model_name=="DF":
            return 5
    elif DataSet_name=="AWF100":
        if CF_Model_name=="VarCNN":
            return 5
        elif CF_Model_name=="AWF":
            return 8
        elif CF_Model_name=="DF":
            return 8
    elif DataSet_name=="AWF200":
        if CF_Model_name=="VarCNN":
            return 8
        elif CF_Model_name=="AWF":
            return 10
        elif CF_Model_name=="DF":
            return 10
def adjust_args(args,DataSet_name,CF_Model_name ):
    # fine-tune hyperparameters based on dataset and classifier model

    args.num_experts = getExpertNum(DataSet_name,CF_Model_name)

    if DataSet_name=="DF":
        args.num_classes = 95 
    elif DataSet_name=="AWF100":
        args.num_classes = 100
    elif DataSet_name=="AWF200":
        args.num_classes = 200

    return args


if __name__ == '__main__':
    # ===================== Experiment Configuration =====================
    DataSet_Name = "AWF100"       #  DataSet:   DF,AWF100,AWF200
    CF_Model_Name = "VarCNN"         #  CF_Model:   DF ; AWF ; VarCNN

    # load and fine-tune hyperparameters
    args = get_args()
    args = adjust_args(args, DataSet_Name, CF_Model_Name)

    # set random seeds for reproducibility
    set_random_seed(args.random_seed)

    # ===================== Logger Setup =====================
    log_dir = f'Defence_Method/FAME/Few_Shot_MoeGen/FileSave/randomseed_{args.random_seed}/{DataSet_Name}_{CF_Model_Name}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_FewShot_MoeGen.log")
    logger = setup_logger(log_file)

    # log experiment startup info
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {DataSet_Name}, Classifier: {CF_Model_Name}")
    logger.info(f"Hyperparameters: {args}")

    # ===================== Run Training =====================
    train_start = time.time()
    try:
        train_generator(DataSet_Name, CF_Model_Name, args, logger)
        train_duration = time.time() - train_start
        logger.info(f"Training completed. Total time: {train_duration:.1f}s")
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Training failed with error:\n{error_msg}")
        raise
