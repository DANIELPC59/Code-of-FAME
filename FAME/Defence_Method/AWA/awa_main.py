#  Main AWA training script.
import os
import datetime
import sys

project_root =os.getcwd()
os.chdir(project_root)
print('project_root:',project_root)
os.chdir(project_root)
sys.path.append(project_root)
#  TensorFlow 2 implementation of AWA training.
from tensorflow.python.keras.backend import clear_session
import numpy as np
import logging
from Defence_Method.AWA.config import *
from Defence_Method.AWA.util import *
from Defence_Method.AWA.awa_class import AWA_Class
from DataTool_Code.DataLoader import *
from WF_Model.CFModel_Loder import  Load_Classfy_Model
from WF_Model.Eva_ClassfyModel import evaluate_model
DataSet_name='DF'
# ClassfyModelList=['AWF','VarCNN','DF']
ClassfyModelList=['VarCNN']

time_list=[]
for ClassfyModelName in ClassfyModelList:
    begin_time=datetime.datetime.now()
    model_=Load_Classfy_Model(ClassfyModelName,DataSet_name,burst_len)
    #  Select the final logit layer according to the classifier model.
    if ClassfyModelName=='DF':
        logit_layer = ['fc3']
    elif ClassfyModelName=='AWF':
        logit_layer =['flatten']
    elif ClassfyModelName=='VarCNN':
        logit_layer=['average_pool']

    #  Select the number of website classes according to the dataset.
    if DataSet_name=='AWF100':
        flow_type=100
    elif DataSet_name=='DF':
        flow_type=94   #  AWA pairs classes, so use an even class count by dropping label 94.

    adv_data_x,adv_labels_y=Load_Data(DataSet_name,'adv')
    test_data_x,test_labels_y=Load_Data(DataSet_name,'test')
    exp_path = f'Defence_Method/AWA/File_Save/Gen_Save/{DataSet_name}/{ClassfyModelName}'
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    #  Logging.
    log_dir = exp_path
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

    #  Randomly generate class pairs.
    cls_list = np.arange(flow_type)     #  Even number of classes.
    keys = np.random.permutation(cls_list)  #  Randomly permute all selected classes.
    key1 = keys[:flow_type//2]
    key2 = keys[flow_type//2:flow_type]

    np.savez(exp_path+'/Keys'+str(datetime.datetime.now()),key1=key1,key2=key2)

    logger.info(f"Pairs: \n{[[i, j] for i, j in zip(key1, key2)]}")

    assert len(key1) == len(key2), "BAD_CONFIG_IN_CLS_LIST"     #  Ensure key1 and key2 have the same length.
    logger.info(f'<==Defence CF_Model:{ClassfyModelName}==>')
    for cls_index in range(len(key1)):
        logger.info(f'CF_Model:{ClassfyModelName},{cls_index}/{len(key1)}')
        
        g1_loss_plot = []; g2_loss_plot = []; best_result = 0
        d_loss_plot = []; acc_list = []; oh_src_test = []; oh_src_train = []; oh_trg_test = []; oh_trg_train = []
        exper_data = data_class()
        src_cls = key1[cls_index]
        trg_cls = key2[cls_index]
        logger.info(f"start {datetime.datetime.now()} {src_cls} <-> {trg_cls}")
        logger.info(f'==>strat {src_cls}<->{trg_cls}')
        directory_path = exp_path+'/Best_Gs_Output_'+str(src_cls)+'-->'+str(trg_cls)+'/'     #+'('+str(datetime.datetime.now())+')'+'/'
        plot_path = directory_path + 'plot/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            os.makedirs(plot_path)

        #  Load source/target class data; training data is later used to validate whether perturbations only insert packets.
        src_adv_data,src_adv_labels = get_tar_class_data(adv_data_x,adv_labels_y, target_c= src_cls)
        src_test_data,src_test_labels =  get_tar_class_data(test_data_x,test_labels_y, target_c= src_cls)

        trg_adv_data,trg_adv_labels =  get_tar_class_data(adv_data_x,adv_labels_y, target_c= trg_cls) 
        trg_test_data,trg_test_labels =  get_tar_class_data(test_data_x,test_labels_y, target_c= trg_cls)


        max_burst_trace_len = 2000#max([max(np.array([ np.count_nonzero(i) for i in src_adv_data])),max(np.array([ np.count_nonzero(i) for i in trg_adv_data]))]) 
        if max_burst_trace_len % 4 != 0:
            max_burst_trace_len += 4 - (max_burst_trace_len % 4)

        src_adv_data = src_adv_data[:,:max_burst_trace_len]
        src_test_data = src_test_data[:,:max_burst_trace_len]
        trg_adv_data = trg_adv_data[:,:max_burst_trace_len]
        trg_test_data = trg_test_data[:,:max_burst_trace_len]

        #max_burst_vector = np.max(np.append(np.abs(src_WF_train_data),np.abs(trg_WF_train_data),axis=0),axis=0)[:max_burst_trace_len]
        logger.info("data info:")
        logger.info(f"Source class ({src_cls}):")
        logger.info(f"  - adversarial-training data: {src_adv_data.shape}")
        logger.info(f"  - test data: {src_test_data.shape}")
        logger.info(f"Target class ({trg_cls}):")
        logger.info(f"  - adversarial-training data: {trg_adv_data.shape}")
        logger.info(f"  - test data: {trg_test_data.shape}")



            #  Clear the previous graph.
        tf.keras.backend.clear_session()

        #  Initialize the model.
        awa = AWA_Class(trace_len=max_burst_trace_len, logit_layer=logit_layer,awa_type=flow_type,trace_length=burst_len,CF_modelname=ClassfyModelName)
        trnsformers_selecting_flag = 0

        #  Training loop.
        for i in range(iterations):
            t_g1_lor = 0
            t_g2_lor = 0
            t_d_l = 0

            #  Train the discriminator.
            for d_i in range(d_iteration):
                #  Randomly select batch_size samples.
                sample_index = np.random.randint(len(src_adv_data), size=batch_size)
                batch_src_x = src_adv_data[sample_index]

                batch_src_noise = np.random.normal(size=batch_src_x.shape)
                sample_index = np.random.randint(len(trg_adv_data), size=batch_size)
                batch_trg_x = trg_adv_data[sample_index]
                batch_trg_noise = np.random.normal(size=batch_trg_x.shape)

                d_l = awa.train_discriminator(batch_src_x, batch_trg_x, batch_src_noise, batch_trg_noise)
                t_d_l += d_l
            d_loss_plot.append(t_d_l / d_iteration)

            #  Train generator 1.
            for g1_i in range(g_iteration):
                sample_index = np.random.randint(len(src_adv_data), size=batch_size)
                batch_src_x = src_adv_data[sample_index]
                batch_src_noise = np.random.normal(size=batch_src_x.shape)

                g1_l, g1_lor, g1_loh, l1_loss, d1_out, pert, adj_new_data = awa.train_generator1(
                    batch_src_x, src_cls, batch_src_noise
                )
                assert np.sum(pert < 0) == 0, "Health issue in perturbation"
                #  logger.info('After perturbation:', adj_new_data[0][:20])
                #  logger.info('Before perturbation:', batch_src_x[0][:20])

                assert np.sum((np.abs(adj_new_data) - np.abs(batch_src_x)) < 0) == 0, "Health issue in sign"
                assert np.array_equal(np.array([np.sum(np.abs(np.sign(i))) for i in adj_new_data]),
                                    np.array([np.sum(np.abs(np.sign(i))) for i in batch_src_x])), "Health issue in size of trace"
                t_g1_lor += g1_lor / g_iteration
            g1_loss_plot.append(t_g1_lor)

            #  Train the discriminator again.
            for d_i in range(d_iteration):
                sample_index = np.random.randint(len(src_adv_data), size=batch_size)
                batch_src_x = src_adv_data[sample_index]
                batch_src_noise = np.random.normal(size=batch_src_x.shape)
                sample_index = np.random.randint(len(trg_adv_data), size=batch_size)
                batch_trg_x = trg_adv_data[sample_index]
                batch_trg_noise = np.random.normal(size=batch_trg_x.shape)

                awa.train_discriminator(batch_src_x, batch_trg_x, batch_src_noise, batch_trg_noise)

            #  Train generator 2.
            for g2_i in range(g_iteration):
                sample_index = np.random.randint(len(trg_adv_data), size=batch_size)
                batch_trg_x = trg_adv_data[sample_index]
                batch_trg_noise = np.random.normal(size=batch_trg_x.shape)

                g2_l, g2_lor, g2_loh, l2_loss, d2_out, pert, adj_new_data = awa.train_generator2(
                    batch_trg_x, trg_cls, batch_trg_noise
                )
                assert np.sum(pert < 0) == 0, "Health issue in perturbation"
                assert np.sum((np.abs(adj_new_data) - np.abs(batch_trg_x)) < 0) == 0, "Health issue in sign"
                assert np.array_equal(np.array([np.sum(np.abs(np.sign(i))) for i in adj_new_data]),
                                    np.array([np.sum(np.abs(np.sign(i))) for i in batch_trg_x])), "Health issue in size of trace"
                t_g2_lor += g2_lor / g_iteration
            g2_loss_plot.append(t_g2_lor)


            #  Evaluate every 50 epochs.
            if (i+1) % 50 == 0 or (i+1) == iterations:
                logger.info(f"{'*'*50} {i+1}/{iterations}")

                noise_src = tf.random.normal(shape=src_adv_data.shape)
                noise_trg = tf.random.normal(shape=trg_adv_data.shape)

                generated_src_adv = awa.adjusted_generated_1(src_adv_data, noise_src, training=False)
                generated_trg_adv = awa.adjusted_generated_2(trg_adv_data, noise_trg, training=False)

                assert tf.reduce_sum(tf.cast(tf.abs(generated_src_adv) - tf.abs(src_adv_data) < 0, tf.int32)) == 0, "Health issue in sign1"
                assert tf.reduce_sum(tf.cast(tf.abs(generated_trg_adv) - tf.abs(trg_adv_data) < 0, tf.int32)) == 0, "Health issue in sign2"

                oh_src_train.append([print_overhead("g1_train", src_adv_data, generated_src_adv.numpy(), pr=1), i])
                oh_trg_train.append([print_overhead("g2_train", trg_adv_data, generated_trg_adv.numpy(), pr=1), i])

                logger.info(f"d_l: {d_l}")
                logger.info(f"g1_l: {g1_l}, g1_lor: {t_g1_lor}, g1_loh: {g1_loh}, l1_loss: {l1_loss}, d1_out: {np.mean(my_sigmoid(d1_out))}")
                logger.info(f"g2_l: {g2_l}, g2_lor: {t_g2_lor}, g2_loh: {g2_loh}, l2_loss: {l2_loss}, d2_out: {np.mean(my_sigmoid(d2_out))}")

                if (oh_src_train[-1][0] < OH * 100 and oh_trg_train[-1][0] < OH * 100) or ((i+1) == iterations and trnsformers_selecting_flag == 0):
                    logger.info("Saving data!!!")
                    trnsformers_selecting_flag = 1

                    generated_src_test = awa.adjusted_generated_1(src_test_data, tf.random.normal(src_test_data.shape), training=False)
                    generated_trg_test = awa.adjusted_generated_2(trg_test_data, tf.random.normal(trg_test_data.shape), training=False)

                    assert tf.reduce_sum(tf.cast(tf.abs(generated_src_test) - tf.abs(src_test_data) < 0, tf.int32)) == 0, "Health issue in sign5"
                    assert tf.reduce_sum(tf.cast(tf.abs(generated_trg_test) - tf.abs(trg_test_data) < 0, tf.int32)) == 0, "Health issue in sign6"

                    F1, TPR, FPR, src_overall_ACC,per_class_acc=evaluate_model(model_,generated_src_test,src_test_labels,batch_size)
                    F1, TPR, FPR, trg_overall_ACC,per_class_acc=evaluate_model(model_,generated_trg_test,trg_test_labels,batch_size)
                    logger.info(f'src_overall_ACC: {src_overall_ACC}')
                    logger.info(f'trg_overall_ACC: {trg_overall_ACC}')
                    logger.info(f'src_overall_ACC{src_overall_ACC}')
                    logger.info(f'trg_overall_ACC{trg_overall_ACC}')
                    
                    awa.generator1.save(directory_path + 'g1.h5')
                    awa.generator2.save(directory_path + 'g2.h5')

        logger.info(f"End:{str(datetime.datetime.now())}")
        end_time=datetime.datetime.now()
        time_list.append(end_time-begin_time)
logger.info(f"time_list: {time_list}")
