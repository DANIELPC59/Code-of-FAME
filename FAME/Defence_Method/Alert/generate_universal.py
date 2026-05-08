#  Generate universal perturbations.

import os
import sys

project_root =os.getcwd()
os.chdir(project_root)
print(project_root)
os.chdir(project_root)
sys.path.append(project_root)

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging


from Defence_Method.Alert import GAN_utility as ganModel
from WF_Model.CFModel_Loder import Load_Classfy_Model
from DataTool_Code.DataLoader import Load_Data
from Defence_Method.Alert.get_target import get_target_adv


#  1. Logging configuration.
log_dir = 'Defence_Method/Alert/File_Save'
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
logger.info(f"Start generate_universal.py, start time:{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")



VERBOSE = 1
model_list=['DF','AWF','VarCNN']

# model_name : DF,AWF,VarCNN
dataset='DF'
flow_size=2000
if dataset == 'AWF100':
    flowtype = 100
elif dataset == 'DF':
    flowtype = 95
time_record=[]
for model_name in model_list:
    logger.info(f'Defince {model_name}==>')
    model=Load_Classfy_Model(model_name,dataset,flow_size)
    train_mode="defence_train"
    start_time = time.time() 
    head_threshold=0.05  
    logger.info(f"in {dataset} for{model_name} do  { train_mode},bandwith threshold:{head_threshold}")
    #  Export the penultimate layer of the classifier.
    if model_name=='DF':
        FE=tf.keras.models.Model(model.input,model.get_layer('fc2').output)
    elif model_name=='AWF':
        FE=tf.keras.models.Model(model.input,model.get_layer('block3_pool').output)
    elif model_name=='VarCNN':
        FE=tf.keras.models.Model(model.input,model.get_layer('average_pool').output)

    def adjust_WF_data(x = None,perturbation = None):
        """
         Add generated perturbations to clean samples to create adversarial samples.
         Args:
            x: Clean samples.
            perturbation: Perturbation values.
        :return:
        """
        perturbation = tf.expand_dims(perturbation, 2)
        perturbation = perturbation * 1.0
        advData = x + perturbation * tf.sign(x)
        return advData

    def get_class_samples(X, Y, C):
        """
         Return samples from class C in dataset (X, Y).
         Args:
            X: Traffic traces as an np.ndarray.
            Y: Labels as an np.ndarray.
            C: Target class.
        :return:
        """
        #  If labels are one-hot, convert them to numeric label IDs with np.argmax(Y, axis=1).
        ind = np.where(Y == C)
        return X[ind], Y[ind]

    #  Loss function construction.
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    #  Optimizer construction.
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def loss_1(adjusted_generated_one, adjusted_generated_two):
        #  Cosine similarity.
        total = 0
        for i in range(0, adjusted_generated_one.shape[0]):
            cos_sim = tf.reduce_sum(adjusted_generated_one[i] * adjusted_generated_two[i]) / (tf.norm(adjusted_generated_one[i],ord=2) * tf.norm(adjusted_generated_two[i],ord=2))
            total = total + cos_sim
        loss = total / batch_size
        loss = 1 - loss
        return loss

    def overHead_loss(X_ori, X_adv, overHead_thresh=0.22):
        overHead = tf.reduce_sum(tf.abs(X_adv)-tf.abs(X_ori))/tf.reduce_sum(tf.abs(X_ori))
        return tf.maximum(0.0, overHead-overHead_thresh)

    def reset_generator_weights(model):
        """
         Reinitialize model parameters.
        """
        for layer in model.layers:
            #  Check whether the layer has kernel and bias tensors, such as Dense layers.
            if hasattr(layer, 'kernel_initializer'):
                #  Reinitialize weights with the layer's default initializer.
                new_kernel = layer.kernel_initializer(shape=layer.kernel.shape)
                layer.kernel.assign(new_kernel)
                
            if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                #  Reinitialize biases with the layer's default initializer.
                new_bias = layer.bias_initializer(shape=layer.bias.shape)
                layer.bias.assign(new_bias)
                
        #  Log the generator reset.
        logger.info("Generator weights have been reset; start training a new label.")

    """ Number of training iterations."""
    batch_size = 512
    # batch_size=64
    # g_iteration = 200
    g_iteration = 40
    
    data_length = 2000

    gen_loss = []
    total_loss = []
    logit_losse = []
    head_one = []

    #  Build the class index list.
    classNum = []
    for i in range(0, flowtype):
        classNum.append(i)


    # data, labels = load_data("./dataset/Burst_Closed World/burst_tor_200w_2500tr_test.npz")
    adv_data,adv_labels = Load_Data(dataset,'adv')
    test_data,test_labels = Load_Data(dataset,'test')

    
    generator = ganModel.generator_model_5()  #  Generator network.
    _=generator(tf.random.normal([1, 2000]))  #  Run one dummy input to initialize generator parameters.
    for label in range(0, flowtype):
        logger.info(f'label:{label}/{flowtype}')
        adv_data_X, adv_data_Y = get_class_samples(adv_data, adv_labels, label)
        test_data_X, test_data_Y = get_class_samples(test_data, test_labels, label)
        #  Split samples from adversarial-training data and test data.

        logger.info(f"{adv_data_X.shape}, {adv_data_Y.shape}")
        logger.info(f"##############  ori_label:{label}")
        reset_generator_weights(generator) #  Reset generator parameters so each label is trained independently.



        targetLabel=get_target_adv(dataset,model_name,train_mode,label)
        logger.info(f"slected target label:{targetLabel}")

        #  Samples from the target class.
        targe_data_X, targe_data_Y = get_class_samples(adv_data, adv_labels, targetLabel)
        #  Use source samples to train the universal perturbation and target samples for feature matching.

        indices = np.random.randint(targe_data_X.shape[0], size=batch_size)
        x_targe_batch = targe_data_X[indices]

        for iter in range(g_iteration):
            logger.info("label :%d, iter :%d" % (label, iter))

            with tf.GradientTape() as G1_tape:
                #  Current source-label samples.
                indices = np.random.randint(adv_data_X.shape[0], size=batch_size)
                x_train_batch = adv_data_X[indices]
                y_train_batch = adv_data_Y[indices]  #  Batch training labels.

                random_noise_one = np.random.normal(size=[batch_size, data_length])
                adv_distribution = generator(random_noise_one, training=True)   #  Raw generated perturbation.


                #  Generator loss.
                generated_one = adjust_WF_data(x_train_batch, adv_distribution)
                head_loss = overHead_loss(x_train_batch, generated_one,head_threshold)
                # adjusted_generated = tf.expand_dims(generated_one, 1)
                pre_one = model(generated_one)  #  Equivalent to model.predict().
                #  adv_pre_one = tf.argmax(pre_one, axis=1) selects the most likely label.
                gen_logit_loss= tf.reduce_mean(tf.maximum(pre_one[:, label], 0))

                origin_target_loss = loss_1(FE(generated_one), FE(x_targe_batch))

                if iter % 10 == 0:
                    logger.info(f"##############gen_logit_loss:{gen_logit_loss.numpy()}")
                    logger.info(f"##############origin_target_loss:{origin_target_loss.numpy()}")
                    logger.info(f"##############overhead_loss:{head_loss.numpy()}")



                #  Total loss.
                loss = gen_logit_loss + origin_target_loss + head_loss
                total_loss.append(loss.numpy())
                gen_loss.append(origin_target_loss.numpy())
                logit_losse.append(gen_logit_loss.numpy())
                head_one.append(head_loss.numpy())


            #  Optimize the generator.
            gradient_gen = G1_tape.gradient(loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))


        # generator.save("./generator/g1/" + str(label) + ".h5")
        # Zpc/Code/Moe_Gen/Defence_Method/Alert/File_Save
        base_save_path = f"Defence_Method/Alert/File_Save/Epoch_30/Gen_BatchSize{batch_size}/{dataset}/{model_name}/{train_mode}"
        if not os.path.exists(base_save_path):
            os.makedirs(base_save_path, exist_ok=True)  #  Use makedirs to create nested directories.
            logger.info(f"Created directory: {base_save_path}")

        gen_save_path = f'{base_save_path}/label_{label}'
        if not os.path.exists(gen_save_path):
            os.makedirs(gen_save_path)
        generator.save_weights(gen_save_path+ '/'+f'ori_{label}_to{targetLabel}' + ".h5")

        plt.figure()
        plt.plot(head_one, label="overhead")
        plt.plot(total_loss, label="total loss")
        plt.plot(logit_losse, label="logit_loss")
        plt.plot(gen_loss, label="origin_target_loss")
        # plt.plot(dis_loss, label="dis_loss")
        plt.title(" Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(gen_save_path+'/' + str(label) + "_loss.png")
        plt.close()
        head_one = []
        total_loss = []
        logit_losse = []
        gen_loss = []

    end_time = time.time()

    #  Compute elapsed time.
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
    time_record.append(elapsed_time)
logger.info(f"Total elapsed times: {time_record}")
