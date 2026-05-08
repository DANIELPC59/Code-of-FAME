import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tqdm.auto import tqdm  



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def label_accuracy(pred, target):
    """Computes the accuracy of model predictions matching the target labels"""
    batch_size = target.shape[0]
    correct = np.sum(pred == target)
    accuracy = correct / batch_size * 100.0
    return accuracy



class WalkieTalkie:

    def __init__(self):
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def generate_walkie_talkie_samples(self, test_bursts, test_y, train_bursts, train_y):
        """
        Generate perturbed samples following the Walkie-Talkie approach.

        Args:
            test_bursts : Tensor, shape [batch_size, seq_len] - burst sequences of the test set.
            test_y      : Tensor, shape [batch_size]          - test set labels.
            train_bursts: Tensor, shape [train_size, seq_len] - burst sequences of the adv training set.
            train_y     : Tensor, shape [train_size]          - adv training set labels.

        Returns:
            perturbed_bursts: Tensor, shape [batch_size, seq_len] - perturbed burst sequences.
        """
     


        # randomly select a training sample from a different class for each test sample
        batch_size = tf.shape(test_bursts)[0]
        train_size = tf.shape(train_bursts)[0]

      
        # select a decoy sample for each test sample
        perturbed_bursts = []
        virtual_packet_counts = []

        print("Begin generate perturbed x")
        for i in tqdm(range(batch_size), desc="Generating WT Samples", unit="flow"):
            # label of the current test sample
            # print(f'{i}/{batch_size}')
            current_label = test_y[i]

            # find training sample indices belonging to a different class
            different_class_indices = tf.where(tf.not_equal(train_y, current_label))
            different_class_indices = tf.reshape(different_class_indices, [-1])

            # randomly pick one index from the different-class candidates
            random_idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(different_class_indices)[0], dtype=tf.int32)
            decoy_idx = different_class_indices[random_idx]

            # retrieve burst sequences for the test sample and the chosen decoy
            test_burst = test_bursts[i]
            decoy_burst = train_bursts[decoy_idx]

            # build supersequence: at each position take the max absolute value
            # and preserve the sign of the test sample
            test_abs = tf.abs(test_burst)
            decoy_abs = tf.abs(decoy_burst)
            max_abs = tf.maximum(test_abs, decoy_abs)
            test_sign = tf.sign(test_burst)

            # supersequence = sign * max absolute value
            supersequence = test_sign * max_abs
            perturbed_bursts.append(supersequence)
        
        # stack all perturbed bursts into a batch tensor
        perturbed_bursts = tf.stack(perturbed_bursts)
        

        return perturbed_bursts


  
    def eval_performance(self, eval_x, eval_y, train_x, train_y):
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataset.setXY(eval_x, eval_y)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(ori_dataset.get_dataset())
        print('Performance before attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')

        x = eval_x
        assert len(x.shape) == 2, f'x.shape={x.shape}, not [batch, Feat].'
        print('x.shape:', x.shape)
        perturbed_x = self.generate_walkie_talkie_samples(
            x, eval_y, train_x, train_y, max_burst_len=4000, max_dir_len=5000)

        pert_dataset = DynamicDataset(perturbed_x, eval_y)
        pert_dataset.setX(perturbed_x)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(pert_dataset.get_dataset())
        print('Performance after attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')


    def eval_performance_for_batch_baseline(self, eval_x, eval_y, train_x, train_y):
        ori_dataset = DynamicDataset(eval_x, eval_y)
        ori_dataset.setXY(eval_x, eval_y)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(ori_dataset.get_dataset())
        print('Performance before attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')

        x = eval_x
        assert len(x.shape) == 2, f'x.shape={x.shape}, not [batch, Feat].'
        print('x.shape:', x.shape)
        if hasattr(self, 'perturbed_x') and self.perturbed_x.shape == eval_x.shape:
            print('Reuse previous perturbed x') # reuse perturbed_x
        else:
            self.perturbed_x = self.generate_walkie_talkie_samples(
                x, eval_y, train_x, train_y, max_burst_len=4000, max_dir_len=5000)

        pert_dataset = DynamicDataset(self.perturbed_x, eval_y)
        pert_dataset.setX(self.perturbed_x)
        loo, tpr, fpr, f1, acc, overall_acc = self.validation_novel(pert_dataset.get_dataset())
        print('Performance after attack:')
        print(f'Overall_acc: {overall_acc}, loss: {loo}, TPR: {tpr}, FPR: {fpr}, F1: {f1}, ACC: {acc}')



# Usage example
if __name__ == "__main__":
    # synthetic data for demonstration; replace with real data in practice
    test_x = tf.random.uniform([10, 100], minval=-1, maxval=2, dtype=tf.int32)  # 10 samples of length 100
    test_y = tf.random.uniform([10], minval=0, maxval=5, dtype=tf.int32)  # 5 classes

    train_x = tf.random.uniform([50, 100], minval=-1, maxval=2, dtype=tf.int32)  # 50 training samples
    train_y = tf.random.uniform([50], minval=0, maxval=5, dtype=tf.int32)  # 5 classes

    # generate perturbed samples
    wt = WalkieTalkie()
    perturbed_samples = wt.generate_walkie_talkie_samples(test_x, test_y, train_x, train_y)

    print("Original sample shape:", test_x.shape)
    print("Perturbed sample shape:", perturbed_samples.shape)
    print("First original sample (first 20 elements):", test_x[0][:20].numpy())
    print("First perturbed sample (first 20 elements):", perturbed_samples[0][:20].numpy())
