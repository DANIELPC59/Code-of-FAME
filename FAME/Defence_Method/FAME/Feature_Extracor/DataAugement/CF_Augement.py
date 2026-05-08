import numpy as np

# --- Data Augmentation ---

class CF_DataAugmentation:
    """Implements Injecting (background traffic injection) and Removing (packet drop simulation) augmentations."""
    def __init__(self, prob=0.1):
        self.prob = prob 

    def augment(self, x):
        # randomly choose augmentation strategy
        if np.random.rand() > 0.5:
            return self.injecting(x)
        else:
            return self.removing(x)

    def injecting(self, x):
        """Randomly inject +1 or -1 values into the sequence."""
        x_aug = x.copy()
        mask = np.random.rand(*x.shape) < self.prob  # boolean mask, True with probability self.prob
        
        # inject packets along the original direction
        direction = np.sign(x)
        x_aug = x_aug + (direction * mask)
        return x_aug

    def removing(self, x):
        """Randomly zero out positions to simulate packet loss."""
        x_aug = x.copy()
        mask = np.random.rand(*x.shape) < self.prob
        direction = np.sign(x)
        # decrement: subtract direction * mask from each value
        temp_aug = x_aug - (direction * mask)
        
        # keep modified value only if sign is preserved or result is 0
        is_same_direction = (np.sign(temp_aug) == direction) | (temp_aug == 0)
        x_aug = np.where(is_same_direction, temp_aug, x_aug) # np.where(condition, x, y) return x if condition is True, else y
        return x_aug