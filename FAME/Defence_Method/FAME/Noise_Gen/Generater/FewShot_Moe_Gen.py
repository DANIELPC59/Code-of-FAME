
from typing import Any
import tensorflow as tf
from tensorflow.keras import layers

class GatingNetwork(tf.keras.Model):
    """
    Strategy mapping network that routes traffic features 
    to a expert weight vector.
    """
    def __init__(self, num_experts=8, temperature=0.1, **kwargs):
        super(GatingNetwork, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.temperature = temperature
        
        # project CF encoder output (256-dim by default) to a compact latent space
        self.feature_projection = layers.Dense(128, activation='relu')
        self.gate_dense = layers.Dense(num_experts)

    def call(self, feature_input):
        """
        Args:
            feature_input: [B, feature_dim] - per-sample feature r or class prototype P.
        Returns:
            gate_weights: [B, num_experts] - mixture weights over experts.
        """
        x = self.feature_projection(feature_input)
        gate_logits = self.gate_dense(x)
        gate_weights = tf.nn.softmax(gate_logits / self.temperature, axis=-1)
        return gate_weights

class ExpertGenerator(tf.keras.Model):
    """
    Perturbation execution network that generates dummy-packet sequences
    from gate weights and random noise, decoupled from traffic features.
    """
    def __init__(self, noise_size, num_experts=8, dropout_rate=0.05, **kwargs):
        super(ExpertGenerator, self).__init__(**kwargs)
        self.noise_size = noise_size
        self.num_experts = num_experts
        
        # build expert generator pool
        self.experts = [self._build_expert(noise_size, dropout_rate) for _ in range(num_experts)]

    def _build_expert(self, noise_size, dropout_rate):
        return tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(2000, activation='relu'), 
            layers.Dense(2000, activation='relu'), 
        ])

    def call(self, gate_weights, training=False):
        """
        Args:
            gate_weights: Pre-computed gate weights [B, num_experts].
        Returns:
            Perturbation tensor of shape [B, output_size, 1].
        Note:
            Random noise z is sampled internally with shape [B, noise_size].
        """
        batch_size = gate_weights.shape[0]
        z = tf.random.normal(shape=(batch_size, self.noise_size))

        # experts generate candidate perturbations in parallel (noise-driven for diversity)
        expert_outputs = tf.stack([expert(z, training=training) for expert in self.experts], axis=1) 

        # aggregate expert outputs weighted by gate weights
        gate_weights_expanded = tf.expand_dims(gate_weights, -1) 
        x = tf.reduce_sum(gate_weights_expanded * expert_outputs, axis=1) 

        # apply uplink mask: perturb only even-indexed positions
        mask = tf.cast(tf.range(tf.shape(x)[1]) % 2 == 0, tf.float32)
        x = x * mask

        # pad or truncate to target output_size
        feature_dim_current = tf.shape(x)[1]
        if feature_dim_current < self.noise_size:
            pad_size = self.noise_size - feature_dim_current
            padding = tf.zeros([tf.shape(x)[0], pad_size], dtype=x.dtype)
            x = tf.concat([x, padding], axis=1)
        else:
            x = x[:, :self.noise_size]

        # ensure non-negative output (only dummy packets can be inserted)
        x = tf.nn.relu(x)
        x = x[:, :, tf.newaxis]
        
        return x

class FewShot_Moe_Gen(tf.keras.Model):
    """
    Joint training wrapper combining GatingNetwork and ExpertGenerator.
    Use during training; instantiate the sub-modules directly at inference.
    """
    def __init__(self, noise_size, num_experts=8, dropout_rate=0.05, temperature=0.1, **kwargs):
        super(FewShot_Moe_Gen, self).__init__(**kwargs)
        self.noise_size = noise_size
        
        # instantiate the two decoupled sub-modules
        self.gating_network = GatingNetwork(num_experts, temperature)
        self.expert_generator = ExpertGenerator(noise_size, num_experts, dropout_rate)

    def call(self, batch_feature, training=False):
        
        # Step 1 (Strategy): derive gate weights from input features
        gate_weights = self.gating_network(batch_feature)
        
        # Step 2 (Execution): generate perturbation from gate weights and random noise
        perturbation = self.expert_generator(gate_weights, training=training)
        
        if training:
            return perturbation, gate_weights
        else:
            return perturbation
