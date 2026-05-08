AWA_type = 'UAWA'
# tau_high = 0.30
# tau_low = 0.05
#  OH = 0.50 sets the bandwidth-overhead upper bound.
tau_high = 0.04 #  Upper insertion threshold.
tau_low = 0.06  #  Lower insertion threshold.
OH = 0.05   #  Bandwidth-overhead upper bound.
exp_num = 1
burst_len=2000

batch_size=512  #  Match the MoE-Gen batch-size setting.

iterations=100  #  Total training epochs.
d_iteration=2 #  Discriminator epochs per outer iteration.
g_iteration=2 #  Generator epochs per outer iteration.

#  Loss weights.
disc_weight=1e2
oh_weight= 1e3
logit_weight = 1e3
