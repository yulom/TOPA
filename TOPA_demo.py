import torch
import torch.nn as nn
from TOPA import TOPA_Loss
# Example usage
cls_num = 4
features_dim = 1000
batch_size = 64

# Generate random target output probabilities and prototypes
random_matrix = torch.rand(batch_size, cls_num)
probability_matrix = random_matrix / random_matrix.sum(dim=1, keepdim=True)
prototypes = torch.rand(features_dim, cls_num)

# Compute TOPA loss
LOSS = TOPA_Loss(probability_matrix, prototypes)

print(LOSS)