'''This script illustrates how to train an SBI model,
and generates a pickle file which is in the same format as the one used in tutorial.ipynb
'''
import os, sys
import numpy as np
import pickle
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi import utils as Ut
from sbi import inference as Inference

nhidden = 500 # architecture
nblocks = 15 # architecture

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

# load training data
# for fitting galaxy photometry: x = thetas; y = fluxes and uncertainties
x_train, y_train =

# train NPE
fanpe = # name for the .pt file where the trained model will be saved
fsumm = # name for the .p file where the training summary will be saved; useful if want to check the convergence, etc.

anpe = Inference.SNPE(
                      density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks),
                      device=device)
# because we append_simulations, training set == prior
anpe.append_simulations(
    torch.as_tensor(x_train.astype(np.float32), device='cpu'),
    torch.as_tensor(y_train.astype(np.float32), device='cpu'))
p_x_y_estimator = anpe.train()

# save trained ANPE
torch.save(p_x_y_estimator.state_dict(), fanpe)

# save training summary
pickle.dump(anpe._summary, open(fsumm, 'wb'))
print(anpe._summary)
