import torch
import os

model = torch.load(os.path.join("/Users/yassine/Desktop/christian/foo/darts_PSO/rnn_PSO/search-EXP_PSO_classic-20181009-143015", 'model.pt'))

genotype = model