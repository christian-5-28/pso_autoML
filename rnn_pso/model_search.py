import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel
import numpy as np


class Particle:

    def __init__(self, args):
        self._initialize_particle()
        self.edge_select_mode = args.edge_select_mode
        print('USING EDGE MODE: {}'.format(self.edge_select_mode))

    def _initialize_particle(self):
        """
        initialization of the alpha parameters
        that describes the architecture representation
        """

        # k is the number of edges for the longest path
        # possible in a DAG having n nodes. n(n-1)/2
        # that is the sum of the first n natural numbers
        self.max_num_edges = sum(i for i in range(1, STEPS + 1))

        self.num_operations = len(PRIMITIVES)

        # returns a torch tensor of dimension k x |operations| initialized randomly
        self.position = Variable(torch.randn(self.max_num_edges, self.num_operations).mul_(1e-3).cuda())

        self._arch_parameters = [self.position]

        # random init of the best position
        self.best_position = Variable(torch.randn(self.max_num_edges, self.num_operations).mul_(1e-3).cuda())

        # random init of the current particle's velocity
        self.velocity = Variable(torch.randn(self.max_num_edges, self.num_operations).mul_(1e-3).cuda())

        self.best_fit = np.Inf

    def update_best_position(self, new_fit):
        if new_fit < self.best_fit:

            self.best_fit = new_fit

            # TODO(PSO): make sure this is deep copy
            self.best_position = self.position.clone()

    def _update_velocity(self, global_best):

        w = 0.5  # constant inertia weight
        c1 = 1  # cognative constant
        c2 = 2  # social constant

        # r1 and r2 are random tensor uniformily distributed in [0, 1]
        r1 = Variable(torch.rand(self.max_num_edges, self.num_operations).cuda())
        r2 = Variable(torch.rand(self.max_num_edges, self.num_operations).cuda())

        # between r1 and the delta position there is an element-wise product
        vel_cognitive = c1 * r1 * (self.best_position - self.position)
        vel_social = c2 * r2 * (global_best - self.position)

        # updating the velocity of the particle
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def update_position(self, global_best):

        # updating the velocity member
        self._update_velocity(global_best)

        # updating the particle's position
        self.position = self.position + self.velocity

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(probs):
            """
            build the discrete representation of the cell
            :param probs: tensor of dim (|max_edges| x |operations|
            representing the prob distribution of the ops
            """

            gene = []
            start = 0

            # 'i' is the index regarding the edge to the ith intermediate node
            for i in range(STEPS):
                end = start + i + 1

                # selecting the alpha vectors dedicated to the incoming edges of intermediate node i
                # for i = 2, get the vectors regarding edges: e(0,2), e(1,2)
                # for i = 3, get the vectors regarding edges: e(0,3), e(1,3), e(2,3)
                W = probs[start:end].copy()

                # among the vectors of the valid edges, select the vector of
                # the edge with the highest probable operation, this is for
                # the constraint that each node has only 1 incoming edge (see paper)
                j = sorted(range(i + 1),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]

                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k

                # appending tuple describing the edge towards the ith node,
                # describing the activation function and the previous node (j)
                gene.append((PRIMITIVES[k_best], j))

                start = end

            return gene

        # preparing the continuous representation based on the selected modality
        if self.edge_select_mode == 'softmax':
            continuous_repr = F.softmax(self.position, dim=-1).data.cpu().numpy()
        elif self.edge_select_mode == 'absolute_max':
            continuous_repr = self.position.data.cpu().numpy()
        else:
            raise NotImplementedError

        # obtaining the discrete representation
        gene = _parse(continuous_repr)
        genotype = Genotype(recurrent=gene, concat=range(STEPS + 1)[-CONCAT:])
        return genotype
