import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel
import numpy as np
import logging

logger = logging.getLogger("train_search")


class Particle:

    def __init__(self, num_nodes, args, genotype_seed=None, block_mask=None):
        self.args = args
        self.block_mask = block_mask
        self.genotype_seed = genotype_seed
        self.num_nodes = num_nodes
        self.concat = args.concat
        self.operations = PRIMITIVES
        self.w = args.w_inertia
        self.c1 = args.c_local
        self.c2 = args.c_global
        self._initialize_particle()

    def _initialize_particle(self):
        """
        initialization of the alpha parameters
        that describes the architecture representation
        """

        # max_num_edges is the number of edges for the longest path
        # possible in a DAG having n nodes. n(n-1)/2
        # that is the sum of the first n natural numbers
        self.max_num_edges = sum(num for num in range(1, self.num_nodes + 1))

        self.num_operations = len(self.operations)
        # self.num_operations = len(PRIMITIVES)

        # returns a torch tensor of dimension k x |operations| initialized randomly
        self.position = Variable(torch.randn(self.max_num_edges, self.num_operations).mul_(1e-1).cuda())
        # self.position = Variable(torch.randn(self.max_num_edges, self.num_operations)).cuda()

        if self.genotype_seed is not None:
            self.initialize_weights(self.position)

        self._arch_parameters = [self.position]

        # random init of the best position
        self.best_position = Variable(torch.randn(self.max_num_edges, self.num_operations).mul_(1e-1).cuda())
        # self.best_position = Variable(torch.randn(self.max_num_edges, self.num_operations)).cuda()
        # if self.genotype_seed is not None:
        #     self.initialize_weights(self.best_position)

        # random init of the current particle's velocity
        self.velocity = Variable(torch.randn(self.max_num_edges, self.num_operations).mul_(1e-1).cuda())
        # self.velocity = Variable(torch.randn(self.max_num_edges, self.num_operations)).cuda()
        # if self.genotype_seed is not None:
        #    self.initialize_weights(self.velocity)

        self.best_fit = np.Inf

        if self.genotype_seed is not None:
            self.check_genotype()

    def copy_particle(self):
        copied_particle = Particle(num_nodes=self.num_nodes,
                                   args=self.args,
                                   genotype_seed=self.genotype_seed,
                                   block_mask=self.block_mask)

        copied_particle.position = self.position.clone()
        copied_particle.velocity = self.velocity.clone()
        copied_particle.best_position = self.best_position.clone()
        copied_particle.best_fit = self.best_fit

        return copied_particle

    def update_best_position(self, new_fit):
        if new_fit < self.best_fit:

            self.best_fit = new_fit

            # TODO(PSO): make sure this is deep copy
            self.best_position = self.position.clone()

    def _update_velocity(self, global_best):

        '''
        w = 0.75  # constant inertia weight
        c1 = 1.5  # cognative constant
        c2 = 1.5  # social constant
        '''

        # r1 and r2 are random tensor uniformily distributed in [0, 1]
        r1 = Variable(torch.rand(self.max_num_edges, self.num_operations).cuda())
        r2 = Variable(torch.rand(self.max_num_edges, self.num_operations).cuda())

        # between r1 and the delta position there is an element-wise product
        vel_cognitive = self.c1 * r1 * (self.best_position - self.position)
        vel_social = self.c2 * r2 * (global_best.position - self.position)

        # updating the velocity of the particle
        if self.block_mask is not None:
            self.velocity = self.block_mask * (self.w * self.velocity + vel_cognitive + vel_social)
            print('velocity: {}'.format(self.velocity))
        else:
            self.velocity = self.w * self.velocity + vel_cognitive + vel_social

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
            for index_node in range(self.num_nodes):
                end = start + index_node + 1

                # selecting the alpha vectors dedicated to the incoming edges of intermediate node i
                # for i = 2, get the vectors regarding edges: e(0,2), e(1,2)
                # for i = 3, get the vectors regarding edges: e(0,3), e(1,3), e(2,3)
                W = probs[start:end].copy()

                # among the vectors of the valid edges, select the vector of
                # the edge with the highest probable operation, this is for
                # the constraint that each node has only 1 incoming edge (see paper)
                j = sorted(range(index_node + 1),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[0]  # if k != self.operations.index('none')))[0]

                # k_best = np.argmax(W[j])

                k_best = None
                for k in range(len(W[j])):
                    # if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k

                # appending tuple describing the edge towards the ith node,
                # describing the activation function and the previous node (j)
                gene.append((self.operations[k_best], j))
                # gene.append((PRIMITIVES[k_best], j))

                start = end

            return gene

        '''
        softmax = torch.nn.Softmax(dim=1)
        gene = _parse(softmax(self.position).data.cpu().numpy())
        genotype = Genotype(recurrent=gene, concat=range(self.num_nodes + 1)[-self.num_nodes:])
        '''
        gene = _parse(F.softmax(self.position, dim=-1).data.cpu().numpy())
        genotype = Genotype(recurrent=gene, concat=range(self.num_nodes + 1)[-self.concat:])
        return genotype

    def initialize_weights(self, weights_tensor=None, desired_genotype=None):

        if desired_genotype is None:
            desired_genotype = self.genotype_seed

        if weights_tensor is None:
            weights_tensor = self.position

        start = 0
        # tensor_np = weights_tensor.data.cpu().numpy().copy()
        softmax = torch.nn.Softmax(dim=1)
        soft_tensor = softmax(weights_tensor).data.cpu().numpy()

        # 'i' is the index regarding the edge to the ith intermediate node
        for step in range(self.num_nodes):

            # genotype retrieval of indices part
            gene_edge_descr = desired_genotype.recurrent[step]

            operation = gene_edge_descr[0]

            # op index to be used for the swap
            seed_operation_id = self.operations.index(operation)

            # previous node index to be used for the swap of the row
            seed_prev_node_id = gene_edge_descr[1]

            end = start + step + 1

            # selecting the alpha vectors dedicated to the incoming edges of intermediate node i
            # for i = 2, get the vectors regarding edges: e(0,2), e(1,2)
            # for i = 3, get the vectors regarding edges: e(0,3), e(1,3), e(2,3)
            possible_incoming_edges = soft_tensor[start:end].copy()

            # among the vectors of the valid edges, select the vector of
            # the edge with the highest probable operation, this is for
            # the constraint that each node has only 1 incoming edge (see paper)
            best_prec_node_id = sorted(range(step + 1),
                       key=lambda x: -max(possible_incoming_edges[x][k] for k in range(len(possible_incoming_edges[x]))))[0]

            best_op_id = np.argmax(possible_incoming_edges[best_prec_node_id])

            # if same ids of the desired gene we continue with the next node
            if best_op_id == seed_operation_id and best_prec_node_id == seed_prev_node_id:
                start = end
                continue

            else:
                '''
                logger.info('forcing the gene! desired node {}, best_node {}, desired op {}, best op {}'.format(seed_prev_node_id,
                                                                                                                best_prec_node_id,
                                                                                                                seed_operation_id,
                                                                                                                best_op_id))
                '''

                if best_prec_node_id != seed_prev_node_id:

                    seed_row = weights_tensor[start + seed_prev_node_id].clone()
                    best_row = weights_tensor[start + best_prec_node_id].clone()

                    # swapping the the best value in the place of seed operation
                    seed_op_value = best_row[seed_operation_id].data[0]
                    best_row[seed_operation_id] = best_row[best_op_id].data[0]
                    best_row[best_op_id] = seed_op_value

                    weights_tensor[start + seed_prev_node_id] = best_row
                    weights_tensor[start + best_prec_node_id] = seed_row

                else:
                    best_row = weights_tensor[start + best_prec_node_id].clone()

                    # swapping the the best value in the place of seed operation
                    seed_op_value = best_row[seed_operation_id].data[0]
                    best_row[seed_operation_id] = best_row[best_op_id].data[0]
                    best_row[best_op_id] = seed_op_value
                    weights_tensor[start + seed_prev_node_id] = best_row

            start = end

        self.check_genotype(compared_genotype=desired_genotype)

    def check_genotype(self, compared_genotype=None, compared_position=None, compare=False):

        softmax = torch.nn.Softmax(dim=1)

        if compared_genotype is None:
            compared_genotype = self.genotype_seed

        current_genotype = self.genotype()
        # seed_genotype = self.genotype_seed
        for index, elem in enumerate(current_genotype.recurrent):
            seed_elem = compared_genotype.recurrent[index]

            if elem[0] != seed_elem[0]:
                logger.info('operation not as the seed! op {}, seed op {}, node {}'.format(elem[0], seed_elem[0], index))
                logger.info('current genotype {}'.format(current_genotype))
                logger.info('genotype to compare {}'.format(compared_genotype))
                logger.info('position tensor: {}'.format(self.position))
                logger.info('softmax position: {}'.format(softmax(self.position)))
                if compare:
                    logger.info('compared position: {}'.format(compared_position))
                    logger.info('compared softmax: {}'.format(softmax(compared_position)))

                raise AssertionError
            if elem[1] != seed_elem[1]:
                logger.info('prec node not as the seed! prec node {}, seed node {}, node {}'.format(elem[1],seed_elem[1],index))
                logger.info('current genotype {}'.format(current_genotype))
                logger.info('genotype to compare {}'.format(compared_genotype))
                logger.info('position tensor: {}'.format(self.position))
                logger.info('softmax position: {}'.format(softmax(self.position)))
                if compare:
                    logger.info('compared position: {}'.format(compared_position))
                    logger.info('compared softmax: {}'.format(softmax(compared_position)))
                raise AssertionError

        logger.info('particle properly initialized!')

    def reset_new_gen_seed(self, new_genotype):

        self.genotype_seed = new_genotype

        # updating our data structure wrt the new genotype seed
        '''
        self.initialize_weights(self.position)
        self.initialize_weights(self.best_position)
        self.initialize_weights(self.velocity)
        self.check_genotype()
        '''
        self._initialize_particle()

    def reset_as_centroid(self, cluster):

        compared_genotype = self.genotype()
        compared_position = self.position.clone()

        centroid_position = Variable(torch.zeros_like(self.position.data)).cuda()
        centroid_best_position = Variable(torch.zeros_like(self.best_position.data)).cuda()
        centroid_velocity = Variable(torch.zeros_like(self.velocity.data)).cuda()

        for particle in cluster:

            centroid_position += particle.position.clone()
            centroid_best_position += particle.best_position.clone()
            centroid_velocity += particle.velocity.clone()

        cluster_size = len(cluster)
        centroid_position = centroid_position / cluster_size
        centroid_best_position = centroid_best_position / cluster_size
        centroid_velocity = centroid_velocity / cluster_size

        self.position = centroid_position
        self.best_position = centroid_best_position
        self.velocity = centroid_velocity
        self._arch_parameters[0] = self.position

        logger.info("CHECKING GENOTYPE OF THE CENTROID")
        self.initialize_weights(self.position, desired_genotype=compared_genotype)
        # self.initialize_weights(self.best_position, desired_genotype=compared_genotype)
        # self.initialize_weights(self.velocity, desired_genotype=compared_genotype)
        self.check_genotype(compared_genotype=compared_genotype,
                            compared_position=compared_position,
                            compare=True)
        logger.info("centroid GENOTYPE CHECK COMPLETED")

    def get_genotype_id(self):

        previous_slot = 0
        num_operations = len(self.operations)

        for index, gene in enumerate(self.genotype().recurrent):
            operation = gene[0]

            possible_relative_slots = (index + 1) * num_operations

            operation_id = self.operations.index(operation)

            prev_node_id = gene[1]

            relative_slot = num_operations * prev_node_id + operation_id

            previous_slot = relative_slot + (previous_slot * possible_relative_slots)

        return previous_slot

'''
geno = Genotype(
    recurrent=[
        ('relu', 0),
        ('relu', 1),
        ('tanh', 2),
        ('relu', 3),
        ('relu', 4),
        ('identity', 1),
        ('relu', 5),
        ('relu', 1)
    ],
    concat=range(1, 9))

particle = Particle(num_nodes=8, genotype_seed=geno)
print(particle.genotype())
print(particle.get_genotype_id())
'''