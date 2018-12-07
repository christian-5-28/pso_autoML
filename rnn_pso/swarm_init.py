import time
from functools import reduce

from particle import Particle
# from model_search import Particle
import numpy as np
import math
from genotypes import PRIMITIVES, Genotype
import logging
import torch
from torch.autograd import Variable

logger = logging.getLogger("train_search")


class Swarm:

    def __init__(self,
                 population_size,
                 intermediate_nodes,
                 num_operations,
                 args,
                 genos_init=False):

        self.population_size = population_size

        self.intermediate_nodes = intermediate_nodes
        self.concat = args.concat

        self.num_operations = num_operations

        self.explored_geno_dict = {}
        self.genotype_survival_epochs = {}

        self.operations = [op for op in PRIMITIVES if op is not 'none']

        # blocks logic
        self.use_blocks = args.use_blocks
        self.max_num_edges = sum(num for num in range(1, self.intermediate_nodes + 1))

        self.block_mask = Variable(torch.ones(self.max_num_edges, self.num_operations).cuda(), requires_grad=False)
        self.args = args

        print('BUILDING SWARM')

        if self.use_blocks:
            self.set_gen_ids = set()
            self.genotypes_seeds, self.set_gen_ids = self.geno_seeds_after_block(genotype_id=self.args.prec_id,
                                                                           num_nodes=self.args.prec_nodes)

            block_edges = sum(num for num in range(1, self.args.prec_nodes + 1))
            self.block_mask[:block_edges] = 0

            self.population = [Particle(genotype_seed=genotype_seed,
                                        num_nodes=self.intermediate_nodes,
                                        args=args,
                                        block_mask=self.block_mask) for
                               genotype_seed in self.genotypes_seeds]
            self.population_size = len(self.population)

        elif genos_init:
            self.set_gen_ids = set()
            self.genotypes_seeds, self.set_gen_ids = self.get_init_genotypes()
            self.population = [Particle(genotype_seed=genotype_seed, num_nodes=self.intermediate_nodes, args=args) for genotype_seed in self.genotypes_seeds]

        elif self.args.evaluate:
            print("EVALUATE POP")

            self.set_gen_ids = set()
            geno_seed = self.genotype_from_id(self.args.genotype_id)
            self.population = [Particle(genotype_seed=geno_seed, num_nodes=self.intermediate_nodes, args=args)]
            print(self.population[0].genotype())
            time.sleep(10)
        else:
            self.population = [Particle(num_nodes=self.intermediate_nodes, args=args) for _ in range(self.population_size)]
            # self.population = [Particle(args) for _ in range(self.population_size)]
            self.set_gen_ids = set()

        # initialize global_best
        '''
        self.gbest_id = np.random.randint(0, self.population_size)
        self.global_best = self.population[self.gbest_id].position.clone()
        self.global_best_fit = np.Inf
        '''
        # initialize global_best
        '''
        self.global_best = self.population[0].position.clone()
        self.gbest_id = 0
        self.global_best_fit = np.Inf
        '''
        self.global_best = self.population[0].copy_particle()
        self.gbest_id = 0
        self.global_best_fit = self.global_best.best_fit
        self.same_best = 0

    def initialize_blocks_logic(self):
        prec_nodes = self.args.prec_nodes
        prec_id = self.args.prec_id
        gen_ids = self.geno_seeds_after_block(genotype_id=prec_id, num_nodes=prec_nodes)
        return gen_ids

    def get_genotype_id(self):
        range_ids = math.factorial(self.intermediate_nodes) * (self.num_operations ** self.intermediate_nodes)
        self.solutions_size = range_ids
        gen_id = np.random.randint(0, range_ids)
        return gen_id

    def get_genotype_ids(self, size):

        gen_ids = []
        new_id = self.get_genotype_id()
        gen_ids.append(new_id)
        self.set_gen_ids.add(new_id)
        len_ids = len(gen_ids)

        while len_ids < size:

            '''
            logger.info('LEN GEN IDS: {}'.format(len(gen_ids)))
            logger.info('SIZE: {}'.format(size))
            logger.info('set: {}'.format(self.set_gen_ids))
            logger.info('new_id: {}'.format(new_id))
            inside = new_id in self.set_gen_ids
            logger.info('inside: {}'.format(inside))
            '''

            while new_id in self.set_gen_ids:
                if len_ids == size:
                    break
                new_id = self.get_genotype_id()
                logger.info('IN ID WHILE, ID: {}'.format(new_id))
                logger.info('IN ID WHILE, SET: {}'.format(self.set_gen_ids))

            self.set_gen_ids.add(new_id)
            gen_ids.append(new_id)
            len_ids = len(gen_ids)

        # print('NUM SOLUTIONS: {}'.format(len(gen_ids)))
        logger.info('genotype seed ids {}'.format(gen_ids))
        return gen_ids

    def get_init_genotypes(self, size=None):

        if size is None:
            size = self.population_size

        '''
        range_ids = math.factorial(self.intermediate_nodes) * (self.num_operations ** self.intermediate_nodes)
        gen_ids = np.random.randint(0, range_ids, size=size)
        '''

        gen_ids = self.get_genotype_ids(size=size)
        genotype_seeds = [self.genotype_from_id(gen_id) for gen_id in gen_ids]
        logger.info('GENOTYPES SEEDS: {}'.format(genotype_seeds))
        return genotype_seeds, set(gen_ids)

    def genotype_from_id(self, genotype_id, num_nodes=None):

        if num_nodes is None:
            num_nodes = self.intermediate_nodes

        gene = [(self.operations[0], 0) for _ in range(num_nodes)]

        current_div_result = genotype_id
        current_node_id = num_nodes - 1

        while current_div_result > 0:
            # print(current_div_result, current_node_id, self.num_operations)
            current_div_result, prec_op_id = divmod(current_div_result, ((current_node_id + 1) * self.num_operations))

            prec_node_id, operation_id = divmod(prec_op_id, self.num_operations)

            # updating the edge for the current node slot with the new ids
            gene[current_node_id] = (self.operations[operation_id], prec_node_id)

            # updating to the next node id of the genotype slot, from bottom to top
            current_node_id -= 1

        return Genotype(recurrent=gene, concat=range(num_nodes + 1)[-num_nodes:])

    def get_genotype_id_from_geno(self, genotype):

        previous_slot = 0
        num_operations = len(self.operations)

        for index, gene in enumerate(genotype.recurrent):
            operation = gene[0]

            possible_relative_slots = (index + 1) * num_operations

            operation_id = self.operations.index(operation)

            prev_node_id = gene[1]

            relative_slot = num_operations * prev_node_id + operation_id

            previous_slot = relative_slot + (previous_slot * possible_relative_slots)

        return previous_slot

    def geno_seeds_after_block(self, genotype_id, num_nodes):

        # new_nodes = [node for node in range(num_nodes + 1, self.intermediate_nodes + 1)]
        new_nodes = [node for node in range(num_nodes, self.intermediate_nodes)]
        print('new nodes: ', new_nodes)
        # taking the old genotype of the block
        old_geno = self.genotype_from_id(genotype_id, num_nodes=num_nodes)

        # initialize the new starting block
        new_start_nodes = []
        for new_node in range(len(new_nodes)):
            new_start_nodes.append((self.operations[0], 1))

        gene = []
        gene.extend(old_geno.recurrent)
        gene.extend(new_start_nodes)
        new_geno = Genotype(recurrent=gene, concat=range(self.intermediate_nodes + 1)[-self.intermediate_nodes:])
        new_geno_id = self.get_genotype_id_from_geno(new_geno)

        end_range = reduce((lambda x, y: x * y), new_nodes) * (self.num_operations ** len(new_nodes))

        gen_ids = [gen_id for gen_id in range(new_geno_id, new_geno_id + end_range)]
        print('gen ids for blocks: ', gen_ids)
        genotype_seeds = [self.genotype_from_id(gen_id) for gen_id in gen_ids]
        logger.info('genotypes seed for blocks: {}'.format(genotype_seeds))
        return genotype_seeds, set(gen_ids)

    def get_new_genotype(self, size=1):
        new_id = self.get_genotype_ids(size=size)[0]

        while new_id in self.set_gen_ids:
            if len(self.set_gen_ids) == self.solutions_size:
                break
            new_id = self.get_genotype_ids(size=size)[0]

        self.set_gen_ids.add(new_id)

        genotype = self.genotype_from_id(genotype_id=new_id)
        return genotype, new_id

    def evaluate_population(self, new_fitnesses_dict, epoch):

        new_gbest_found = False

        # updating the local best for each of the particles
        for particle_id, new_fit in new_fitnesses_dict.items():

            if new_fit < self.population[particle_id].best_fit:
                self.population[particle_id].best_fit = new_fit
                self.population[particle_id].best_position = self.population[particle_id].position.clone()

            # checking for the new global best
            if new_fit < self.global_best_fit:
                logger.info('NEW GBEST FOUND, previous best gen: {}'.format(self.global_best.genotype()))
                logger.info('NEW best gen: {}'.format(self.population[particle_id].genotype()))
                logger.info('previous fit: {}'.format(self.global_best_fit))
                logger.info('new fit: {}'.format(new_fit))
                self.same_best = 0

                # updating the global best particle
                self.global_best = self.population[particle_id].copy_particle()
                self.global_best_fit = self.global_best.best_fit
                self.gbest_id = particle_id
                new_gbest_found = True

        if not new_gbest_found:
            logger.info('NO NEW GBEST FOUND...')
            logger.info('GBEST GEN: {}'.format(self.global_best.genotype()))
            logger.info('GBEST GEN FIT: {}'.format(self.global_best.best_fit))
            if epoch >= self.args.start_using_pso and epoch % self.args.pso_window == 0:
                logger.info('SAME BEST PARTICLE!')
                self.same_best += 1

    def update_particles_position(self):

        for particle_id, particle in enumerate(self.population):
            particle.update_position(self.global_best)

    def update_survival_dict(self, epoch):
        for particle in self.population:
            genotype_id = particle.get_genotype_id()

            if genotype_id not in self.genotype_survival_epochs.keys():
                self.genotype_survival_epochs[genotype_id] = []

            self.genotype_survival_epochs[genotype_id].append(epoch)

    def compute_clusters(self):

        clusters_dict = {}

        for particle in self.population:
            genotype_id = particle.get_genotype_id()

            if genotype_id not in clusters_dict.keys():
                clusters_dict[genotype_id] = 1
            else:
                clusters_dict[genotype_id] += 1

        return clusters_dict

    def compute_centroids(self):

        genotype_clusters_dict = {}

        for particle in self.population:
            genotype_id = particle.get_genotype_id()
            genotype_str = str(particle.genotype())

            if genotype_str not in self.explored_geno_dict.keys():
                self.explored_geno_dict[genotype_str] = genotype_id

            if genotype_id not in genotype_clusters_dict.keys():
                genotype_clusters_dict[genotype_id] = []

            if genotype_id == self.explored_geno_dict[genotype_str]:
                genotype_clusters_dict[genotype_id].append(particle)

            else:
                logger.info('ERROR WITH THE ID GENOTYPE:')
                logger.info('with genotype {}, supposed to have this id: {}'.format(genotype_str, self.explored_geno_dict[genotype_str]))
                logger.info('instead, we have this id: {}'.format(genotype_id))
                logger.info('postion of current particle: {}'.format(particle.position))
                logger.info('softmax pos curr particle: {}'.format(torch.nn.Softmax(dim=1)(particle.position)))
                logger.info('particle current geno: {}'.format(particle.genotype()))

        for gen_id, cluster in genotype_clusters_dict.items():

            # logging infos about each cluster
            logger.info('\n Cluster of the genotype with id {}'.format(gen_id))
            logger.info(cluster)
            logger.info('size of the cluster: {}'.format(len(cluster)))

            if len(cluster) == 1:
                continue

            for index, particle in enumerate(cluster):

                # needed for the check of the gbest id
                particle_index = self.population.index(particle)
                logger.info('genotype of particle {}: {}'.format(particle_index, particle.genotype()))

                # using the first particle object as container of the centroid
                if index == 0:

                    # taking the first particle as representative of the cluster
                    # (the particles of each cluster have the same validation results
                    # because in the discrete they are a unique network
                    # particle.reset_as_centroid(cluster)

                    # needed for the check of the gbest id
                    centroid_index = self.population.index(particle)

                else:
                    new_genotype, _ = self.get_new_genotype()
                    particle.reset_new_gen_seed(new_genotype)

                # taking care of the cluster containing the global best
                if self.gbest_id == particle_index:
                    logger.info('updated particle gbest centroid ! centroid_id {}, gbest id before {}'.format(centroid_index, self.gbest_id))
                    self.global_best = self.population[centroid_index].position.clone()
                    self.gbest_id = centroid_index
                    logger.info('gbest id after {}'.format(self.gbest_id))




'''
swarm = Swarm(population_size=5,
              num_operations=4,
              intermediate_nodes=8)

start = time.time()
for particle in swarm.population:
    print('particle genotype id', particle.get_genotype_id())

elapsed = (time.time() - start)

print('time per particle in seconds', elapsed / 5.)
'''