import time
import os, sys, glob
import math
from collections import namedtuple
from functools import reduce

import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from swarm_init import Swarm
from tensorboardX import SummaryWriter
import operator

import gc

import data
import model as model_module
import model_wpl

from utils import batchify, get_batch, repackage_hidden, create_dir, save_checkpoint, create_viz_dir

# from visualize import plot

Rank = namedtuple('Rank', 'valid_ppl geno_id')


class TrainerSearch:

    def __init__(self, args):
        self.args = args
        self.initialize_run()

    def initialize_run(self):
        """
        utility method for directories and plots
        :return:
        """

        if not self.args.continue_train:

            self.sub_directory_path = '{}_SEED_{}_geno_id_{}_{}'.format(self.args.save,
                                                                        self.args.seed,
                                                                        self.args.genotype_id,
                                                                        time.strftime("%Y%m%d-%H%M%S")
                                                                        )
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            create_dir(self.exp_dir)

        if self.args.visualize:
            self.viz_dir_path = create_viz_dir(self.exp_dir)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger("train_search")
        self.logger.addHandler(fh)

        # Set the random seed manually for reproducibility.
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                cudnn.benchmark = True
                cudnn.enabled = True
                torch.cuda.manual_seed_all(self.args.seed)

    def validate_model(self, current_model, data_source, batch_size=10):

        current_model.eval()
        total_loss = 0
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, self.args.bptt):
            data, targets = get_batch(data_source, i, self.args, evaluation=True)
            targets = targets.view(-1)

            log_prob, hidden = self.model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)

            hidden = repackage_hidden(hidden)

        avg_valid_loss = total_loss[0] / len(data_source)
        avg_valid_ppl = math.exp(avg_valid_loss)

        return avg_valid_loss, avg_valid_ppl

    def evaluate(self, data_source, fitnesses_dict, batch_size=10):
        """
        Evaluates all the current architectures in the population
        """

        particles_indices = [i for i in range(self.swarm.population_size)]
        np.random.shuffle(particles_indices)

        avg_particle_loss = 0
        avg_particle_ppl = 0
        cluster_validated = set()

        genotypes_fit_dict = {}

        # rank dict for the possible solutions
        genotypes_rank = {}

        for particle_id in particles_indices:

            geno_id = self.swarm.population[particle_id].get_genotype_id()

            # computing the genotype of the next particle
            new_genotype = self.swarm.population[particle_id].genotype()

            if geno_id in cluster_validated:
                self.logger.info('particle already validated, genotype: {} {}'.format(geno_id, new_genotype))
                fitnesses_dict[particle_id] = genotypes_fit_dict[geno_id]
                continue

            # add the geno id to the seen clusters
            cluster_validated.add(geno_id)

            # selecting the current subDAG in our DAG to train
            self.model.change_genotype(genotype=new_genotype)

            avg_valid_loss, avg_valid_ppl = self.validate_model(current_model=self.model,
                                                                data_source=data_source,
                                                                batch_size=batch_size)

            self.logger.info("VALIDATE PARTICLE with genotype id {}, valid_ppl: {}, {}".format(geno_id,
                                                                                               avg_valid_ppl,
                                                                                               self.model.genotype())
                             )

            avg_particle_loss += avg_valid_loss
            avg_particle_ppl += avg_valid_ppl

            # saving the particle fit in our dictionaries
            fitnesses_dict[particle_id] = avg_valid_ppl
            genotypes_fit_dict[geno_id] = avg_valid_ppl
            gen_key = str(self.model.genotype().recurrent)
            genotypes_rank[gen_key] = Rank(avg_valid_ppl, geno_id)

        rank_gens = sorted(genotypes_rank.items(), key=operator.itemgetter(1))

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            temp_gen = elem[0]
            temp_fit = elem[1].valid_ppl
            temp_id = elem[1].geno_id

            self.logger.info('particle gen id: {}, ppl: {}, gen: {}'.format(temp_fit,
                                                                            temp_id,
                                                                            temp_gen)
                             )

        # validate the current config of the global best particle
        glob_best_gen = self.swarm.global_best.genotype() if not self.args.use_random else self.best_so_far[0]
        self.model.change_genotype(genotype=glob_best_gen)
        self.logger.info("VALIDATE BEST GEN: {}".format(glob_best_gen))
        _, best_fit = self.validate_model(current_model=self.model,
                                          data_source=data_source,
                                          batch_size=batch_size)

        self.logger.info('GBEST ID: {}, {}'.format(self.swarm.global_best.get_genotype_id(),
                                                   self.swarm.global_best.genotype()
                                                   )
                         )

        # updating the best fitness of the gbest with respect to the current epoch
        self.logger.info('PREVIOUS BEST FIT: {}'.format(self.swarm.global_best_fit))
        self.swarm.global_best.best_fit = best_fit
        self.swarm.global_best_fit = best_fit
        self.logger.info('NEW BEST FIT: {}'.format(self.swarm.global_best_fit))

        return fitnesses_dict, avg_particle_loss / len(self.cluster_dict.keys()), avg_particle_ppl / len(self.cluster_dict.keys())

    def validate_arch_for_fisher(self, data_source, model, parallel_model, sample_arch,
                                 optimizer, params, epoch):

        """
        validation of different models to update the fisher information
        Code from WPL applied on NAO
        """

        batch_size = 10
        small_batch_size = 10
        hidden = [model.init_hidden(small_batch_size) for _ in
                  range(batch_size // small_batch_size)]
        batch = np.random.randint(0, data_source.size(0) // 35)
        # data, targets = get_batch(data_source, batch, 35, evaluation=False)
        bptt = params.bptt
        seq_len = int(bptt)
        data, targets = get_batch(data_source, batch, params.bptt, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, small_batch_size, 0
        while start < batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])
            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data,
                                                                            hidden[s_id],
                                                                            return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss

            loss *= small_batch_size / batch_size
            loss.backward()
            s_id += 1
            start = end
            end = start + small_batch_size
            gc.collect()
            # DO NOT Clip gradient, only clip fisher if necessary.
            torch.nn.utils.clip_grad_norm(model.parameters(), params.clip)
            model.rnns[0].update_fisher(sample_arch, epoch, params)
            model.rnns[0].update_optimal_weights()

    def train(self, epoch):
        assert self.args.batch_size % self.args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

        # Turn on training mode which enables dropout.
        total_loss = 0
        epoch_loss = 0
        tb_wpl_loss = 0
        start_time = time.time()
        ntokens = len(self.corpus.dictionary)

        # initializing first hidden state
        hidden = [self.model.init_hidden(self.args.small_batch_size) for _ in range(self.args.batch_size // self.args.small_batch_size)]

        batch, i, pop_index = 0, 0, 0

        # shuffle order to train particles
        particles_indices = [i for i in range(self.swarm.population_size)]
        np.random.shuffle(particles_indices)

        # dynamic slot of batches computation
        self.logger.info('clusters size for the current epoch {}: {}'.format(epoch, self.cluster_dict))

        num_clusters = len(self.cluster_dict.keys())
        total_batches = len(self.train_data) // self.args.bptt

        if num_clusters > total_batches:
            total_batches = self.args.train_num_batches * num_clusters
        slot_batches = int(math.floor(total_batches / num_clusters))
        remaining_batches = total_batches % num_clusters

        self.logger.info('num custers: {}, total batches: {}, slot batches: {}, module: {}'.format(num_clusters,
                                                                                                   total_batches,
                                                                                                   slot_batches,
                                                                                                   remaining_batches
                                                                                                   )
                         )

        clusters_seen = set()
        gbest_gen_id = self.swarm.global_best.get_genotype_id() if not self.args.use_random else self.best_so_far[1]
        gbest_gen_trained = False

        self.compute_fisher = epoch > self.args.fisher_after and self.args.use_wpl
        while i < self.train_data.size(0) - 1 - 1:

            for pos, particle_id in enumerate(particles_indices):

                if i >= self.train_data.size(0) - 1 - 1:
                    # break
                    i = 0
                    self.logger.info('REINIT THE INDEX I IN PARTICLE LOOP')

                if len(clusters_seen) == num_clusters:
                    self.logger.info('EXITING FROM TRAINING PHASE, ALL CLUSTERS TRAINED')
                    break

                genotype_id = self.swarm.population[particle_id].get_genotype_id()

                # info regarding the best config training
                if not gbest_gen_trained:
                    gbest_gen_trained = genotype_id == gbest_gen_id

                # if the particle has same discrete configuration
                # already trained during this epoch, we skip it
                if genotype_id in clusters_seen and (
                    len(clusters_seen) < len(self.cluster_dict)):  # and not args.use_fixed_slot:
                    self.logger.info('particle already trained, genotype: {}, data len {}, i {}'.format(genotype_id,
                                                                                                        self.train_data.size(0),
                                                                                                        i))
                    continue

                # logic to distribute the module of the division total_batches / num_clusters
                add_sample = len(clusters_seen) < remaining_batches
                train_slot_batch = slot_batches
                if add_sample:
                    train_slot_batch += 1

                # adding the gen id in the set of the clusters already seen
                clusters_seen.add(genotype_id)

                previous_genotype = self.model.genotype()

                # computing the genotype of the next particle
                new_genotype = self.swarm.population[particle_id].genotype()

                # selecting the current subDAG in our DAG to train
                self.model.change_genotype(genotype=new_genotype)

                # train this subDAG for slot of batches
                self.logger.info('train_slot_batch: {}, genotype: {}, steps seen: {} '.format(train_slot_batch,
                                                                                              genotype_id,
                                                                                              i))
                for b in range(train_slot_batch):

                    if i >= self.train_data.size(0) - 1 - 1:
                        # break
                        i = 0
                        self.logger.info('REINIT THE INDEX I IN BATCH SLOT')

                    bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
                    # Prevent excessively small or negative sequence lengths
                    # seq_len = max(5, int(np.random.normal(bptt, 5)))
                    # # There's a very small chance that it could select a very long sequence length resulting in OOM
                    # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
                    seq_len = int(bptt)

                    lr2 = self.optimizer.param_groups[0]['lr']
                    self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt

                    # training mode activated
                    self.model.train()

                    # preparing batch of data for training
                    data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)

                    self.optimizer.zero_grad()

                    start, end, s_id = 0, self.args.small_batch_size, 0
                    while start < self.args.batch_size:
                        cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

                        # Starting each batch, we detach the hidden state from how it was previously produced.
                        # If we didn't, the model would try backpropagating all the way to start of the dataset.
                        hidden[s_id] = repackage_hidden(hidden[s_id])
                        # hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

                        # assuming small_batch_size = batch_size so we don't accumulate gradients
                        self.optimizer.zero_grad()
                        # hidden[s_id] = repackage_hidden(hidden[s_id])

                        # forward pass
                        log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = self.model(cur_data,
                                                                                    hidden[s_id],
                                                                                    return_h=True)

                        # loss using negative-log-likelihood
                        raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

                        loss = raw_loss

                        # applying the WPL
                        if self.compute_fisher:
                            wpl = self.model.rnns[0].compute_weight_plastic_loss_with_update_fisher(genotype=self.model.genotype(),
                                                                                                    params=self.args)
                            wpl = 0.5 * wpl
                            tb_wpl_loss += wpl.data[0] * self.args.small_batch_size / self.args.batch_size
                            loss = loss + wpl

                        try:
                            # Activation Regularization
                            if self.args.alpha > 0:
                                loss = loss + sum(
                                    self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

                            if batch % self.args.log_interval == 0 and batch > 0:
                                # for step in range(len(rnn_hs[0])):
                                #    print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                                self.logger.info("max hidden value of all steps: " + str(rnn_hs[0].max()))

                            # Temporal Activation Regularization (slowness)
                            loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                        except:

                            self.logger.info("max hidden value of all steps: " + str(rnn_hs[0].max()))
                            print('RNN_HS: {}'.format(rnn_hs))
                            self.logger.info("genotype who caused the error:  ")
                            self.logger.info(self.model.genotype())
                            # print(model.genotype())
                            for name_param, param in self.model.rnns[0].named_parameters():
                                self.logger.info("param name: " + str(name_param))
                                self.logger.info("max value in the param matrix: " + str(param.max()))
                            raise

                        loss *= self.args.small_batch_size / self.args.batch_size
                        total_loss += raw_loss.data * self.args.small_batch_size / self.args.batch_size

                        epoch_loss += raw_loss.data * self.args.small_batch_size / self.args.batch_size

                        loss.backward()

                        s_id += 1
                        start = end
                        end = start + self.args.small_batch_size

                        gc.collect()

                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)

                    # weight balance logic for blocks search
                    if self.args.use_blocks and self.args.use_balance_grad:
                        if batch % self.args.log_interval == 0 and batch > 0:
                            self.logger.info('APPLYING BALANCED WEIGHTS TO THE GRADIENTS!')
                            for name_param, param in self.model.rnns[0]._Ws.named_parameters():
                                self.logger.info('grad norm before BALANCE, matrix node {}: {}'.format(name_param, param.grad.norm()))

                        self.apply_balance_weight_grad()

                        if batch % self.args.log_interval == 0 and batch > 0:
                            for name_param, param in self.model.rnns[0]._Ws.named_parameters():
                                self.logger.info('grad norm AFTER BALANCE, matrix node {}: {}'.format(name_param,
                                                                                                       param.grad.norm()))
                    # applying the gradient updates
                    self.optimizer.step()

                    # total_loss += raw_loss.data
                    self.optimizer.param_groups[0]['lr'] = lr2

                    if batch % self.args.log_interval == 0 and batch > 0:
                        self.logger.info(self.model.genotype())
                        # print(F.softmax(parallel_model.weights, dim=-1))
                        cur_loss = total_loss[0] / self.args.log_interval
                        cur_wpl_loss = tb_wpl_loss / self.args.log_interval

                        elapsed = time.time() - start_time

                        self.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, len(self.train_data) // self.args.bptt, self.optimizer.param_groups[0]['lr'],
                                          elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss)))

                        if self.compute_fisher:
                            self.logger.info('| fisher norm {:5.4f} | alpha {:5.4f} | wpl loss {:5.8f} '.format(
                                self.model.rnns[0].fisher_norm(), self.model.rnns[0].fisher_alpha(epoch, self.args), cur_wpl_loss
                            ))

                        total_loss = 0
                        tb_wpl_loss = 0
                        start_time = time.time()

                    batch += 1
                    i += seq_len

                # updating the fisher info and the optimal weights
                if self.compute_fisher:
                    self.validate_arch_for_fisher(self.val_data, self.model, self.parallel_model, self.model.genotype(),
                                                  self.optimizer, self.args, epoch)

            self.logger.info('ALL CLUSTERS TRAINED. EPOCH CONCLUDED')
            break

        self.logger.info('GENOTYPE GBEST TRAINED: {}'.format(gbest_gen_trained))
        self.logger.info('GENOTYPE GBEST ID: {}'.format(gbest_gen_id))

        self.writer.add_scalar('train_loss', epoch_loss[0] / batch, epoch)
        self.writer.add_scalar('train_ppl', math.exp(epoch_loss[0] / batch), epoch)

    def compute_balanced_weights(self):
        """
        computes the balance for the weight matrices for the fixed nodes
        """
        genotype = self.model.genotype()
        counter_dict = {}
        previous_dict = {}

        # input node count
        counter_dict[1] = 1

        for node_id in range(1, self.args.prec_nodes):

            counter_dict[node_id + 1] = 1
            previous_node = genotype.recurrent[node_id][1]
            previous_dict[node_id + 1] = previous_node
            if previous_node == 0:
                continue
            if previous_node not in counter_dict.keys():
                counter_dict[previous_node] = 1
            else:
                counter_dict[previous_node] += 1

            while previous_node > 1:
                print('node id:', node_id, 'previous:', previous_node, 'dict', previous_dict)
                previous_node = previous_dict[previous_node]
                counter_dict[previous_node] += 1

            self.logger.info('count dict: {}'.format(counter_dict))

        sorted_count = sorted(counter_dict.items(), key=operator.itemgetter(1))

        min_id, min_count = sorted_count[0]
        self.balancing_weights = {}

        self.balancing_weights[min_id] = 1
        for prec_id, count in sorted_count[1:]:
            self.balancing_weights[prec_id] = min_count / count

        self.logger.info('balancing weights computed! {}'.format(self.balancing_weights))

    def apply_balance_weight_grad(self):
        """
        applying the balance on the gradients due to the fixed block logic
        """

        for prec_id, balance in self.balancing_weights.items():

            # applying balance on the gradients of the cell
            self.model.rnns[0]._Ws[prec_id - 1].grad = self.model.rnns[0]._Ws[prec_id - 1].grad * balance

    def run_search(self):
        """
        main method for the train search handling all the epochs
        """

        if self.args.nhidlast < 0:
            self.args.nhidlast = self.args.emsize
        if self.args.small_batch_size < 0:
            self.args.small_batch_size = self.args.batch_size

        self.eval_batch_size = 10
        self.test_batch_size = 1
        self.corpus = data.Corpus(self.args.data)

        self.train_data = batchify(self.corpus.train, self.args.batch_size, self.args)
        self.search_data = batchify(self.corpus.valid, self.args.batch_size, self.args)
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, self.args)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, self.args)

        # Tensorboard writer
        tboard_dir = os.path.join(self.args.tboard_dir,self.sub_directory_path)
        self.writer = SummaryWriter(tboard_dir)

        ntokens = len(self.corpus.dictionary)

        # initialize the swarm
        if self.args.evaluate:

            self.args.population_size = 1
            self.swarm = Swarm(population_size=self.args.population_size,
                               num_operations=self.args.num_operations,
                               intermediate_nodes=self.args.num_intermediate_nodes,
                               args=self.args,
                               genos_init=self.args.uniform_genos_init)

            genotype = self.swarm.genotype_from_id(self.args.genotype_id)
            logging.info('CELL gen id USED, GENO: {}'.format(genotype))

        else:
            self.swarm = Swarm(population_size=self.args.population_size,
                               num_operations=self.args.num_operations,
                               intermediate_nodes=self.args.num_intermediate_nodes,
                               args=self.args,
                               genos_init=self.args.uniform_genos_init)
            # initial genotype
            genotype = self.swarm.global_best.genotype()

        self.particles_fitnesses = {}

        # initializing the model
        if self.args.use_pretrained:
            self.logger.info('PRETRAINED MODEL LOADED!')
            self.model = torch.load(os.path.join(self.args.pretrained_dir, 'model.pt'))

        elif self.args.use_wpl:
            self.model = model_wpl.RNNModel(ntokens,
                                            self.args,
                                            genotype=genotype)

        else:
            self.model = model_module.RNNModel(ntokens,
                                               self.args,
                                               genotype=genotype)

        size = 0
        for p in self.model.parameters():
            size += p.nelement()
        self.logger.info('param size: {}'.format(size))
        self.logger.info('initial genotype:')
        self.logger.info(self.model.genotype())

        if self.args.cuda:
            if self.args.single_gpu:
                self.parallel_model = self.model.cuda()
            else:
                self.parallel_model = nn.DataParallel(self.model, dim=1).cuda()
        else:
            self.parallel_model = self.model

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Model total parameters: {}'.format(total_params))

        # compute the balance weights for the gradients of shared weights
        if self.args.use_blocks and self.args.use_balance_grad:
            self.compute_balanced_weights()

        # Loop over epochs.
        self.lr = self.args.lr
        best_val_loss = []
        stored_loss = 100000000

        if self.args.continue_train:
            optimizer_state = torch.load(os.path.join(self.exp_dir, 'optimizer.pt'))
            if 't0' in optimizer_state['param_groups'][0]:
                self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.args.wdecay)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.args.wdecay)

        # epochs logic starts here
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            self.logger.info('\n EPOCH {} STARTED.'.format(epoch))

            # update the clusters dictionary of unique solutions
            self.cluster_dict = self.swarm.compute_clusters()
            number_clusters = len(self.cluster_dict.keys())
            self.writer.add_scalar('number_of_clusters', number_clusters, epoch)
            max_size = 0
            gen_id = 0
            for item in self.cluster_dict.items():
                key = item[0]
                value = item[1]
                if value > max_size:
                    max_size = value
                    gen_id = key

            self.logger.info('cluster with max size has id {} and size {}'.format(gen_id, max_size))
            self.writer.add_scalar('max_size_cluster', max_size, epoch)

            geno_best_id = self.swarm.global_best.get_genotype_id() if not self.args.use_random else self.best_so_far[1]
            size_cluster_gbest = self.cluster_dict.get(geno_best_id)
            if size_cluster_gbest is None:
                size_cluster_gbest = 1
                self.logger.info('THE GBEST GENOTYPE IS UNIQUE, NOT IN THE SWARM!')

            self.logger.info('cluster of gbest has id {} and size {}'.format(geno_best_id, size_cluster_gbest))
            self.writer.add_scalar('cluster_size_gbest', size_cluster_gbest, epoch)

            # TRAINING LOGIC FOR ONE EPOCH
            self.train(epoch=epoch)

            # FISHER LOGIC TO ZEROING THE FISHER INFO
            if self.compute_fisher and epoch % self.args.zero_fisher == 0:
                self.model.rnns[0].set_fisher_zero()

            # VALIDATION PART STARTS HERE

            if 't0' in self.optimizer.param_groups[0]:
                self.tmp = {}
                for prm in self.model.parameters():
                    if prm.grad is not None:
                        self.tmp[prm] = prm.data.clone()
                        prm.data = self.optimizer.state[prm]['ax'].clone()
                self.logger.info('CLONING SUCCESSFULL')

            # validation at the end of the epoch
            self.particles_fitnesses, val_loss, val_ppl = self.evaluate(self.val_data,
                                                              fitnesses_dict=self.particles_fitnesses,
                                                              batch_size=self.eval_batch_size)

            # updating the particles fitnesses and their best position
            self.swarm.evaluate_population(self.particles_fitnesses, epoch)

            if self.args.early_stopping_search is not None and self.swarm.same_best > self.args.early_stopping_search and \
                    (number_clusters == 1 and not self.args.evaluate):

                self.logger.info('CONVERGENCE REACHED, STOP SEARCH EARLY!')
                self.logger.info("global best genotype: ")
                self.logger.info(self.swarm.global_best.genotype())
                return self.swarm.global_best.genotype(), self.swarm.global_best.get_genotype_id(), self.swarm.global_best.best_fit

            self.logger.info("global best genotype after evaluation:")
            self.logger.info(self.swarm.global_best.genotype())

            if self.args.visualize:
                # saving the viz of the current gbest for this epoch
                file_name = os.path.join(self.viz_dir_path, 'epoch ' + str(epoch))
                graph_name = 'epoch_{}'.format(epoch)
                plot(self.swarm.global_best.genotype().recurrent, file_name, graph_name)

            # updating the global fit
            self.best_global_fit = self.swarm.global_best.best_fit

            # updating the particle current position
            if not self.args.evaluate and epoch >= self.args.start_using_pso and epoch % self.args.pso_window == 0:
                self.logger.info('UPDATING THE PARTICLE REPRESENTATIONS WITH PSO!')
                self.swarm.update_particles_position()

            # tensorboard logs
            self.writer.add_scalar('avg_swarm_validation_loss', val_loss, epoch)
            self.writer.add_scalar('avg_swarm_validation_ppl', val_ppl, epoch)

            self.writer.add_scalar('global_best_validation_loss', math.log(self.best_global_fit), epoch)
            self.writer.add_scalar('global_best_validation_ppl', self.best_global_fit, epoch)

            self.writer.add_scalar('validation_loss', math.log(self.best_global_fit), epoch)
            self.writer.add_scalar('validation_ppl', self.best_global_fit, epoch)

            # logs info
            self.logger.info('-' * 89)
            self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   math.log(self.best_global_fit), self.best_global_fit))
            self.logger.info('-' * 89)

            # call checkpoint if better model found
            if val_loss < stored_loss:
                save_checkpoint(self.model, self.optimizer, epoch, self.exp_dir)
                self.logger.info('Saving Normal!')
                stored_loss = val_loss

            if 't0' in self.optimizer.param_groups[0]:
                for prm in self.model.parameters():
                    if prm.grad is not None:
                        prm.data = self.tmp[prm].clone()

            # ASGD KICKS LOGIC
            if 't0' not in self.optimizer.param_groups[0] and (
                            len(best_val_loss) > self.args.nonmono and self.best_global_fit > min(best_val_loss[:-self.args.nonmono])):
                self.logger.info('Switching!')
                self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)

            best_val_loss.append(self.best_global_fit)
            if len(best_val_loss) > self.args.nonmono:
                self.logger.info('minimum fitness: {}'.format(min(best_val_loss[:-self.args.nonmono])))

        self.logger.info("global best genotype: ")
        self.logger.info(self.swarm.global_best.genotype())

        return self.swarm.global_best.genotype(), self.swarm.global_best.get_genotype_id(), self.swarm.global_best.best_fit