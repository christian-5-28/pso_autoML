import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from swarm_init import Swarm
# from swarm import Swarm
from tensorboardX import SummaryWriter

import gc

import data
import model as model_module

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint, create_viz_dir
# from visualize import plot

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=850,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=850,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=850,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--train_num_batches', type=int, default=18)
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='EXP_PSO_SLOT',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false',
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')

#### PSO args ####

parser.add_argument('--start_using_pso', type=int, default=8)
parser.add_argument('--updates_pso', type=int, default=1)
parser.add_argument('--population_size', type=int, default=25, help='number_of_particles')
parser.add_argument('--handle_hidden_mode', type=str, default='ACTIVATION')
parser.add_argument('--use_training_phase', default=True, action='store_false')
parser.add_argument('--visualize', default=False, action='store_false')
parser.add_argument('--num_operations', type=int, default=4, help='valid operations in search space')
parser.add_argument('--num_intermediate_nodes', type=int, default=8)
parser.add_argument('--concat', type=int, default=8)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--w_inertia', type=float, default=0.5)
parser.add_argument('--c_local', type=int, default=1)
parser.add_argument('--c_global', type=int, default=2)
parser.add_argument('--edge_select_mode', type=str, default='softmax')
parser.add_argument('--reduce_clusters', default=False, action='store_false')
parser.add_argument('--uniform_genos_init', default=True, action='store_false')
parser.add_argument('--use_pretrained', action='store_true')
parser.add_argument('--use_fixed_slot', action='store_true')
parser.add_argument('--use_random', action='store_true')
parser.add_argument('--minimum', type=int, default=None)
parser.add_argument('--range_coeff', type=int, default=2)
parser.add_argument('--use_matrices_on_edge', action='store_true')
parser.add_argument('--use_glorot', default=True, action='store_false')

# pso blocks logic
parser.add_argument('--use_blocks', default=False, action='store_true')
parser.add_argument('--prec_id', type=int, default=1)
parser.add_argument('--prec_nodes', type=int, default=1)



parser.add_argument('--pretrained_dir',
                    type=str,
                    default='search_EXP_PSO_PREPROCESS_nodes_8_hid_ACTIVATION_pop_46_edges_softmax_SEED_1267__20181108-115018')


args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

####


if not args.continue_train:
    args.save = 'search_{}_start_{}_fixed_{}_random_{}_nodes_{}_useBlocks_{}_edgeMatrices_{}_hid_{}_pop_{}_edges_{}_reducCL_{}_genInit_{}_SEED_{}_stepsPSO_{}_w_{}_cl_{}_cg_{}_{}'.format(args.save,
                                                                                                                                                                    args.start_using_pso,
                                                                                                                                                                    args.use_fixed_slot,
                                                                                                                                                                    args.use_random,
                                                                                                                                                                             args.num_intermediate_nodes,
                                                                                                                                                                                          args.use_blocks,
                                                                                                                                                                    args.use_matrices_on_edge,
                                                                                                                                                                    args.handle_hidden_mode,
                                                                                                                                                                    args.population_size,
                                                                                                                                                                    args.edge_select_mode,
                                                                                                                                                                    args.reduce_clusters,
                                                                                                                                                                    args.uniform_genos_init,
                                                                                                                                                                    args.seed,
                                                                                                                                                                    args.updates_pso,
                                                                                                                                                                    args.w_inertia,
                                                                                                                                                                    args.c_local,
                                                                                                                                                                    args.c_global,
                                                                                                                                                                    time.strftime("%Y%m%d-%H%M%S")
                                                                                                                                                                    )
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

if args.visualize:
    viz_dir_path = create_viz_dir(args.save)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger("train_search")
logger.addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, args)
search_data = batchify(corpus.valid, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

# Tensorboard writer
writer = SummaryWriter('runs/tensor_log_' + args.save)

ntokens = len(corpus.dictionary)

# initialize the swarm

swarm = Swarm(population_size=args.population_size,
              num_operations=args.num_operations,
              intermediate_nodes=args.num_intermediate_nodes,
              args=args,
              genos_init=args.uniform_genos_init)

# swarm = Swarm(args, population_size=args.population_size)

# initial genotype
genotype = swarm.global_best.genotype()

# initializing the model
if args.use_pretrained:
    logger.info('PRETRAINED MODEL LOADED!')
    model = torch.load(os.path.join(args.pretrained_dir, 'model.pt'))
else:
    model = model_module.RNNModel(ntokens,
                                  args,
                                  genotype=genotype)

size = 0
for p in model.parameters():
    size += p.nelement()
logger.info('param size: {}'.format(size))
logger.info('initial genotype:')
logger.info(model.genotype())

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))


def validate_model(current_model, data_source, batch_size=10):
    current_model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)

    avg_valid_loss = total_loss[0] / len(data_source)
    avg_valid_ppl = math.exp(avg_valid_loss)

    return avg_valid_loss, avg_valid_ppl


def evaluate(data_source, fitnesses_dict, batch_size=10):
    particles_indices = [i for i in range(swarm.population_size)]
    np.random.shuffle(particles_indices)
    avg_particle_loss = 0
    avg_particle_ppl = 0
    cluster_validated = set()
    genotypes_fit_dict = {}

    for particle_id in particles_indices:

        geno_id = swarm.population[particle_id].get_genotype_id()

        # computing the genotype of the next particle
        new_genotype = swarm.population[particle_id].genotype()

        if geno_id in cluster_validated:
            logger.info('particle already validated, genotype: {} {}'.format(geno_id, new_genotype))
            fitnesses_dict[particle_id] = genotypes_fit_dict[geno_id]
            continue

        # add the geno id to the seen clusters
        cluster_validated.add(geno_id)

        # selecting the current subDAG in our DAG to train
        model.change_genotype(genotype=new_genotype)
        logger.info("VALIDATE PARTICLE: {}, with genotype id {}, {}".format(particle_id, geno_id, new_genotype))

        avg_valid_loss, avg_valid_ppl = validate_model(current_model=model,
                                                       data_source=data_source,
                                                       batch_size=batch_size)

        avg_particle_loss += avg_valid_loss
        avg_particle_ppl += avg_valid_ppl

        # saving the particle fit in our dictionaries
        fitnesses_dict[particle_id] = avg_valid_ppl
        genotypes_fit_dict[geno_id] = avg_valid_ppl

    # validate the current config of the global best particle
    glob_best_gen = swarm.global_best.genotype() if not args.use_random else best_so_far[0]
    model.change_genotype(genotype=glob_best_gen)
    logger.info("VALIDATE BEST GEN: {}".format(glob_best_gen))
    _, best_fit = validate_model(current_model=model,
                                 data_source=data_source,
                                 batch_size=batch_size)
    if not args.use_random:
        logger.info('PREVIOUS BEST FIT: {}'.format(swarm.global_best_fit))
        swarm.global_best.best_fit = best_fit
        swarm.global_best_fit = best_fit
        logger.info('NEW BEST FIT: {}'.format(swarm.global_best_fit))
    else:
        logger.info('PREVIOUS BEST FIT: {}'.format(best_so_far[2]))
        best_so_far[2] = best_fit
        logger.info('NEW BEST FIT: {}'.format(best_so_far[2]))

    return fitnesses_dict, avg_particle_loss / len(cluster_dict.keys()), avg_particle_ppl / len(cluster_dict.keys())


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    epoch_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # initializing first hidden state
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    # hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]

    batch, i, pop_index = 0, 0, 0

    # shuffle order to train particles
    particles_indices = [i for i in range(swarm.population_size)]
    np.random.shuffle(particles_indices)

    if not args.use_fixed_slot:
        # dynamic slot of batches computation
        logger.info('clusters size for the current epoch {}: {}'.format(epoch, cluster_dict))

        num_clusters = len(cluster_dict.keys())
        total_batches = len(train_data) // args.bptt
        slot_batches = int(math.floor(total_batches / num_clusters))
        remaining_batches = total_batches % num_clusters

        logger.info('num custers: {}, total batches: {}, slot batches: {}, module: {}'.format(num_clusters,
                                                                                              total_batches,
                                                                                              slot_batches,
                                                                                              remaining_batches))

    else:
        slot_batches = args.train_num_batches
        num_clusters = len(cluster_dict.keys())

    clusters_seen = set()
    gbest_gen_id = swarm.global_best.get_genotype_id() if not args.use_random else best_so_far[1]
    gbest_gen_trained = False

    while i < train_data.size(0) - 1 - 1:

        if args.use_fixed_slot and len(clusters_seen) == num_clusters:
            logger.info('EXITING FROM TRAINING PHASE, FIXED SLOT, ALL CLUSTERS TRAINED')
            break

        for particle_id in particles_indices:

            if i >= train_data.size(0) - 1 - 1:
                break

            if args.use_fixed_slot and len(clusters_seen) == num_clusters:
                logger.info('EXITING FROM TRAINING PHASE, FIXED SLOT, ALL CLUSTERS TRAINED')
                break

            genotype_id = swarm.population[particle_id].get_genotype_id()
            if not gbest_gen_trained:
                gbest_gen_trained = genotype_id == gbest_gen_id

            # if the particle has same discrete configuration
            # already trained during this epoch, we skip it
            if genotype_id in clusters_seen and (len(clusters_seen) < len(cluster_dict)):  # and not args.use_fixed_slot:
                logger.info('particle already trained, genotype: {}, data len {}, i {}'.format(genotype_id,
                                                                                               train_data.size(0),
                                                                                               i))
                continue

            if not args.use_fixed_slot:
                # logic to distribute the module of the division total_batches / num_clusters
                add_sample = len(clusters_seen) < remaining_batches
                train_slot_batch = slot_batches
                if add_sample:
                    train_slot_batch += 1
            else:
                train_slot_batch = slot_batches

            # adding the gen id in the set of the clusters already seen
            clusters_seen.add(genotype_id)

            previous_genotype = model.genotype()

            # random search
            '''
            if args.use_random:
                new_genotype, genotype_id = swarm.get_new_genotype()
                swarm.population[particle_id].initialize_weights(desired_genotype=new_genotype)
                new_genotype = swarm.population[particle_id].genotype()
            
            if epoch < args.start_using_pso and not args.use_random:
                # computing the next genotype
                new_genotype, genotype_id = swarm.get_new_genotype()
            '''

            # else:
            # computing the genotype of the next particle
            new_genotype = swarm.population[particle_id].genotype()

            # selecting the current subDAG in our DAG to train
            model.change_genotype(genotype=new_genotype)

            # train this subDAG for slot of batches
            logger.info('train_slot_batch: {}, genotype: {}, steps seen: {} '.format(train_slot_batch,
                                                                                     genotype_id,
                                                                                     i))
            for b in range(train_slot_batch):

                if i >= train_data.size(0) - 1 - 1:
                    break

                bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
                # Prevent excessively small or negative sequence lengths
                # seq_len = max(5, int(np.random.normal(bptt, 5)))
                # # There's a very small chance that it could select a very long sequence length resulting in OOM
                # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
                seq_len = int(bptt)

                lr2 = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt

                # training mode activated
                model.train()

                # preparing batch of data from validation for the architecture optimizer step
                # data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args)

                # preparing batch of data for training
                data, targets = get_batch(train_data, i, args, seq_len=seq_len)

                optimizer.zero_grad()

                start, end, s_id = 0, args.small_batch_size, 0
                while start < args.batch_size:
                    cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

                    # Starting each batch, we detach the hidden state from how it was previously produced.
                    # If we didn't, the model would try backpropagating all the way to start of the dataset.
                    hidden[s_id] = repackage_hidden(hidden[s_id])
                    # hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

                    # assuming small_batch_size = batch_size so we don't accumulate gradients
                    optimizer.zero_grad()
                    # hidden[s_id] = repackage_hidden(hidden[s_id])

                    # forward pass
                    log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model(cur_data, hidden[s_id], return_h=True)

                    # loss using negative-log-likelihood
                    raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

                    loss = raw_loss
                    try:
                        # Activation Regularization
                        if args.alpha > 0:
                            loss = loss + sum(
                                args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

                        if batch % args.log_interval == 0 and batch > 0:
                            # for step in range(len(rnn_hs[0])):
                            #    print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                            logger.info("max hidden value of all steps: " + str(rnn_hs[0].max()))

                        # Temporal Activation Regularization (slowness)
                        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                    except:
                        # for step in range(len(rnn_hs[0])):
                        # print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                        logger.info("max hidden value of all steps: " + str(rnn_hs[0].max()))
                        print('RNN_HS: {}'.format(rnn_hs))
                        logger.info("genotype who caused the error:  ")
                        logger.info(model.genotype())
                        # print(model.genotype())
                        for name_param, param in model.rnns[0].named_parameters():
                            logger.info("param name: " + str(name_param))
                            logger.info("max value in the param matrix: " + str(param.max()))
                        raise
                    loss *= args.small_batch_size / args.batch_size
                    total_loss += raw_loss.data * args.small_batch_size / args.batch_size

                    epoch_loss += raw_loss.data * args.small_batch_size / args.batch_size

                    loss.backward()

                    s_id += 1
                    start = end
                    end = start + args.small_batch_size

                    gc.collect()

                    if batch > 0 and (epoch_loss[0] / batch) > 15:
                        logger.info('TRAIN LOSS IS HIGH {}'.format(epoch_loss[0] / batch))
                        logger.info('current particle used has GENOTYPE id {}'.format(swarm.population[particle_id].get_genotype_id()))

                        logger.info('PREVIOUS genotype {}'.format(previous_genotype))
                        logger.info('genotype {}'.format(swarm.population[particle_id].genotype()))
                        logger.info('POSITION {}'.format(swarm.population[particle_id].position))
                        logger.info('SOFTMAX: {}'.format(F.softmax(swarm.population[particle_id].position, dim=-1)))

                        logger.info('MODEL INFOS BEFORE CLIPPING GRAD')
                        for name, module_ in model.named_children():
                            for name_param, param in module_.named_parameters():
                                if param.grad is not None:
                                    logger.info('model module name: {}, param name: {}, max layer value: {}, max grad: {}'.format(name,
                                                                                                                                  name_param,
                                                                                                                                 param.max(),
                                                                                                                                 param.grad.max()))

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

                if batch > 0 and (epoch_loss[0] / batch) > 15:
                    logger.info('TRAIN LOSS IS HIGH: {}'.format(epoch_loss[0] / batch))
                    logger.info('MODEL INFOS AFTER GRAD CLIP')
                    for name, module_ in model.named_children():
                        for name_param, param in module_.named_parameters():
                            if param.grad is not None:
                                logger.info(
                                    'model module name: {}, param name: {}, max layer value: {}, max grad: {}'.format(
                                        name,
                                        name_param,
                                        param.max(),
                                        param.grad.max()))

                optimizer.step()

                # total_loss += raw_loss.data
                optimizer.param_groups[0]['lr'] = lr2
                if batch % args.log_interval == 0 and batch > 0:
                    logger.info(model.genotype())
                    # print(F.softmax(parallel_model.weights, dim=-1))
                    cur_loss = total_loss[0] / args.log_interval

                    # TODO: debug clusters error
                    if cur_loss != cur_loss:
                        logger.info('TRAIN LOSS IS NAN!')
                        logger.info('current particle used has GENOTYPE id {}'.format(
                            swarm.population[particle_id].get_genotype_id()))

                        logger.info('PREVIOUS genotype {}'.format(previous_genotype))
                        logger.info('genotype {}'.format(swarm.population[particle_id].genotype()))
                        logger.info('POSITION {}'.format(swarm.population[particle_id].position))
                        logger.info('SOFTMAX: {}'.format(F.softmax(swarm.population[particle_id].position, dim=-1)))

                        logger.info('MODEL INFOS BEFORE CLIPPING GRAD')
                        for name, module_ in model.named_children():
                            for name_param, param in module_.named_parameters():
                                if param.grad is not None:
                                    logger.info(
                                        'model module name: {}, param name: {}, max layer value: {}, max grad: {}'.format(
                                            name,
                                            name_param,
                                            param.max(),
                                            param.grad.max()))

                    elapsed = time.time() - start_time
                    logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

                    total_loss = 0
                    start_time = time.time()
                batch += 1
                i += seq_len

    logger.info('GENOTYPE GBEST TRAINED: {}'.format(gbest_gen_trained))
    logger.info('GENOTYPE GBEST ID: {}'.format(gbest_gen_id))

    writer.add_scalar('train_loss', epoch_loss[0] / batch, epoch)
    writer.add_scalar('train_ppl', math.exp(epoch_loss[0] / batch), epoch)


def update_best_solution(fitnesses, best_solution_so_far):
    #best_gen, best_id, best_ppl = best_solution_so_far

    for particle_id, validation_ppl in fitnesses.items():

        if validation_ppl < best_solution_so_far[2]:
            logger.info('new best solution found: previous gen: {}'.format(best_solution_so_far[0]))
            new_gen = swarm.population[particle_id].genotype()
            new_id = swarm.population[particle_id].get_genotype_id()
            logger.info('new best solution found: new gen: {}'.format(new_gen))
            logger.info('new best solution found: previous ppl: {}, new ppl: {}'.format(best_solution_so_far[2],
                                                                                        validation_ppl))
            best_solution_so_far[0] = new_gen
            best_solution_so_far[1] = new_id
            best_solution_so_far[2] = validation_ppl

    return best_solution_so_far


def update_random_swarm(swarm, minimum, coeff, best_gen):

    logger.info('UPDATE THE RANDOM SWARM!')

    particle_index = 0
    particles_available = len(swarm.population)
    best_so_far_used = False

    while particle_index < len(swarm.population):

        prob = np.random.rand()

        if prob < 0.4:
            new_geno = swarm.population[particle_index].genotype()
            # repeat = min(np.random.randint(minimum, coeff * minimum), particles_available)

        elif not best_so_far_used:
            new_geno = best_gen
            best_so_far_used = True
            # repeat = min(np.random.randint(1, 3), particles_available)

        else:
            new_geno, _ = swarm.get_new_genotype()
            # repeat = min(np.random.randint(minimum, coeff*minimum), particles_available)

        repeat = min(np.random.randint(minimum, coeff * minimum), particles_available)
        logger.info('REPEAT VALUE: {}'.format(repeat))

        for _ in range(repeat):

            swarm.population[particle_index].initialize_weights(desired_genotype=new_geno)
            particle_index += 1
            particles_available -= 1


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

geno_init, geno_init_id = swarm.get_new_genotype(size=1)
best_so_far = [geno_init, geno_init_id, np.Inf]
particles_fitnesses = {}

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    logger.info('\n EPOCH {} STARTED.'.format(epoch))

    # update the clusters_dict
    cluster_dict = swarm.compute_clusters()
    number_clusters = len(cluster_dict.keys())
    writer.add_scalar('number_of_clusters', number_clusters, epoch)
    max_size = 0
    gen_id = 0
    for item in cluster_dict.items():
        key = item[0]
        value = item[1]
        if value > max_size:
            max_size = value
            gen_id = key

    logger.info('cluster with max size has id {} and size {}'.format(gen_id, max_size))
    writer.add_scalar('max_size_cluster', max_size, epoch)

    geno_best_id = swarm.global_best.get_genotype_id() if not args.use_random else best_so_far[1]
    size_cluster_gbest = cluster_dict.get(geno_best_id)
    if size_cluster_gbest is None:
        size_cluster_gbest = 1
        logger.info('THE GBEST GENOTYPE IS UNIQUE, NOT IN THE SWARM!')

    logger.info('cluster of gbest has id {} and size {}'.format(geno_best_id, size_cluster_gbest))
    writer.add_scalar('cluster_size_gbest', size_cluster_gbest, epoch)

    # training pipeline for one epoch
    train()

    if epoch >= args.start_using_pso and not args.use_random:
        # PSO UPDATES PHASE
        # TODO: refactor in one method called PSO_update_phase
        for step in range(args.updates_pso):
            logger.info('UPDATE PSO STEP {}:'.format(step))

            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    if prm.grad is not None:
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer.state[prm]['ax'].clone()
                logger.info('CLONING SUCCESSFULL')

            # validation at the end of the epoch
            particles_fitnesses, val_loss, val_ppl = evaluate(val_data,
                                                              fitnesses_dict=particles_fitnesses,
                                                              batch_size=eval_batch_size)

            # updating the particles fitnesses and their best position
            swarm.evaluate_population(particles_fitnesses)

            logger.info("global best genotype after evaluation:")
            logger.info(swarm.global_best.genotype())

            if args.visualize:
                # saving the viz of the current gbest for this epoch
                file_name = os.path.join(viz_dir_path, 'epoch ' + str(epoch))
                graph_name = 'epoch_{}'.format(epoch)
                plot(swarm.global_best.genotype().recurrent, file_name, graph_name)

            best_global_fit = swarm.global_best.best_fit

            # updating the particle current position
            swarm.update_particles_position()

            '''
            logger.info("global best genotype after updating the particles: ")
            logger.info(swarm.global_best.genotype())
            logger.info('Best particle representation: {}'.format(swarm.population[swarm.gbest_id].position))
            logger.info('SOFTMAX: {}'.format(F.softmax(swarm.population[swarm.gbest_id].position, dim=-1)))
            '''

            if args.reduce_clusters:
                # computing the centroids for each discrete configuration in the swarm
                swarm.compute_centroids()

                logger.info('current set of genotypes ids covered by the swarm after centroids {}'.format(swarm.set_gen_ids))
                logger.info('size of gene ids set: {}'.format(len(swarm.set_gen_ids)))

                # TODO: store in a better way the survival of genotypes
                swarm.update_survival_dict(epoch=epoch)
                logger.info('survival_dict {}'.format(swarm.genotype_survival_epochs))

            # tensorboard logs
            writer.add_scalar('avg_swarm_validation_loss', val_loss, epoch)
            writer.add_scalar('avg_swarm_validation_ppl', val_ppl, epoch)

            writer.add_scalar('global_best_validation_loss', math.log(best_global_fit), epoch)
            writer.add_scalar('global_best_validation_ppl', best_global_fit, epoch)

            writer.add_scalar('validation_loss', math.log(best_global_fit), epoch)
            writer.add_scalar('validation_ppl', best_global_fit, epoch)

            # writer.add_scalar('global_best_id', swarm.gbest_id, epoch)

            # logs info
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   math.log(best_global_fit), best_global_fit))
            logger.info('-' * 89)

            # call checkpoint if better model found
            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, epoch, args.save)
                logger.info('Saving Normal!')
                stored_loss = val_loss

            if 't0' in optimizer.param_groups[0]:
                for prm in model.parameters():
                    if prm.grad is not None:
                        prm.data = tmp[prm].clone()

            if 't0' not in optimizer.param_groups[0] and (
                            len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logger.info('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            best_val_loss.append(val_loss)

            # update the clusters_dict
            # TODO: refactor the handling of the cluster logs in one utility method
            cluster_dict = swarm.compute_clusters()
            # dynamic slot of batches computation
            logger.info('clusters size for the current STEP {}: {}'.format(step, cluster_dict))
            num_clusters = len(cluster_dict.keys())
            logger.info('number of cluster at STEP {}: {}'.format(step, num_clusters))
            max_size = 0
            gen_id = 0
            for item in cluster_dict.items():
                key = item[0]
                value = item[1]
                if value > max_size:
                    max_size = value
                    gen_id = key

            logger.info('cluster with max size has id {} and size {}'.format(gen_id, max_size))
            geno_best_id = swarm.global_best.get_genotype_id()
            size_cluster_gbest = cluster_dict.get(geno_best_id)
            if size_cluster_gbest is None:
                size_cluster_gbest = 1
                logger.info('THE GBEST GENOTYPE IS UNIQUE, NOT IN THE SWARM!')
            logger.info('cluster of gbest has id {} and size {}'.format(geno_best_id, size_cluster_gbest))

    elif args.use_random:

        logger.info('START VALIDATION RANDOM SEARCH')

        # validation at the end of the epoch
        particles_fitnesses, val_loss, val_ppl = evaluate(val_data,
                                                          fitnesses_dict=particles_fitnesses,
                                                          batch_size=eval_batch_size)

        logger.info("global best genotype before validation: {}".format(best_so_far[0]))

        # update the best solution found so far
        best_so_far = update_best_solution(particles_fitnesses, best_so_far)
        best_global_fit = best_so_far[2]
        logger.info("global best genotype after validation: {}".format(best_so_far[0]))

        # mimick pso clustering randomly
        minimum = epoch if args.minimum is None else args.minimum
        update_random_swarm(swarm=swarm,
                            minimum=minimum,
                            coeff=args.range_coeff,
                            best_gen=best_so_far[0])

        # tensorboard logs
        writer.add_scalar('avg_swarm_validation_loss', val_loss, epoch)
        writer.add_scalar('avg_swarm_validation_ppl', val_ppl, epoch)

        writer.add_scalar('global_best_validation_loss', math.log(best_global_fit), epoch)
        writer.add_scalar('global_best_validation_ppl', best_global_fit, epoch)

        writer.add_scalar('validation_loss', math.log(best_global_fit), epoch)
        writer.add_scalar('validation_ppl', best_global_fit, epoch)

        # logs info
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               math.log(best_global_fit), best_global_fit))
        logger.info('-' * 89)

        # call checkpoint if better model found
        if val_loss < stored_loss:
            save_checkpoint(model, optimizer, epoch, args.save)
            logger.info('Saving Normal!')
            stored_loss = val_loss

        best_val_loss.append(val_loss)

if not args.use_random:
    logger.info("global best genotype: ")
    logger.info(swarm.global_best.genotype())

else:
    logger.info("global best genotype: ")
    logger.info(best_so_far[0])



