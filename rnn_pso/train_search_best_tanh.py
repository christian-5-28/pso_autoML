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
from swarm import Swarm
from tensorboardX import SummaryWriter

import gc

import data
import model as model_module

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint, create_viz_dir
# from visualize import plot

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
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
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='EXP_PSO_classic',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
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
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')

#### PSO args ####
parser.add_argument('--population_size', type=int, default=46, help='number_of_particles')
parser.add_argument('--handle_hidden_mode', type=str, default='ACTIVATION')
parser.add_argument('--visualize', default=False, action='store_false')
parser.add_argument('--edge_select_mode', type=str, default='softmax')
parser.add_argument('--nlayers', type=int, default=1)

args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = 'search_{}_bs{}_nb{}_hid_{}_pop_{}_edges_{}_seed_{}_{}'.format(args.save,
                                                                               args.batch_size,
                                                                               args.train_num_batches,
                                                                               args.handle_hidden_mode,
                                                                               args.population_size,
                                                                               args.edge_select_mode,
                                                                               args.seed,
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
swarm = Swarm(args, population_size=args.population_size)

# initial genotype
genotype = swarm.population[swarm.gbest_id].genotype()

# initializing the model
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
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

# initializing the architecture search optimizer
# architect = Architect(parallel_model, args)

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))


def evaluate(data_source, fitnesses_dict, batch_size=10):
    particles_indices = [i for i in range(swarm.population_size)]
    np.random.shuffle(particles_indices)
    avg_particle_loss = 0
    avg_particle_ppl = 0

    for particle_id in particles_indices:
        # computing the genotype of the next particle
        new_genotype = swarm.population[particle_id].genotype()

        # selecting the current subDAG in our DAG to train
        model.change_genotype(genotype=new_genotype)
        print("VALIDATE PARTICLE: ", particle_id)
        # Turn on evaluation mode which disables dropout.
        model.eval()
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

        avg_particle_loss += avg_valid_loss
        avg_particle_ppl += avg_valid_ppl

        # saving the particle fit in our dictionary
        fitnesses_dict[particle_id] = avg_valid_ppl

    return fitnesses_dict, avg_particle_loss / swarm.population_size, avg_particle_ppl / swarm.population_size


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # initializing first hidden state
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    # hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]

    batch, i = 0, 0

    particles_indices = [i for i in range(swarm.population_size)]
    np.random.shuffle(particles_indices)

    while i < train_data.size(0) - 1 - 1:

        for particle_id in particles_indices:

            # computing the genotype of the next particle
            new_genotype = swarm.population[particle_id].genotype()

            # selecting the current subDAG in our DAG to train
            model.change_genotype(genotype=new_genotype)

            # train this subDAG for 4 batches
            for b in range(args.train_num_batches):

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
                    # cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

                    # Starting each batch, we detach the hidden state from how it was previously produced.
                    # If we didn't, the model would try backpropagating all the way to start of the dataset.
                    hidden[s_id] = repackage_hidden(hidden[s_id])
                    # hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

                    # architecture optimizer step, updating the alphas weights
                    '''
                    hidden_valid[s_id], grad_norm = architect.step(hidden[s_id],
                                                                   cur_data,
                                                                   cur_targets,
                                                                   hidden_valid[s_id],
                                                                   cur_data_valid,
                                                                   cur_targets_valid,
                                                                   optimizer,
                                                                   args.unrolled)

                    '''

                    # assuming small_batch_size = batch_size so we don't accumulate gradients
                    optimizer.zero_grad()
                    # hidden[s_id] = repackage_hidden(hidden[s_id])

                    # forward pass
                    if epoch == 7:
                        logger.info(" EPOCH 7, FORWARD PASS OF BATCH ", batch)

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
                            #     print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                            logger.info("max hidden value of all steps: " + str(rnn_hs[0].max()))

                        # Temporal Activation Regularization (slowness)
                        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                    except:
                        # for step in range(len(rnn_hs[0])):
                        # print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                        logger.info("max hidden value of all steps: " + str(rnn_hs[0].max()))
                        logger.info("genotype who caused the error:  ")
                        logger.info(model.genotype())
                        # print(model.genotype())
                        for name_param, param in model.rnns[0].named_parameters():
                            logger.info("param name: " + str(name_param))
                            logger.info("max value in the param matrix: " + str(param.max()))
                        raise
                    loss *= args.small_batch_size / args.batch_size
                    total_loss += raw_loss.data * args.small_batch_size / args.batch_size
                    loss.backward()

                    s_id += 1
                    start = end
                    end = start + args.small_batch_size

                    gc.collect()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
                total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                """
                if batch % args.log_interval == 0 and batch > 0:
                    logger.info("norm of the gradients of the model: " + str(total_norm))

                    clip_coef = args.clip / (total_norm + 1e-6)

                    logger.info("clip coef of the gradients of the model: " + str(clip_coef))

                    for name_param, param in model.rnns[0].named_parameters():
                        logger.info("param name: " + str(name_param))
                        logger.info("max value in the param matrix: " + str(param.max()))
                """
                optimizer.step()

                # total_loss += raw_loss.data
                optimizer.param_groups[0]['lr'] = lr2
                if batch % args.log_interval == 0 and batch > 0:
                    logger.info(model.genotype())
                    # print(F.softmax(parallel_model.weights, dim=-1))
                    cur_loss = total_loss[0] / args.log_interval
                    elapsed = time.time() - start_time
                    logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
                batch += 1
                i += seq_len
    writer.add_scalar('train_loss', cur_loss, epoch)
    writer.add_scalar('train_ppl', math.exp(cur_loss), epoch)


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

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

    # training pipeline for one epoch
    train()

    # validation at the end of the epoch
    particles_fitnesses, val_loss, val_ppl = evaluate(val_data,
                                                      fitnesses_dict=particles_fitnesses,
                                                      batch_size=eval_batch_size)

    # updating the particles fitnesses and their best position
    swarm.evaluate_population(particles_fitnesses)

    logger.info("global best genotype: " + str(swarm.gbest_id))
    logger.info(swarm.population[swarm.gbest_id].genotype())

    if args.visualize:
        # saving the viz of the current gbest for this epoch
        file_name = os.path.join(viz_dir_path, 'epoch ' + str(epoch))
        graph_name = 'epoch_{}'.format(epoch)
        plot(swarm.population[swarm.gbest_id].genotype().recurrent, file_name, graph_name)

    best_global_fit = particles_fitnesses[swarm.gbest_id]

    # updating the particle current position
    swarm.update_particles_position()

    # tensorboard logs
    writer.add_scalar('avg_swarm_validation_loss', val_loss, epoch)
    writer.add_scalar('avg_swarm_validation_ppl', val_ppl, epoch)

    writer.add_scalar('global_best_validation_loss', math.log(best_global_fit), epoch)
    writer.add_scalar('global_best_validation_ppl', best_global_fit, epoch)

    writer.add_scalar('global_best_id', swarm.gbest_id, epoch)

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

logger.info("global best genotype: ")
logger.info(swarm.population[swarm.gbest_id].genotype())
