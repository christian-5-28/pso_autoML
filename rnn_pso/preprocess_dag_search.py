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
parser.add_argument('--epochs', type=int, default=350,
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
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='EXP_PSO_PREPROCESS',
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

parser.add_argument('--start_using_pso', type=int, default=3)
parser.add_argument('--stop_training_pso', type=int, default=20)
parser.add_argument('--updates_pso', type=int, default=5)
parser.add_argument('--population_size', type=int, default=46, help='number_of_particles')
parser.add_argument('--handle_hidden_mode', type=str, default='ACTIVATION')
parser.add_argument('--use_training_phase', default=True, action='store_false')
parser.add_argument('--visualize', default=False, action='store_false')
parser.add_argument('--num_operations', type=int, default=4, help='valid operations in search space')
parser.add_argument('--num_intermediate_nodes', type=int, default=8)
parser.add_argument('--edge_select_mode', type=str, default='softmax')
parser.add_argument('--reduce_clusters', default=False, action='store_false')
parser.add_argument('--uniform_genos_init', default=False, action='store_false')

args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = 'search_{}_nodes_{}_hid_{}_pop_{}_edges_{}_SEED_{}__{}'.format(
        args.save,
        args.num_intermediate_nodes,
        args.handle_hidden_mode,
        args.population_size,
        args.edge_select_mode,
        args.seed,
        time.strftime("%Y%m%d-%H%M%S")
        )
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


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
              genos_init=args.uniform_genos_init)

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

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))


def evaluate(data_source, batch_size=10):

    # computing the next genotype
    new_genotype = swarm.get_new_genotype()

    # selecting the current subDAG in our DAG to train
    model.change_genotype(genotype=new_genotype)

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

    return avg_valid_loss, avg_valid_ppl


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # initializing first hidden state
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    # hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]

    total_batches = len(train_data) // args.bptt
    slot_batches = int(math.floor(total_batches / args.population_size))
    remaining_batches = total_batches % args.population_size

    batch, i = 0, 0

    while i < train_data.size(0) - 1 - 1:

        for individual in range(args.population_size):

            if i >= train_data.size(0) - 1 - 1:
                break

            # logic to distribute the module of the division total_batches / num_clusters
            add_sample = individual < remaining_batches
            train_slot_batch = slot_batches
            if add_sample:
                train_slot_batch += 1

            # computing the next genotype
            new_genotype = swarm.get_new_genotype()

            if batch % args.log_interval == 0 and batch > 0:
                logger.info('new genotype used: {}'.format(new_genotype))

            # selecting the current subDAG in our DAG to train
            model.change_genotype(genotype=new_genotype)

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
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

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

    writer.add_scalar('pre_train_loss', cur_loss, epoch)
    writer.add_scalar('pre_train_ppl', math.exp(cur_loss), epoch)


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

particles_fitnesses = {}

# At any point you can hit Ctrl + C to break out of training early.
try:

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

        # training pipeline for one epoch
        train()

        # validation logic
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            # model validation at the end of epoch
            val_loss2, val_ppl2 = evaluate(val_data)
            writer.add_scalar('validation_loss', val_loss2, epoch)
            writer.add_scalar('validation_ppl', val_ppl2, epoch)
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss2, math.exp(val_loss2)))
            logging.info('-' * 89)

            if val_loss2 < stored_loss:
                save_checkpoint(model, optimizer, epoch, args.save)
                logging.info('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss, val_ppl = evaluate(val_data, eval_batch_size)
            writer.add_scalar('pre_validation_loss', val_loss, epoch)
            writer.add_scalar('pre_validation_ppl', val_ppl, epoch)

            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss, math.exp(val_loss)))
            logging.info('-' * 89)

            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, epoch, args.save)
                logging.info('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (
                            len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logging.info('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')