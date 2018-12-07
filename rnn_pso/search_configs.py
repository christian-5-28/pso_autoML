import argparse

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
parser.add_argument('--save', type=str, default='EXP_PSO_SLOT', help='path to save the final model')
parser.add_argument('--main_path', type=str, default='BLOCKS_TEST', help='path to save the final model')
parser.add_argument('--tboard_dir', type=str, default='runs', help='path to save the tboard logs')
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
parser.add_argument('--max_seq_len_delta', type=int, default=20, help='max sequence length')
parser.add_argument('--seed_range_start', type=int, default=None)
parser.add_argument('--seed_range_end', type=int, default=None)
parser.add_argument('--early_stopping_search', type=int, default=None)
parser.add_argument('--gen_ids_start', type=int, default=None)
parser.add_argument('--gen_ids_range', type=int, default=None)
parser.add_argument('--single_gpu', default=True, action='store_false', help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')


#### PSO args ####

parser.add_argument('--start_using_pso', type=int, default=5)
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
parser.add_argument('--random_id_eval', action='store_true')
parser.add_argument('--minimum', type=int, default=None)
parser.add_argument('--range_coeff', type=int, default=2)
parser.add_argument('--use_matrices_on_edge', action='store_true')
parser.add_argument('--use_glorot', default=True, action='store_false')
parser.add_argument('--evaluate', default=False, action='store_true')
parser.add_argument('--genotype_id', type=int, default=None)

# pso blocks logic
parser.add_argument('--use_blocks', default=False, action='store_true')
parser.add_argument('--use_balance_grad', default=False, action='store_true')
parser.add_argument('--prec_id', type=int, default=1)
parser.add_argument('--prec_nodes', type=int, default=1)
parser.add_argument('--pso_window', type=int, default=5)

parser.add_argument('--pretrained_dir',
                    type=str,
                    default='search_EXP_PSO_PREPROCESS_nodes_8_hid_ACTIVATION_pop_46_edges_softmax_SEED_1267__20181108-115018')

# wpl args
parser.add_argument('--alpha_decay', type=float, default=1)
parser.add_argument('--alpha_decay_after', type=float, default=100)
parser.add_argument('--alpha_fisher', type=float, default=170000.0, help='increase penalty')
parser.add_argument('--fisher_clip_by_norm', type=float, default=50, help='Clip fisher before it is too large')
parser.add_argument('--lambda_fisher', type=float, default=0.9, help='momentum to update fisher')
parser.add_argument('--zero_fisher', type=int, default=5, help='zero fisher')
parser.add_argument('--fisher_after', type=int, default=2, help='fisher after??')
parser.add_argument('--use_wpl', default=False, action='store_true')
