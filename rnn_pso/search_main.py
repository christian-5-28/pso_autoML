import glob
from time import strftime

import math
import search_configs as configs
from trainer import TrainerSearch
import os
import pickle
from utils import create_exp_dir, create_dir
import random


# retrieving the args
args = configs.parser.parse_args()

args_dict = vars(args)

# creating the main directory

args.main_path = '{}_start_{}_fixed_{}_random_{}_nodes_{}_useBlocks_{}_edgeMatrices_{}_hid_{}_pop_{}_edges_{}_reducCL_{}_genInit_{}_SEED_{}_stepsPSO_{}_w_{}_cl_{}_cg_{}_{}'.format(
                args.main_path,
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
                strftime("%Y%m%d-%H%M%S")
                )

create_exp_dir(args.main_path, scripts_to_save=glob.glob('*.py'))

# creating the tensorboard directory
tboard_path = os.path.join(args.main_path, args.tboard_dir)
args.tboard_dir = tboard_path
create_dir(tboard_path)

# list for the final dictionary to be used for dataframe analysis
gen_ids = []
genotypes = []
validation_ppl = []

if args.seed_range_end is not None:
    range_tests = [value for value in range(args.seed_range_start, args.seed_range_end)]

elif args.gen_ids_range is not None:
    range_tests = [value for value in range(args.gen_ids_start, args.gen_ids_range)]
    print(range_tests)

else:
    raise NotImplementedError

for value in range_tests:

    # modifying the configs
    if args.seed_range_end is not None:
        args.seed = value

    elif args.gen_ids_range is not None:
        args.genotype_id = value

        # logic for random solution sampled on all the possible solutions
        if args.random_id_eval:
            random.seed(value)
            range_ids = math.factorial(args.num_intermediate_nodes) * (
            args.num_operations ** args.num_intermediate_nodes)
            args.genotype_id = random.randint(0, range_ids)
            print('random gen id: {}'.format(args.genotype_id))

    trainer_search = TrainerSearch(args)

    try:

        # run the trainer
        best_geno, best_gen_id, best_val_ppl = trainer_search.run_search()

        gen_ids.append(best_gen_id)
        genotypes.append(best_geno.recurrent)
        validation_ppl.append(best_val_ppl)
        print('search concluded! genotype: {}, gen_id: {}, validation ppl: {}'.format(best_geno,
                                                                                      best_gen_id,
                                                                                      best_val_ppl))

    except Exception as e:
        print(e)
        print('ERROR ENCOUNTERED IN THIS SEARCH, MOVING ON NEXT SEARCH')
        # raise
        continue

dataframe_dict = {'gen_ids': gen_ids, 'genotypes': genotypes, 'last_valid_ppl': validation_ppl}

save_path = os.path.join(args.main_path, 'dataframe_dict')

with open(save_path, 'wb') as handle:
    pickle.dump(dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(dataframe_dict)



