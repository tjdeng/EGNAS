from LaNAS.MCTS import MCTS
import copy
import torch
import json
import egnas.arch_process as dp
from egnas.mlp import MLP_Manager
import numpy as np
import time


def mcts_ppo(args, trnr, arch_train_info_trace):
    # get info about arch's mask.
    arch_mask_info = dp.arch_info(args.layers_of_child_model)
    # mcts = MCTS(arch_mask_info, trnr, 5, args)
    mcts = MCTS(arch_mask_info, trnr, args.tree_height, args)

    results = {}
    for i in range(args.init_samples):
        print(20 * '-', 'init_samples:', i, 20 * '-')

        if len(trnr.total_samples) % 100 == 0:
            trnr.init_controller()

        while True:
            structure_list, log_probs, entropies, actions_index = trnr.controller_sample()
            tmp_action = copy.deepcopy(structure_list[0])
            tmp_action[-1] = trnr.submodel_manager.n_classes

            if str(tmp_action) not in trnr.total_samples:
                break
            else:
                print('Arise a repeated arch', tmp_action)

        trnr.total_samples.append(str(tmp_action))

        mask = dp.encode_arch_to_mask(structure_list)[0]

        # calculate reward
        np_entropies = entropies.data.cpu().numpy()

        start_time = time.time()
        rewards, val_acc = trnr.get_reward(structure_list, np_entropies)
        results[json.dumps(mask)] = val_acc
        end_time = time.time()
        arch_train_info_trace.append((structure_list, val_acc, end_time - start_time))

        torch.cuda.empty_cache()

        trnr.collect_trajectory(rewards, log_probs, actions_index)
        print(20 * '-', 'episodes: ', trnr.buf.get_buf_size(), 20 * '-')

        if trnr.buf.get_buf_size() % args.episodes == 0:
            trnr.train_controller_by_ppo2()

    mcts.collect_samples(results)
    train_mcts_time_costs = mcts.run(arch_train_info_trace)
    # mcts.run_use_period_predictor()

    return train_mcts_time_costs


def ppo(args, trnr):

    arch_mask_info = dp.arch_info(args.layers_of_child_model)
    mask_samples = {}

    if args.is_predictor:
        mlp_mg = MLP_Manager(args, arch_mask_info['mask_len'])

    while len(trnr.total_samples) < args.search_samples:

        if len(trnr.total_samples) % 100 == 0:
            trnr.init_controller()

        if args.is_predictor and len(trnr.total_samples) >= args.num_mlp_start and flag == 1:
            flag = 0
            mlp_mg.train_mlp(mask_samples)
            for i in range(args.select_samples):

                while True:
                    structure_list, log_probs, entropies, actions_index = trnr.controller_sample()
                    tmp_action = copy.deepcopy(structure_list[0])
                    tmp_action[-1] = trnr.submodel_manager.n_classes
                    if str(tmp_action) not in trnr.total_samples:
                        break
                    else:
                        print('Arise a repeated arch', tmp_action)

                trnr.total_samples.append(str(tmp_action))
                np_entropies = entropies.data.cpu().numpy()
                mask = dp.encode_arch_to_mask(structure_list)[0]
                rewards = mlp_mg.mlp_pred_acc(mask)

                path = args.log_output_dir + '_predictor' + '.txt'
                trnr.submodel_manager.record_action_info(path, structure_list[0], rewards)
                print('pred_acc:{0}'.format(rewards))

                if args.entropy_mode == 'reward':
                    rewards = rewards + args.entropy_coeff * np_entropies
                elif args.entropy_mode == 'regularizer':
                    rewards = rewards * np.ones_like(np_entropies)
                else:
                    raise NotImplementedError(f'Unkown entropy mode: {args.entropy_mode}')

                torch.cuda.empty_cache()

                trnr.collect_trajectory(rewards, log_probs, actions_index)
                print(20 * '-', 'episodes: ', trnr.buf.get_buf_size(), 20 * '-')

                if trnr.buf.get_buf_size() % args.episodes == 0:
                    trnr.train_controller_by_ppo2()
        else:
            flag = 1
            for i in range(args.select_samples):

                while True:
                    structure_list, log_probs, entropies, actions_index = trnr.controller_sample()
                    tmp_action = copy.deepcopy(structure_list[0])
                    tmp_action[-1] = trnr.submodel_manager.n_classes

                    if str(tmp_action) not in trnr.total_samples:
                        break
                    else:
                        print('Arise a repeated arch', tmp_action)

                trnr.total_samples.append(str(tmp_action))
                mask = dp.encode_arch_to_mask(structure_list)[0]
                # calculate reward
                np_entropies = entropies.data.cpu().numpy()
                rewards, mask_samples[json.dumps(mask)] = trnr.get_reward(structure_list, np_entropies)
                torch.cuda.empty_cache()

                trnr.collect_trajectory(rewards, log_probs, actions_index)
                print(20 * '-', 'episodes: ', trnr.buf.get_buf_size(), 20 * '-')

                if trnr.buf.get_buf_size() % args.episodes == 0:
                    trnr.train_controller_by_ppo2()


def pg(args, trnr, arch_train_info_trace):
    for i in range(args.search_samples):
        print(20 * '-', 'RL-', i, 20 * '-')

        while True:
            individual = trnr._generate_random_individual()
            tmp_action = copy.deepcopy(individual)
            tmp_action[-1] = trnr.submodel_manager.n_classes
            if str(tmp_action) not in trnr.total_samples:
                break
            else:
                print('Arise a repeated arch', tmp_action)

        trnr.total_samples.append(str(tmp_action))

        ind_actions = trnr._construct_action([individual])
        gnn = trnr.form_gnn_info(ind_actions[0])

        start_time = time.time()
        _, ind_acc = \
            trnr.submodel_manager.train(gnn, format=args.format)
        end_time = time.time()
        arch_train_info_trace.append((gnn, ind_acc, end_time - start_time))

        print("individual:", individual, " val_score:", ind_acc)


