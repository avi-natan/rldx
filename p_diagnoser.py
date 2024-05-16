import copy
import platform
import random
import time

import numpy as np
import gym

from h_consts import DETERMINISTIC
from h_rl_models import models


def diagnose_deterministic_faults_full_obs_wfm(env_name,
                                               render_mode,
                                               ml_model_name,
                                               total_timesteps,
                                               fault_model,
                                               lst_actions,
                                               lst_states,
                                               fault_probability,
                                               instance_seed,
                                               fault_models,
                                               sample_size):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_full_obs_wfm\n========================================================================================')
    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # determined faulty action
    determined_faulty_action_index = -1

    start_time = time.time()
    # running the diagnosis loop
    for i in range(1, len(lst_states)):
        # print(f'{i}/{len(lst_states)}')
        a_i_gag, _ = model.predict(lst_states[i - 1], deterministic=DETERMINISTIC)
        # a_i = lst_actions[i-1]
        # if a_i == a_i_gag:
        #     print(f'[ACTION SAME] a_{i}: {a_i}, a_{i}_gag: {a_i_gag}')
        # else:
        #     print(f'[ACTION DIFF] a_{i}: {a_i}, a_{i}_gag: {a_i_gag}')
        s_i_gag, reward, done, trunc, info = env.step(a_i_gag)
        if not np.array_equal(lst_states[i], s_i_gag):
            # print(f'[STATE  DIFF] s_{i}: {list(lst_states[i])}, s_{i}_gag: {list(s_i_gag)}')
            determined_faulty_action_index = i
            break
        # else:
        #     print(f'[STATE  SAME] s_{i}: {list(lst_states[i])}, s_{i}_gag: {list(s_i_gag)}')
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_ms = elapsed_time_sec * 1000
    output = {
        "i": determined_faulty_action_index,
        "a_i": lst_actions[determined_faulty_action_index - 1],
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output


def diagnose_deterministic_faults_full_obs_sfm(env_name,
                                               render_mode,
                                               ml_model_name,
                                               total_timesteps,
                                               fault_model,
                                               lst_actions,
                                               lst_states,
                                               fault_probability,
                                               instance_seed,
                                               fault_models,
                                               sample_size):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_full_obs_sfm\n========================================================================================')
    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # preparing the environments for each of the fault models
    fault_model_trajectories = {}
    for key in fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [fault_models[key], fm_env, [fm_s_0]]

    start_time = time.time()
    # running the diagnosis loop
    for i in range(1, len(lst_states)):
        a_i_gag, _ = model.predict(lst_states[i - 1], deterministic=DETERMINISTIC)
        irrelevant_keys = []
        for fm_i, fm in enumerate(fault_model_trajectories.keys()):
            # print(f'running {i}/{len(lst_states)}, fm {fm} ({fm_i}/{len(fault_model_trajectories.keys())})')
            a_i_gag_fm = fault_model_trajectories[fm][0](a_i_gag)
            fault_model_trajectories[fm][2].append(a_i_gag_fm)
            s_i_gag_fm, reward, done, trunc, info = fault_model_trajectories[fm][1].step(a_i_gag_fm)
            if not np.array_equal(lst_states[i], s_i_gag_fm):
                irrelevant_keys.append(fm)
            else:
                fault_model_trajectories[fm][2].append(s_i_gag_fm)
        for fm in irrelevant_keys:
            fault_model_trajectories.pop(fm)

    # finilizing the output
    determined_fms = []
    for key in fault_model_trajectories.keys():
        determined_fms.append(key)
    determined_trajectories = []
    for determined_fm in determined_fms:
        determined_trajectory = []
        for i in fault_model_trajectories[determined_fm][2]:
            if isinstance(i, int):
                determined_trajectory.append(i)
            else:
                determined_trajectory.append(list(i))
        determined_trajectories.append(determined_trajectory)
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_ms = elapsed_time_sec * 1000
    output = {
        "fm": determined_fms,
        # "traj": determined_trajectories,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output


def diagnose_deterministic_faults_part_obs_wfm(env_name,
                                               render_mode,
                                               ml_model_name,
                                               total_timesteps,
                                               fault_model,
                                               lst_actions,
                                               lst_states,
                                               fault_probability,
                                               instance_seed,
                                               fault_models,
                                               sample_size):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_part_obs_wfm\n========================================================================================')

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # determined range where fault occured
    fault_occurence_range = [0, None]

    start_time = time.time()
    # running the diagnosis loop
    s_i_gag = lst_states[0]
    for i in range(1, len(lst_states)):
        # print(f'{i}/{len(lst_states)}')
        a_i_gag, _ = model.predict(s_i_gag, deterministic=DETERMINISTIC)
        # a_i = lst_actions[i-1]
        # if a_i == a_i_gag:
        #     print(f'[ACTION SAME] a_{i}: {a_i}, a_{i}_gag: {a_i_gag}')
        # else:
        #     print(f'[ACTION DIFF] a_{i}: {a_i}, a_{i}_gag: {a_i_gag}')
        s_i_gag, reward, done, trunc, info = env.step(a_i_gag)
        if lst_states[i] is not None:
            if np.array_equal(lst_states[i], s_i_gag):
                # print(f'[STATE  DIFF] s_{i}: {list(lst_states[i])}, s_{i}_gag: {list(s_i_gag)}')
                fault_occurence_range[0] = i
            else:
                #     print(f'[STATE  SAME] s_{i}: {list(lst_states[i])}, s_{i}_gag: {list(s_i_gag)}')
                fault_occurence_range[1] = i
                break
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_ms = elapsed_time_sec * 1000
    output = {
        "fault_occurence_range": fault_occurence_range,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output


def diagnose_deterministic_faults_part_obs_sfm(env_name,
                                               render_mode,
                                               ml_model_name,
                                               total_timesteps,
                                               fault_model,
                                               lst_actions,
                                               lst_states,
                                               fault_probability,
                                               instance_seed,
                                               fault_models,
                                               sample_size):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_part_obs_sfm\n========================================================================================')

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # preparing the environments for each of the fault models
    fault_model_trajectories = {}
    for key in fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [fault_models[key], fm_env, [fm_s_0]]

    start_time = time.time()
    # running the diagnosis loop
    for fm_i, fm in enumerate(fault_model_trajectories.keys()):
        s_fm = lst_states[0]
        for i in range(1, len(lst_states)):
            # print(f'running fm {fm} ({fm_i}/{len(fault_model_trajectories.keys())}), {i}/{len(lst_states)}')
            a_i_gag, _ = model.predict(s_fm, deterministic=DETERMINISTIC)
            a_i_gag_fm = fault_model_trajectories[fm][0](a_i_gag)
            fault_model_trajectories[fm][2].append(a_i_gag_fm)
            s_fm, reward, done, trunc, info = fault_model_trajectories[fm][1].step(a_i_gag_fm)
            fault_model_trajectories[fm][2].append(s_fm)

    # finilizing the output
    equality_dict = {}
    for fm_i, fm in enumerate(fault_model_trajectories.keys()):
        equal = True
        for i in range(len(lst_states)):
            # print(f'finilizing fm {fm} ({fm_i}/{len(fault_model_trajectories.keys())}), {i}/{len(lst_states)}')
            s_lst_states = lst_states[i]
            s_trajectory_execution = fault_model_trajectories[fm][2][i * 2]
            a = s_lst_states is not None
            b = s_trajectory_execution is not None
            c = not np.array_equal(s_lst_states, s_trajectory_execution)
            if a and b and c:
                equal = False
                break
        equality_dict[fm] = equal
    determined_fms = []
    for key in equality_dict.keys():
        if equality_dict[key]:
            determined_fms.append(key)
    determined_trajectories = []
    for determined_fm in determined_fms:
        determined_trajectory = []
        for i in fault_model_trajectories[determined_fm][2]:
            if isinstance(i, int):
                determined_trajectory.append(i)
            else:
                determined_trajectory.append(list(i))
        determined_trajectories.append(determined_trajectory)
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_ms = elapsed_time_sec * 1000
    output = {
        "fm": determined_fms,
        # "traj": determined_trajectories,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output


def sfm_stofm_fobs_partical(env_name,
                            render_mode,
                            ml_model_name,
                            total_timesteps,
                            fault_model,
                            lst_actions,
                            lst_states,
                            fault_probability,
                            instance_seed,
                            fault_models,
                            sample_size):
    # welcome message
    print(f'diagnosing with diagnoser: sfm_stofm_fobs_partical\n========================================================================================')
    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # preparing the environments for each of the fault models
    fault_model_trajectories = {}
    for key in fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [fault_models[key], fm_env, [fm_s_0], [], []]

    total_time = 0.0
    # running the diagnosis loop
    for i in range(1, len(lst_states)):
        # print(i)
        ts0 = time.time()
        a_gag_i, _ = model.predict(lst_states[i - 1], deterministic=DETERMINISTIC)
        irrelevant_keys = []
        te0 = time.time()
        total_time += te0 - ts0
        for fm_j, fm in enumerate(fault_model_trajectories.keys()):
            ts1 = time.time()
            # print(f'running {i}/{len(lst_states)}, fm {fm} ({fm_j}/{len(fault_model_trajectories.keys())})')
            a_gag_ij = fault_model_trajectories[fm][0](a_gag_i)
            te1 = time.time()
            total_time += te1 - ts1

            # reconstruct the states
            a_gag_i_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
            a_gag_i_initial_obs, _ = a_gag_i_env.reset(seed=instance_seed)
            a_gag_i_s_0, _ = a_gag_i_env.reset()
            a_gag_ij_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
            a_gag_ij_initial_obs, _ = a_gag_ij_env.reset(seed=instance_seed)
            a_gag_ij_s_0, _ = a_gag_ij_env.reset()
            a_gag_i_s = a_gag_i_s_0
            for i_rec in range(1, len(fault_model_trajectories[fm][2]), 2):
                a = fault_model_trajectories[fm][2][i_rec]
                a_gag_i_s, reward, done, trunc, info = a_gag_i_env.step(a)
            a_gag_ij_s = a_gag_ij_s_0
            for i_rec in range(1, len(fault_model_trajectories[fm][2]), 2):
                a = fault_model_trajectories[fm][2][i_rec]
                a_gag_ij_s, reward, done, trunc, info = a_gag_ij_env.step(a)

            ts2 = time.time()
            a_gag_i_s, reward, done, trunc, info = a_gag_i_env.step(a_gag_i)
            a_gag_ij_s, reward, done, trunc, info = a_gag_ij_env.step(a_gag_ij)

            if np.array_equal(a_gag_i_s, lst_states[i]) and np.array_equal(a_gag_ij_s, lst_states[i]):
                fault_model_trajectories[fm][2].append(a_gag_i)
                fault_model_trajectories[fm][2].append(a_gag_i_s)
                fault_model_trajectories[fm][3].append(1)
            elif np.array_equal(a_gag_i_s, lst_states[i]) and not np.array_equal(a_gag_ij_s, lst_states[i]):
                fault_model_trajectories[fm][2].append(a_gag_i)
                fault_model_trajectories[fm][2].append(a_gag_i_s)
                fault_model_trajectories[fm][3].append(1 - fault_probability)
            elif not np.array_equal(a_gag_i_s, lst_states[i]) and not np.array_equal(a_gag_ij_s, lst_states[i]):
                irrelevant_keys.append(fm)
            elif not np.array_equal(a_gag_i_s, lst_states[i]) and np.array_equal(a_gag_ij_s, lst_states[i]):
                fault_model_trajectories[fm][2].append(a_gag_ij)
                fault_model_trajectories[fm][2].append(a_gag_ij_s)
                fault_model_trajectories[fm][3].append(fault_probability)
                fault_model_trajectories[fm][4].append(i)
            te2 = time.time()
            total_time += te2 - ts2

        ts3 = time.time()
        for fm in irrelevant_keys:
            fault_model_trajectories.pop(fm)
        te3 = time.time()
        total_time += te3 - ts3

    # finilizing the output
    ts4 = time.time()
    determined_fms = []
    for key in fault_model_trajectories.keys():
        determined_fms.append(key)
    determined_trajectories = []
    determined_fault_probabilities = []
    determined_fault_probabilities_products = []
    determined_points_of_failure = []
    for determined_fm in determined_fms:
        determined_trajectory = []
        for i in fault_model_trajectories[determined_fm][2]:
            if isinstance(i, int):
                determined_trajectory.append(i)
            else:
                if i.ndim == 0:
                    determined_trajectory.append(int(i))
                else:
                    determined_trajectory.append(list(i))
        determined_trajectories.append(determined_trajectory)
        determined_fault_probabilities.append(fault_model_trajectories[determined_fm][3])
        determined_fault_probabilities_products.append(np.prod(fault_model_trajectories[determined_fm][3]))
        determined_points_of_failure.append(fault_model_trajectories[determined_fm][4])
    te4 = time.time()
    total_time += te4 - ts4
    elapsed_time_sec = total_time
    elapsed_time_ms = elapsed_time_sec * 1000

    zipped_lists = zip(determined_fms, determined_fault_probabilities, determined_points_of_failure, determined_fault_probabilities_products)
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[-1], reverse=True)
    determined_fms, determined_fault_probabilities, determined_points_of_failure, determined_fault_probabilities_products = zip(*sorted_zipped_lists)

    try:
        fm_rank = determined_fms.index(fault_model)
    except ValueError:
        fm_rank = len(fault_models)
    output = {
        "fm": determined_fms,
        # "traj": determined_trajectories,
        # "fp": determined_fault_probabilities,
        # "fp_prod": determined_fault_probabilities_products,
        # "pof": determined_points_of_failure,
        "fm_rank": fm_rank,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output


def sfm_stofm_fobs_sample(env_name,
                          render_mode,
                          ml_model_name,
                          total_timesteps,
                          fault_model,
                          lst_actions,
                          lst_states,
                          fault_probability,
                          instance_seed,
                          fault_models,
                          sample_size):
    # welcome message
    print(f'diagnosing with diagnoser: sfm_stofm_fobs_sample\n========================================================================================')
    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # preparing the environments for each of the fault models
    fault_model_trajectories = {}
    for key in fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [fault_models[key], fm_env, [fm_s_0], []]

    start_time = time.time()
    # running the diagnosis loop
    Tj_gags = {}
    for fm_j, fm in enumerate(fault_models.keys()):
        Tj_gag = []
        for i in range(sample_size):
            # print(f'{fm_j}/{i}')
            fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
            fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
            fm_s_0, _ = fm_env.reset()
            t_j_i = [fm_s_0]
            S_r_minus_1 = fm_s_0
            for r in range(1, len(lst_states)):
                if random.random() < fault_probability:
                    a_r, _ = model.predict(S_r_minus_1, deterministic=DETERMINISTIC)
                    a_r = fault_models[fm](a_r)
                else:
                    a_r, _ = model.predict(S_r_minus_1, deterministic=DETERMINISTIC)
                t_j_i.append(a_r)
                S_r, reward, done, trunc, info = fm_env.step(a_r)
                t_j_i.append(S_r)
                S_r_minus_1 = S_r
            Tj_gag.append(t_j_i)
        Tj_gags[fm] = Tj_gag
    simis = {}
    lst_states_flat = [s.flatten() for s in lst_states]
    for fm_j, fm in enumerate(fault_models.keys()):
        simis_j = []
        for i in range(sample_size):
            t_j_i_states = [s for s_i, s in enumerate(Tj_gags[fm][i]) if s_i % 2 == 0]
            t_j_i_states_flat = [s.flatten() for s in t_j_i_states]
            # similarity
            simi_j_i = 0.0
            for r in range(len(t_j_i_states_flat)):
                dot_st = sum(a * b for a, b in zip(lst_states_flat[r], t_j_i_states_flat[r]))
                dot_ss = sum(a * b for a, b in zip(lst_states_flat[r], lst_states_flat[r]))
                dot_tt = sum(a * b for a, b in zip(t_j_i_states_flat[r], t_j_i_states_flat[r]))
                simi_j_i_r = dot_st / ((dot_ss ** .5) * (dot_tt ** .5))
                simi_j_i += simi_j_i_r
            simis_j.append(simi_j_i)
        simis[fm] = simis_j
    scores = {}
    for fm_j, fm in enumerate(fault_models.keys()):
        score_j = sum(simis[fm]) / len(simis[fm])
        scores[fm] = score_j
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_ms = elapsed_time_sec * 1000

    # finilizing the output
    sorted_zipped_lists = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    determined_fms, ranks = zip(*sorted_zipped_lists)

    try:
        fm_rank = determined_fms.index(fault_model)
    except ValueError:
        fm_rank = len(determined_fms)
    output = {
        "fm": determined_fms,
        "fm_rank": fm_rank,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }

    return output


diagnosers = {
    # abstract fault models (suitable for all environments)
    "diagnose_deterministic_faults_full_obs_wfm": diagnose_deterministic_faults_full_obs_wfm,
    "diagnose_deterministic_faults_full_obs_sfm": diagnose_deterministic_faults_full_obs_sfm,
    "diagnose_deterministic_faults_part_obs_wfm": diagnose_deterministic_faults_part_obs_wfm,
    "diagnose_deterministic_faults_part_obs_sfm": diagnose_deterministic_faults_part_obs_sfm,
    "sfm_stofm_fobs_partical": sfm_stofm_fobs_partical,
    "sfm_stofm_fobs_sample": sfm_stofm_fobs_sample
}
