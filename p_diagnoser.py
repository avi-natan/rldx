import copy
import platform
import time

import numpy as np
import gym

from h_consts import DETERMINISTIC
from h_rl_models import models


def diagnose_deterministic_faults_full_obs_wfm(env_name, render_mode, ml_model_name, total_timesteps, lst_actions, lst_states, available_fault_models, instance_seed):
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
        a_i_gag, _ = model.predict(lst_states[i-1], deterministic=DETERMINISTIC)
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

def diagnose_deterministic_faults_full_obs_sfm(env_name, render_mode, ml_model_name, total_timesteps, lst_actions, lst_states, available_fault_models, instance_seed):
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
    for key in available_fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [available_fault_models[key], fm_env, [fm_s_0]]

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
        "traj": determined_trajectories,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output

def diagnose_deterministic_faults_part_obs_wfm(env_name, render_mode, ml_model_name, total_timesteps, lst_actions, lst_states, available_fault_models, instance_seed):
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

def diagnose_deterministic_faults_part_obs_sfm(env_name, render_mode, ml_model_name, total_timesteps, lst_actions, lst_states, available_fault_models, instance_seed):
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
    for key in available_fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [available_fault_models[key], fm_env, [fm_s_0]]

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
            s_trajectory_execution = fault_model_trajectories[fm][2][i*2]
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
        "traj": determined_trajectories,
        "runtime_sec": elapsed_time_sec,
        "runtime_ms": elapsed_time_ms
    }
    return output

def sfm_stofm_fobs_brute(env_name, render_mode, ml_model_name, total_timesteps, lst_actions, lst_states, available_fault_models, instance_seed):
    # welcome message
    print(f'diagnosing with diagnoser: sfm_stofm_fobs_brute\n========================================================================================')
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
    for key in available_fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
        fm_initial_obs, _ = fm_env.reset(seed=instance_seed)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [available_fault_models[key], fm_env, [fm_s_0], []]

    start_time = time.time()
    # running the diagnosis loop
    for i in range(1, len(lst_states)):
        print(i)
        a_gag_i, _ = model.predict(lst_states[i - 1], deterministic=DETERMINISTIC)
        irrelevant_keys = []
        for fm_j, fm in enumerate(fault_model_trajectories.keys()):
            # print(f'running {i}/{len(lst_states)}, fm {fm} ({fm_j}/{len(fault_model_trajectories.keys())})')
            a_gag_ij = fault_model_trajectories[fm][0](a_gag_i)

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

            a_gag_i_s, reward, done, trunc, info = a_gag_i_env.step(a_gag_i)
            a_gag_ij_s, reward, done, trunc, info = a_gag_ij_env.step(a_gag_ij)

            if np.array_equal(a_gag_i_s, lst_states[i]) and np.array_equal(a_gag_ij_s, lst_states[i]):
                fault_model_trajectories[fm][2].append(a_gag_i)
                fault_model_trajectories[fm][2].append(a_gag_i_s)
            elif np.array_equal(a_gag_i_s, lst_states[i]) and not np.array_equal(a_gag_ij_s, lst_states[i]):
                fault_model_trajectories[fm][2].append(a_gag_i)
                fault_model_trajectories[fm][2].append(a_gag_i_s)
            elif not np.array_equal(a_gag_i_s, lst_states[i]) and not np.array_equal(a_gag_ij_s, lst_states[i]):
                irrelevant_keys.append(fm)
            elif not np.array_equal(a_gag_i_s, lst_states[i]) and np.array_equal(a_gag_ij_s, lst_states[i]):
                fault_model_trajectories[fm][2].append(a_gag_ij)
                fault_model_trajectories[fm][2].append(a_gag_ij_s)
                fault_model_trajectories[fm][3].append(i)

        for fm in irrelevant_keys:
            fault_model_trajectories.pop(fm)

    # finilizing the output
    determined_fms = []
    for key in fault_model_trajectories.keys():
        determined_fms.append(key)
    determined_trajectories = []
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
        determined_points_of_failure.append(fault_model_trajectories[determined_fm][3])
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    elapsed_time_ms = elapsed_time_sec * 1000
    output = {
        "fm": determined_fms,
        "traj": determined_trajectories,
        "pof": determined_points_of_failure,
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
    "sfm_stofm_fobs_brute": sfm_stofm_fobs_brute
}
