import numpy as np
import gym

from h_common import os_compatible_render_mode
from h_consts import SEED, DETERMINISTIC
from h_rl_models import models


def diagnose_deterministic_faults_full_obs_wfm(env_name, model_name, total_timesteps, lst_actions, lst_states, available_fault_models):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_full_obs_wfm\n========================================================================================\n')
    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=os_compatible_render_mode())
    initial_obs, _ = env.reset(seed=SEED)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # determined faulty action
    determined_faulty_action_index = -1

    # running the diagnosis loop
    for i in range(1, len(lst_states)):
        print(f'{i}/{len(lst_states)}')
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
    output = {
        "i": determined_faulty_action_index,
        "a_i": lst_actions[determined_faulty_action_index - 1]
    }
    return output

def diagnose_deterministic_faults_full_obs_sfm(env_name, model_name, total_timesteps, lst_actions, lst_states, available_fault_models):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_full_obs_sfm\n========================================================================================\n')
    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=os_compatible_render_mode())
    initial_obs, _ = env.reset(seed=SEED)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # preparing the environments for each of the fault models
    fault_model_trajectories = {}
    for key in available_fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=os_compatible_render_mode())
        fm_initial_obs, _ = fm_env.reset(seed=SEED)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [available_fault_models[key], fm_env, [fm_s_0]]

    # running the diagnosis loop
    for i in range(1, len(lst_states)):
        a_i_gag, _ = model.predict(lst_states[i - 1], deterministic=DETERMINISTIC)
        irrelevant_keys = []
        for fm_i, fm in enumerate(fault_model_trajectories.keys()):
            print(f'running {i}/{len(lst_states)}, fm {fm} ({fm_i}/{len(fault_model_trajectories.keys())})')
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
    determined_fm = ""
    for key in fault_model_trajectories.keys():
        determined_fm = key
    determined_trajectory = []
    for i in fault_model_trajectories[determined_fm][2]:
        if isinstance(i, int):
            determined_trajectory.append(i)
        else:
            determined_trajectory.append(list(i))
    output = {
        "fm": determined_fm,
        "traj": determined_trajectory
    }
    return output

def diagnose_deterministic_faults_part_obs_wfm(env_name, model_name, total_timesteps, lst_actions, lst_states, available_fault_models):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_part_obs_wfm\n========================================================================================\n')

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=os_compatible_render_mode())
    initial_obs, _ = env.reset(seed=SEED)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # determined range where fault occured
    fault_occurence_range = [0, None]

    # running the diagnosis loop
    s_i_gag = lst_states[0]
    for i in range(1, len(lst_states)):
        print(f'{i}/{len(lst_states)}')
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

    output = {
        "fault_occurence_range": fault_occurence_range
    }
    return output

def diagnose_deterministic_faults_part_obs_sfm(env_name, model_name, total_timesteps, lst_actions, lst_states, available_fault_models):
    # welcome message
    print(f'diagnosing with diagnoser: diagnose_deterministic_faults_part_obs_sfm\n========================================================================================\n')

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=os_compatible_render_mode())
    initial_obs, _ = env.reset(seed=SEED)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[model_name].load(model_path, env=env)

    # make sure the first state is the same in obs and simulation
    s_0, _ = env.reset()
    assert s_0.all() == lst_states[0].all()

    # preparing the environments for each of the fault models
    fault_model_trajectories = {}
    for key in available_fault_models.keys():
        fm_env = gym.make(env_name.replace('_', '-'), render_mode=os_compatible_render_mode())
        fm_initial_obs, _ = fm_env.reset(seed=SEED)
        fm_s_0, _ = fm_env.reset()
        fault_model_trajectories[key] = [available_fault_models[key], fm_env, [fm_s_0]]

    # running the diagnosis loop
    for fm_i, fm in enumerate(fault_model_trajectories.keys()):
        s_fm = lst_states[0]
        for i in range(1, len(lst_states)):
            print(f'running fm {fm} ({fm_i}/{len(fault_model_trajectories.keys())}), {i}/{len(lst_states)}')
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
            print(f'finilizing fm {fm} ({fm_i}/{len(fault_model_trajectories.keys())}), {i}/{len(lst_states)}')
            s_lst_states = lst_states[i]
            s_trajectory_execution = fault_model_trajectories[fm][2][i*2]
            a = s_lst_states is not None
            b = s_trajectory_execution is not None
            c = not np.array_equal(s_lst_states, s_trajectory_execution)
            if a and b and c:
                equal = False
                break
        equality_dict[fm] = equal
    determined_fm = ""
    for key in equality_dict.keys():
        if equality_dict[key]:
            determined_fm = key
    determined_trajectory = []
    for i in fault_model_trajectories[determined_fm][2]:
        if isinstance(i, int):
            determined_trajectory.append(i)
        else:
            determined_trajectory.append(list(i))
    output = {
        "fm": determined_fm,
        "traj": determined_trajectory
    }
    return output


diagnosers = {
    # abstract fault models (suitable for all environments)
    "diagnose_deterministic_faults_full_obs_wfm": diagnose_deterministic_faults_full_obs_wfm,
    "diagnose_deterministic_faults_full_obs_sfm": diagnose_deterministic_faults_full_obs_sfm,
    "diagnose_deterministic_faults_part_obs_wfm": diagnose_deterministic_faults_part_obs_wfm,
    "diagnose_deterministic_faults_part_obs_sfm": diagnose_deterministic_faults_part_obs_sfm
}
