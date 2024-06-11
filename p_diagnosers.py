import time

import numpy as np
import gym

from h_consts import DETERMINISTIC
from h_rl_models import models

def W(debug_print, render_mode, instance_seed, ml_model_name, total_timesteps, domain_name, observations, candidate_fault_modes):
    # welcome message
    print(f'diagnosing with diagnoser: W\n========================================================================================')

    # initialize simulator
    simulator = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    policy = models[ml_model_name].load(model_path)

    # make sure the first state is the same in obs and simulation
    s_0, _ = simulator.reset()
    assert s_0.all() == observations[0].all()

    diagnosis_start_time = time.time()
    b = 0
    e = len(observations) - 1
    S = observations[0]
    for i in range(1, len(observations)):
        a, _ = policy.predict(S, deterministic=DETERMINISTIC)
        S, reward, done, trunc, info = simulator.step(a)
        if observations[i] is not None:
            if np.array_equal(observations[i], S):
                b = i
            else:
                e = i
                if debug_print:
                    print(f"i broke at {i}")
                break
    D = []
    for i in range(b + 1, e + 1):
        D.append(i)
    diagnosis_end_time = time.time()
    diagnosis_runtime_sec = diagnosis_end_time - diagnosis_start_time
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    output = {
        "diagnoses": D,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms
    }

    return output


def SN(debug_print, render_mode, instance_seed, ml_model_name, total_timesteps, domain_name, observations, candidate_fault_modes):
    # welcome message
    print(f'diagnosing with diagnoser: SN\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    policy = models[ml_model_name].load(model_path)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        T_j = [observations[0]]
        simulator_j = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
        initial_obs_j, _ = simulator_j.reset(seed=instance_seed)
        S_0_j, _ = simulator_j.reset()
        assert np.array_equal(T_j[0], S_0_j)
        G[key_j] = [candidate_fault_modes[key_j], simulator_j, T_j]
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    # running the diagnosis loop
    for i in range(1, len(observations)):
        irrelevant_keys = []
        for key_j in G.keys():
            ts1 = time.time()
            a_gag_i, _ = policy.predict(G[key_j][2][-1], deterministic=DETERMINISTIC)
            a_gag_i_j = G[key_j][0](a_gag_i)
            S_gag_i_j, reward, done, trunc, info = G[key_j][1].step(a_gag_i_j)
            G[key_j][2].append(int(a_gag_i_j))
            G[key_j][2].append(S_gag_i_j)
            if observations[i] is not None:
                if not np.array_equal(observations[i], S_gag_i_j):
                    irrelevant_keys.append(key_j)
            te1 = time.time()
            diagnosis_runtime_sec += te1 - ts1

        # remove the irrelevant fault modes
        ts2 = time.time()
        for key in irrelevant_keys:
            G.pop(key)
        te2 = time.time()
        diagnosis_runtime_sec += te2 - ts2

        if debug_print:
            print(f'STEP {i}/{len(observations)}: KICKED {len(irrelevant_keys)} ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break

    # finilizing the runtime in ms
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms
    }

    return raw_output


def SIF(debug_print, render_mode, instance_seed, ml_model_name, total_timesteps, domain_name, observations, candidate_fault_modes):
    # welcome message
    print(f'diagnosing with diagnoser: SIF\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    policy = models[ml_model_name].load(model_path)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        T_j = [observations[0]]
        simulator_j = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
        initial_obs_j, _ = simulator_j.reset(seed=instance_seed)
        S_0_j, _ = simulator_j.reset()
        assert np.array_equal(T_j[0], S_0_j)
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], simulator_j, T_j]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    for i in range(1, len(observations)):
        irrelevant_keys = []
        new_relevant_keys = {}
        for key_j in G.keys():
            ts1 = time.time()
            a_gag_i, _ = policy.predict(G[key_j][2][-1], deterministic=DETERMINISTIC)
            a_gag_i_j = G[key_j][0](a_gag_i)
            te1 = time.time()
            diagnosis_runtime_sec += te1 - ts1

            # reconstruct the states
            simulator_j_to_normal = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
            initial_obs_j_to_normal, _ = simulator_j_to_normal.reset(seed=instance_seed)
            S_0_j_to_normal, _ = simulator_j_to_normal.reset()
            assert np.array_equal(G[key_j][2][0], S_0_j_to_normal)
            S_to_normal = S_0_j_to_normal
            for i_rec in range(1, len(G[key_j][2]), 2):
                a = G[key_j][2][i_rec]
                S_to_normal, reward, done, trunc, info = simulator_j_to_normal.step(a)
            simulator_j_to_fault = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
            initial_obs_j_to_fault, _ = simulator_j_to_fault.reset(seed=instance_seed)
            S_0_j_to_fault, _ = simulator_j_to_fault.reset()
            assert np.array_equal(G[key_j][2][0], S_0_j_to_fault)
            T_j_to_fault = [S_0_j_to_fault]
            S_to_fault = S_0_j_to_fault
            for i_rec in range(1, len(G[key_j][2]), 2):
                a = G[key_j][2][i_rec]
                S_to_fault, reward, done, trunc, info = simulator_j_to_fault.step(a)
                T_j_to_fault.append(a)
                T_j_to_fault.append(S_to_fault)

            # apply the normal and the faulty action on the reconstructed states, respectively
            ts2 = time.time()
            S_gag_i, reward, done, trunc, info = simulator_j_to_normal.step(a_gag_i)
            S_gag_i_j, reward, done, trunc, info = simulator_j_to_fault.step(a_gag_i_j)
            if observations[i] is not None:
                # the case where there is an observation that can be checked
                if np.array_equal(S_gag_i, observations[i]) and np.array_equal(S_gag_i_j, observations[i]):
                    # a_gag_i not changed, f_j cannot change a_gag_i
                    G[key_j][2].append(int(a_gag_i))
                    G[key_j][2].append(S_gag_i)
                elif np.array_equal(S_gag_i, observations[i]) and not np.array_equal(S_gag_i_j, observations[i]):
                    # a_gag_i not changed, f_j can    change a_gag_i
                    G[key_j][2].append(int(a_gag_i))
                    G[key_j][2].append(S_gag_i)
                elif not np.array_equal(S_gag_i, observations[i]) and not np.array_equal(S_gag_i_j, observations[i]):
                    # a_gag_i     changed, f_j cannot change a_gag_i
                    irrelevant_keys.append(key_j)
                elif not np.array_equal(S_gag_i, observations[i]) and np.array_equal(S_gag_i_j, observations[i]):
                    # a_gag_i     changed, f_j can    change a_gag_i
                    G[key_j][2].append(a_gag_i_j)
                    G[key_j][2].append(S_gag_i_j)
            else:
                # the case where there is no observation to be checked - insert the normal action and state to the original key
                G[key_j][2].append(int(a_gag_i))
                G[key_j][2].append(S_gag_i)
                if a_gag_i != a_gag_i_j:
                    # if the action was changed - create new trajectory and insert it as well
                    T_j_to_fault.append(a_gag_i_j)
                    T_j_to_fault.append(S_gag_i_j)
                    k_j = key_j.split('_')[0]
                    new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j], simulator_j_to_fault, T_j_to_fault]
                    I[k_j] = I[k_j] + 1
            te2 = time.time()
            diagnosis_runtime_sec += te2 - ts2

        # add new relevant fault modes
        ts3 = time.time()
        for key in new_relevant_keys:
            G[key] = new_relevant_keys[key]
        # remove the irrelevant fault modes
        for key in irrelevant_keys:
            G.pop(key)
        te3 = time.time()
        diagnosis_runtime_sec += te3 - ts3

        if debug_print:
            print(f'STEP {i}/{len(observations)}: ADDED  {len(new_relevant_keys)} ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
            print(f'STEP {i}/{len(observations)}: KICKED {len(irrelevant_keys)} ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break

    # finilizing the runtime in ms
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms
    }

    return raw_output


diagnosers = {
    # new fault models
    "W": W,
    "SN": SN,
    "SIF": SIF
}
