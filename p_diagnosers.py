import copy
import time
import tracemalloc

import numpy as np
import gym

from h_consts import DETERMINISTIC
from h_raw_state_comparators import comparators
from h_rl_models import models
from h_state_refiners import refiners
from h_wrappers import wrappers


def W(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # starting the monitoring
    tracemalloc.start()
    exp_rt_begin = time.time()
    # welcome message
    print(f'diagnosing with diagnoser: W\n========================================================================================')

    # initialize simulator
    simulator = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = simulator.reset(seed=instance_seed)

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # make sure the first state is the same in obs and simulation
    s_0, _ = simulator.reset()
    assert np.array_equal(s_0, observations[0])

    diagnosis_start_time = time.time()
    b = 0
    e = len(observations) - 1
    S = observations[0]
    for i in range(1, len(observations)):
        a, _ = policy.predict(S, deterministic=DETERMINISTIC)
        a = int(a)
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

    exp_rt_end = time.time()
    exp_duration_sec = exp_rt_end - exp_rt_begin
    exp_duration_ms = exp_duration_sec * 1000

    memory_tracked = tracemalloc.get_traced_memory()
    # stopping the library
    tracemalloc.stop()

    output = {
        "diagnoses": D,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "exp_duration_sec": exp_duration_sec,
        "exp_duration_ms": exp_duration_ms,
        "exp_memory_at_end": memory_tracked[0],
        "exp_memory_max": memory_tracked[1],
        "G_max_size":0
    }

    return output


def SN(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # starting the monitoring
    tracemalloc.start()
    exp_rt_begin = time.time()
    # welcome message
    print(f'diagnosing with diagnoser: SN\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        simulator_j = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
        initial_obs_j, _ = simulator_j.reset(seed=instance_seed)
        S_0_j, _ = simulator_j.reset()
        assert np.array_equal(observations[0], S_0_j)
        G[key_j] = [candidate_fault_modes[key_j], simulator_j, S_0_j, A_j, S_0_j]
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    # running the diagnosis loop
    for i in range(1, len(observations)):
        irrelevant_keys = []
        for key_j in G.keys():
            ts1 = time.time()
            a_gag_i, _ = policy.predict(G[key_j][4], deterministic=DETERMINISTIC)
            a_gag_i_j = G[key_j][0](a_gag_i)
            S_gag_i_j, reward, done, trunc, info = G[key_j][1].step(a_gag_i_j)
            G[key_j][3].append(int(a_gag_i_j))
            G[key_j][4] = S_gag_i_j
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

    exp_rt_end = time.time()
    exp_duration_sec = exp_rt_end - exp_rt_begin
    exp_duration_ms = exp_duration_sec * 1000

    memory_tracked = tracemalloc.get_traced_memory()
    # stopping the library
    tracemalloc.stop()

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "exp_duration_sec": exp_duration_sec,
        "exp_duration_ms": exp_duration_ms,
        "exp_memory_at_end": memory_tracked[0],
        "exp_memory_max": memory_tracked[1],
        "G_max_size": len(candidate_fault_modes)
    }

    return raw_output


def SIF(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # starting the monitoring
    tracemalloc.start()
    exp_rt_begin = time.time()
    # welcome message
    print(f'diagnosing with diagnoser: SIF\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        simulator_j = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
        initial_obs_j, _ = simulator_j.reset(seed=instance_seed)
        S_0_j, _ = simulator_j.reset()
        assert np.array_equal(observations[0], S_0_j)
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], simulator_j, S_0_j, A_j, S_0_j]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    for i in range(1, len(observations)):
        irrelevant_keys = []
        new_relevant_keys = {}
        for key_j in G.keys():
            ts1 = time.time()
            a_gag_i, _ = policy.predict(G[key_j][4], deterministic=DETERMINISTIC)
            a_gag_i = int(a_gag_i)
            a_gag_i_j = G[key_j][0](a_gag_i)
            te1 = time.time()
            diagnosis_runtime_sec += te1 - ts1

            # reconstruct the states
            simulator_j_to_normal = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
            initial_obs_j_to_normal, _ = simulator_j_to_normal.reset(seed=instance_seed)
            S_0_j_to_normal, _ = simulator_j_to_normal.reset()
            assert np.array_equal(observations[0], S_0_j_to_normal)
            S_to_normal = S_0_j_to_normal
            for i_rec in range(len(G[key_j][3])):
                a = G[key_j][3][i_rec]
                S_to_normal, reward, done, trunc, info = simulator_j_to_normal.step(a)
            simulator_j_to_fault = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
            initial_obs_j_to_fault, _ = simulator_j_to_fault.reset(seed=instance_seed)
            S_0_j_to_fault, _ = simulator_j_to_fault.reset()
            assert np.array_equal(observations[0], S_0_j_to_fault)
            A_j_to_fault = []
            S_to_fault = S_0_j_to_fault
            for i_rec in range(len(G[key_j][3])):
                a = G[key_j][3][i_rec]
                S_to_fault, reward, done, trunc, info = simulator_j_to_fault.step(a)
                A_j_to_fault.append(a)

            # apply the normal and the faulty action on the reconstructed states, respectively
            ts2 = time.time()
            S_gag_i, reward, done, trunc, info = simulator_j_to_normal.step(a_gag_i)
            S_gag_i_j, reward, done, trunc, info = simulator_j_to_fault.step(a_gag_i_j)
            if observations[i] is not None:
                # the case where there is an observation that can be checked
                S_gag_i_eq_S_i = np.array_equal(S_gag_i, observations[i])
                S_gag_i_j_eq_S_i = np.array_equal(S_gag_i_j, observations[i])
                if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][3].append(int(a_gag_i))
                    G[key_j][4] = S_gag_i
                elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][3].append(int(a_gag_i))
                    G[key_j][4] = S_gag_i
                elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    irrelevant_keys.append(key_j)
                elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][3].append(int(a_gag_i_j))
                    G[key_j][4] = S_gag_i_j
            else:
                # the case where there is no observation to be checked - insert the normal action and state to the original key
                if debug_print:
                    print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                G[key_j][3].append(int(a_gag_i))
                G[key_j][4] = S_gag_i
                if a_gag_i != a_gag_i_j:
                    # if the action was changed - create new trajectory and insert it as well
                    if debug_print:
                        print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    A_j_to_fault.append(a_gag_i_j)
                    k_j = key_j.split('_')[0]
                    new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j], simulator_j_to_fault, observations[0],  A_j_to_fault, S_gag_i_j]
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

        # update the maximum size of G
        G_max_size = max(G_max_size, len(G))

        if debug_print:
            if observations[i] is not None:
                print(f'STEP {i}/{len(observations)}: OBSERVED')
            else:
                print(f'STEP {i}/{len(observations)}: HIDDEN')
            print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
            print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
            print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break

    # finilizing the runtime in ms
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    exp_rt_end = time.time()
    exp_duration_sec = exp_rt_end - exp_rt_begin
    exp_duration_ms = exp_duration_sec * 1000

    memory_tracked = tracemalloc.get_traced_memory()
    # stopping the library
    tracemalloc.stop()

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "exp_duration_sec": exp_duration_sec,
        "exp_duration_ms": exp_duration_ms,
        "exp_memory_at_end": memory_tracked[0],
        "exp_memory_max": memory_tracked[1],
        "G_max_size": G_max_size
    }

    return raw_output


def W2(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # starting the monitoring
    tracemalloc.start()
    exp_rt_begin = time.time()
    # welcome message
    print(f'diagnosing with diagnoser: W2\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    diagnosis_start_time = time.time()
    b = 0
    e = len(observations) - 1
    S = observations[0]
    for i in range(1, len(observations)):
        a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
        a = int(a)
        S, reward, done, trunc, info = simulator.step(a)
        if observations[i] is not None:
            if comparators[domain_name](observations[i], S):
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

    exp_rt_end = time.time()
    exp_duration_sec = exp_rt_end - exp_rt_begin
    exp_duration_ms = exp_duration_sec * 1000

    memory_tracked = tracemalloc.get_traced_memory()
    # stopping the library
    tracemalloc.stop()

    output = {
        "diagnoses": D,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "exp_duration_sec": exp_duration_sec,
        "exp_duration_ms": exp_duration_ms,
        "exp_memory_at_end": memory_tracked[0],
        "exp_memory_max": memory_tracked[1],
        "G_max_size":0
    }

    return output


def SN2(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # starting the monitoring
    tracemalloc.start()
    exp_rt_begin = time.time()
    # welcome message
    print(f'diagnosing with diagnoser: SN2\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        G[key_j] = [candidate_fault_modes[key_j], A_j, S_0]
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    # running the diagnosis loop
    for i in range(1, len(observations)):
        irrelevant_keys = []
        for key_j in G.keys():
            ts1 = time.time()
            a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
            a_gag_i_j = G[key_j][0](a_gag_i)
            simulator.set_state(G[key_j][2])
            S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
            G[key_j][1].append(int(a_gag_i_j))
            G[key_j][2] = S_gag_i_j
            if observations[i] is not None:
                if not comparators[domain_name](observations[i], S_gag_i_j):
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

    exp_rt_end = time.time()
    exp_duration_sec = exp_rt_end - exp_rt_begin
    exp_duration_ms = exp_duration_sec * 1000

    memory_tracked = tracemalloc.get_traced_memory()
    # stopping the library
    tracemalloc.stop()

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "exp_duration_sec": exp_duration_sec,
        "exp_duration_ms": exp_duration_ms,
        "exp_memory_at_end": memory_tracked[0],
        "exp_memory_max": memory_tracked[1],
        "G_max_size": len(candidate_fault_modes)
    }

    return raw_output


def fm_and_state_in_set(key_raw, state, FG):
    for fkey in FG.keys():
        fkey_raw = fkey.split('_')[0]
        fstate = FG[fkey][2]
        if key_raw == fkey_raw and state == fstate:
            return True
    return False


def SIF2(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
    # starting the monitoring
    tracemalloc.start()
    exp_rt_begin = time.time()
    # welcome message
    print(f'diagnosing with diagnoser: SIF2\n========================================================================================')

    # load trained model as policy
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    policy = models[ml_model_name].load(model_path)

    # load the environment as simulator
    simulator = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=render_mode))
    initial_obs, _ = simulator.reset(seed=instance_seed)
    S_0, _ = simulator.reset()
    assert comparators[domain_name](observations[0], S_0)

    # initialize time counting
    diagnosis_runtime_sec = 0.0

    # initialize maximum size of G
    G_max_size = 0

    # initialize unique ID's for each fault mode in order to represent different branchings
    I = {}
    for key_j in candidate_fault_modes:
        I[key_j] = 0

    # initialize G
    ts0 = time.time()
    G = {}
    for key_j in candidate_fault_modes:
        A_j = []
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], A_j, S_0]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    for i in range(1, len(observations)):
        irrelevant_keys = []
        new_relevant_keys = {}
        for key_j in G.keys():
            ts1 = time.time()
            a_gag_i, _ = policy.predict(refiners[domain_name](G[key_j][2]), deterministic=DETERMINISTIC)
            a_gag_i = int(a_gag_i)
            a_gag_i_j = G[key_j][0](a_gag_i)
            te1 = time.time()
            diagnosis_runtime_sec += te1 - ts1

            # apply the normal and the faulty action on the reconstructed states, respectively
            ts2 = time.time()
            simulator.set_state(G[key_j][2])
            S_gag_i, reward, done, trunc, info = simulator.step(a_gag_i)
            simulator.set_state(G[key_j][2])
            S_gag_i_j, reward, done, trunc, info = simulator.step(a_gag_i_j)
            if observations[i] is not None:
                # the case where there is an observation that can be checked
                S_gag_i_eq_S_i = comparators[domain_name](S_gag_i, observations[i])
                S_gag_i_j_eq_S_i = comparators[domain_name](S_gag_i_j, observations[i])
                if S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 1: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i))
                    G[key_j][2] = S_gag_i
                elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i not changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i))
                    G[key_j][2] = S_gag_i
                elif not S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j cannot change a_gag_i
                    if debug_print:
                        print(f'case 3: kicking                     (a_gag_i     changed, f_j cannot change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    irrelevant_keys.append(key_j)
                elif not S_gag_i_eq_S_i and S_gag_i_j_eq_S_i:
                    # a_gag_i     changed, f_j can    change a_gag_i
                    if debug_print:
                        print(f'case 4: adding a_gag_i_j, S_gag_i_j (a_gag_i     changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1].append(int(a_gag_i_j))
                    G[key_j][2] = S_gag_i_j
            else:
                # the case where there is no observation to be checked - insert the normal action and state to the original key
                if debug_print:
                    print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                G[key_j][1].append(int(a_gag_i))
                G[key_j][2] = S_gag_i
                if a_gag_i != a_gag_i_j:
                    # if the action was changed - create new trajectory and insert it as well
                    if debug_print:
                        print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    A_j_to_fault = copy.deepcopy(G[key_j][1])
                    A_j_to_fault[-1] = a_gag_i_j
                    k_j = key_j.split('_')[0]
                    new_relevant_keys[k_j + f'_{I[k_j]}'] = [candidate_fault_modes[k_j],  A_j_to_fault, S_gag_i_j]
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
        # filter out similar trajectories (applies to taxi only)
        if domain_name == "Taxi_v3":
            FG = {}
            for key in G.keys():
                key_raw = key.split('_')[0]
                state = G[key][2]
                if not fm_and_state_in_set(key_raw, state, FG):
                    FG[key] = G[key]
            G = FG

        # update the maximum size of G
        G_max_size = max(G_max_size, len(G))

        if debug_print:
            if observations[i] is not None:
                print(f'STEP {i}/{len(observations)}: OBSERVED')
            else:
                print(f'STEP {i}/{len(observations)}: HIDDEN')
            print(f'STEP {i}/{len(observations)}: ADDED   {len(new_relevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(new_relevant_keys.keys()))}')
            print(f'STEP {i}/{len(observations)}: KICKED  {len(irrelevant_keys)}\t ({len(G)}) at time {diagnosis_runtime_sec}: {str(irrelevant_keys)}')
            print(f'STEP {i}/{len(observations)}: G         \t ({len(G)}) at time {diagnosis_runtime_sec}: {str(list(G.keys()))}\n')

        if len(G) == 1:
            if debug_print:
                print(f"i broke at {i}")
            break

    # finilizing the runtime in ms
    diagnosis_runtime_ms = diagnosis_runtime_sec * 1000

    exp_rt_end = time.time()
    exp_duration_sec = exp_rt_end - exp_rt_begin
    exp_duration_ms = exp_duration_sec * 1000

    memory_tracked = tracemalloc.get_traced_memory()
    # stopping the library
    tracemalloc.stop()

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "exp_duration_sec": exp_duration_sec,
        "exp_duration_ms": exp_duration_ms,
        "exp_memory_at_end": memory_tracked[0],
        "exp_memory_max": memory_tracked[1],
        "G_max_size": G_max_size
    }

    return raw_output


diagnosers = {
    # new fault models
    "W": W,
    "SN": SN,
    "SIF": SIF,
    "W2": W2,
    "SN2": SN2,
    "SIF2": SIF2
}
