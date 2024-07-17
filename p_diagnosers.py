import copy
import time

import gym

from h_consts import DETERMINISTIC
from h_raw_state_comparators import comparators
from h_rl_models import models
from h_state_refiners import refiners
from h_wrappers import wrappers


def W(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
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

    output = {
        "diagnoses": D,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "G_max_size":0
    }

    return output


def SN(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
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

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
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


def SIF(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
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

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
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

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    ts = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = j - i
            i = j
    sorted_index_pairs = sorted(index_pairs.keys(), key=lambda k: (index_pairs[k], k))
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in sorted_index_pairs]
    te = time.time()
    diagnosis_runtime_sec += te - ts

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
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
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
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
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
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
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

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU2(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
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

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    ts = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = j - i
            i = j
    useful_index_pairs = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            useful_index_pairs[pair] = index_pairs[pair]
    sorted_useful_index_pairs = sorted(useful_index_pairs.keys(), key=lambda k: (useful_index_pairs[k], k))
    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in sorted_useful_index_pairs]
    te = time.time()
    diagnosis_runtime_sec += te - ts

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
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
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
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
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
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
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

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


def SIFU3(debug_print, render_mode, instance_seed, ml_model_name, domain_name, observations, candidate_fault_modes):
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

    # compute index queue (the computed is of the form: [(b1,e1), (b2,e2), ..., (bm,em)]  )
    # at the same time, collect the action types to be tested
    ts = time.time()
    index_pairs = {}
    i = 0
    for j in range(1, len(observations)):
        if observations[j] is None:
            continue
        else:
            i_s = str(i).zfill(3)
            j_s = str(j).zfill(3)
            index_pairs[f"{i_s}_{j_s}"] = [j - i, None]
            i = j

    index_pairs_failed = {}
    for pair in index_pairs:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b + i}: a {a}')
            S, reward, done, trunc, info = simulator.step(a)
        if not comparators[domain_name](observations[e], S):
            index_pairs_failed[pair] = [index_pairs[pair][0], set()]
            # index_pairs[pair][1] = 'FAIL'
            # print(f'pair {pair}: FAIL\n')
        # else:
        #     index_pairs[pair][1] = '  OK'
            # print(f'pair {pair}: OK\n')
    index_pairs_failed_sorted = {k: v for k, v in sorted(index_pairs_failed.items(), key=lambda item: (item[1][0], -len(item[1][1]), item[0]))}

    index_pairs_failed_sorted_useful = {}
    action_types_combined = set()
    for pair in index_pairs_failed_sorted:
        b = int(pair.split("_")[0])
        e = int(pair.split("_")[1])
        S = observations[b]
        simulator.set_state(S)
        for i in range(e - b):
            a, _ = policy.predict(refiners[domain_name](S), deterministic=DETERMINISTIC)
            a = int(a)
            # print(f'i {b+i}: a {a}')
            index_pairs_failed_sorted[pair][1].add(a)
            S, reward, done, trunc, info = simulator.step(a)
        if len(index_pairs_failed_sorted[pair][1].difference(action_types_combined)) != 0:
            index_pairs_failed_sorted_useful[pair] = [index_pairs_failed_sorted[pair][0], index_pairs_failed_sorted[pair][1]]
            action_types_combined.update(index_pairs_failed_sorted[pair][1])

    index_queue = [(int(item.split("_")[0]), int(item.split("_")[1])) for item in index_pairs_failed_sorted_useful.keys()]
    te = time.time()
    diagnosis_runtime_sec += te - ts

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
        G[key_j + f'_{I[key_j]}'] = [candidate_fault_modes[key_j], [None] * (len(observations)-1), None]
        I[key_j] = I[key_j] + 1
    te0 = time.time()
    diagnosis_runtime_sec += te0 - ts0

    for irk in index_queue:
        if len(G) == 1:
            break
        for key in G.keys():
            G[key][2] = observations[irk[0]]
        for i in range(irk[0]+1, irk[1]+1):
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
                        G[key_j][1][i-1] = int(a_gag_i)
                        G[key_j][2] = S_gag_i
                    elif S_gag_i_eq_S_i and not S_gag_i_j_eq_S_i:
                        # a_gag_i not changed, f_j can    change a_gag_i
                        if debug_print:
                            print(f'case 2: adding a_gag_i, S_gag_i     (a_gag_i not changed, f_j can    change a_gag_i) (a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        G[key_j][1][i-1] = int(a_gag_i)
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
                        G[key_j][1][i-1] = int(a_gag_i_j)
                        G[key_j][2] = S_gag_i_j
                else:
                    # the case where there is no observation to be checked - insert the normal action and state to the original key
                    if debug_print:
                        print(f'case 5: adding a_gag_i, S_gag_i     (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                    G[key_j][1][i-1] = int(a_gag_i)
                    G[key_j][2] = S_gag_i
                    if a_gag_i != a_gag_i_j:
                        # if the action was changed - create new trajectory and insert it as well
                        if debug_print:
                            print(f'case 6: adding a_gag_i_j, S_gag_i_j (no observation, a_gag_i: {a_gag_i}, a_gag_i_j: {a_gag_i_j}) [fault model: {key_j}]')
                        A_j_to_fault = copy.deepcopy(G[key_j][1])
                        A_j_to_fault[i-1] = a_gag_i_j
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

    raw_output = {
        "diagnoses": G,
        "diagnosis_runtime_sec": diagnosis_runtime_sec,
        "diagnosis_runtime_ms": diagnosis_runtime_ms,
        "G_max_size": G_max_size
    }

    return raw_output


diagnosers = {
    # new fault models
    "W": W,
    "SN": SN,
    "SIF": SIF,
    "SIFU": SIFU,
    "SIFU2": SIFU2,
    "SIFU3": SIFU3
}
