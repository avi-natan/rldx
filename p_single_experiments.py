import math

from p_pipeline import run_W_single_experiment, run_SN_single_experiment, run_SIF_single_experiment


# =================================================================================================
# ========================================== LunerLander ==========================================
# =================================================================================================
def single_experiment_LunarLander_W():
    # changable test settings - weak fault model (W)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        w_domain_name = "LunarLander_v2"
        w_debug_print = False
        w_execution_fault_mode_name = "[0,0,2,3]"
        w_instance_seed = 42
        w_fault_probability = 1.0
        w_percent_visible_states = 100
        w_possible_fault_mode_names = []
        w_num_candidate_fault_modes = 0
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_W_single_experiment(domain_name=w_domain_name,
                                                                                        debug_print=w_debug_print,
                                                                                        execution_fault_mode_name=w_execution_fault_mode_name,
                                                                                        instance_seed=w_instance_seed,
                                                                                        fault_probability=w_fault_probability,
                                                                                        percent_visible_states=w_percent_visible_states,
                                                                                        possible_fault_mode_names=w_possible_fault_mode_names,
                                                                                        num_candidate_fault_modes=w_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_LunarLander_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sn_domain_name = "LunarLander_v2"
        sn_debug_print = False
        sn_execution_fault_mode_name = "[0,0,2,3]"
        sn_instance_seed = 42
        sn_fault_probability = 1.0
        sn_percent_visible_states = 100
        sn_possible_fault_mode_names = ["[0,0,2,3]",  # shutting down jets
                                        "[0,1,0,3]",
                                        "[0,1,2,0]",
                                        "[0,0,0,3]",
                                        "[0,0,2,0]",
                                        "[0,1,0,0]",
                                        "[0,0,0,0]",
                                        "[0,3,2,1]",  # swapping jets
                                        "[0,2,1,3]",
                                        "[0,1,3,2]"
                                        ]
        sn_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SN_single_experiment(domain_name=sn_domain_name,
                                                                                         debug_print=sn_debug_print,
                                                                                         execution_fault_mode_name=sn_execution_fault_mode_name,
                                                                                         instance_seed=sn_instance_seed,
                                                                                         fault_probability=sn_fault_probability,
                                                                                         percent_visible_states=sn_percent_visible_states,
                                                                                         possible_fault_mode_names=sn_possible_fault_mode_names,
                                                                                         num_candidate_fault_modes=sn_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_LunarLander_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "LunarLander_v2"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,0,2,3]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = ["[0,0,2,3]",  # shutting down jets
                                         "[0,1,0,3]",
                                         "[0,1,2,0]",
                                         "[0,0,0,3]",
                                         "[0,0,2,0]",
                                         "[0,1,0,0]",
                                         "[0,0,0,0]",
                                         "[0,3,2,1]",  # swapping jets
                                         "[0,2,1,3]",
                                         "[0,1,3,2]"
                                         ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SIF_single_experiment(domain_name=sif_domain_name,
                                                                                          debug_print=sif_debug_print,
                                                                                          execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                          instance_seed=sif_instance_seed,
                                                                                          fault_probability=sif_fault_probability,
                                                                                          percent_visible_states=sif_percent_visible_states,
                                                                                          possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                          num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


# =================================================================================================
# ============================================ Acrobot ============================================
# =================================================================================================
def single_experiment_Acrobot_W():
    # changable test settings - weak fault model (W)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        w_domain_name = "Acrobot_v1"
        w_debug_print = False
        w_execution_fault_mode_name = "[1,1,2]"
        w_instance_seed = 42
        w_fault_probability = 1.0
        w_percent_visible_states = 100
        w_possible_fault_mode_names = []
        w_num_candidate_fault_modes = 0
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_W_single_experiment(domain_name=w_domain_name,
                                                                                        debug_print=w_debug_print,
                                                                                        execution_fault_mode_name=w_execution_fault_mode_name,
                                                                                        instance_seed=w_instance_seed,
                                                                                        fault_probability=w_fault_probability,
                                                                                        percent_visible_states=w_percent_visible_states,
                                                                                        possible_fault_mode_names=w_possible_fault_mode_names,
                                                                                        num_candidate_fault_modes=w_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_Acrobot_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sn_domain_name = "Acrobot_v1"
        sn_debug_print = False
        sn_execution_fault_mode_name = "[1,1,2]"
        sn_instance_seed = 42
        sn_fault_probability = 1.0
        sn_percent_visible_states = 100
        sn_possible_fault_mode_names = ["[1,1,2]",
                                        "[0,1,1]",
                                        "[0,2,1]",
                                        "[1,0,2]",
                                        "[1,2,0]",
                                        "[2,0,1]",
                                        "[2,1,0]",
                                        "[0,0,0]",
                                        "[1,1,1]",
                                        "[2,2,2]"
                                        ]
        sn_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SN_single_experiment(domain_name=sn_domain_name,
                                                                                         debug_print=sn_debug_print,
                                                                                         execution_fault_mode_name=sn_execution_fault_mode_name,
                                                                                         instance_seed=sn_instance_seed,
                                                                                         fault_probability=sn_fault_probability,
                                                                                         percent_visible_states=sn_percent_visible_states,
                                                                                         possible_fault_mode_names=sn_possible_fault_mode_names,
                                                                                         num_candidate_fault_modes=sn_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_Acrobot_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "Acrobot_v1"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[1,1,2]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = ["[1,1,2]",
                                         "[0,1,1]",
                                         "[0,2,1]",
                                         "[1,0,2]",
                                         "[1,2,0]",
                                         "[2,0,1]",
                                         "[2,1,0]",
                                         "[0,0,0]",
                                         "[1,1,1]",
                                         "[2,2,2]"
                                         ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SIF_single_experiment(domain_name=sif_domain_name,
                                                                                          debug_print=sif_debug_print,
                                                                                          execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                          instance_seed=sif_instance_seed,
                                                                                          fault_probability=sif_fault_probability,
                                                                                          percent_visible_states=sif_percent_visible_states,
                                                                                          possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                          num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


# =================================================================================================
# =========================================== CartPole ============================================
# =================================================================================================
def single_experiment_CartPole_W():
    # changable test settings - weak fault model (W)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        w_domain_name = "CartPole_v1"
        w_debug_print = False
        w_execution_fault_mode_name = "[1,1]"
        w_instance_seed = 42
        w_fault_probability = 1.0
        w_percent_visible_states = 100
        w_possible_fault_mode_names = []
        w_num_candidate_fault_modes = 0
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_W_single_experiment(domain_name=w_domain_name,
                                                                                        debug_print=w_debug_print,
                                                                                        execution_fault_mode_name=w_execution_fault_mode_name,
                                                                                        instance_seed=w_instance_seed,
                                                                                        fault_probability=w_fault_probability,
                                                                                        percent_visible_states=w_percent_visible_states,
                                                                                        possible_fault_mode_names=w_possible_fault_mode_names,
                                                                                        num_candidate_fault_modes=w_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_CartPole_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sn_domain_name = "CartPole_v1"
        sn_debug_print = False
        sn_execution_fault_mode_name = "[1,1]"
        sn_instance_seed = 42
        sn_fault_probability = 1.0
        sn_percent_visible_states = 100
        sn_possible_fault_mode_names = ["[0,0]",
                                        "[1,1]",
                                        "[1,0]"
                                        ]
        sn_num_candidate_fault_modes = 3
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SN_single_experiment(domain_name=sn_domain_name,
                                                                                         debug_print=sn_debug_print,
                                                                                         execution_fault_mode_name=sn_execution_fault_mode_name,
                                                                                         instance_seed=sn_instance_seed,
                                                                                         fault_probability=sn_fault_probability,
                                                                                         percent_visible_states=sn_percent_visible_states,
                                                                                         possible_fault_mode_names=sn_possible_fault_mode_names,
                                                                                         num_candidate_fault_modes=sn_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_CartPole_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "CartPole_v1"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[1,1]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = ["[0,0]",
                                         "[1,1]",
                                         "[1,0]"
                                         ]
        sif_num_candidate_fault_modes = 3
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SIF_single_experiment(domain_name=sif_domain_name,
                                                                                          debug_print=sif_debug_print,
                                                                                          execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                          instance_seed=sif_instance_seed,
                                                                                          fault_probability=sif_fault_probability,
                                                                                          percent_visible_states=sif_percent_visible_states,
                                                                                          possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                          num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


# =================================================================================================
# ============================================ Assault ============================================
# =================================================================================================
def single_experiment_Assault_W():
    # changable test settings - weak fault model (W)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/Assault_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5,6]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = []
        sif_num_candidate_fault_modes = 0
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_W_single_experiment(domain_name=sif_domain_name,
                                                                                        debug_print=sif_debug_print,
                                                                                        execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                        instance_seed=sif_instance_seed,
                                                                                        fault_probability=sif_fault_probability,
                                                                                        percent_visible_states=sif_percent_visible_states,
                                                                                        possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                        num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_Assault_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/Assault_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5,6]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = [
                                        "[0,0,2,3,4,5,6]",
                                        "[0,1,0,3,4,5,6]",
                                        "[0,1,2,0,4,5,6]",
                                        "[0,1,2,3,0,5,6]",
                                        "[0,1,2,3,4,0,6]",
                                        "[0,1,2,3,4,5,0]",
                                        "[0,2,1,3,4,5,6]",
                                        "[0,3,2,1,4,5,6]",
                                        "[0,4,2,3,1,5,6]",
                                        "[0,5,2,3,4,1,6]",
                                        "[0,6,2,3,4,5,1]"
                                        ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SN_single_experiment(domain_name=sif_domain_name,
                                                                                         debug_print=sif_debug_print,
                                                                                         execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                         instance_seed=sif_instance_seed,
                                                                                         fault_probability=sif_fault_probability,
                                                                                         percent_visible_states=sif_percent_visible_states,
                                                                                         possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                         num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_Assault_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/Assault_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5,6]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = [
                                        "[0,0,2,3,4,5,6]",
                                        "[0,1,0,3,4,5,6]",
                                        "[0,1,2,0,4,5,6]",
                                        "[0,1,2,3,0,5,6]",
                                        "[0,1,2,3,4,0,6]",
                                        "[0,1,2,3,4,5,0]",
                                        "[0,2,1,3,4,5,6]",
                                        "[0,3,2,1,4,5,6]",
                                        "[0,4,2,3,1,5,6]",
                                        "[0,5,2,3,4,1,6]",
                                        "[0,6,2,3,4,5,1]"
                                        ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SIF_single_experiment(domain_name=sif_domain_name,
                                                                                          debug_print=sif_debug_print,
                                                                                          execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                          instance_seed=sif_instance_seed,
                                                                                          fault_probability=sif_fault_probability,
                                                                                          percent_visible_states=sif_percent_visible_states,
                                                                                          possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                          num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


# =================================================================================================
# =========================================== Carnival ============================================
# =================================================================================================
def single_experiment_Carnival_W():
    # changable test settings - weak fault model (W)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/Carnival_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = []
        sif_num_candidate_fault_modes = 0
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_W_single_experiment(domain_name=sif_domain_name,
                                                                                        debug_print=sif_debug_print,
                                                                                        execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                        instance_seed=sif_instance_seed,
                                                                                        fault_probability=sif_fault_probability,
                                                                                        percent_visible_states=sif_percent_visible_states,
                                                                                        possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                        num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_Carnival_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/Carnival_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = [
                                        "[0,0,2,3,4,5]",
                                        "[0,1,0,3,4,5]",
                                        "[0,1,2,0,4,5]",
                                        "[0,1,2,3,0,5]",
                                        "[0,1,2,3,4,0]",
                                        "[0,2,1,3,4,5]",
                                        "[0,3,2,1,4,5]",
                                        "[0,4,2,3,1,5]",
                                        "[0,5,2,3,4,1]",
                                        "[1,0,2,3,4,5]"
                                        ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SN_single_experiment(domain_name=sif_domain_name,
                                                                                         debug_print=sif_debug_print,
                                                                                         execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                         instance_seed=sif_instance_seed,
                                                                                         fault_probability=sif_fault_probability,
                                                                                         percent_visible_states=sif_percent_visible_states,
                                                                                         possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                         num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_Carnival_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/Carnival_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = [
                                        "[0,0,2,3,4,5]",
                                        "[0,1,0,3,4,5]",
                                        "[0,1,2,0,4,5]",
                                        "[0,1,2,3,0,5]",
                                        "[0,1,2,3,4,0]",
                                        "[0,2,1,3,4,5]",
                                        "[0,3,2,1,4,5]",
                                        "[0,4,2,3,1,5]",
                                        "[0,5,2,3,4,1]",
                                        "[1,0,2,3,4,5]"
                                        ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SIF_single_experiment(domain_name=sif_domain_name,
                                                                                          debug_print=sif_debug_print,
                                                                                          execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                          instance_seed=sif_instance_seed,
                                                                                          fault_probability=sif_fault_probability,
                                                                                          percent_visible_states=sif_percent_visible_states,
                                                                                          possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                          num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


# =================================================================================================
# ========================================= SpaceInvaders =========================================
# =================================================================================================
def single_experiment_SpaceInvaders_W():
    # changable test settings - weak fault model (W)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/SpaceInvaders_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = []
        sif_num_candidate_fault_modes = 0
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_W_single_experiment(domain_name=sif_domain_name,
                                                                                        debug_print=sif_debug_print,
                                                                                        execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                        instance_seed=sif_instance_seed,
                                                                                        fault_probability=sif_fault_probability,
                                                                                        percent_visible_states=sif_percent_visible_states,
                                                                                        possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                        num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_SpaceInvaders_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/SpaceInvaders_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = [
                                        "[0,0,2,3,4,5]",
                                        "[0,1,0,3,4,5]",
                                        "[0,1,2,0,4,5]",
                                        "[0,1,2,3,0,5]",
                                        "[0,1,2,3,4,0]",
                                        "[0,2,1,3,4,5]",
                                        "[0,3,2,1,4,5]",
                                        "[0,4,2,3,1,5]",
                                        "[0,5,2,3,4,1]",
                                        "[1,0,2,3,4,5]"
                                        ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SN_single_experiment(domain_name=sif_domain_name,
                                                                                         debug_print=sif_debug_print,
                                                                                         execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                         instance_seed=sif_instance_seed,
                                                                                         fault_probability=sif_fault_probability,
                                                                                         percent_visible_states=sif_percent_visible_states,
                                                                                         possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                         num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')


def single_experiment_SpaceInvaders_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    exp_durations_in_ms = []
    exp_memories_at_end = []
    exp_memories_max = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        print(f'try {i}/{NUM_TRIES}')
        sif_domain_name = "ALE/SpaceInvaders_v5"
        sif_debug_print = False
        sif_execution_fault_mode_name = "[0,1,0,3,4,5]"
        sif_instance_seed = 42
        sif_fault_probability = 1.0
        sif_percent_visible_states = 100
        sif_possible_fault_mode_names = [
                                        "[0,0,2,3,4,5]",
                                        "[0,1,0,3,4,5]",
                                        "[0,1,2,0,4,5]",
                                        "[0,1,2,3,0,5]",
                                        "[0,1,2,3,4,0]",
                                        "[0,2,1,3,4,5]",
                                        "[0,3,2,1,4,5]",
                                        "[0,4,2,3,1,5]",
                                        "[0,5,2,3,4,1]",
                                        "[1,0,2,3,4,5]"
                                        ]
        sif_num_candidate_fault_modes = 10
        exp_duration_in_ms, exp_memory_at_end, exp_memory_max = run_SIF_single_experiment(domain_name=sif_domain_name,
                                                                                          debug_print=sif_debug_print,
                                                                                          execution_fault_mode_name=sif_execution_fault_mode_name,
                                                                                          instance_seed=sif_instance_seed,
                                                                                          fault_probability=sif_fault_probability,
                                                                                          percent_visible_states=sif_percent_visible_states,
                                                                                          possible_fault_mode_names=sif_possible_fault_mode_names,
                                                                                          num_candidate_fault_modes=sif_num_candidate_fault_modes)
        exp_durations_in_ms.append(exp_duration_in_ms)
        exp_memories_at_end.append(exp_memory_at_end)
        exp_memories_max.append(exp_memory_max)

    for e in exp_durations_in_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(exp_durations_in_ms) / len(exp_durations_in_ms))}')
    for e in exp_memories_at_end:
        print(math.floor(e))
    print(f'avg memory at end: {math.floor(sum(exp_memories_at_end) / len(exp_memories_at_end))}')
    for e in exp_memories_max:
        print(math.floor(e))
    print(f'avg memory max: {math.floor(sum(exp_memories_max) / len(exp_memories_max))}')
