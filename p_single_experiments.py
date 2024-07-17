import math
from datetime import datetime

from p_pipeline import run_SIF_single_experiment, run_SN_single_experiment, run_W_single_experiment, run_SIFU_single_experiment, run_SIFU2_single_experiment, run_SIFU3_single_experiment


# =================================================================================================
# ========================================== LunerLander ==========================================
# =================================================================================================
def single_experiment_LunarLander_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "LunarLander_v2"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_LunarLander_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "LunarLander_v2"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3]",  # shutting down jets
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_LunarLander_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "LunarLander_v2"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0,2,3]",  # shutting down jets
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# ============================================ Acrobot ============================================
# =================================================================================================
def single_experiment_Acrobot_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU():
    # changable test settings - strong fault model intermittent faults smart (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Acrobot_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 1
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Acrobot_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# =========================================== CartPole ============================================
# =================================================================================================
def single_experiment_CartPole_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU():
    # changable test settings - strong fault model intermittent faults smart (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_CartPole_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "CartPole_v1"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[0,0]",
            "[1,1]",
            "[1,0]"
        ]
        num_candidate_fault_modes = 3
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# ========================================== MountainCar ==========================================
# =================================================================================================
def single_experiment_MountainCar_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU():
    # changable test settings - strong fault model intermittent faults (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_MountainCar_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "MountainCar_v0"
        ml_model_name = "DQN"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[1,1,2]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
            "[1,1,2]",
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


# =================================================================================================
# ============================================= Taxi ==============================================
# =================================================================================================
def single_experiment_Taxi_W():
    # changable test settings - weak fault model (W)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = []
        num_candidate_fault_modes = 0
        diagnosis_runtime_ms = run_W_single_experiment(domain_name=domain_name,
                                                       ml_model_name=ml_model_name,
                                                       render_mode=render_mode,
                                                       max_exec_len=max_exec_len,
                                                       debug_print=debug_print,
                                                       execution_fault_mode_name=execution_fault_mode_name,
                                                       instance_seed=instance_seed,
                                                       fault_probability=fault_probability,
                                                       percent_visible_states=percent_visible_states,
                                                       possible_fault_mode_names=possible_fault_mode_names,
                                                       num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SN():
    # changable test settings - strong fault model non-intermittent faults (SN)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SN_single_experiment(domain_name=domain_name,
                                                        ml_model_name=ml_model_name,
                                                        render_mode=render_mode,
                                                        max_exec_len=max_exec_len,
                                                        debug_print=debug_print,
                                                        execution_fault_mode_name=execution_fault_mode_name,
                                                        instance_seed=instance_seed,
                                                        fault_probability=fault_probability,
                                                        percent_visible_states=percent_visible_states,
                                                        possible_fault_mode_names=possible_fault_mode_names,
                                                        num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIF():
    # changable test settings - strong fault model intermittent faults (SIF)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIF_single_experiment(domain_name=domain_name,
                                                         ml_model_name=ml_model_name,
                                                         render_mode=render_mode,
                                                         max_exec_len=max_exec_len,
                                                         debug_print=debug_print,
                                                         execution_fault_mode_name=execution_fault_mode_name,
                                                         instance_seed=instance_seed,
                                                         fault_probability=fault_probability,
                                                         percent_visible_states=percent_visible_states,
                                                         possible_fault_mode_names=possible_fault_mode_names,
                                                         num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU():
    # changable test settings - strong fault model intermittent faults (SIFS)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU_single_experiment(domain_name=domain_name,
                                                          ml_model_name=ml_model_name,
                                                          render_mode=render_mode,
                                                          max_exec_len=max_exec_len,
                                                          debug_print=debug_print,
                                                          execution_fault_mode_name=execution_fault_mode_name,
                                                          instance_seed=instance_seed,
                                                          fault_probability=fault_probability,
                                                          percent_visible_states=percent_visible_states,
                                                          possible_fault_mode_names=possible_fault_mode_names,
                                                          num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU2():
    # changable test settings - strong fault model intermittent faults upgraded 2 (SIFU2)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU2_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')


def single_experiment_Taxi_SIFU3():
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
    diagnosis_runtimes_ms = []

    NUM_TRIES = 10
    for i in range(NUM_TRIES):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string}: try {i}/{NUM_TRIES}')
        domain_name = "Taxi_v3"
        ml_model_name = "PPO"                         # "PPO", "DQN"
        render_mode = "rgb_array"                     # "human", "rgb_array"
        max_exec_len = 200
        debug_print = False
        execution_fault_mode_name = "[0,0,2,3,4,5]"
        instance_seed = 42
        fault_probability = 1.0
        percent_visible_states = 100
        possible_fault_mode_names = [
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
        num_candidate_fault_modes = 10
        diagnosis_runtime_ms = run_SIFU3_single_experiment(domain_name=domain_name,
                                                           ml_model_name=ml_model_name,
                                                           render_mode=render_mode,
                                                           max_exec_len=max_exec_len,
                                                           debug_print=debug_print,
                                                           execution_fault_mode_name=execution_fault_mode_name,
                                                           instance_seed=instance_seed,
                                                           fault_probability=fault_probability,
                                                           percent_visible_states=percent_visible_states,
                                                           possible_fault_mode_names=possible_fault_mode_names,
                                                           num_candidate_fault_modes=num_candidate_fault_modes)
        diagnosis_runtimes_ms.append(diagnosis_runtime_ms)

    for e in diagnosis_runtimes_ms:
        print(math.floor(e))
    print(f'avg duration in ms: {math.floor(sum(diagnosis_runtimes_ms) / len(diagnosis_runtimes_ms))}')

