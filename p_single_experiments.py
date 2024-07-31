import math
from datetime import datetime

from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import diagnosers
from p_executor import execute_manual
from p_pipeline import run_SIF_single_experiment, run_SN_single_experiment, run_W_single_experiment, run_SIFU_single_experiment, run_SIFU2_single_experiment, run_SIFU3_single_experiment, run_SIFU4_single_experiment, separate_trajectory, calculate_largest_hidden_gap, mask_states, rank_diagnoses_WFM, rank_diagnoses_SFM, prepare_record, write_records_to_excel


# =================================================================================================
# ============================================ manual =============================================
# =================================================================================================
def single_experiment_manual():
    domain_name = "CartPole_v1"
    ml_model_name = "PPO"  # "PPO", "DQN"
    render_mode = "rgb_array"  # "human", "rgb_array"
    debug_print = False
    execution_fault_mode_name = "[0,0]"
    instance_seed = 6
    fault_probability = 0.4
    percent_visible_states = 30

    # ###########################
    faulty_actions_indices = [7, 20, 21, 23, 25, 27, 31, 32, 36, 39, 40, 41]
    execution_length = 40
    observation_mask = [0, 4, 6, 7, 9, 10, 15, 17, 32, 33, 37, 39, 40]
    diagnoser_name = "SIFU3"
    candidate_fault_modes_names = [
        '[0,0]',
        '[1,0]'
    ]
    # ###########################

    fault_mode_generator = FaultModelGeneratorDiscrete()
    trajectory_execution, faulty_actions_indices = execute_manual(domain_name,
                                                                  debug_print,
                                                                  execution_fault_mode_name,
                                                                  instance_seed,
                                                                  fault_probability,
                                                                  render_mode,
                                                                  ml_model_name,
                                                                  fault_mode_generator,
                                                                  execution_length,
                                                                  faulty_actions_indices)
    registered_actions, observations = separate_trajectory(trajectory_execution)
    print(f'registered_actions: {[f"{i}:{a}" for i, a in enumerate(registered_actions)]}')
    print(f'faulty actions indices: {faulty_actions_indices}')

    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
    masked_observations = mask_states(observations, observation_mask)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    candidate_fault_modes = {}
    for fmn in candidate_fault_modes_names:
        fm = fault_mode_generator.generate_fault_model(fmn)
        candidate_fault_modes[fmn] = fm

    diagnoser = diagnosers[diagnoser_name]

    raw_output = diagnoser(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    if diagnoser_name == "W":
        output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)
    else:
        output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    records = []
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, candidate_fault_modes_names, len(candidate_fault_modes_names),
                            render_mode, ml_model_name, execution_length, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, diagnoser_name, longest_hidden_state_sequence)
    records.append(record)
    write_records_to_excel(records, f"single_experiment_manual_{domain_name.split('_')[0]}_{diagnoser_name}")

    print(f'duration in ms: {raw_output["diag_rt_ms"]}')




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


def single_experiment_Acrobot_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
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
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
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


def single_experiment_CartPole_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
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
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
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
    # changable test settings - strong fault model intermittent faults upgraded 3 (SIFU3)
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


def single_experiment_MountainCar_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
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
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
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


def single_experiment_Taxi_SIFU4():
    # changable test settings - strong fault model intermittent faults upgraded 4 (SIFU4)
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
        diagnosis_runtime_ms = run_SIFU4_single_experiment(domain_name=domain_name,
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

