import copy
import json
import math
import os
import random
import time
from datetime import datetime

import numpy
import xlsxwriter

from h_common import read_json_data
from h_fault_model_generator import FaultModelGeneratorDiscrete
from p_diagnosers import W, SIF, SN, diagnosers
from p_executor import execute


# separating trajectory to actions and states
def separate_trajectory(trajectory_execution):
    registered_actions = []
    observations = []
    for i in range(len(trajectory_execution)):
        if i % 2 == 1:
            registered_actions.append(trajectory_execution[i])
        else:
            observations.append(trajectory_execution[i])
    if len(registered_actions) == len(observations):
        registered_actions = registered_actions[:-1]
    return registered_actions, observations


def generate_observation_mask(observations_length, percent_visible_states):
    mask = [0] * observations_length
    ones = math.floor((observations_length - 2) * percent_visible_states / 100.0)
    indices = random.sample(range(1, observations_length - 1), ones)
    for i in indices:
        mask[i] = 1
    mask[0] = 1
    mask[len(mask) - 1] = 1
    observation_mask = []
    for i in range(len(mask)):
        if mask[i] == 1:
            observation_mask.append(i)

    # print(f"{mask}; {obs_p_visible},{obs_p_mean},{obs_p_dev};;;{mean_index},{deviation};;;{ones},{start_index},{end_index}")
    return observation_mask


def calculate_largest_hidden_gap(observation_mask):
    largest_hidden_gap = 0
    for i in range(1, len(observation_mask)):
        largest_hidden_gap = max(largest_hidden_gap, observation_mask[i] - observation_mask[i - 1] - 1)
    return largest_hidden_gap


def mask_states(observations, observation_mask):
    masked_observations = [None] * len(observations)
    for i in observation_mask:
        masked_observations[i] = copy.deepcopy(observations[i])
    return masked_observations


def single_experiment_prepare_inputs(domain_name,
                                     ml_model_name,
                                     render_mode,
                                     max_exec_len,
                                     debug_print,
                                     execution_fault_mode_name,
                                     instance_seed,
                                     fault_probability):
    # ### initialize fault model generator
    fault_mode_generator = FaultModelGeneratorDiscrete()

    # ### execute to get trajectory
    trajectory_execution = []
    faulty_actions_indices = []
    while len(faulty_actions_indices) == 0:
        trajectory_execution, faulty_actions_indices = execute(domain_name,
                                                               debug_print,
                                                               execution_fault_mode_name,
                                                               instance_seed,
                                                               fault_probability,
                                                               render_mode,
                                                               ml_model_name,
                                                               fault_mode_generator,
                                                               max_exec_len)

    # ### separating trajectory to actions and states
    registered_actions, observations = separate_trajectory(trajectory_execution)

    return fault_mode_generator, trajectory_execution, faulty_actions_indices, registered_actions, observations


def prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator):
    if num_candidate_fault_modes == 0:
        candidate_fault_mode_names = []
    else:
        candidate_fault_mode_names = [execution_fault_mode_name]
        rest = copy.deepcopy(possible_fault_mode_names)
        rest.remove(execution_fault_mode_name)
        i = 0
        while i < num_candidate_fault_modes - 1:
            fmr = random.choice(rest)
            candidate_fault_mode_names.append(fmr)
            rest.remove(fmr)
            i += 1
    fault_modes = {}
    for fmr in candidate_fault_mode_names:
        fm = fault_mode_generator.generate_fault_model(fmr)
        fault_modes[fmr] = fm

    l = list(fault_modes.items())
    random.shuffle(l)
    fault_modes = dict(l)
    return fault_modes


def rank_diagnoses_WFM(raw_output, registered_actions, debug_print):
    # TODO think about more sophisticated ranking elgorithm
    ranking_start_time = time.time()
    G = raw_output['diagnoses']
    # diagnoses = []
    # ranks = []

    diagnoses = G
    ranks = [0] * len(diagnoses)

    ranking_end_time = time.time()
    ranking_runtime_sec = ranking_end_time - ranking_start_time
    ranking_runtime_ms = ranking_runtime_sec * 1000

    output = {
        "diagnoses": diagnoses,
        "diagnosis_runtime_sec": raw_output['diagnosis_runtime_sec'],
        "diagnosis_runtime_ms": raw_output['diagnosis_runtime_ms'],
        "exp_duration_sec": raw_output['exp_duration_sec'],
        "exp_duration_ms": raw_output['exp_duration_ms'],
        "exp_memory_at_end": raw_output['exp_memory_at_end'],
        "exp_memory_max": raw_output['exp_memory_max'],
        "G_max_size": raw_output['G_max_size'],
        "ranks": ranks,
        "ranking_runtime_sec": ranking_runtime_sec,
        "ranking_runtime_ms": ranking_runtime_ms
    }
    return output


def rank_diagnoses_SFM(raw_output, registered_actions, debug_print):
    ranking_start_time = time.time()
    G = raw_output['diagnoses']
    diagnoses = []
    ranks = []

    for key_j in G:
        actions_j = G[key_j][3]
        num_actual_faults = 0
        for i in range(len(actions_j)):
            if registered_actions[i] != actions_j[i]:
                num_actual_faults += 1
        num_potential_faults = 0
        for i in range(len(actions_j)):
            a = registered_actions[i]
            fa = G[key_j][0](a)
            if a != fa:
                num_potential_faults += 1

        if debug_print:
            print(f'num_actual_faults / num_potential_faults: {num_actual_faults} / {num_potential_faults}')

        if num_potential_faults == 0:
            rank = 1.0
        else:
            rank = num_actual_faults * 1.0 / num_potential_faults

        k_j = key_j.split('_')[0]
        diagnoses.append(k_j)
        ranks.append(rank)

    zipped_lists = zip(diagnoses, ranks)
    sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[-1], reverse=True)
    diagnoses, ranks = zip(*sorted_zipped_lists)

    ranking_end_time = time.time()
    ranking_runtime_sec = ranking_end_time - ranking_start_time
    ranking_runtime_ms = ranking_runtime_sec * 1000

    output = {
        "diagnoses": diagnoses,
        "diagnosis_runtime_sec": raw_output['diagnosis_runtime_sec'],
        "diagnosis_runtime_ms": raw_output['diagnosis_runtime_ms'],
        "exp_duration_sec": raw_output['exp_duration_sec'],
        "exp_duration_ms": raw_output['exp_duration_ms'],
        "exp_memory_at_end": raw_output['exp_memory_at_end'],
        "exp_memory_max": raw_output['exp_memory_max'],
        "G_max_size": raw_output['G_max_size'],
        "ranks": ranks,
        "ranking_runtime_sec": ranking_runtime_sec,
        "ranking_runtime_ms": ranking_runtime_ms,
    }

    return output


def prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                   render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                   candidate_fault_modes, output, diagnoser, longest_hidden_state_sequence):
    record = {
        "domain_name": domain_name,
        "debug_print": debug_print,
        "execution_fault_mode_name": execution_fault_mode_name,
        "instance_seed": instance_seed,
        "fault_probability": fault_probability,
        "percent_visible_states": percent_visible_states,
        "possible_fault_mode_names": possible_fault_mode_names,
        "num_candidate_fault_modes": num_candidate_fault_modes,
        "render_mode": render_mode,
        "ml_model_name": ml_model_name,
        "max_exec_len": max_exec_len,
        "trajectory_execution": [str(list(item)) if isinstance(item, numpy.ndarray) else item for item in trajectory_execution],
        "faulty_actions_indices": faulty_actions_indices,
        "registered_actions": registered_actions,
        "observations": [str(list(item)) for item in observations],
        "observation_mask": observation_mask,
        "masked_observations": [str(list(item)) if item is not None else "None" for item in masked_observations],
        "candidate_fault_modes": list(candidate_fault_modes.keys()),
        "output": output,
        "diagnoser": diagnoser,
        "longest_hidden_state_sequence": longest_hidden_state_sequence
    }
    return record


def get_ordinal_rank(diagnoses, ranks, d):
    index_d = diagnoses.index(d)
    rank_d = ranks[index_d]

    unique_ranks_set = set(ranks)
    unique_ranks_list = sorted(list(unique_ranks_set), reverse=True)

    res = unique_ranks_list.index(rank_d)
    return res


def write_records_to_excel(records, experimental_filename):
    columns = [
        {'header': '01_i_domain_name'},
        {'header': '02_i_execution_fault_mode_name'},
        {'header': '03_i_fault_probability'},
        {'header': '04_i_instance_seed'},
        {'header': '05_O_faulty_actions_indices'},
        {'header': '06_O_num_faulty_actions'},
        {'header': '07_O_registered_actions'},
        {'header': '08_O_execution_length'},
        {'header': '09_O_observations'},
        {'header': '10_i_percent_visible_states'},
        {'header': '11_O_observation_mask'},
        {'header': '12_O_num_visible_states'},
        {'header': '13_O_masked_observations'},
        {'header': '14_i_num_candidate_fault_modes'},
        {'header': '15_O_candidate_fault_modes'},
        {'header': '16_i_diagnoser'},
        {'header': '17_O_diagnoses'},
        {'header': '18_O_ranks'},
        {'header': '19_O_num_diagnoses'},
        {'header': '20_O_correct_diagnosis_rank'},
        {'header': '21_O_diagnosis_runtime_sec'},
        {'header': '22_O_diagnosis_runtime_ms'},
        {'header': '23_O_ranking_runtime_sec'},
        {'header': '24_O_ranking_runtime_ms'},
        {'header': '25_O_total_runtime_sec'},
        {'header': '26_O_total_runtime_ms'},
        {'header': '27_O_G_max_size'},
        {'header': '28_O_longest_hidden_state_sequence'},
        {'header': '29_D_exp_duration_sec'},
        {'header': '30_D_exp_duration_ms'},
        {'header': '31_D_exp_memory_at_end'},
        {'header': '32_D_exp_memory_max'}
    ]
    rows = []
    for i in range(len(records)):
        record_i = records[i]
        row = [
            record_i['domain_name'],  # 01_i_domain_name
            record_i['execution_fault_mode_name'],  # 02_i_execution_fault_mode_name
            float(record_i['fault_probability']),  # 03_i_fault_probability
            record_i['instance_seed'],  # 04_i_instance_seed
            str(record_i['faulty_actions_indices']),  # 05_O_faulty_actions_indices
            len(record_i['faulty_actions_indices']),  # 06_O_len_faulty_actions_indices
            str(record_i['registered_actions']),  # 07_O_registered_actions
            len(record_i['registered_actions']),  # 08_O_len_registered_actions
            str(record_i['observations']) if record_i['debug_print'] else 'Omitted',  # 09_O_observations
            record_i['percent_visible_states'],  # 10_i_percent_visible_states
            str(record_i['observation_mask']),  # 11_O_observation_mask
            len(record_i['observation_mask']),  # 12_O_num_visible_states
            str(record_i['masked_observations']) if record_i['debug_print'] else 'Omitted',  # 13_O_masked_observations
            record_i['num_candidate_fault_modes'],  # 14_i_num_candidate_fault_modes
            str(record_i['candidate_fault_modes']),  # 15_O_candidate_fault_modes
            record_i['diagnoser'],  # 16_i_diagnoser
            str(list(record_i['output']['diagnoses'])),  # 17_O_diagnoses
            str(list(record_i['output']['ranks'])),  # 18_O_ranks
            len(record_i['output']['diagnoses']),  # 19_O_num_diagnoses
            get_ordinal_rank(list(record_i['output']['diagnoses']), list(record_i['output']['ranks']), record_i['execution_fault_mode_name']) if record_i['diagnoser'] != "W" else "Irrelevant",  # 20_O_correct_diagnosis_rank
            record_i['output']['diagnosis_runtime_sec'],  # 21_O_diagnosis_runtime_sec
            record_i['output']['diagnosis_runtime_ms'],  # 22_O_diagnosis_runtime_ms
            record_i['output']['ranking_runtime_sec'],  # 23_O_ranking_runtime_sec
            record_i['output']['ranking_runtime_ms'],  # 24_O_ranking_runtime_ms
            record_i['output']['diagnosis_runtime_sec'] + record_i['output']['ranking_runtime_sec'],  # 25_O_total_runtime_sec
            record_i['output']['diagnosis_runtime_ms'] + record_i['output']['ranking_runtime_ms'],  # 26_O_total_runtime_ms
            record_i['output']['G_max_size'] if record_i['diagnoser'] == "SIF" else "Irrelevant",  # 27_O_G_max_size
            record_i['longest_hidden_state_sequence'],  # 28_O_longest_hidden_state_sequence
            record_i['output']['exp_duration_sec'],  # 29_D_exp_duration_sec
            record_i['output']['exp_duration_ms'],  # 30_D_exp_duration_ms
            record_i['output']['exp_memory_at_end'],  # 31_D_exp_memory_at_end
            record_i['output']['exp_memory_max']  # 32_D_exp_memory_max
        ]
        rows.append(row)
    workbook = xlsxwriter.Workbook(f"experimental results/{experimental_filename.replace('/', '_')}.xlsx")
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(rows), len(columns) - 1, {'data': rows, 'columns': columns})
    workbook.close()


def run_W_single_experiment(domain_name,
                            ml_model_name,
                            render_mode,
                            max_exec_len,
                            debug_print,
                            execution_fault_mode_name,
                            instance_seed,
                            fault_probability,
                            percent_visible_states,
                            possible_fault_mode_names,
                            num_candidate_fault_modes
                            ):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run W
    raw_output = W(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, [], 0,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            {}, output, "W"
                            )
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_W")

    return raw_output["exp_duration_in_ms"], raw_output["exp_memory_at_end"], raw_output["exp_memory_max"]


def run_SN_single_experiment(domain_name,
                             ml_model_name,
                             render_mode,
                             max_exec_len,
                             debug_print,
                             execution_fault_mode_name,
                             instance_seed,
                             fault_probability,
                             percent_visible_states,
                             possible_fault_mode_names,
                             num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SN
    raw_output = SN(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SN"
                            )
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SN")

    return raw_output["exp_duration_in_ms"], raw_output["exp_memory_at_end"], raw_output["exp_memory_max"]


def run_SIF_single_experiment(domain_name,
                              ml_model_name,
                              render_mode,
                              max_exec_len,
                              debug_print,
                              execution_fault_mode_name,
                              instance_seed,
                              fault_probability,
                              percent_visible_states,
                              possible_fault_mode_names,
                              num_candidate_fault_modes):
    # ### prepare the records database to be written to the excel file
    records = []

    # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for rnking
    fault_mode_generator, trajectory_execution, \
        faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                    ml_model_name,
                                                                                                    render_mode,
                                                                                                    max_exec_len,
                                                                                                    debug_print,
                                                                                                    execution_fault_mode_name,
                                                                                                    instance_seed,
                                                                                                    fault_probability)
    print(f'faulty actions indices: {faulty_actions_indices}')

    # ### generate observation mask
    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
    print(f'OBSERVATION MASK: {str(observation_mask)}')
    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
    print(f'observed {len(observation_mask)}/{len(observations)} states')

    # ### mask the states list
    masked_observations = mask_states(observations, observation_mask)

    # ### prepare candidate fault modes
    candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, possible_fault_mode_names, fault_mode_generator)

    # ### run SIF
    raw_output = SIF(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

    # ### ranking the diagnoses
    output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

    # ### preparing record for writing to excel file
    record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, possible_fault_mode_names, num_candidate_fault_modes,
                            render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                            candidate_fault_modes, output, "SIF"
                            )
    records.append(record)

    # ### write records to an excel file
    write_records_to_excel(records, f"single_experiment_{domain_name.split('_')[0]}_SIF")

    return raw_output["exp_duration_in_ms"], raw_output["exp_memory_at_end"], raw_output["exp_memory_max"]


def run_experimental_setup(arguments, render_mode, debug_print, continue_from_saved_file):
    # ### parameters dictionary
    experimental_file_name = arguments[1]
    param_dict = read_json_data(f"experimental inputs/{experimental_file_name}")

    # ### the domain name of this experiment (each experiment file has only one associated domain)
    domain_name = param_dict['domain_name']

    # ### the machine learning model name of this experiment (each experiment file has one associated ml model)
    ml_model_name = param_dict['ml_model_name']

    # ### maximum length of the execution for the experiment (each experiment file has one associated length)
    max_exec_len = 200

    # ### initialize or load indices
    if continue_from_saved_file:
        records_so_far_json = read_json_data('records_so_far.json')
        calculated_instance_numbers = records_so_far_json['calculated_instance_numbers']
        records = records_so_far_json['records']
    else:
        calculated_instance_numbers = []
        records = []

    # ### run the experimental loop
    current_instance_number = 1
    for execution_fault_mode_name_i, execution_fault_mode_name in enumerate(param_dict['possible_fault_mode_names']):
        for fault_probability_i, fault_probability in enumerate(param_dict['fault_probabilities']):
            for instance_seed_i, instance_seed in enumerate(param_dict['instance_seeds']):
                # ### prepare the inputs to the algorithm based on the instance inputs, including inputs for ranking
                fault_mode_generator, trajectory_execution, \
                    faulty_actions_indices, registered_actions, observations = single_experiment_prepare_inputs(domain_name,
                                                                                                                ml_model_name,
                                                                                                                render_mode,
                                                                                                                max_exec_len,
                                                                                                                debug_print,
                                                                                                                execution_fault_mode_name,
                                                                                                                instance_seed,
                                                                                                                fault_probability)
                for percent_visible_states_i, percent_visible_states in enumerate(param_dict['percent_visible_states']):
                    # ### generate observation mask
                    observation_mask = generate_observation_mask(len(observations), percent_visible_states)
                    # ### calculate largest hidden gap
                    longest_hidden_state_sequence = calculate_largest_hidden_gap(observation_mask)
                    print(f'OBSERVATION MASK: {str(observation_mask)}')
                    print(f'LONGEST HIDDEN STATE SEQUENCE: {longest_hidden_state_sequence}')
                    print(f'HIDDEN STATES: {[oi for oi in range(len(observations)) if oi not in observation_mask]}')
                    print(f'observed {len(observation_mask)}/{len(observations)} states')

                    # ### mask the states list
                    masked_observations = mask_states(observations, observation_mask)

                    # ### prepare the next batch of records to be added to the main dataset
                    next_batch_records = []
                    next_batch_calculated_instance_numbers = []
                    for num_candidate_fault_modes_i, num_candidate_fault_modes in enumerate(param_dict['num_candidate_fault_modes']):
                        # skipping calculated instance numbers
                        if current_instance_number in calculated_instance_numbers:
                            # logging
                            now = datetime.now()
                            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                            print(f"{dt_string}: {current_instance_number}/{len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'])}")
                            print(f"execution_fault_mode_name: {execution_fault_mode_name}, fault_probability: {fault_probability}, instance_seed: {instance_seed}, percent_visible_states: {percent_visible_states}, num_candidate_fault_modes: {num_candidate_fault_modes}")
                            print(f"    calculated. SKIPPING")
                            current_instance_number += 1
                        else:
                            # logging
                            now = datetime.now()
                            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                            print(f"{dt_string}: {current_instance_number}/{len(param_dict['possible_fault_mode_names']) * len(param_dict['fault_probabilities']) * len(param_dict['instance_seeds']) * len(param_dict['percent_visible_states']) * len(param_dict['num_candidate_fault_modes'])}")
                            print(f"execution_fault_mode_name: {execution_fault_mode_name}, fault_probability: {fault_probability}, instance_seed: {instance_seed}, percent_visible_states: {percent_visible_states}, num_candidate_fault_modes: {num_candidate_fault_modes}")
                            print(f"not calculated. DIAGNOSING")

                            # ### prepare candidate fault modes
                            candidate_fault_modes = prepare_fault_modes(num_candidate_fault_modes, execution_fault_mode_name, param_dict['possible_fault_mode_names'], fault_mode_generator)

                            # ### run the algorithm
                            diagnoser = diagnosers[param_dict["diagnoser_name"]]
                            raw_output = diagnoser(debug_print=debug_print, render_mode=render_mode, instance_seed=instance_seed, ml_model_name=ml_model_name, domain_name=domain_name, observations=masked_observations, candidate_fault_modes=candidate_fault_modes)

                            # ### ranking the diagnoses
                            if param_dict["diagnoser_name"] == "W":
                                output = rank_diagnoses_WFM(raw_output, registered_actions, debug_print)
                            else:
                                output = rank_diagnoses_SFM(raw_output, registered_actions, debug_print)

                            # ### preparing record for writing to excel file
                            record = prepare_record(domain_name, debug_print, execution_fault_mode_name, instance_seed, fault_probability, percent_visible_states, param_dict['possible_fault_mode_names'], num_candidate_fault_modes,
                                                    render_mode, ml_model_name, max_exec_len, trajectory_execution, faulty_actions_indices, registered_actions, observations, observation_mask, masked_observations,
                                                    candidate_fault_modes, output, param_dict["diagnoser_name"], longest_hidden_state_sequence)
                            next_batch_records.append(record)
                            next_batch_calculated_instance_numbers.append(current_instance_number)
                            current_instance_number += 1
                    records = records + next_batch_records
                    calculated_instance_numbers = calculated_instance_numbers + next_batch_calculated_instance_numbers

                    # ### saving records so far
                    records_so_far = {
                        'calculated_instance_numbers': calculated_instance_numbers,
                        'records': records
                    }
                    with open('records_so_far.json', 'w') as f:
                        json.dump(records_so_far, f)
                    print(f"saved records up to instance number {calculated_instance_numbers[-1]}")
                    print(9)

    # ### write records to an excel file
    write_records_to_excel(records, experimental_file_name.split(".")[0])

    print(9)
