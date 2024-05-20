import copy
import json
import math
import platform
import random
import sys
import xlsxwriter

from h_common import read_experimental_params
from h_fault_model_generator_list import fault_model_generators
from p_diagnoser import diagnosers
from p_executor import execute
from p_trainer import train_on_environment


# from pipeline import run_pipeline

def run_single_experiment(env_name,
                          render_display,
                          render_terminal,
                          ml_model_name,
                          total_timesteps,
                          action_type,
                          max_exec_len,
                          fault_model_type,
                          diagnoser,
                          debug_print,
                          instance_seed,
                          fault_model,
                          fault_probability,
                          obs_percent_visible,
                          obs_percent_mean,
                          obs_percent_dev,
                          fault_model_names,
                          sample_size):
    # determine the render mode
    render_mode = determine_render_mode(render_display, render_terminal)
    # initialize fault model generator
    fault_model_generator = fault_model_generators[action_type]
    # execute to get trajectory
    trajectory_execution, faulty_actions_indices = execute_trajectory(env_name,
                                                                      render_mode,
                                                                      ml_model_name,
                                                                      total_timesteps,
                                                                      fault_model_generator,
                                                                      max_exec_len,
                                                                      fault_model_type,
                                                                      debug_print,
                                                                      instance_seed,
                                                                      fault_model,
                                                                      fault_probability)
    # separating trajectory to actions and states
    lst_actions, lst_states = separate_trajectory(trajectory_execution)
    # generate observation mask
    observation_mask = generate_observation_mask(len(lst_states), obs_percent_visible, obs_percent_mean, obs_percent_dev)
    # mask the states list
    lst_states_masked = mask_states(lst_states, observation_mask)
    # prepare fault models
    # prepare fault models
    fault_models = {}
    for f in fault_model_names:
        fm = fault_model_generator.generate_fault_model(f)
        fault_models[f] = fm
    # use the diagnoser to diagnose
    diagnose = diagnosers[diagnoser]
    output = diagnose(env_name=env_name,
                      render_mode=render_mode,
                      ml_model_name=ml_model_name,
                      total_timesteps=total_timesteps,
                      fault_model=fault_model,
                      lst_actions=lst_actions,
                      lst_states=lst_states_masked,
                      instance_seed=instance_seed,
                      fault_probability=fault_probability,
                      fault_models=fault_models,
                      sample_size=sample_size)
    print(f'faulty_actions_indices: {faulty_actions_indices}')
    for key in output.keys():
        print(f'{key}: {output[key]}')


# determine the render mode
def determine_render_mode(render_display, render_terminal):
    if platform.system() == "Windows":
        return render_display
    else:
        return render_terminal


# execute to get trajectory
def execute_trajectory(env_name,
                       render_mode,
                       ml_model_name,
                       total_timesteps,
                       fault_model_generator,
                       max_exec_len,
                       fault_model_type,
                       debug_print,
                       instance_seed,
                       fault_model,
                       fault_probability):
    trajectory_execution = []
    faulty_actions_indices = []
    while len(faulty_actions_indices) == 0:
        trajectory_execution, faulty_actions_indices = execute(env_name,
                                                               render_mode,
                                                               ml_model_name,
                                                               total_timesteps,
                                                               fault_model_generator,
                                                               max_exec_len,
                                                               fault_model_type,
                                                               debug_print,
                                                               instance_seed,
                                                               fault_model,
                                                               fault_probability)
    if len(trajectory_execution) % 2 != 1:
        if len(faulty_actions_indices) > 0 and faulty_actions_indices[-1] * 2 == len(trajectory_execution):
            trajectory_execution = trajectory_execution[:-1]
            faulty_actions_indices = faulty_actions_indices[:-1]
        else:
            trajectory_execution = trajectory_execution[:-1]
    return trajectory_execution, faulty_actions_indices


# separating trajectory to actions and states
def separate_trajectory(trajectory_execution):
    lst_actions = []
    lst_states = []
    for i in range(len(trajectory_execution)):
        if i % 2 == 1:
            lst_actions.append(trajectory_execution[i])
        else:
            lst_states.append(trajectory_execution[i])
    return lst_actions, lst_states


# prepare fault models
def prepare_fault_models(num_fault_models, fault_model_representation, all_fault_models_representations, fault_model_generator):
    if num_fault_models == 0:
        fault_models_representations = []
    else:
        fault_models_representations = [fault_model_representation]
        rest = copy.deepcopy(all_fault_models_representations)
        rest.remove(fault_model_representation)
        i = 0
        while i < num_fault_models - 1:
            fmr = random.choice(rest)
            fault_models_representations.append(fmr)
            rest.remove(fmr)
            i += 1
    fault_models = {}
    for fmr in fault_models_representations:
        fm = fault_model_generator.generate_fault_model(fmr)
        fault_models[fmr] = fm
    return fault_models


def generate_observation_mask(obs_size, obs_p_visible, obs_p_mean, obs_p_dev):
    mask = [0] * obs_size
    ones = math.floor(obs_size * obs_p_visible / 100.0)
    mean_index = math.floor(obs_size * obs_p_mean / 100.0)  # Adjust this to set the mean index
    deviation = max(math.ceil(ones / 2), obs_p_dev)
    # Calculate the range around the mean index
    start_index = max(0, mean_index - deviation)
    end_index = min(len(mask) - 1, mean_index + deviation)
    # fix start and end indices at the edges
    if end_index - start_index < ones:
        missing = ones - (end_index - start_index)
        if start_index == 0:
            end_index = end_index + missing
        elif end_index == obs_size - 1:
            start_index = start_index - missing
        else:
            raise Exception("this shouldnt happen (main line 171)")
    # Distribute the ones within the range
    indices = random.sample(range(start_index, end_index), ones)
    for i in indices:
        mask[i] = 1
    mask[0] = 1
    mask[len(mask) - 1] = 1

    result = []
    for i in range(len(mask)):
        if mask[i] == 1:
            result.append(i)

    # print(f"{mask}; {obs_p_visible},{obs_p_mean},{obs_p_dev};;;{mean_index},{deviation};;;{ones},{start_index},{end_index}")
    return result


def mask_states(lst_states, observation_mask):
    lst_states_masked = copy.deepcopy(lst_states)
    for i in range(1, len(lst_states) - 1):
        if i not in observation_mask:
            lst_states_masked[i] = None
    return lst_states_masked


def run_experimental_setup(arguments):
    # parameters dictionary
    experimental_file_name = arguments[1]
    param_dict = read_experimental_params(f"experimental inputs/{experimental_file_name}")

    # prepare the outputs database
    database = []

    # run the experimental loop
    total = 1
    for pci, instance_seed in enumerate(param_dict["pci_instance_seed"]):
        for pcf, fault_model in enumerate(param_dict["pcf_fault_models"]):
            for ip1, fault_probability in enumerate(param_dict["ip1_fault_probabilities"]):
                # determine the render mode
                render_mode = determine_render_mode(param_dict["c2_render_display"], param_dict["c3_render_terminal"])
                # initialize fault model generator
                fault_model_generator = fault_model_generators[param_dict["c6_action_type"]]
                # execute to get trajectory
                trajectory_execution, faulty_actions_indices = execute_trajectory(param_dict["c1_env_name"],
                                                                                  render_mode,
                                                                                  param_dict["c4_ml_model_name"],
                                                                                  param_dict["c5_total_timesteps"],
                                                                                  fault_model_generator,
                                                                                  param_dict["c9_max_exec_len"],
                                                                                  param_dict["c10_debug_print"],
                                                                                  param_dict["c11_fault_model_type"],
                                                                                  instance_seed,
                                                                                  fault_model,
                                                                                  fault_probability)
                print(f'faulty_actions_indices: {faulty_actions_indices}')
                # separating trajectory to actions and states
                lst_actions, lst_states = separate_trajectory(trajectory_execution)
                for ip2, obs_p_visible in enumerate(param_dict["ip2_obs_percent_visible"]):
                    for ip3, obs_p_mean in enumerate(param_dict["ip3_obs_percent_mean"]):
                        for ip4, obs_p_dev in enumerate(param_dict["ip4_obs_percent_dev"]):
                            # generate observation mask
                            observation_mask = generate_observation_mask(len(lst_states), obs_p_visible, obs_p_mean, obs_p_dev)
                            # mask the states list
                            lst_states_masked = mask_states(lst_states, observation_mask)
                            for ip5, num_fault_models in enumerate(param_dict["ip5_num_fault_models"]):
                                for ip6, sample_size in enumerate(param_dict["ip6_sample_sizes"]):
                                    print(f"({total}/{len(param_dict['pci_instance_seed']) * len(param_dict['pcf_fault_models']) * len(param_dict['ip1_fault_probabilities']) * len(param_dict['ip2_obs_percent_visible']) * len(param_dict['ip3_obs_percent_mean']) * len(param_dict['ip4_obs_percent_dev']) * len(param_dict['ip5_num_fault_models']) * len(param_dict['ip6_sample_sizes'])})"
                                          f" {pci + 1}/{pcf + 1}/{ip1 + 1}/{ip2 + 1}/{ip3 + 1}/{ip4 + 1}/{ip5 + 1}/{ip6 + 1} "
                                          f"instance_seed: {instance_seed}, fault_model: {fault_model}, fault_probability: {fault_probability}, obs_p_visible: {obs_p_visible}, obs_p_mean: {obs_p_mean}, obs_p_dev: {obs_p_dev}, num_fault_models: {num_fault_models}, sample_size: {sample_size}")

                                    # prepare fault models
                                    fault_models = prepare_fault_models(num_fault_models, fault_model, param_dict["pcf_fault_models"], fault_model_generator)

                                    # use the diagnoser to diagnose
                                    diagnose = diagnosers[param_dict["c7_diagnoser"]]
                                    output = diagnose(env_name=param_dict["c1_env_name"],
                                                      render_mode=render_mode,
                                                      ml_model_name=param_dict["c4_ml_model_name"],
                                                      total_timesteps=param_dict["c5_total_timesteps"],
                                                      fault_model=fault_model,
                                                      lst_actions=lst_actions,
                                                      lst_states=lst_states_masked,
                                                      instance_seed=instance_seed,
                                                      fault_probability=fault_probability,
                                                      fault_models=fault_models,
                                                      sample_size=sample_size)

                                    # # print out the output
                                    # print(f'faulty_actions_indices: {faulty_actions_indices}')
                                    # print(f'fm: {output["fm"]}')
                                    # print(f'pof: {output["pof"]}')
                                    for key in output.keys():
                                        print(f'{key}: {output[key]}')
                                    print("")

                                    # register a database line
                                    database_row = [
                                        total,
                                        param_dict["c1_env_name"],
                                        param_dict["c2_render_display"],
                                        param_dict["c3_render_terminal"],
                                        param_dict["c4_ml_model_name"],
                                        param_dict["c5_total_timesteps"],
                                        param_dict["c6_action_type"],
                                        param_dict["c7_diagnoser"],
                                        param_dict["c8_normal_model"],
                                        param_dict["c9_max_exec_len"],
                                        param_dict["c10_debug_print"],
                                        param_dict["c11_fault_model_type"],
                                        instance_seed,
                                        fault_model,
                                        fault_probability,
                                        lst_actions if param_dict["c10_debug_print"] == 1 else "Omitted",
                                        len(lst_actions),
                                        lst_states if param_dict["c10_debug_print"] == 1 else "Omitted",
                                        len(lst_states),
                                        obs_p_visible,
                                        obs_p_mean,
                                        obs_p_dev,
                                        observation_mask if param_dict["c10_debug_print"] == 1 else "Omitted",
                                        lst_states_masked if param_dict["c10_debug_print"] == 1 else "Omitted",
                                        len([x for x in lst_states_masked if x is not None]),
                                        next((i for i, x in enumerate(lst_states_masked[1:], 1) if x is not None), None),
                                        num_fault_models,
                                        str(list(fault_models.keys())),
                                        sample_size,
                                        json.dumps(output) if param_dict["c10_debug_print"] == 1 else "Omitted",
                                        str(output['fault_occurence_range']) if param_dict["c7_diagnoser"] == 'W' else "Irrelevant",
                                        output['fault_occurence_range'][0] if param_dict["c7_diagnoser"] == 'W' else "Irrelevant",
                                        output['fault_occurence_range'][1] if param_dict["c7_diagnoser"] == 'W' else "Irrelevant",
                                        output['fault_occurence_range'][1]-output['fault_occurence_range'][0] if param_dict["c7_diagnoser"] == 'W' else "Irrelevant",
                                        str(output['fm']) if param_dict["c7_diagnoser"] == 'SD' or param_dict["c7_diagnoser"] == 'SSF_SMP' or param_dict["c7_diagnoser"] == 'SSF_PF' else "Irrelevant",
                                        len(output['fm']) if param_dict["c7_diagnoser"] == 'SD' or param_dict["c7_diagnoser"] == 'SSF_SMP' or param_dict["c7_diagnoser"] == 'SSF_PF' else "Irrelevant",
                                        output['fm_rank'] if param_dict["c7_diagnoser"] == 'SD' or param_dict["c7_diagnoser"] == 'SSF_SMP' or param_dict["c7_diagnoser"] == 'SSF_PF' else "Irrelevant",
                                        output['runtime_sec'],
                                        output['runtime_ms']
                                    ]
                                    database.append(database_row)

                                    # increase the number of experiments count
                                    total += 1

    # output the database to a result file
    columns = [
        {'header': 'id'},
        {'header': 'c1_env_name'},
        {'header': 'c2_render_display'},
        {'header': 'c3_render_terminal'},
        {'header': 'c4_ml_model_name'},
        {'header': 'c5_total_timesteps'},
        {'header': 'c6_action_type'},
        {'header': 'c7_diagnoser'},
        {'header': 'c8_normal_model'},
        {'header': 'c9_max_exec_len'},
        {'header': 'c10_debug_print'},
        {'header': 'c11_fault_model_type'},
        {'header': 'pci_instance_seed'},
        {'header': 'pcf_fault_model'},
        {'header': 'ip1_fault_probability'},
        {'header': 'bpp_lst_actions'},
        {'header': 'bpp_len_actions'},
        {'header': 'bpp_lst_states'},
        {'header': 'bpp_len_states'},
        {'header': 'ip2_obs_percent_visible'},
        {'header': 'ip3_obs_percent_mean'},
        {'header': 'ip4_obs_percent_dev'},
        {'header': 'bpp_observation_mask'},
        {'header': 'bpp_lst_states_masked'},
        {'header': 'bpp_num_known_states'},
        {'header': 'bpp_first_known_post_s0'},
        {'header': 'ip5_num_fault_models'},
        {'header': 'bpp_fault_models'},
        {'header': 'ip6_sample_size'},
        {'header': 'dp1_output'},
        {'header': 'dp2_W_fault_occurence_range'},
        {'header': 'dp3_W_range_start'},
        {'header': 'dp4_W_range_end'},
        {'header': 'dp5_W_range_length'},
        {'header': 'dp6_SD_SSF_determined_fault_models'},
        {'header': 'dp7_SD_SSF_determined_fault_models_num'},
        {'header': 'dp8_SD_SSF_determined_fault_models_rank'},
        {'header': 'dp9_algall_runtime_sec'},
        {'header': 'dp10_algall_runtime_ms'}
    ]
    workbook = xlsxwriter.Workbook(f"experimental results/{experimental_file_name[:-5]}.xlsx")
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(database), len(columns) - 1, {'data': database, 'columns': columns})
    workbook.close()
    print(9)


if __name__ == '__main__':
    '''
    envs: LunarLander_v2, Ant_v4, ALE/SpaceInvaders_v5, ALE/AirRaid_v5 (debug)
    models: PPO, A2C, DQN
    fault_model_generators: discrete, box
    '''

    # train_on_environment("LunarLander_v2", "human", None, "PPO")
    # train_on_environment("Ant_v4", "human", None, "PPO")
    # train_on_environment("ALE/SpaceInvaders_v5", "human", "rgb_array", "PPO")
    # train_on_environment("CliffWalking-v0", "human", "rgb_array", "PPO")

    # ================== single experiments ==================
    ''' run_pipeline(env_name, render_display, render_terminal, ml_model_name, total_timesteps, action_type, diagnoser, fault_models, fault_model, observation_mask) '''
    '''
    run_single_experiment(env_name, 
                          render_display, 
                          render_terminal, 
                          ml_model_name, 
                          total_timesteps, 
                          action_type, 
                          max_exec_len, 
                          fault_model_type
                          diagnoser, 
                          debug_print, 
                          instance_seed, 
                          fault_model, 
                          fault_probability, 
                          obs_percent_visible, 
                          obs_percent_mean, 
                          obs_percent_dev, 
                          fault_model_names,
                          sample_size)
    '''

    # ================== debug setup algs domains ============
    # ALG1: DWF
    # ALG1DOM1
    # run_single_experiment("LunarLander_v2",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DWF",
    #                       0,
    #                       42,
    #                       "[0,0,2,3]",
    #                       1.0,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #                        "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"  # overworking jets
    #                        ],
    #                       -1)
    # ALG1DOM2
    # run_single_experiment("Ant_v4",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "box",
    #                       200,
    #                       "deterministic",
    #                       "DWF",
    #                       0,
    #                       42,
    #                       "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #                       1.0,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #                        "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #                        "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #                        "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #                        "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 1
    #                        "[1,1,-1,-1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 2
    #                        "[1,1,1,1,-1,-1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 3
    #                        "[1,1,1,1,1,1,-1,-1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 4
    #                        "[1,1,1,1,1,1,1,1];[0.5,0.5,0,0,0,0,0,0];-1;1",  # restraining leg 1
    #                        "[1,1,1,1,1,1,1,1];[0,0,0.5,0.5,0,0,0,0];-1;1",  # restraining leg 2
    #                        ],
    #                       -1)
    # ALG1DOM3
    # run_single_experiment("ALE/SpaceInvaders_v5",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DWF",
    #                       0,
    #                       42,
    #                       "[0,1,2,0,4,0]",
    #                       1.0,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #                        "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,2,0,4,0]"  # shutting down 2 actions
    #                        # "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,0,0,4,5]", "[0,1,2,3,0,0]",
    #                        # "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #                        # "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #                        # "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #                        # "[0,0,0,0,0,0]",  # shutting down 5 actions
    #                        # "[0,2,1,3,4,5]"  # swapping fire for going right
    #                        ],
    #                       -1)
    # ALG2: DSF
    # ALG2DOM1
    # run_single_experiment("LunarLander_v2",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DSF",
    #                       0,
    #                       42,
    #                       "[0,0,2,3]",
    #                       1.0,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #                        "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"  # overworking jets
    #                        ],
    #                       -1)
    # ALG2DOM2
    # run_single_experiment("Ant_v4",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "box",
    #                       200,
    #                       "deterministic",
    #                       "DSF",
    #                       0,
    #                       42,
    #                       "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #                       1.0,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #                        "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #                        "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #                        "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #                        "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 1
    #                        "[1,1,-1,-1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 2
    #                        "[1,1,1,1,-1,-1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 3
    #                        "[1,1,1,1,1,1,-1,-1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 4
    #                        "[1,1,1,1,1,1,1,1];[0.5,0.5,0,0,0,0,0,0];-1;1",  # restraining leg 1
    #                        "[1,1,1,1,1,1,1,1];[0,0,0.5,0.5,0,0,0,0];-1;1",  # restraining leg 2
    #                        ],
    #                       -1)
    # ALG2DOM3
    # run_single_experiment("ALE/SpaceInvaders_v5",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DSF",
    #                       0,
    #                       42,
    #                       "[0,1,2,0,4,0]",
    #                       1.0,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #                        "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,2,0,4,0]"  # shutting down 2 actions
    #                        # "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,0,0,4,5]", "[0,1,2,3,0,0]",
    #                        # "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #                        # "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #                        # "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #                        # "[0,0,0,0,0,0]",  # shutting down 5 actions
    #                        # "[0,2,1,3,4,5]"  # swapping fire for going right
    #                        ],
    #                       -1)
    # ALG3: DWP
    # ALG3DOM1
    # run_single_experiment("LunarLander_v2",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DWP",
    #                       0,
    #                       42,
    #                       "[0,0,2,3]",
    #                       1.0,
    #                       0,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #                        "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"  # overworking jets
    #                        ],
    #                       -1)
    # ALG3DOM2
    # run_single_experiment("Ant_v4",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "box",
    #                       200,
    #                       "deterministic",
    #                       "DWP",
    #                       0,
    #                       42,
    #                       "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #                       1.0,
    #                       0,
    #                       50,
    #                       100,
    #                       ["[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #                        "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #                        "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #                        "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #                        "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 1
    #                        "[1,1,-1,-1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 2
    #                        "[1,1,1,1,-1,-1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 3
    #                        "[1,1,1,1,1,1,-1,-1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 4
    #                        "[1,1,1,1,1,1,1,1];[0.5,0.5,0,0,0,0,0,0];-1;1",  # restraining leg 1
    #                        "[1,1,1,1,1,1,1,1];[0,0,0.5,0.5,0,0,0,0];-1;1",  # restraining leg 2
    #                        ],
    #                       -1)
    # ALG3DOM3
    # run_single_experiment("ALE/SpaceInvaders_v5",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DWP",
    #                       0,
    #                       42,
    #                       "[0,1,2,0,4,0]",
    #                       1.0,
    #                       0,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #                        "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,2,0,4,0]"  # shutting down 2 actions
    #                        # "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,0,0,4,5]", "[0,1,2,3,0,0]",
    #                        # "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #                        # "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #                        # "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #                        # "[0,0,0,0,0,0]",  # shutting down 5 actions
    #                        # "[0,2,1,3,4,5]"  # swapping fire for going right
    #                        ],
    #                       -1)
    # ALG4: DSP
    # ALG4DOM1
    # run_single_experiment("LunarLander_v2",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DSP",
    #                       0,
    #                       42,
    #                       "[0,0,2,3]",
    #                       1.0,
    #                       0,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #                        "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"  # overworking jets
    #                        ],
    #                       -1)
    # ALG4DOM2
    # run_single_experiment("Ant_v4",
    #                       "human",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "box",
    #                       200,
    #                       "deterministic",
    #                       "DSP",
    #                       0,
    #                       42,
    #                       "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #                       1.0,
    #                       0,
    #                       50,
    #                       100,
    #                       ["[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #                        "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #                        "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #                        "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #                        "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 1
    #                        "[1,1,-1,-1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 2
    #                        "[1,1,1,1,-1,-1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 3
    #                        "[1,1,1,1,1,1,-1,-1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 4
    #                        "[1,1,1,1,1,1,1,1];[0.5,0.5,0,0,0,0,0,0];-1;1",  # restraining leg 1
    #                        "[1,1,1,1,1,1,1,1];[0,0,0.5,0.5,0,0,0,0];-1;1",  # restraining leg 2
    #                        ],
    #                       -1)
    # ALG4DOM3
    # run_single_experiment("ALE/SpaceInvaders_v5",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "deterministic",
    #                       "DSP",
    #                       0,
    #                       42,
    #                       "[0,1,2,0,4,0]",
    #                       1.0,
    #                       0,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #                        "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,2,0,4,0]"  # shutting down 2 actions
    #                        # "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,0,0,4,5]", "[0,1,2,3,0,0]",
    #                        # "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #                        # "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #                        # "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #                        # "[0,0,0,0,0,0]",  # shutting down 5 actions
    #                        # "[0,2,1,3,4,5]"  # swapping fire for going right
    #                        ],
    #                       -1)
    # ALG5: SSF_PF
    # ALG5DOM1
    # run_single_experiment("LunarLander_v2",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "stochastic",
    #                       "SSF_PF",
    #                       0,
    #                       42,
    #                       "[0,0,2,3]",
    #                       0.5,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #                        "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"  # overworking jets
    #                        ],
    #                       -1)
    # ALG5DOM2
    # run_single_experiment("Ant_v4",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "box",
    #                       200,
    #                       "stochastic",
    #                       "SSF_PF",
    #                       0,
    #                       42,
    #                       "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #                       0.5,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #                        "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #                        "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #                        "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #                        "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 1
    #                        "[1,1,-1,-1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 2
    #                        "[1,1,1,1,-1,-1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 3
    #                        "[1,1,1,1,1,1,-1,-1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 4
    #                        "[1,1,1,1,1,1,1,1];[0.5,0.5,0,0,0,0,0,0];-1;1",  # restraining leg 1
    #                        "[1,1,1,1,1,1,1,1];[0,0,0.5,0.5,0,0,0,0];-1;1",  # restraining leg 2
    #                        ],
    #                       -1)
    # ALG5DOM3
    # run_single_experiment("ALE/SpaceInvaders_v5",  # todo check later maybe
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "stochastic",
    #                       "SSF_PF",
    #                       0,
    #                       42,
    #                       "[0,1,2,0,4,0]",
    #                       0.5,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #                        "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,2,0,4,0]"  # shutting down 2 actions
    #                        # "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,0,0,4,5]", "[0,1,2,3,0,0]",
    #                        # "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #                        # "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #                        # "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #                        # "[0,0,0,0,0,0]",  # shutting down 5 actions
    #                        # "[0,2,1,3,4,5]"  # swapping fire for going right
    #                        ],
    #                       -1)
    # ALG6: SSF_SMP
    # ALG6DOM1
    # run_single_experiment("LunarLander_v2",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "stochastic",
    #                       "SSF_SMP",
    #                       0,
    #                       42,
    #                       "[0,0,2,3]",
    #                       0.5,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #                        "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"  # overworking jets
    #                        ],
    #                       10)
    # ALG6DOM2
    # run_single_experiment("Ant_v4",
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "box",
    #                       200,
    #                       "stochastic",
    #                       "SSF_SMP",
    #                       0,
    #                       42,
    #                       "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #                       0.5,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #                        "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #                        "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #                        "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #                        "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 1
    #                        "[1,1,-1,-1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 2
    #                        "[1,1,1,1,-1,-1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 3
    #                        "[1,1,1,1,1,1,-1,-1];[0,0,0,0,0,0,0,0];-1;1",  # inverting leg 4
    #                        "[1,1,1,1,1,1,1,1];[0.5,0.5,0,0,0,0,0,0];-1;1",  # restraining leg 1
    #                        "[1,1,1,1,1,1,1,1];[0,0,0.5,0.5,0,0,0,0];-1;1",  # restraining leg 2
    #                        ],
    #                       10)
    # ALG6DOM3
    # run_single_experiment("ALE/SpaceInvaders_v5",  # todo check later maybe
    #                       "rgb_array",
    #                       "rgb_array",
    #                       "PPO",
    #                       90000,
    #                       "discrete",
    #                       200,
    #                       "stochastic",
    #                       "SSF_PF",
    #                       0,
    #                       42,
    #                       "[0,1,2,0,4,0]",
    #                       0.5,
    #                       100,
    #                       50,
    #                       100,
    #                       ["[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #                        "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,2,0,4,0]"  # shutting down 2 actions
    #                        # "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,0,0,4,5]", "[0,1,2,3,0,0]",
    #                        # "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #                        # "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #                        # "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #                        # "[0,0,0,0,0,0]",  # shutting down 5 actions
    #                        # "[0,2,1,3,4,5]"  # swapping fire for going right
    #                        ],
    #                       10)

    # ================== experimental setup ==================
    run_experimental_setup(sys.argv)

    print(9)
