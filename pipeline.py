import platform

from h_common import output_trajectory_to_file
from h_fault_model_generator_list import fault_model_generators
from p_diagnoser import diagnosers
from p_executor import execute


def run_pipeline(env_name, render_display, render_terminal, model_name, total_timesteps, fault_model_generator_name, available_fault_model_representations, execution_fault_model_representation, diagnoser_name, observation_mask):
    # determine the render mode
    if platform.system() == "Windows":
        render_mode = render_display
    else:
        render_mode = render_terminal

    # initialize fault model generator
    fault_model_generator = fault_model_generators[fault_model_generator_name]

    # execute to get observations
    trajectory_execution = execute(env_name, render_mode, model_name, total_timesteps, fault_model_generator, execution_fault_model_representation)
    if len(trajectory_execution) % 2 != 1:
        trajectory_execution = trajectory_execution[:-1]

    # save trajectory
    output_trajectory_to_file(f'environments/{env_name}/trajectories/{model_name}', f'traj-{total_timesteps}-exec-{execution_fault_model_representation}.txt', trajectory_execution)

    # separating trajectory to actions and states
    lst_actions = []
    lst_states = []
    for i in range(len(trajectory_execution)):
        if i % 2 == 1:
            lst_actions.append(trajectory_execution[i])
        else:
            lst_states.append(trajectory_execution[i])

    # prepare fault models
    available_fault_models = {}
    for rep in available_fault_model_representations:
        fault_model = fault_model_generator.generate_fault_model(rep)
        available_fault_models[rep] = fault_model

    # make the observation partial if this is required
    if observation_mask == [-1]:
        for i in range(1, len(lst_states)-1):
            lst_states[i] = None
    elif len(observation_mask) != 0:
        for i in range(1, len(lst_states)-1):
            if i not in observation_mask:
                lst_states[i] = None
    print(9)

    # use the diagnoser to diagnose
    diagnose = diagnosers[diagnoser_name]
    output = diagnose(env_name, render_mode, model_name, total_timesteps, lst_actions, lst_states, available_fault_models)
    for key in output.keys():
        print(f'{key}: {output[key]}')
    print(9)
