import random

import gym

from h_consts import DETERMINISTIC
from h_rl_models import models


def execute(domain_name,
            debug_print,
            execution_fault_mode_name,
            instance_seed,
            fault_probability,
            render_mode,
            ml_model_name,
            fault_mode_generator,
            max_exec_len):
    print(f'executing with fault mode: {execution_fault_mode_name}\n========================================================================================')

    # initialize environment
    env = gym.make(domain_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{domain_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{domain_name}__{ml_model_name}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode = fault_mode_generator.generate_fault_model(execution_fault_mode_name)

    # initializing empty trajectory
    trajectory = []

    faulty_actions_indices = []
    action_number = 1
    done = False
    exec_len = 1
    obs, _ = env.reset()
    while not done and exec_len < max_exec_len:
        trajectory.append(obs)
        if debug_print:
            print(f'a#:{action_number} [PREVOBS]: {obs.tolist() if not isinstance(obs, int) else obs}')
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        action = int(action)
        trajectory.append(action)
        if random.random() < fault_probability:
            faulty_action = execution_fault_mode(action)
        else:
            faulty_action = action
        if faulty_action != action:
            faulty_actions_indices.append(action_number)
        if debug_print:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}\n')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}\n')
        obs, reward, done, trunc, info = env.step(faulty_action)
        action_number += 1
        exec_len += 1

    env.close()

    return trajectory, faulty_actions_indices
