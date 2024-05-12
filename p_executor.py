import random

import gym

from h_consts import EPISODES, DETERMINISTIC
from h_fault_model_generator import FaultModelGeneratorDiscrete, same_box_action
from h_rl_models import models


def execute(env_name,
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
    print(f'executing with fault model: {fault_model}\n========================================================================================')

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    initial_obs, _ = env.reset(seed=instance_seed)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{ml_model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[ml_model_name].load(model_path, env=env)

    # initialize execution fault model
    execution_fault_model = fault_model_generator.generate_fault_model(fault_model)

    # initializing empty trajectory
    trajectory = []

    episodes = EPISODES
    action_number = 1
    faulty_actions_indices = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        exec_len = 0
        while not done and exec_len < max_exec_len:
            # env.render()
            trajectory.append(obs)
            # print(f'EP:{ep} a#:{action_number} [PREVOBS]: {obs.tolist()}')
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            if isinstance(fault_model_generator, FaultModelGeneratorDiscrete):
                trajectory.append(int(action))
            else:
                trajectory.append(action)
            if fault_model_type == "deterministic":
                faulty_action = execution_fault_model(action)
                if isinstance(fault_model_generator, FaultModelGeneratorDiscrete):
                    if faulty_action != action:
                        faulty_actions_indices.append(action_number)
                else:
                    if not same_box_action(action, faulty_action):
                        faulty_actions_indices.append(action_number)
            else:
                if random.random() < fault_probability:
                    faulty_action = execution_fault_model(action)
                else:
                    faulty_action = action
                if isinstance(fault_model_generator, FaultModelGeneratorDiscrete):
                    if faulty_action != action:
                        faulty_actions_indices.append(action_number)
                else:
                    if not same_box_action(action, faulty_action):
                        faulty_actions_indices.append(action_number)
            # if isinstance(fault_model_generator, FaultModelGeneratorDiscrete):
            #     if action != faulty_action:
            #         print(f'EP:{ep} a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}\n')
            #     else:
            #         print(f'EP:{ep} a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}\n')
            # else:
            #     if not same_box_action(action, faulty_action):
            #         print(f'EP:{ep} a#:{action_number} [FAILURE] - planned: {action.tolist()}, \nEP:{ep} a#:{action_number} [FAILURE] - actual : {faulty_action.tolist()}\n')
            #     else:
            #         print(f'EP:{ep} a#:{action_number} [SUCCESS] - planned: {action.tolist()}, \nEP:{ep} a#:{action_number} [SUCCESS] - actual : {faulty_action.tolist()}\n')
            obs, reward, done, trunc, info = env.step(faulty_action)
            action_number += 1
            exec_len += 1

    env.close()

    return trajectory, faulty_actions_indices
