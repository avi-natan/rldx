import gym

from h_consts import EPISODES, DETERMINISTIC, SEED, RENDER_MODE
from h_fault_model_generator import FaultModelGeneratorDiscrete, same_box_action
from h_rl_models import models


def execute(env_name, model_name, total_timesteps, fault_model_generator, execution_fault_model_representation):
    print(f'executing with fault model: {execution_fault_model_representation}\n========================================================================================\n')

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode=RENDER_MODE)
    initial_obs, _ = env.reset(seed=42)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[model_name].load(model_path, env=env)

    # initialize execution fault model
    execution_fault_model = fault_model_generator.generate_fault_model(execution_fault_model_representation)

    # initializing empty trajectory
    trajectory = []

    episodes = EPISODES
    action_number = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            env.render()
            trajectory.append(obs)
            print(f'EP:{ep} a#:{action_number} [PREVOBS]: {obs.tolist()}')
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            if isinstance(fault_model_generator, FaultModelGeneratorDiscrete):
                trajectory.append(int(action))
            else:
                trajectory.append(action)
            faulty_action = execution_fault_model(action)
            if isinstance(fault_model_generator, FaultModelGeneratorDiscrete):
                if action != faulty_action:
                    print(f'EP:{ep} a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}\n')
                else:
                    print(f'EP:{ep} a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}\n')
            else:
                if not same_box_action(action, faulty_action):
                    print(f'EP:{ep} a#:{action_number} [FAILURE] - planned: {action.tolist()}, \nEP:{ep} a#:{action_number} [FAILURE] - actual : {faulty_action.tolist()}\n')
                else:
                    print(f'EP:{ep} a#:{action_number} [SUCCESS] - planned: {action.tolist()}, \nEP:{ep} a#:{action_number} [SUCCESS] - actual : {faulty_action.tolist()}\n')
            obs, reward, done, trunc, info = env.step(faulty_action)
            action_number += 1

    env.close()

    return trajectory
