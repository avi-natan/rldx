import gym

from h_common import output_trajectory_to_file
from h_fault_model_generator import FaultModelGeneratorDiscrete, same_box_action
from h_fault_model_generator_list import fault_model_generators
from h_rl_models import models


def run_pipeline(env_name, model_name, total_timesteps, fault_model_generator_name, available_fault_model_representations, execution_fault_model_representation):
    # initialize fault model generator
    fault_model_generator = fault_model_generators[fault_model_generator_name]

    # initialize available fault models
    available_fault_models = {}
    for rep in available_fault_model_representations:
        fault_model = fault_model_generator.generate_fault_model(rep)
        available_fault_models[rep] = fault_model

    # running the actual execution
    print(f'simulating with fault model: {execution_fault_model_representation}\n========================================================================================\n')
    trajectory_execution = simulate_with_fault_model(env_name, model_name, total_timesteps, fault_model_generator, available_fault_models[execution_fault_model_representation])
    output_trajectory_to_file(f'environments/{env_name}/trajectories/{model_name}', f'traj-{total_timesteps}-exec-{execution_fault_model_representation}.txt', trajectory_execution)

    # simulating with the list of fault models
    for fault_model_name in available_fault_models.keys():
        print(f'simulating with fault model: {fault_model_name}\n========================================================================================\n')
        trajectory_simulation = simulate_with_fault_model(env_name, model_name, total_timesteps, fault_model_generator, available_fault_models[fault_model_name])
        output_trajectory_to_file(f'environments/{env_name}/trajectories/{model_name}', f'traj-{total_timesteps}-simu-{fault_model_name}.txt', trajectory_simulation)

    print(9)

def simulate_with_fault_model(env_name, model_name, total_timesteps, fault_model_generator, execution_fault_model):
    # initializing empty trajectory
    trajectory = []

    # initialize environment
    env = gym.make(env_name.replace('_', '-'), render_mode="human")
    initial_obs, _ = env.reset(seed=42)
    print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    models_dir = f"environments/{env_name}/models/{model_name}"
    model_path = f"{models_dir}/{total_timesteps}.zip"
    model = models[model_name].load(model_path, env=env)

    episodes = 1
    action_number = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            env.render()
            trajectory.append(obs)
            print(f'EP:{ep} a#:{action_number} [PREVOBS]: {obs.tolist()}')
            action, _ = model.predict(obs, deterministic=True)
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
            trajectory.append(faulty_action)
            obs, reward, done, trunc, info = env.step(faulty_action)
            action_number += 1

    env.close()

    return trajectory
