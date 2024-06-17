import os

import gym

from h_rl_models import models


def train_on_environment(env_name, render_mode, model_name):
    models_dir = f"environments/{env_name}/models/{model_name}"
    log_dir = f"environments/{env_name}/logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    env.reset()

    model = models[model_name]("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    TIMESTEPS = 10000
    for i in range(1, 11):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
        print(f'########################### learned {TIMESTEPS} timesteps for iteration number {i} ###########################')
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    env.close()


if __name__ == '__main__':
    domain_names = [
        'ALE/Assault_v5',
        'ALE/Atlantis_v5',
        'ALE/Bowling_v5',
        'ALE/Breakout_v5',
        'ALE/Carnival_v5',
        'ALE/NameThisGame_v5',
        'ALE/Pong_v5',
        'ALE/Pooyan_v5',
        'ALE/Qbert_v5',
        'ALE/StarGunner_v5',
        'ALE/UpNDown_v5'
    ]

    # p_render_mode = "human"
    p_render_mode = "rgb_array"
    p_model_name = "PPO"
    for p_domain_name in domain_names:
        train_on_environment(p_domain_name, p_render_mode, p_model_name)
