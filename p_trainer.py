import os
import platform

import gym

from h_rl_models import models


def train_on_environment(env_name, render_display, render_terminal, model_name):
    models_dir = f"environments/{env_name}/models/{model_name}"
    log_dir = f"environments/{env_name}/logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if platform.system() == "Windows":
        render_mode = render_display
    else:
        render_mode = render_terminal

    env = gym.make(env_name.replace('_', '-'), render_mode=render_mode)
    env.reset()

    model = models[model_name]("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    TIMESTEPS = 10000
    for i in range(1, 30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    env.close()
