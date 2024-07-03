import gym
import numpy


# class LunarLanderSetStepWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def reset(self, seed=None):
#         next_state, info = self.env.reset(seed=seed)
#
#         fps = 50
#         scale = 30.0
#
#         leg_down = 18
#
#         viewport_w = 600
#         viewport_h = 400
#
#         helipad_y = self.unwrapped.helipad_y
#
#         pos_x = self.unwrapped.lander.position.x
#         pos_y = self.unwrapped.lander.position.y
#         vel_x = self.unwrapped.lander.linearVelocity.x
#         vel_y = self.unwrapped.lander.linearVelocity.y
#
#         next_state2 = [
#             (pos_x - viewport_w / scale / 2) / (viewport_w / scale / 2),
#             (pos_y - (helipad_y + leg_down / scale)) / (viewport_h / scale / 2),
#             vel_x * (viewport_w / scale / 2) / fps,
#             vel_y * (viewport_h / scale / 2) / fps,
#             self.unwrapped.lander.angle,
#             20.0 * self.unwrapped.lander.angularVelocity / fps,
#             1.0 if self.unwrapped.legs[0].ground_contact else 0.0,
#             1.0 if self.unwrapped.legs[1].ground_contact else 0.0,
#         ]
#         print(9)
#         next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
#         return next_state2_as_ndarray, info
#
#     def set_state(self, state):
#         self.unwrapped.state = state
#
#     def step(self, action):
#         next_state, reward, done, trunc, info = self.env.step(action)
#         # modify ...
#
#         next_state2 = self.unwrapped.state
#         next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
#         return next_state2_as_ndarray, reward, done, trunc, info


# class AcrobotSetStepWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def reset(self, seed=None):
#         next_state, info = self.env.reset(seed=seed)
#
#         next_state2 = self.unwrapped.state
#         next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
#         return next_state2_as_ndarray, info
#
#     def set_state(self, state):
#         self.unwrapped.state = state
#
#     def step(self, action):
#         next_state, reward, done, trunc, info = self.env.step(action)
#         # modify ...
#
#         next_state2 = self.unwrapped.state
#         next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
#         return next_state2_as_ndarray, reward, done, trunc, info


class CartPoleSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        next_state, info = self.env.reset(seed=seed)

        next_state2 = self.unwrapped.state
        next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
        return next_state2_as_ndarray, info

    def set_state(self, state):
        self.unwrapped.state = state

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        # modify ...

        next_state2 = self.unwrapped.state
        next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
        return next_state2_as_ndarray, reward, done, trunc, info


class MountainCarSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        next_state, info = self.env.reset(seed=seed)

        next_state2 = self.unwrapped.state
        next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
        return next_state2_as_ndarray, info

    def set_state(self, state):
        self.unwrapped.state = state

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        # modify ...

        next_state2 = self.unwrapped.state
        next_state2_as_ndarray = numpy.array(next_state2, dtype=numpy.float64)
        return next_state2_as_ndarray, reward, done, trunc, info


class TaxiSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        next_state, info = self.env.reset(seed=seed)

        next_state2 = self.unwrapped.s
        next_state2_as_int = int(next_state2)
        return next_state2_as_int, info

    def set_state(self, state):
        self.unwrapped.s = numpy.int64(state)

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        # modify ...

        next_state2 = self.unwrapped.s
        next_state2_as_int = int(next_state2)
        return next_state2_as_int, reward, done, trunc, info


class FrozenLakeSetStepWrapper(gym.Wrapper):
    def __init__(self, env):
        print(9)
        super().__init__(env)

    def reset(self, seed=None):
        next_state, info = self.env.reset(seed=seed)

        next_state2 = self.unwrapped.s
        next_state2_as_int = int(next_state2)
        return next_state2_as_int, info

    def set_state(self, state):
        self.unwrapped.s = numpy.int64(state)

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        # modify ...

        next_state2 = self.unwrapped.s
        next_state2_as_int = int(next_state2)
        return next_state2_as_int, reward, done, trunc, info



wrappers = {
    # "LunarLander_v2": LunarLanderSetStepWrapper,
    # "Acrobot_v1": AcrobotSetStepWrapper,
    "CartPole_v1": CartPoleSetStepWrapper,
    "MountainCar_v0": MountainCarSetStepWrapper,
    "Taxi_v3": TaxiSetStepWrapper,
    "FrozenLake_v1": FrozenLakeSetStepWrapper
}
