import gym
import numpy


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


wrappers = {
    "CartPole_v1": CartPoleSetStepWrapper,
    "MountainCar_v0": MountainCarSetStepWrapper,
    "Taxi_v3": TaxiSetStepWrapper
}
