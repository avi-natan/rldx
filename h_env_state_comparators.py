import numpy as np


# def acrobot_compare(s1, s2):
#     return np.array_equal(s1, s2)

def cart_pole_compare(s1, s2):
    return np.array_equal(s1, s2)

def mountain_car_compare(s1, s2):
    return np.array_equal(s1, s2)

def taxi_compare(s1, s2):
    return s1 == s2

def frozen_lake_compare(s1, s2):
    return s1 == s2


comparators = {
    # "LunarLander_v2": LunarLanderSetStepWrapper
    # "Acrobot_v1": acrobot_compare,
    "CartPole_v1": cart_pole_compare,
    "MountainCar_v0": mountain_car_compare,
    "Taxi_v3": taxi_compare,
    "FrozenLake_v1": frozen_lake_compare
}