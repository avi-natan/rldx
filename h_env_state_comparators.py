import numpy as np

def cart_pole_compare(s1, s2):
    return np.array_equal(s1, s2)

def mountain_car_compare(s1, s2):
    return np.array_equal(s1, s2)

def taxi_compare(s1, s2):
    return s1 == s2


comparators = {
    "CartPole_v1": cart_pole_compare,
    "MountainCar_v0": mountain_car_compare,
    "Taxi_v3": taxi_compare
}
