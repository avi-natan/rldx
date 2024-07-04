import sys
import time

from pygame import mixer  # Load the popular external library

from p_pipeline import run_experimental_setup2
from p_single_experiments import single_experiment_LunarLander_W, single_experiment_LunarLander_SN, single_experiment_LunarLander_SIF, \
    single_experiment_Acrobot_W2, single_experiment_Acrobot_SN2, single_experiment_Acrobot_SIF2, \
    single_experiment_CartPole_SIF2, single_experiment_CartPole_W2, single_experiment_CartPole_SN2, \
    single_experiment_MountainCar_W2, single_experiment_MountainCar_SN2, single_experiment_MountainCar_SIF2, \
    single_experiment_Taxi_W2, single_experiment_Taxi_SN2, single_experiment_Taxi_SIF2


if __name__ == '__main__':
    try:
        # == single experiments (for coding and debug purposes) ==
        # single_experiment_LunarLander_W()         # OK ALL
        # single_experiment_LunarLander_SN()        # OK ALL
        # single_experiment_LunarLander_SIF()       # OK ALL

        # single_experiment_Acrobot_W()             # OK ALL
        # single_experiment_Acrobot_SN()            # OK ALL
        # single_experiment_Acrobot_SIF()           # OK ALL
        # single_experiment_Acrobot_W2()            # OK ALL
        # single_experiment_Acrobot_SN2()           # OK ALL
        # single_experiment_Acrobot_SIF2()          # OK ALL

        # single_experiment_CartPole_W()            # OK ALL
        # single_experiment_CartPole_SN()           # OK ALL
        # single_experiment_CartPole_SIF()          # OK ALL
        # single_experiment_CartPole_W2()           # OK ALL
        # single_experiment_CartPole_SN2()          # OK ALL
        # single_experiment_CartPole_SIF2()         # OK ALL

        # single_experiment_MountainCar_W()         # OK ALL
        # single_experiment_MountainCar_SN()        # OK ALL
        # single_experiment_MountainCar_SIF()       # OK ALL
        # single_experiment_MountainCar_W2()        # OK ALL
        # single_experiment_MountainCar_SN2()       # OK ALL
        # single_experiment_MountainCar_SIF2()      # OK ALL

        # single_experiment_Taxi_W()                # OK ALL
        # single_experiment_Taxi_SN()               # OK ALL
        # single_experiment_Taxi_SIF()              # OK ALL
        # single_experiment_Taxi_W2()               # OK ALL
        # single_experiment_Taxi_SN2()              # OK ALL
        # single_experiment_Taxi_SIF2()             # OK ALL

        # ================== experimental setup ==================
        render_mode = "rgb_array"       # "human", "rgb_array"
        debug_print = False             # False, True
        # run_experimental_setup(arguments=sys.argv, render_mode=render_mode, debug_print=debug_print)
        run_experimental_setup2(arguments=sys.argv, render_mode=render_mode, debug_print=debug_print)

        print(f'finisehd gracefully')
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
    except ValueError as e:
        print(f'Value error: {e}')
        mixer.init()
        mixer.music.load('alarm.mp3')
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)

    print(9)
