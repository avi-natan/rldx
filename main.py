import sys
import time

from pygame import mixer  # Load the popular external library

from p_pipeline import run_experimental_setup
from p_single_experiments import single_experiment_LunarLander_W, single_experiment_LunarLander_SN, single_experiment_LunarLander_SIF, \
    single_experiment_Acrobot_W, single_experiment_Acrobot_SN, single_experiment_Acrobot_SIF, \
    single_experiment_CartPole_W, single_experiment_CartPole_SN, single_experiment_CartPole_SIF, \
    single_experiment_Assault_W, single_experiment_Assault_SN, single_experiment_Assault_SIF, \
    single_experiment_Carnival_W, single_experiment_Carnival_SN, single_experiment_Carnival_SIF, \
    single_experiment_SpaceInvaders_W, single_experiment_SpaceInvaders_SN, single_experiment_SpaceInvaders_SIF, single_experiment_MountainCar_W, single_experiment_MountainCar_SN, single_experiment_MountainCar_SIF, single_experiment_Taxi_W, single_experiment_Taxi_SN, single_experiment_Taxi_SIF, single_experiment_FrozenLake_W, single_experiment_FrozenLake_SN, single_experiment_FrozenLake_SIF

if __name__ == '__main__':
    try:
        # == single experiments (for coding and debug purposes) ==
        # single_experiment_LunarLander_W()         # OK ALL
        # single_experiment_LunarLander_SN()        # OK ALL
        # single_experiment_LunarLander_SIF()       # OK ALL

        # single_experiment_Acrobot_W()             # OK ALL
        # single_experiment_Acrobot_SN()            # OK ALL
        # single_experiment_Acrobot_SIF()           # OK ALL

        # single_experiment_CartPole_W()            # OK ALL
        # single_experiment_CartPole_SN()           # OK ALL
        # single_experiment_CartPole_SIF()          # OK ALL

        # single_experiment_MountainCar_W()         # OK
        # single_experiment_MountainCar_SN()        # OK
        # single_experiment_MountainCar_SIF()       # OK

        # single_experiment_Taxi_W()                # OK
        # single_experiment_Taxi_SN()               # OK
        # single_experiment_Taxi_SIF()              # OK

        # single_experiment_FrozenLake_W()          # OK
        # single_experiment_FrozenLake_SN()         # OK
        # single_experiment_FrozenLake_SIF()        # OK

        # single_experiment_Assault_W()             # TODO problem: "[0,0,2,3,4,5,6]"
        # single_experiment_Assault_SN()            # OK ALL
        # single_experiment_Assault_SIF()           # TODO problem: "[0,0,2,3,4,5,6]",#X wrong output | "[0,1,0,3,4,5,6]",#X error, empty G | "[0,1,2,0,4,5,6]",#X error, empty G | "[0,1,2,3,0,5,6]",# but correct diagnosis didnt get 1.0 | "[0,1,2,3,4,0,6]",# but correct diagnosis didnt get 1.0 | "[0,1,2,3,4,5,0]",# | "[0,2,1,3,4,5,6]",#X error, empty G | "[0,3,2,1,4,5,6]",# | "[0,4,2,3,1,5,6]",# | "[0,5,2,3,4,1,6]"#X error, empty G

        # single_experiment_Carnival_W()            # OK ALL
        # single_experiment_Carnival_SN()           # OK ALL
        # single_experiment_Carnival_SIF()          # TODO problem: "[0,0,2,3,4,5]",#X error, empty G

        # single_experiment_SpaceInvaders_W()       # TODO problem: "[0,0,2,3,4,5]", "[0,1,2,3,0,5]", "[0,4,2,3,1,5]", "[1,0,2,3,4,5]"
        # single_experiment_SpaceInvaders_SN()      # OK ALL
        # single_experiment_SpaceInvaders_SIF()     # TODO problem: "[0,1,2,3,0,5]",# but correct diagnosis didnt get 1.0 | "[0,3,2,1,4,5]",# but correct diagnosis didnt get 1.0 | "[0,4,2,3,1,5]",# but correct diagnosis didnt get 1.0 | "[0,5,2,3,4,1]",# but correct diagnosis didnt get 1.0 | "[1,0,2,3,4,5]"# but correct diagnosis didnt get 1.0

        print(f'finisehd gracefully')
        # mixer.init()
        # mixer.music.load('alarm.mp3')
        # mixer.music.play()
        # while mixer.music.get_busy():  # wait for music to finish playing
        #     time.sleep(1)
    except ValueError as e:
        print(f'Value error: {e}')
        # mixer.init()
        # mixer.music.load('alarm.mp3')
        # mixer.music.play()
        # while mixer.music.get_busy():  # wait for music to finish playing
        #     time.sleep(1)

    # ================== experimental setup ==================
    # do_debug_print = False
    # run_experimental_setup(arguments=sys.argv, debug_print=do_debug_print)
    print(9)
