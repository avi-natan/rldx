import sys

from p_pipeline import run_experimental_setup
from p_single_experiments import single_experiment_LunarLander_W, single_experiment_LunarLander_SN, single_experiment_LunarLander_SIF, \
    single_experiment_Acrobot_W, single_experiment_Acrobot_SN, single_experiment_Acrobot_SIF, \
    single_experiment_CartPole_W, single_experiment_CartPole_SN, single_experiment_CartPole_SIF, \
    single_experiment_Assault_W, single_experiment_Assault_SN, single_experiment_Assault_SIF, \
    single_experiment_Carnival_W, single_experiment_Carnival_SN, single_experiment_Carnival_SIF, \
    single_experiment_SpaceInvaders_W, single_experiment_SpaceInvaders_SN, single_experiment_SpaceInvaders_SIF

if __name__ == '__main__':
    # == single experiments (for coding and debug purposes) ==
    # single_experiment_LunarLander_W()         # OK
    # single_experiment_LunarLander_SN()        # OK
    # single_experiment_LunarLander_SIF()       # OK

    # single_experiment_Acrobot_W()             # OK
    # single_experiment_Acrobot_SN()            # OK
    # single_experiment_Acrobot_SIF()           # OK

    # single_experiment_CartPole_W()            # OK
    # single_experiment_CartPole_SN()           # OK
    # single_experiment_CartPole_SIF()          # OK

    # single_experiment_Assault_W()             # OK
    # single_experiment_Assault_SN()            # OK
    # single_experiment_Assault_SIF()           # TODO problem

    # single_experiment_Carnival_W()            # OK
    # single_experiment_Carnival_SN()           # OK
    # single_experiment_Carnival_SIF()          # OK

    # single_experiment_SpaceInvaders_W()       # OK
    # single_experiment_SpaceInvaders_SN()      # OK
    # single_experiment_SpaceInvaders_SIF()     # OK

    # ================== experimental setup ==================
    do_debug_print = False
    run_experimental_setup(arguments=sys.argv, debug_print=do_debug_print)
    print(9)
