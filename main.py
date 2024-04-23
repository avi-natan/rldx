from p_trainer import train_on_environment
from pipeline import run_pipeline

if __name__ == '__main__':
    '''
    envs: LunarLander_v2, Ant_v4, ALE/SpaceInvaders_v5, ALE/AirRaid_v5
    models: PPO, A2C, DQN
    fault_model_generators: discrete, box
    '''

    # train_on_environment("ALE/AirRaid_v5", "PPO")

    ''' run_pipeline(env_name, model_name, total_timesteps, fault_model_generator_name, available_fault_models, execution_fault_model, diagnoser_name, observation_mask) '''
    # run_pipeline("LunarLander_v2", "PPO", 90000, "discrete",
    #              ["[0,1,2,3]",  # normal behavior
    #               "[0,0,2,3]", "[0,1,0,3]", "[0,1,2,0]", "[0,0,0,3]", "[0,0,2,0]", "[0,1,0,0]", "[0,0,0,0]",  # shutting down jets
    #               "[1,1,2,3]", "[2,1,2,3]", "[3,1,2,3]"],  # overworking jets
    #              "[0,0,2,3]",
    #              "diagnose_deterministic_faults_part_obs_sfm",
    #              [-1])
    # run_pipeline("Ant_v4", "PPO", 90000, "box",
    #              ["[1,1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # normal behaviour
    #               "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 1
    #               "[1,1,0,0,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 2
    #               "[1,1,1,1,0,0,1,1];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 3
    #               "[1,1,1,1,1,1,0,0];[0,0,0,0,0,0,0,0];-1;1",  # shut down leg 4
    #               "[-1,-1,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1"  # inverting leg 1
    #               ],
    #              "[0,0,1,1,1,1,1,1];[0,0,0,0,0,0,0,0];-1;1",
    #              "diagnose_deterministic_faults_part_obs_sfm",
    #              [-1])
    # run_pipeline("ALE/SpaceInvaders_v5", "PPO", 90000, "discrete",
    #              ["[0,1,2,3,4,5]",  # normal behavior
    #               "[0,0,2,3,4,5]", "[0,1,0,3,4,5]", "[0,1,2,0,4,5]", "[0,1,2,3,0,5]", "[0,1,2,3,4,0]",  # shutting down 1 action
    #               "[0,0,0,3,4,5]", "[0,0,2,0,4,5]", "[0,0,2,3,0,5]", "[0,0,2,3,4,0]", "[0,1,0,0,4,5]",  # shutting down 2 actions
    #               "[0,1,0,3,0,5]", "[0,1,0,3,4,0]", "[0,1,2,0,0,5]", "[0,1,2,0,4,0]", "[0,1,2,3,0,0]",
    #               "[0,0,0,0,4,5]", "[0,0,0,3,0,5]", "[0,0,0,3,4,0]", "[0,0,2,0,0,5]", "[0,0,2,0,4,0]",  # shutting down 3 actions
    #               "[0,0,2,3,0,0]", "[0,1,0,0,0,5]", "[0,1,0,0,4,0]", "[0,1,0,3,0,0]", "[0,1,2,0,0,0]",
    #               "[0,0,0,0,0,5]", "[0,0,0,0,4,0]", "[0,0,0,3,0,0]", "[0,0,2,0,0,0]", "[0,1,0,0,0,0]",  # shutting down 4 actions
    #               "[0,0,0,0,0,0]",  # shutting down 5 actions
    #               "[0,2,1,3,4,5]"  # swapping fire for going right
    #               ],
    #              "[0,1,2,0,4,0]",
    #              "diagnose_deterministic_faults_part_obs_sfm",
    #              [-1])
