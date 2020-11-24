from utils.networks import get_session
from Env.UsersEnvCluster import UsersEnvCluster
from utils.continuous_environments import Environment
from keras.backend.tensorflow_backend import set_session
from DDPG.ddpg import DDPG
from DDQN.ddqn import DDQN
import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import *
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def k_subset(arr, k):
    s_arr = sorted(arr)
    return list(set([i for i in combinations(subsets(arr), k) if sorted(chain(*i)) == s_arr]))


def create_cluster_mapping(n_t, n_c):
    k_partitions = k_subset(range(1, n_t + 1), n_c)
    mapping = {}
    for i in range(len(k_partitions)):
        mapping[i] = k_partitions[i]
    return mapping


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--step', type=str, default='train', help="The machine learning step", choices=['train', 'inference'])
    parser.add_argument('--nb_episodes', type=int, default=2500, help="Number of training episodes")
    parser.add_argument('--episode_length', type=int, default=500, help="Length of one episode")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--out_dir', type=str, default='experiments', help="Name of the output directory")

    parser.add_argument('--consecutive_frames', type=int, default=2, help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")

    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true', help="Compute Average reward per episode (slower)")
    parser.add_argument('--env', type=str, default='UsersEnv', help="Wireless environment")
    parser.add_argument('--M1', type=int, default='4', help="The number of transmitters")
    parser.add_argument('--M2', type=int, default='4', help="The number of receivers")
    parser.add_argument('--snr_M1', type=int, default='37', help="SNR")
    parser.add_argument('--snr_M2', type=int, default='37', help="SNR")
    parser.add_argument('--gpu', type=str, default="0,1", help='GPU ID')

    parser.set_defaults(render=False)
    return parser.parse_args(args)


def main(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())

    summary_writer = tf.summary.FileWriter("{}/tensorboard_M1_{}_M1_{}_snr1_{}_snr2_{}".format(args.out_dir, args.M1, args.M1, args.snr_M1, args.snr_M2))

    # Initialize the wireless environment
    users_env = UsersEnvCluster(args.M1, args.M2, args.snr_M1, args.snr_M2, fixed_channel=False)
    print(users_env)

    # Wrap the environment to use consecutive frames
    env = Environment(users_env, args.consecutive_frames)
    env.reset()

    # Define parameters for the DDQN and DDPG algorithms
    state_dim = env.get_state_size()
    action_dim = users_env.action_dim
    act_range = 1
    act_min = 0

    # Initialize the DQN algorithm for the clustering optimization
    n_clusters = users_env.n_clusters
    algo_clustering = DDQN(n_clusters, state_dim, args)

    # Initialize the DDPG algorithm for the beamforming optimization
    algo = DDPG(action_dim, state_dim, act_range, act_min, args.consecutive_frames, algo_clustering, episode_length=args.episode_length)

    if args.step == "train":
        # Train
        stats = algo.train(env, args, summary_writer)

        # Export results to CSV
        if(args.gather_stats):
            df = pd.DataFrame(np.array(stats))
            df.to_csv(args.out_dir + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

        # Save weights and close environments
        exp_dir = '{}/models_M1_{}_M2_{}_snr1_{}_snr2_{}/'.format(args.out_dir, args.M1, args.M2, args.snr_M1, args.snr_M2)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        # Save DDPG
        export_path = '{}_{}_NB_EP_{}_BS_{}'.format(exp_dir, "DDPG", args.nb_episodes, args.batch_size)
        algo.save_weights(export_path)

        # Save DDQN
        export_path = '{}_{}_NB_EP_{}_BS_{}'.format(exp_dir, "DDQN", args.nb_episodes, args.batch_size)
        algo.ddqn_clustering.save_weights(export_path)

    elif args.step == "inference":
        print("Loading the DDPG networks (actor and critic) and the DDQN policy network ...")
        path_actor = '<add the path of the .h5 file of the DDPG actor>'
        path_critic = '<add the path of the .h5 file of the DDPG critic>'
        path_ddqn = '<add the path of the .h5 file of the DDQN actor>'
        algo.load_weights(path_actor, path_critic, path_ddqn)

        # run a random policy during inference as an example
        s = np.random.rand(1, args.Nr)
        s_1 = np.zeros_like(s)
        s = np.vstack((s_1, s))

        while True:
            W = algo.policy_action(s)
            cluster_index = algo.ddqn_clustering.policy_action(s)
            a_and_c = {'a': W, 'c': cluster_index}
            new_state, r, done, _ = env.step(a_and_c)
            print("RL min rate = {}".format(r))
            print("RL state = {}".format(np.log(1 + new_state)))
            s = new_state
            input('Press Enter to continue ...')


if __name__ == "__main__":
    main()
