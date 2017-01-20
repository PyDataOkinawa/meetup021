#!/usr/bin/env python
# basic_rl.py (v0.0.5)
#
# New in v0.0.5
# - implemented a simple eligibility trace (Q-lambda and SARSA-lambda).
# - Set the minimum value of epsilon.
#
# New in v0.0.4
# - This version uses `gym.wrapper` to store results of the simulaiton.
# - A random seed can be set through command line argument.
# - Fixed the update rule used in the episodic terminal.

import argparse
parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-a', '--algorithm', default='q_learning', choices=['sarsa', 'q_learning'],
                    help="Type of learning algorithm. (Default: sarsa)")
parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-e', '--environment', default='FrozenLake-v0',
                    help="Name of the environment provided in the OpenAI Gym. (Default: FrozenLake-v0)")
parser.add_argument('-n', '--nepisode', default='20000', type=int,
                    help="Number of episode. (Default: 20000)")
parser.add_argument('-al', '--alpha', default='0.05', type=float,
                    help="Learning rate. (Default: 0.05)")
parser.add_argument('-be', '--beta', default='0.0', type=float,
                    help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.01', type=float,
                    help="Linear increase rate of an inverse temperature. (Default: 0.01)")
parser.add_argument('-ga', '--gamma', default='0.99', type=float,
                    help="Discount rate. (Default: 0.99)")
parser.add_argument('-ep', '--epsilon', default='0.8', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.8)")
parser.add_argument('-ed', '--epsilondecay', default='0.995', type=float,
                    help="Decay rate of epsilon in the epsilon greedy. (Default: 0.995)")
parser.add_argument('-em', '--epsilonmin', default=0.001, type=float,
                    help="Minimum epsilon value. (Default: 0.01)")
parser.add_argument('-ms', '--maxstep', default='200', type=int,
                    help="Maximum step allowed in a episode. (Default: 200)")
parser.add_argument('-ka', '--kappa', default='0.01', type=float,
                    help="Weight of the most recent cumulative reward for computing its running average. (Default: 0.01)")
parser.add_argument('-qm', '--qmean', default='0.0', type=float,
                    help="Mean of the Gaussian used for initializing Q table. (Default: 0.0)")
parser.add_argument('-qs', '--qstd', default='0.1', type=float,
                    help="Standard deviation of the Gaussian used for initializing Q table. (Default: 1.0)")
parser.add_argument('-se', '--seed', default='42', type=int,
                    help="Seed value used for env.seed() and np.random.seed() (Default: 42) ")
parser.add_argument('-tr', '--tracedecay', default='0.8', type=float,
                    help="A parameter controlling the decay rate of the eligibility trace")

args = parser.parse_args()

import gym
from gym import wrappers
import numpy as np
import os

import matplotlib.pyplot as plt

def softmax(q_value, beta=1.0):
    assert beta >= 0.0
    q_tilde = q_value - np.max(q_value)
    factors = np.exp(beta * q_tilde)
    return factors / np.sum(factors)

def select_a_with_softmax(curr_s, q_value, beta=1.0):
    prob_a = softmax(q_value[curr_s, :], beta=beta)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

def select_a_with_epsilon_greedy(curr_s, q_value, epsilon=0.1):
    a = np.argmax(q_value[curr_s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q_value.shape[1])
    return a

def main():

    env_type = args.environment
    algorithm_type = args.algorithm
    policy_type = args.policy

    # Random seed
    np.random.seed(args.seed)

    # Selection of the problem
    env = gym.envs.make(env_type)
    env.seed(args.seed)

    # Constraints imposed by the environment
    n_s = env.observation_space.n
    n_a = env.action_space.n

    # Meta parameters for the RL agent
    alpha = args.alpha
    beta = args.beta
    beta_inc = args.betainc
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_decay = args.epsilondecay
    epsilon_min = args.epsilonmin
    q_mean = args.qmean
    q_std = args.qstd
    trace_decay = args.tracedecay

    # Experimental setup
    n_episode = args.nepisode
    print "n_episode ", n_episode
    max_step = args.maxstep

    # Running average of the cumulative reward, which is used for controlling an exploration rate
    # (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
    # See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
    kappa = args.kappa
    ave_cumu_r = None

    # Initialization of a Q-value table
    #q_value = np.zeros([n_s, n_a])
    # Optimistic initialization
    q_value = q_mean + q_std * np.random.randn(n_s, n_a)

    # Initialization of a list for storing simulation history
    history = []

    print "algorithm_type: {}".format(algorithm_type)
    print "policy_type: {}".format(policy_type)

    env.reset()

    np.set_printoptions(precision=3, suppress=True)

    result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    env = wrappers.Monitor(env, result_dir, force=True)

    for i_episode in xrange(n_episode):

        # Initialize eligibility trace
        #
        # There is a typo in Sutton & Barto's book, so be careful. See the following for detail.
        # Eligibility trace reinitialization between episodes in SARSA-Lambda implementation
        # http://stackoverflow.com/questions/29904270/eligibility-trace-reinitialization-between-episodes-in-sarsa-lambda-implementati
        trace = np.zeros([n_s, n_a])

        # Reset a cumulative reward for this episode
        cumu_r = 0

        # Start a new episode and sample the initial state
        curr_s = env.reset()

        # Select the first action in this episode
        if policy_type == 'softmax':
            curr_a = select_a_with_softmax(curr_s, q_value, beta=beta)
        elif policy_type == 'epsilon_greedy':
            curr_a = select_a_with_epsilon_greedy(curr_s, q_value, epsilon=epsilon)
        else:
            raise ValueError("Invalid policy_type: {}".format(policy_type))

        for i_step in xrange(max_step):

            # Get a result of your action from the environment
            next_s, r, done, info = env.step(curr_a)

            # Modification of reward
            # CAUTION: Changing this part of the code in order to get a fast convergence
            # is not a good idea because it is essentially changing the problem setting itself.
            # This part of code was kept not to get fast convergence but to show the
            # influence of a reward function on the convergence speed for pedagogical reason.
            #if done & (r == 0):
            #    # Punishment for falling into a hall
            #    r = 0.0
            #elif not done:
            #    # Cost per step
            #    r = 0.0

            # Update a cummulative reward
            cumu_r = r + gamma * cumu_r

            # Select an action
            if policy_type == 'softmax':
                next_a = select_a_with_softmax(next_s, q_value, beta=beta)
            elif policy_type == 'epsilon_greedy':
                next_a = select_a_with_epsilon_greedy(next_s, q_value, epsilon=epsilon)
            else:
                raise ValueError("Invalid policy_type: {}".format(policy_type))

            # Calculation of TD error
            if algorithm_type == 'sarsa':
                if done:
                    target = r
                else:
                    target = r + gamma * q_value[next_s, next_a]
                delta = target - q_value[curr_s, curr_a]
            elif algorithm_type == 'q_learning':
                if done:
                    target = r
                else:
                    next_greedy_a = np.argmax(q_value[next_s, :])
                    target = r + gamma * q_value[next_s, next_greedy_a]
                delta = target - q_value[curr_s, curr_a]
            else:
                raise ValueError("Invalid algorithm_type: {}".format(algorithm_type))

            # Update eligibility trace
            trace[curr_s, curr_a] = 1

            # Update a Q value table
            #q_value[curr_s, curr_a] += alpha * delta
            q_value = q_value + alpha * delta * trace

            if algorithm_type == 'q_learning':
                if next_a == next_greedy_a:
                    # Decay eligibility trace when the executed action is the greedy action
                    trace = gamma * trace_decay * trace
                else:
                    # Reset eligibility trace when the executed action is not the greedy action
                    trace = np.zeros([n_s, n_a])
            elif algorithm_type == 'sarsa':
                trace = gamma * trace_decay * trace
            else:
                raise ValueError("Invalid algorithm_type: {}".format(algorithm_type))

            curr_s = next_s
            curr_a = next_a

            if done:

                # Running average of the terminal reward, which is used for controlling an exploration rate
                # (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
                # See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
                kappa = 0.01
                if ave_cumu_r == None:
                    ave_cumu_r = cumu_r
                else:
                    ave_cumu_r = kappa * cumu_r + (1 - kappa) * ave_cumu_r

                if cumu_r > ave_cumu_r:
                    # Bias the current policy toward exploitation

                    if policy_type == 'epsilon_greedy':

                        if epsilon > epsilon_min:
                            # epsilon is decayed expolentially
                            epsilon = epsilon * epsilon_decay
                        else:
                            epsilon = epsilon_min
                    elif policy_type == 'softmax':
                        # beta is increased linearly
                        beta = beta + beta_inc

                if policy_type == 'softmax':
                    print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tBeta: {5:.3f}".format(
                        i_episode, i_step, cumu_r, r, ave_cumu_r, beta)
                    history.append([i_episode, i_step, cumu_r, r, ave_cumu_r, beta])
                elif policy_type == 'epsilon_greedy':
                    print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tEpsilon: {5:.3f}".format(
                        i_episode, i_step, cumu_r, r, ave_cumu_r, epsilon)
                    history.append([i_episode, i_step, cumu_r, r, ave_cumu_r, epsilon])
                else:
                    raise ValueError("Invalid policy_type: {}".format(policy_type))

                break

    # Stop monitoring the simulation for OpenAI Gym
    env.close()

    history = np.array(history)

    window_size = 100
    def running_average(x, window_size, mode='valid'):
        return np.convolve(x, np.ones(window_size)/window_size, mode=mode)

    fig, ax = plt.subplots(2, 2, figsize=[12, 8])
    # Number of steps
    ax[0, 0].plot(history[:, 0], history[:, 1], '.')
    ax[0, 0].set_xlabel('Episode')
    ax[0, 0].set_ylabel('Number of steps')
    ax[0, 0].plot(history[window_size-1:, 0], running_average(history[:, 1], window_size))
    # Cumulative reward
    ax[0, 1].plot(history[:, 0], history[:, 2], '.')
    ax[0, 1].set_xlabel('Episode')
    ax[0, 1].set_ylabel('Cumulative rewards')
    ax[0, 1].plot(history[:, 0], history[:, 4], '--')
    #ax[0, 1].plot(history[window_size-1:, 0], running_average(history[:, 2], window_size))
    # Terminal reward
    ax[1, 0].plot(history[:, 0], history[:, 3], '.')
    ax[1, 0].set_xlabel('Episode')
    ax[1, 0].set_ylabel('Terminal rewards')
    ax[1, 0].plot(history[window_size-1:, 0], running_average(history[:, 3], window_size))
    # Epsilon/Beta
    ax[1, 1].plot(history[:, 0], history[:, 5], '.')
    ax[1, 1].set_xlabel('Episode')
    if policy_type == 'softmax':
        ax[1, 1].set_ylabel('Beta')
    elif policy_type == 'epsilon_greedy':
        ax[1, 1].set_ylabel('Epsilon')
    fig.savefig('./'+result_dir+'.png')

    print "Q value table:"
    print q_value

    if policy_type == 'softmax':
        print "Action selection probability:"
        print np.array([softmax(q, beta=beta) for q in q_value])
    elif policy_type == 'epsilon_greedy':
        print "Greedy action"
        greedy_action = np.zeros([n_s, n_a])
        greedy_action[np.arange(n_s), np.argmax(q_value, axis=1)] = 1
        print greedy_action

if __name__ == "__main__":

    main()
