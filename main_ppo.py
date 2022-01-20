# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# envFile func: main func
import os

import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
import json

from arguments import parse_args
from replay_buffer import ReplayBuffer
import magentEnv.envFile.scenarios as scenarios
from model_PPO import ActorNN,CriticNN
from magentEnv.envFile.environment import MultiAgentEnv
import adversary_control


def make_env(scenario_name, arglist, benchmark=False):
    """ 
    create the environment from script 
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        if scenario.done and arglist.use_done:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actor = ActorNN(obs_shape_n[0], action_shape_n[0])  # ALG STEP 1
    critic = CriticNN(obs_shape_n[0], 1)

    # Initialize optimizers for actor and critic
    actor_optim = optim.Adam(actor.parameters(), lr=arglist.lr_a)
    critic_optim = optim.Adam(critic.parameters(), lr=arglist.lr_c)

    # Initialize the covariance matrix used to query the actor for actions
    cov_var = torch.full(size=(action_shape_n[0],), fill_value=2)
    cov_mat = torch.diag(cov_var)
    std = torch.full(size=(action_shape_n[0],),fill_value=0.5)

    return actor, critic, actor_optim, critic_optim, cov_var, cov_mat, std


def compute_rtgs(arglist,batch_rews):
    """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
            batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
            batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    """
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    batch_rtgs = []

    # Iterate through each episode
    for ep_rews in reversed(batch_rews):

        discounted_reward = 0  # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * arglist.gamma
            batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs


def evaluate(batch_obs, batch_acts,critic,actor,cov_mat,std):
    """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)

        Return:
            V - the predicted values of batch_obs
            log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
    """
    # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
    V = critic(batch_obs).squeeze()

    # Calculate the log probabilities of batch actions using most recent actor network.
    # This segment of code is similar to that in get_action()
    # mean = actor(batch_obs)
    # dist = MultivariateNormal(mean, cov_mat)
    # dist = Normal(mean, std)
    mean, log_std = actor(batch_obs)
    dist = Normal(mean,log_std.exp())
    log_probs = dist.log_prob(batch_acts).sum(1)

    # Return the value vector V of each observation in the batch
    # and log probabilities log_probs of each action in the batch
    return V, log_probs

def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')
    time_now = time.strftime('%y_%m%d')
    model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
        arglist.scenario_name, time_now, arglist.exp_name))
    if not os.path.exists(model_file_dir):  # make the path
        os.mkdir(model_file_dir)
    learning_curve_file = os.path.join(model_file_dir,'data.txt')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[0].shape[0]]
    #obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shape_n = [env.action_space[0].n ] # no need for stop bit
    #action_shape_n = [env.action_space[i].n for i in range(env.n)]  # no need for stop bit
    #num_adversaries = min(env.n, arglist.num_adversaries)
    num_adversaries = 4
    actor,critic,actor_optim,critic_optim,cov_var,cov_mat,std = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    t_so_far = 0  # Timesteps simulated so far
    i_so_far = 0  # Iterations ran so far
    log_episode_reward = []
    log_actorloss = []
    log_criticloss = []
    while t_so_far < arglist.max_episode:  # ALG STEP 2
        """         
        batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
        batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
        batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
        batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
        batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < arglist.per_episode_max_len * 10:  # 每个batch的数量，取每轮的上限的2倍
            ep_rews = []  # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation.
            obs = env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(arglist.per_episode_max_len):

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs[0])

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                # mean = actor(obs[0])
                # dist = MultivariateNormal(mean, cov_mat)
                # dist = Normal(mean, std)
                # action_origin = dist.sample().detach()
                # log_prob = dist.log_prob(action_origin).detach()
                # action = torch.tanh(action_origin)
                # log_prob -= torch.log(torch.tensor(1+1e-6) - action.pow(2))
                action, log_prob = actor.sample(obs[0])
                action = action.detach().numpy()
                #log_prob = log_prob.sum()
                action_good_clip = []  # 输入到环境的动作空间
                action_good_clip_real = []  # 训练时候使用的动作空间
                for i in range(env.num_good):
                    a = [0, action[i * 2], 0, action[i * 2 + 1], 0]
                    #b = [action[i * 2], action[i * 2 + 1]]
                    action_good_clip.append(a)
                    #action_good_clip_real.append(b)

                action_adversary_clip = adversary_control.adversary_action(env)     #外部固定策略
                action_all = np.concatenate((action_good_clip, action_adversary_clip))
                obs, rew, done, _ = env.step(action_all)

                # Track recent reward, action, and action log probability
                ep_rews.append(np.sum(rew))
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                done = any(done)
                # If the environment tells us the episode is terminated, break
                terminal = (ep_t >= arglist.per_episode_max_len - 1)
                if done or terminal:
                    obs = env.reset()

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            log_episode_reward.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = compute_rtgs(arglist,batch_rews)  # ALG STEP 4

        t_so_far += np.sum(batch_lens)

        # Increment the number of iterations
        i_so_far += 1


        V, _ = evaluate(batch_obs, batch_acts,critic,actor,cov_mat,std)
        A_k = batch_rtgs - V.detach()  # ALG STEP 5

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases the variance of
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it.
        A_k = A_k - A_k.mean()


        for _ in range(5):  # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = evaluate(batch_obs, batch_acts,critic,actor,cov_mat,std)

            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation:
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            #surr1 = torch.mul(ratios * A_k)
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - arglist.clip, 1 + arglist.clip) * A_k

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for actor network
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            log_actorloss.append(actor_loss.detach().numpy())
            log_criticloss.append(critic_loss.detach().numpy())

        # 打印这一batch的reward
        fre = 20
        if (i_so_far % fre == 0):
            print("=Training=episode:{} average batch reward:{}".format(t_so_far, np.mean(log_episode_reward[-10*fre:])),end="\n")
            print("current actor loss=={}".format(np.mean(log_actorloss[-fre:])))
            print("current critic loss=={}".format(np.mean(log_criticloss[-fre:])))
            # for name, parms in actor.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,' -->grad_value:', parms.grad)


        if(i_so_far % 50 == 0):
            model_file_dir_ep = os.path.join(model_file_dir,'{}'.format(i_so_far))
            print(model_file_dir_ep," saved")
            if not os.path.exists(model_file_dir_ep): # make the path
                os.mkdir(model_file_dir_ep)
            torch.save(actor.state_dict(), os.path.join(model_file_dir_ep, 'a_c_0.pt'))
            torch.save(critic.state_dict(), os.path.join(model_file_dir_ep, 'c_c_0.pt'))







if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
