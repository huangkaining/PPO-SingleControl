import os
import sys

import torch
import torch.nn.functional as F
import magentEnv.envFile.scenarios as scenarios
from magentEnv.envFile.environment import MultiAgentEnv
import time
import  adversary_control
import numpy as np
import matplotlib.pyplot as plt
import threading
import model_PPO
from torch.distributions import Normal


from model import actor_agent, critic_agent
from arguments import parse_args


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
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                done_callback=scenario.done,info_callback=scenario.get_score)
        else:
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,info_callback=scenario.get_score)
    return env

def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    actors_tar = model_PPO.ActorNN(obs_dim, act_dim)
    actors_tar.load_state_dict(torch.load(arglist.old_model_name+'a_c_0.pt'))
    actors_tar = [actors_tar]

    return actors_tar

def enjoy(arglist):
    """ 
    This func is used for testing the model
    """
    episode_step = 0
    """ init the env """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    """ init the agents """
    obs_shape_n = [env.observation_space[0].shape ]
    actors_tar = get_trainers(env, arglist)

    #th1 = curve_Thread()    #绘图线程
    #th1.start()

    """ interact with the env """
    obs_n = env.reset()
    while(1):

        # update the episode step number
        episode_step += 1
        #plt.ion()
        # get action
        action_n = []
        for actor, obs in zip(actors_tar, obs_n):
            model_out, _= actor(obs)
            #action_n.append(F.softmax(model_out,dim=-1).detach().cpu().numpy())
            action_n.append(model_out)

        action_good_clip = []
        for i in range(env.num_good):
            a = [0 , action_n[0][i* 2], 0 , action_n[0][i * 2 +1], 0]
            action_good_clip.append(a)
        #print(action_good_clip)
        action_adversary_clip = adversary_control.adversary_action(env)
        action_n = np.concatenate((action_good_clip, action_adversary_clip))
        # interact with env
        obs_n, rew_n, done_n, info_n = env.step(action_n)
        #th1.update_fig(info_n)

        # update the flag
        done = any(done_n)
        #print(done_n)
        terminal = (episode_step >= arglist.per_episode_max_len)

        # reset the env
        if done or terminal: 
            episode_step = 0
            obs_n = env.reset()
            #plt.ioff()
            #th1.reset_fig()

        # render the env
        #print(rew_n)
        env.render()
        time.sleep(0.1)

if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)
