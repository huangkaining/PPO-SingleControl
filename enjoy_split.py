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


from model import actor_agent, critic_agent
from arguments import parse_args

class curve_Thread(threading.Thread):
    def __init__(self):
        super(curve_Thread, self).__init__()
        self.score_list = [[],[]]
        self.score_good = 0
        self.score_adversary = 0
        #plt.ion()

    def run(self):
        while(1):
            time.sleep(0.1)

    def update_fig(self,score):
        self.score_good += score[0]
        self.score_adversary += score[1]
        self.score_list[0].append(self.score_good)
        self.score_list[1].append(self.score_adversary)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.move(900, 300)
        plt.subplot(1, 2, 1)
        plt.title("Good Agent Score")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.plot(range(len(self.score_list[0])), self.score_list[0])
        plt.subplot(1, 2, 2)
        plt.title("Adversary Agent Score")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.plot(range(len(self.score_list[1])), self.score_list[1])
        #plt.show()

    def reset_fig(self):

        plt.subplot(1, 2, 1)
        plt.title("Good Agent Score")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.plot(range(len(self.score_list[0])), self.score_list[0])
        plt.subplot(1, 2, 2)
        plt.title("Adversary Agent Score")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.plot(range(len(self.score_list[1])), self.score_list[1])
        plt.show()
        self.score_good = 0
        self.score_adversary = 0
        self.score_list = [[], []]

class render_Thread(threading.Thread):
    def __init__(self,env,arglist):
        super(render_Thread,self).__init__()
        self.env = env
        self.arglist = arglist

    def run(self):
        env = self.env
        arglist = self.arglist
        episode_step = 0
        actors_tar = get_trainers(env, arglist)
        obs_n = env.reset()
        while (1):

            # update the episode step number

            episode_step += 1

            # get action
            action_n = []
            for actor, obs in zip(actors_tar, obs_n):
                model_out, _ = actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)
                # action_n.append(F.softmax(model_out,dim=-1).detach().cpu().numpy())
                action_n.append(torch.clamp(model_out, -1.0, 1.0))

            action_good_clip = []
            for i in range(env.num_good):
                a = [0, action_n[0][i * 2], 0, action_n[0][i * 2 + 1], 0]
                action_good_clip.append(a)
            # print(action_good_clip)
            action_adversary_clip = adversary_control.adversary_action(env)
            action_n = np.concatenate((action_good_clip, action_adversary_clip))
            # interact with env
            obs_n, rew_n, done_n, info_n = env.step(action_n)

            # update the flag
            done = any(done_n)
            # print(done_n)
            terminal = (episode_step >= arglist.per_episode_max_len)

            # reset the env
            if done or terminal:
                episode_step = 0
                obs_n = env.reset()

            # render the env
            # print(rew_n)
            env.render()
            time.sleep(0.1)


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
    actors_tar = [torch.load(arglist.old_model_name+'a_c_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(1)]

    return actors_tar

def enjoy2(arglist):
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)
    th1 = curve_Thread()  # 绘图线程
    th2 = render_Thread(env,arglist)
    th2.start()
    #th1.start()

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
            model_out,_ = actor(torch.from_numpy(obs).to(arglist.device, torch.float),model_original_out=True)
            #action_n.append(F.softmax(model_out,dim=-1).detach().cpu().numpy())
            action_n.append(torch.clamp(model_out,-1.0 , 1.0))

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
