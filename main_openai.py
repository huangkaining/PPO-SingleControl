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
import json

from arguments import parse_args
from replay_buffer import ReplayBuffer
import magentEnv.envFile.scenarios as scenarios
from model import openai_actor, openai_critic
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
    actors_cur = [None for _ in range(num_adversaries)]
    critics_cur = [None for _ in range(num_adversaries)]
    actors_tar = [None for _ in range(num_adversaries)]
    critics_tar = [None for _ in range(num_adversaries)]
    optimizers_c = [None for _ in range(num_adversaries)]
    optimizers_a = [None for _ in range(num_adversaries)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    if arglist.restore == True: # restore the model
        for idx in arglist.restore_idxs:
            trainers_cur[idx] = torch.load(arglist.old_model_name+'c_{}'.format(agent_idx))
            trainers_tar[idx] = torch.load(arglist.old_model_name+'t_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(num_adversaries):
        #print("obs shape,action shape:",obs_shape_n[i],action_shape_n[i])
        #print("sum(obs_shape_n),sum(action_shape_n):",sum(obs_shape_n),sum(action_shape_n))
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0) # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0) # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c

def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tao + \
                    (1-tao)*state_dict_t[key] 
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """ 
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
        (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...'+' '*100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx) # Note_The func is not the same as others
                
            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float) # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device) # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                for idx, a_t in enumerate(actors_tar)], dim=1)
            #print("obs_n_o action_cur_o:",obs_n_o.shape,action_cur_o.shape)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1) # q 
            q_ = critic_t(obs_n_n, action_tar).reshape(-1) # q_ 
            tar_value = q_*arglist.gamma*done_n + rew # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            '''for name, parms in critic_c.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                    ' -->grad_value:', parms.grad)'''
            #print(loss_c)
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new 
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

            opt_a.zero_grad()
            (1e-3*loss_pse+loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y_%m%d')
            print('=time:{} step:{}'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                arglist.scenario_name, time_now, arglist.exp_name),'{}'.format(\
                game_step))
            if not os.path.exists(model_file_dir): # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao) 
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao) 

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

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
    num_adversaries = min(env.n, arglist.num_adversaries)
    num_adversaries = 1
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    #memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)
    
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    episode_cnt = 0
    update_cnt = 0
    t_start = time.time()
    episode_len = 0
    rew_n_old = [0.0 for _ in range(env.n)] # set the init reward
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [0.0] # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(1)] # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape 
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')
    obs_n = env.reset()

    for episode_gone in range(arglist.max_episode):
        #print("episode_gone:{}".format(episode_gone),end='\n')
        episode_len = 0
        for episode_cnt in range(arglist.per_episode_max_len):
            #if episode_cnt%10==0: print("episode_cnt:{}".format(episode_cnt))
            episode_len += 1
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                for agent, obs in zip(actors_cur, obs_n)]

            action_good_clip = []   #??????????????????????????????
            action_good_clip_real = [] #?????????????????????????????????
            for i in range(env.num_good):
                #a = action_n[0][i * 5:i * 5 + 5]
                a = [0 , action_n[0][i* 2], 0 , action_n[0][i * 2 +1], 0]
                b = [action_n[0][i*2],action_n[0][i*2+1]]
                action_good_clip.append(a)
                action_good_clip_real.append(b)

            action_adversary_clip = adversary_control.adversary_action(env)
            action_n = np.concatenate((action_good_clip , action_adversary_clip))
            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            # save the experience
            memory.add(obs_n, np.concatenate(action_good_clip_real), rew_n , new_obs_n, done_n)#???????????????????????????????????????
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents 
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(\
                arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = any(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len-1)
            if done or terminal:
                episode_step = 0
                obs_n = env.reset()
                agent_info.append([[]])
                episode_rewards.append(0)
                for a_r in agent_rewards:   
                    a_r.append(0)
                #continue
        '''if game_step > 1 and game_step % 100 == 0:
                    mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 2) for idx in range(env.n)]
                    mean_ep_r = round(np.mean(episode_rewards[-200:-1]), 3)
                    print(" "*43 + 'episode reward:{} agents mean reward:{}'.format(mean_ep_r, mean_agents_r), end='\r')
                print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')'''
        if episode_gone > 1 and episode_gone % 100 == 0:
            print("=Training=episode:{} average reward:{}".format(episode_gone,\
                  np.mean(episode_rewards[-100:])), end="\n")
            if arglist.store_learning_curve:
                save_data = {}
                save_data['episode_gone'] = episode_gone
                save_data['episode_rewards'] = episode_rewards[-100:]
                for i in range(len(agent_rewards)):
                    save_data[str(i)] = agent_rewards[i][-100:]
                with open(learning_curve_file,'a') as f:
                    ss = json.dumps(save_data)
                    f.write(ss + "\n")




if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
