import numpy as np


def cal_angle(agent_source, agent_target):
    # 计算从source到target的角度（返回角度值角度）
    vec = agent_target.state.p_pos - agent_source.state.p_pos
    angle = np.arctan2(vec[1],vec[0])
    return angle

def adversary_action(env):
    good_agent = env.good_agents
    adversary_agent = env.adversary_agents
    action_n = []
    for adv in adversary_agent:
        tar = good_agent[0]
        dis_min = np.sqrt(np.sum(np.square(adv.state.p_pos - good_agent[0].state.p_pos)))
        for agent in good_agent:
            if agent is tar: continue
            dis = np.sqrt(np.sum(np.square(adv.state.p_pos - agent.state.p_pos)))
            if dis < dis_min:
                tar = agent
                dis_min = dis
        ang = cal_angle(adv,tar)
        action = np.array([0 , 2 * np.cos(ang), 0 , 2 * np.sin(ang) , 0])
        action_n.append(action)
    assert len(action_n) == 4
    return action_n