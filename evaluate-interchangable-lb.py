import torch
import rware
import lbforaging
import gym
from time import sleep
from seac.a2c import A2C
from seac.wrappers import RecordEpisodeStatistics, TimeLimit, Monitor
import numpy as np
import matplotlib.pyplot as plt


path = "./LB8x8-4p-2f-coop/trained_models/TEST1/u7500000"
# path = "./RWARE16x29-16p/trained_models/TEST1/u2000000"
# path = "./RWARE16x29-16p-hard/trained_models/TEST1/u1000000"

env_name = "Foraging-8x8-4p-2f-coop-v1"
# env_name = "rware-medium-16ag-v1"
# env_name = "rware-medium-16ag-hard-v1"

time_limit = 25
# time_limit = 25 # 25 for LBF

EPISODES = 100

total_rewards = np.zeros((35,EPISODES))

CONF = ['0004', '0013', '0022', '0031', '0040', '0103', '0112', '0121', '0130', '0202', '0211', '0220', '0301', '0310', '0400', '1003', '1012', '1021', '1030', '1102', '1111', '1120', '1201', '1210', '1300', '2002', '2011', '2020', '2101', '2110', '2200', '3001', '3010', '3100', '4000']

# for it, conf in enumerate(CONF):
    
#     env = gym.make(env_name)
    
#     agents = [
#         A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
#         for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
#     ]
    
#     restored = 0
    
#     for I in range(len(agents)):
#         for _ in range(int(conf[I])):
#             agents[restored].restore(path + f"/agent{I}")
#             restored += 1     

#     for ep in range(EPISODES):
#         env = gym.make(env_name)
#         # env = Monitor(env, f"seac_rware-small-4ag_eval/video_ep{ep+1}", mode="evaluation", force=True)
#         env = TimeLimit(env, time_limit)
#         env = RecordEpisodeStatistics(env)

#         obs = env.reset()
#         done = [False] * len(agents)

#         while not all(done):
#             obs = [torch.from_numpy(o) for o in obs]
#             _, actions, _ , _, dist = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
#             actions = [a.item() for a in actions]
#             obs, _, done, info = env.step(actions)

#         # obs = env.reset()
#         print(f"--- Episode {conf}:{ep} Finished ---")
#         print(f"Episode rewards: {sum(info['episode_reward'])}")
#         print(info)
#         print(" --- ")
        
        
#         total_rewards[it][ep] = sum(info['episode_reward'])

# np.save("./tr.npy",total_rewards)

# print(total_rewards)

total_rewards = np.load("./tr.npy")
avg = [np.average(total_rewards[i]) for i in range(len(total_rewards))]
std = [np.std(total_rewards[i]) for i in range(len(total_rewards))]

items = list(zip(CONF, avg, std))
items.sort(key= lambda x: (-x[1], x[2]))
X,y,std = zip(*items)
plt.bar(X,y)
plt.errorbar(X, y, yerr=std, fmt="o", color="r")
plt.title("Interchangability of 4 trained agents on LBForaging-8x8-4ag-2f-coop")
plt.xticks(rotation=90)
plt.show()