import torch
import rware
import lbforaging
import gym
from time import sleep
from seac.a2c import A2C
from seac.wrappers import RecordEpisodeStatistics, TimeLimit, Monitor
import numpy as np
import matplotlib.pyplot as plt


time_limit = 500
EPISODES = 100

path = "./RWARE16x20-4p/trained_models/TEST1/u7500000"
total_rewards_4p = np.zeros((20-1,EPISODES))

# for agent_count in range(1,20):
    
#     env_name = f"rware-medium-{agent_count}ag-v1"
#     env = gym.make(env_name)
    
#     agents = [
#         A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
#         for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
#     ]
#     for agent in agents:
#         agent.restore(path + f"/agent{agent.agent_id%4}")

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
#         print(f"--- Episode {agent_count}:{ep} Finished ---")
#         print(f"Episode rewards: {sum(info['episode_reward'])}")
#         print(info)
#         print(" --- ")
        
        
#         total_rewards_4p[agent_count-4][ep] = sum(info['episode_reward'])

# np.save("./tr_curriculum-4p.npy",total_rewards_4p)

# path = "./RWARE16x20-16p/trained_models/TEST1/u2000000"
# total_rewards_16p = np.zeros((20-1,EPISODES))

# for agent_count in range(1,20):
    
#     env_name = f"rware-medium-{agent_count}ag-v1"
#     env = gym.make(env_name)
    
#     agents = [
#         A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
#         for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
#     ]
#     for agent in agents:
#         agent.restore(path + f"/agent{agent.agent_id%16}")

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
#         print(f"--- Episode {agent_count}:{ep} Finished ---")
#         print(f"Episode rewards: {sum(info['episode_reward'])}")
#         print(info)
#         print(" --- ")
        
        
#         total_rewards_16p[agent_count-4][ep] = sum(info['episode_reward'])

# np.save("./tr_curriculum-16p.npy",total_rewards_16p)

# path = "./RWARE16x20-16p-hard/trained_models/TEST1/u1000000"
# total_rewards_16p_hard = np.zeros((20-1,EPISODES))

# for agent_count in range(1,20):
    
#     env_name = f"rware-medium-{agent_count}ag-v1"
#     env = gym.make(env_name)
    
#     agents = [
#         A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
#         for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
#     ]
#     for agent in agents:
#         agent.restore(path + f"/agent{agent.agent_id%16}")

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
#         print(f"--- Episode {agent_count}:{ep} Finished ---")
#         print(f"Episode rewards: {sum(info['episode_reward'])}")
#         print(info)
#         print(" --- ")
        
        
#         total_rewards_16p_hard[agent_count-4][ep] = sum(info['episode_reward'])

# np.save("./tr_curriculum-16p-hard.npy",total_rewards_16p_hard)

# path = "./RWARE16x20-16p-cur/trained_models/TEST1/u850000"
# total_rewards_16p_cur = np.zeros((20-1,EPISODES))

# for agent_count in range(1,20):
    
#     env_name = f"rware-medium-{agent_count}ag-v1"
#     env = gym.make(env_name)
    
#     agents = [
#         A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
#         for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
#     ]
#     for agent in agents:
#         agent.restore(path + f"/agent{agent.agent_id%16}")

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
#         print(f"--- Episode {agent_count}:{ep} Finished ---")
#         print(f"Episode rewards: {sum(info['episode_reward'])}")
#         print(info)
#         print(" --- ")
        
        
#         total_rewards_16p_cur[agent_count-4][ep] = sum(info['episode_reward'])

# np.save("./tr_curriculum-16p-cur.npy",total_rewards_16p_cur)

total_rewards_4p = np.load("./RWARE16x20-16p/tr_curriculum-4p.npy")
total_rewards_16p = np.load("./RWARE16x20-16p/tr_curriculum-16p.npy")
total_rewards_16p_hard = np.load("./RWARE16x20-16p/tr_curriculum-16p-hard.npy")
total_rewards_16p_cur = np.load("./RWARE16x20-16p/tr_curriculum-16p-cur.npy")

# The reason for doing this, is because the last line of each evaluation loop, I save it to some total_rewards array
# But I set the index to agent_count-4. This is of course not correct when starting from 1 with the number of agents (instead of 4)
# Therefore, the first 3 runs are set at the -3, -2 and -1 index. These lines will change them back to the beginning to the array
# To line them up. This way it all matches up again.
total_rewards_4p = np.array(list(total_rewards_4p)[-3:] + list(total_rewards_4p)[:-3])
total_rewards_16p = np.array(list(total_rewards_16p)[-3:] + list(total_rewards_16p)[:-3])
total_rewards_16p_hard = np.array(list(total_rewards_16p_hard)[-3:] + list(total_rewards_16p_hard)[:-3])
total_rewards_16p_cur = np.array(list(total_rewards_16p_cur)[-3:] + list(total_rewards_16p_cur)[:-3])

for i in range(len(total_rewards_16p_hard)):
    total_rewards_16p[i] /= (i+1)
    total_rewards_16p_hard[i] /= (i+1)
    total_rewards_4p[i] /= (i+1)
    total_rewards_16p_cur[i] /= (i+1)

print(total_rewards_4p,total_rewards_16p,total_rewards_16p_hard)

ind = np.arange(1,20)  
width = 0.2
  
xvals = [np.average(total_rewards_4p[i]) for i in range(len(total_rewards_4p))]
bar1 = plt.bar(ind, xvals, width, color = 'r') 
c = [np.std(total_rewards_4p[i]) for i in range(len(total_rewards_4p))]
plt.errorbar(ind, xvals, yerr=c, capsize=1, ecolor='black', ls='', lw=1, capthick=1)

yvals = [np.average(total_rewards_16p[i]) for i in range(len(total_rewards_16p))]
bar2 = plt.bar(ind+width, yvals, width, color='g') 
c = [np.std(total_rewards_16p[i]) for i in range(len(total_rewards_16p))]
plt.errorbar(ind+width, yvals, yerr=c, capsize=1, ecolor='black', ls='', lw=1, capthick=1)

zvals = [np.average(total_rewards_16p_hard[i]) for i in range(len(total_rewards_16p_hard))]
bar3 = plt.bar(ind+width*2, zvals, width, color = 'b') 
c = [np.std(total_rewards_16p_hard[i]) for i in range(len(total_rewards_16p_hard))]
plt.errorbar(ind+width*2, zvals, yerr=c, capsize=1, ecolor='black', ls='', lw=1, capthick=1)

wvals = [np.average(total_rewards_16p_cur[i]) for i in range(len(total_rewards_16p_cur))]
bar4 = plt.bar(ind+width*3, wvals, width, color = 'orange', hatch='/') 
c = [np.std(total_rewards_16p_cur[i]) for i in range(len(total_rewards_16p_cur))]
plt.errorbar(ind+width*3, wvals, yerr=c, capsize=1, ecolor='black', ls='', lw=1, capthick=1)

plt.xlabel("Number of agents")
plt.ylabel("Average return per agent")
plt.title("Comparison scaling number of agents structurally via transfer learning or curriculum learning")

  
plt.xticks(ind+width*2,ind) 
plt.legend((bar1, bar2, bar3,bar4),('RWARE16x20-4p', 'RWARE16x20-16p', 'RWARE16x20-16p-hard','RWARE16x20-16p-curriculum')) 
plt.show() 


