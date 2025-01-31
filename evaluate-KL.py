import torch
import rware
import lbforaging
import gym
import matplotlib.pyplot as plt
from a2c import A2C
import numpy as np
from wrappers import RecordEpisodeStatistics, TimeLimit, Monitor

AGENTS = 16
ACTION_SPACE = 5
OBSERVATION_SPACE = 71
path = f"./RWARE10x11-{AGENTS}p-40mil/trained_models/TEST1/u2000000/"
path = f"./RWARE16x20-{AGENTS}p/trained_models/TEST1/u2000000/"
# path = "./u1000000"
env_name = "rware-medium-16ag-v1"

time_limit = 500 # 25 for LBF

EPISODES = 5

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

plt.ion()
fig, ax = plt.subplots()


data_heatmap = np.zeros((AGENTS,AGENTS))
heatmap = ax.imshow(data_heatmap)

iteration = 1

for ep in range(EPISODES):
    env = gym.make(env_name)
    env = Monitor(env, f"seac_rware-small-4ag_eval/video_ep{ep+1}", mode="evaluation", force=True)
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    obs = env.reset()
    done = [False] * len(agents)

    while not all(done):
        obs = [torch.from_numpy(o) for o in obs]

        for o in obs:
            # Return the distribution as well (adjustment from the original code)
            _, actions, _ , _, dist = zip(*[agent.model.act(o, None, None) for agent in agents])

            # Get the distributions of each agent, given the state, saved in dist
            # Apply Kullback–Leibler divergence 
            # Add the weighted average term
            with torch.no_grad():
                for i in range(AGENTS):
                    for j in range(AGENTS):
                        KL = 0
                        Px = np.zeros(ACTION_SPACE)
                        Qx = np.zeros(ACTION_SPACE)
                        for a in range(ACTION_SPACE):
                            a = torch.tensor([a])
                            Px[a] = np.exp(dist[i].log_probs(a))
                            Qx[a] = np.exp(dist[j].log_probs(a))
                        
                        
                        Px /= np.sum(Px)
                        Qx /= np.sum(Qx)

                        E = 1e-8
                        Px += E
                        Qx += E
                        mask = Qx > 0
                            
                        KL += np.sum(Px[mask] * np.log(Px[mask] / Qx[mask]))

                        data_heatmap[i][j] = data_heatmap[i][j] + (KL - data_heatmap[i][j])/iteration

                iteration += 1

                heatmap.set_data(data_heatmap)
                heatmap.set_clim(vmin=data_heatmap.min(), vmax=data_heatmap.max())
                plt.title(f"Average KL Divergence on {iteration*len(obs)} states")
                plt.draw()
                plt.pause(0.005)
        
        _, actions, _ , _, dist = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        env.render()
        obs, _, done, info = env.step(actions)
    obs = env.reset()
    print("--- Episode Finished ---")
    print(f"Episode rewards: {sum(info['episode_reward'])}")
    print(info)
    print(" --- ")
