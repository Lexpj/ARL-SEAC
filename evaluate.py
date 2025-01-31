import torch
import rware
import lbforaging
import gym
from seac.a2c import A2C
from seac.wrappers import RecordEpisodeStatistics, TimeLimit, Monitor
from gym.wrappers.record_video import RecordVideo
from PIL import Image

path = "./LB10x10-3p-3f/trained_models/TEST1/u2000000"
path = "./LB15x15-3p-4f/trained_models/TEST1/u1500000"
path = "./RWARE10x11-2p/trained_models/TEST2/u4000000"
path = "./RWARE10x11-4p-40mil/trained_models/TEST1/u2000000"
# path = "./LB15x15-3p-4f/trained_models/TEST1/u1500000"

# path = "./LB8x8-2p-2f-coop/trained_models/TEST2/u7000000"
# path = "./RWARE10x11-4p-40mil/trained_models/TEST1/u2000000"
# path = "./RWARE16x20-16p/trained_models/TEST1/u2000000"
# path = "./RWARE16x20-4p/trained_models/TEST1/u7500000"
# path = "./RWARE16x20-16p-hard/trained_models/TEST1/u1000000"

env_name = "Foraging-10x10-3p-3f-v1"
env_name = "Foraging-15x15-3p-4f-v1"
env_name = "rware-tiny-2ag-v1"
env_name = "rware-tiny-4ag-v1"
# env_name = "rware-medium-4ag-v1"
# env_name = "rware-medium-16ag-hard-v1"

time_limit = 500
# time_limit = 25 # 25 for LBF

EPISODES = 1

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

rgbs = []

for ep in range(EPISODES):
    env = gym.make(env_name)
    # env = Monitor(env, f"./videos/{env_name}/video_ep{ep+1}", mode="evaluation", force=True)
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    obs = env.reset()
    done = [False] * len(agents)
    
    while not all(done):
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        
        rgb = env.render(mode="rgb_array")
        rgbs.append(Image.fromarray(rgb))
        obs, _, done, info = env.step(actions)
        
    rgb = env.render(mode="rgb_array")
    rgbs.append(Image.fromarray(rgb))
    rgbs.append(Image.fromarray(rgb))
    rgbs.append(Image.fromarray(rgb))
    rgbs.append(Image.fromarray(rgb))
    
    print("--- Episode Finished ---")
    print(f"Episode rewards: {sum(info['episode_reward'])}")
    print(info)
    print(" --- ")

rgbs[0].save(f"./videos/{env_name}.gif", save_all=True, append_images=rgbs[1:], duration=100, loop=0)