from env import make_env
from hrl.trajectory_generator import TrajectoryGenerator
from hrl.ppo import Memory
from hrl.ppo import PPO
from hrl.video_saver import VideoSaver
import numpy as np
import matplotlib.pyplot as plt
import torch
from config.furniture import get_default_config

obs_dim = 17
act_dim = 6
max_iter = 200
update_episode = 10
num_of_traj = 180
robot_ob_mask = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23]
ckpts_path = './hrl/ckpts'

def train():
    traj_generator = TrajectoryGenerator(pretrained_path='./hrl/save_model/aae-decoder-6000.pt', lr=4e-5)
    ppo = PPO(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, K_epochs=100)

    config = get_default_config()
    config.unity = False
    env = make_env('FurnitureBaxterHRLEnv', config)
    obs = env.reset()
    memory = Memory()

    video_saver = VideoSaver()

    for i in range(max_iter):
        # meta policy
        start = obs['robot_ob'][2:5]
        end = obs['object_ob'][0:3]
        trajectories = traj_generator.generate(start, end, num_of_traj)

        for episode, traj in enumerate(trajectories):
            done = False
            time_step = 0
            running_reward = 0.
            env.trajectory = traj

            while not done:
                robot_ob = obs['robot_ob'][robot_ob_mask]
                next_anchor = traj[time_step::100]
                custom_obs = np.concatenate((robot_ob, next_anchor))

                # run using old policy
                action = env.action_space.sample()
                action[:6] = ppo.select_action(custom_obs, memory)
                obs, reward, done, info = env.step(action)
                running_reward += reward

                # save to memory
                memory.rewards.append(reward)
                memory.done_mask.append(done)

                time_step += 1

            print(f'Meta round: {i}, Episode: {episode}, Total reward: {running_reward}')

            if (episode + 1) % update_episode == 0:
                ppo.update(memory)
                memory.clear()

                # save videos and images
                video_saver.set(env, ppo, traj)
                video_saver.record(fname=f'{i}-{episode}')
                save_traj(env.trajectory, env.robot_trajectory, f'{i}-{episode}')

                # save ppo model
                torch.save(ppo.actor.state_dict(), f'{ckpts_path}/actor-{i}-{episode}.pt')
                torch.save(ppo.critic.state_dict(), f'{ckpts_path}/critic-{i}-{episode}.pt')

            obs = env.reset()

        # update meta policy and save model
        traj_generator.update()
        torch.save(traj_generator.decoder.state_dict(), f'{ckpts_path}/traj-generator-{i}.pt')


def save_traj(real_traj, robot_traj, fname):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca(projection='3d')

    ax.plot3D(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], 'red')
    ax.plot3D(robot_traj[:, 0], robot_traj[:, 1], robot_traj[:, 2], 'blue')
    plt.savefig(f'./hrl/results/{fname}.png')


if __name__ == '__main__':
    train()
