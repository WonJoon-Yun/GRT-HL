from collections import defaultdict
from hrl.ppo import Memory

import numpy as np
import cv2
import os
import moviepy.editor as mpy


robot_ob_mask = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23]


class VideoSaver:
    def __init__(self):
        self.env = None
        self.policy = None
        self.traj = None
        self.record_frames = None

    def set(self, env, policy, trajectory):
        self.env = env
        self.policy = policy
        self.traj = trajectory
        self.record_frames = []

    def record(self, fname, record_dir='hrl/videos', fps=15.):
        running_reward = self._run_episode()
        path = os.path.join(record_dir, f'{fname}-{running_reward}.mp4')

        def f(t):
            frame_length = len(self.record_frames)
            new_fps = 1. / (1. / fps + 1. / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return self.record_frames[idx]

        video = mpy.VideoClip(f, duration=len(self.record_frames) / fps + 2)
        video.write_videofile(path, fps, verbose=False)
        print('Video saved')

    def _run_episode(self):
        done = False
        obs = self.env.reset()
        self.env.trajectory = self.traj
        timestep = 0
        memory = Memory()
        running_reward = 0.

        while not done:
            robot_ob = obs['robot_ob'][robot_ob_mask]
            next_anchor = self.env.trajectory[timestep]
            custom_obs = np.concatenate((robot_ob, next_anchor))

            # sample action from policy
            action = self.env.action_space.sample()
            action[:6] = self.policy.select_action(custom_obs, memory)

            # take a step
            obs, reward, done, info = self.env.step(action)
            running_reward += reward

            self._store_frame()

            timestep += 1

        return running_reward

    def _store_frame(self):
        # render video frame
        frame = self.env.render('rgb_array') * 255.
        fheight, fwidth = frame.shape[:2]
        frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)
        self.record_frames.append(frame)
