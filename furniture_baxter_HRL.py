from env.furniture_baxter import FurnitureBaxterEnv
import numpy as np
import env.transform_utils as T
from tslearn.metrics import dtw


class FurnitureBaxterHRLEnv(FurnitureBaxterEnv):
    """
    Custom Baxter robot environment.
    """

    def __init__(self, config):
        """
        Args:
            config: configurations for the environment.
        """
        # set the furniture & background
        config.furniture_id = 0
        config.background = 'Lab'

        super().__init__(config)

        self._env_config.update({
            'max_episode_steps': 100,
            'trajectory_follow_reward': 1,
            'grip_reward': 300,
            'success_reward': 1000,
        })

        self._trajectory = None
        self._robot_trajectory = []
        # self._scale_factor = 0.
        # self._curr_traj = None
        # self._prev_dtw = 0.
        self._curr_step = 0

    @property
    def trajectory(self):
        """
        Returns the current trajectory which is baxter to follow.
        """
        return self._trajectory

    @trajectory.setter
    def trajectory(self, traj):
        self._trajectory = traj.reshape(3, 100).T
        # self._scale_factor = np.linalg.norm(self._trajectory[0] - self._trajectory[-1]) / 100.
        # self._curr_traj = []
        # self._prev_dtw = 0.
        self._curr_step = 0

    @property
    def robot_trajectory(self):
        return np.stack(self._robot_trajectory)

    def _step(self, a):
        """
        Takes a simulation step with @a and computes reward.
        """
        # zero out left arm's act and only use right arm
        a[6:] = 0.
        ob, _, done, _ = super(FurnitureBaxterEnv, self)._step(a)

        reward, done, info = self._compute_reward(a)

        if self._success:
            print('Success!')

        info['right_action'] = a[0:6]
        info['left_action'] = a[6:12]
        info['gripper'] = a[12:]

        return ob, reward, done, info

    def _reset(self, furniture_id=None, background=None):
        """
        Resets simulation and variables to compute reward.

        Args:
            furniture_id: ID of the furniture model to reset.
            background: name of the background scene to reset.
        """
        super()._reset(furniture_id, background)

        # set two bodies for picking or assemblying
        id1 = self.sim.model.eq_obj1id[0]
        id2 = self.sim.model.eq_obj2id[0]
        self._target_body = [
            self.sim.model.body_id2name(id1), self.sim.model.body_id2name(id2)
        ]

        self._trajectory = None
        self._robot_trajectory = []
        # self._scale_factor = 0.
        # self._curr_traj = None
        # self._prev_dtw = 0.
        self._curr_step = 0

        self.start = np.array(self.sim.data.site_xpos[self.right_eef_site_id])
        self.end = self._get_pos(self._target_body[0])

    def _place_objects(self):
        """
        Returns the fixed initial positions and rotations of furniture parts.

        Returns:
            xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
            xquat((float * 4) * n_obj): quaternion of the objects
        """
        # pos_init = [[-0.4, 0.0, 0.05], [0.0, -0.5, 0.05], [0.4, 0.2, 0.05]]
        # quat_init = [[1, 0, 0.7, 0.3], [1, 0, 0.3, 0.7], [1, 0.5, 0.2, 0.3]]
        pos_init = [[0.0, -0.2, 0.05], [0.4, -0.0, 0.05]]
        quat_init = [[1, 0, 0, 0], [1, 0, 0, 0]]
        return pos_init, quat_init

    def _compute_reward(self, a):
        """
        Computes the intrinsic reward.
        """
        info = {}

        # control penalty
        ctrl_reward = self._ctrl_reward(a)
        info['reward_ctrl'] = ctrl_reward

        # compute positions and rotations
        hand_pos = [
            np.array(self.sim.data.site_xpos[self.right_eef_site_id]),
            np.array(self.sim.data.site_xpos[self.left_eef_site_id])
        ]
        info['right_hand'] = hand_pos[0]
        info['left_hand'] = hand_pos[1]

        finger_pos = [
            [self._get_pos('r_fingertip_g0'), self._get_pos('l_fingertip_g0')],
            [self._get_pos('l_g_r_fingertip_g0'), self._get_pos('l_g_l_fingertip_g0')]
        ]
        info['right_r_finger'] = finger_pos[0][0]
        info['right_l_finger'] = finger_pos[0][1]

        # target position
        pos = self._get_pos(self._target_body[0])
        info['target_pos_0'] = pos

        self._robot_trajectory.append(info['right_hand'])

        # euclidean reward
        curr_anchor = self._trajectory[self._curr_step]
        next_anchor = self._trajectory[self._curr_step + 1]
        dist_from_anchor = T.l2_dist(info['right_hand'], curr_anchor)

        threshold_coef = 1.
        threshold = threshold_coef * T.l2_dist(curr_anchor, next_anchor)
        positive_reward_coef = 1.
        negative_reward_coef = 0.1
        if dist_from_anchor < threshold:
            traj_reward = (threshold**2 - dist_from_anchor**2) / (0.05 * (threshold**2))
            traj_reward *= positive_reward_coef
        else:
            traj_reward = -dist_from_anchor / threshold
            # traj_reward = -np.power((dist_from_anchor - threshold) / threshold, 2) / 0.05
            traj_reward *= negative_reward_coef

        info['reward_traj'] = traj_reward

        # # dtw reward
        # self._curr_traj.append(info['right_hand'])
        # curr_dtw = dtw(np.array(self._curr_traj), self._trajectory[:self._curr_step+1])
        # traj_reward = self._scale_factor / (curr_dtw - self._prev_dtw + 0.1 * self._scale_factor)
        # info['reward_traj'] = traj_reward

        # # trajectory following reward
        # curr_anchor = self._trajectory[self._curr_step]
        # next_anchor = self._trajectory[self._curr_step + 1]
        # dist_anchor = T.l2_dist(curr_anchor, next_anchor)
        #
        # threshold_coef = 0.5
        # threshold = threshold_coef * dist_anchor
        # dist_from_anchor = T.l2_dist(curr_anchor, info['right_hand'])
        #
        # positive_reward_coef = 1.
        # negative_reward_coef = 0.05
        # if dist_from_anchor <= threshold:
        #     dist_from_line = self._linseg_dist(info['right_hand'], curr_anchor, next_anchor)
        #     # dist_from_line = np.abs(np.cross(next_anchor - curr_anchor, curr_anchor - info['right_hand'])) / T.l2_dist(curr_anchor, next_anchor)
        #     traj_reward = (positive_reward_coef * threshold) / dist_from_line
        # else:
        #     traj_reward = - (negative_reward_coef * dist_from_anchor) / threshold

        # clip reward
        # traj_reward = np.clip(traj_reward, -5., 5.)

        reward = info['reward_ctrl'] + info['reward_traj']

        self._curr_step += 1
        # self._prev_dtw = curr_dtw

        is_done = False
        if self._curr_step == 99:
            is_done = True

        return reward, is_done, info
