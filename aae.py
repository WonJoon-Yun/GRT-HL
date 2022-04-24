#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy.matlib as npm
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import time


# Probability distribution for prior z
class HyperSphereInsideSampler(object):
    def __init__(self, name='hyper_sphere_inside_sampler', r=3, z_dim=5):
        self.name = name
        self.r = r
        self.z_dim = z_dim

    def sample(self, n):
        z = np.random.randn(n, self.z_dim)
        z_norm = np.linalg.norm(z, axis=1)
        z_unit = z / npm.repmat(z_norm.reshape((-1, 1)), 1, self.z_dim)  # on the surface of a hypersphere
        u = np.power(np.random.rand(n, 1), (1 / self.z_dim) * np.ones(shape=(n, 1)))
        z_sphere = self.r * z_unit * npm.repmat(u, 1, self.z_dim)  # samples inside the hypersphere
        samples = z_sphere
        return samples

    def plot(self, n=1000, tfs=20):
        samples = self.sample(n=n)
        plt.figure(figsize=(6, 6))
        plt.plot(samples[:, 0], samples[:, 1], 'k.')
        plt.xlim(-self.r, self.r)
        plt.ylim(-self.r, self.r)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.name, fontsize=tfs)
        plt.show()

class GaussianSampler:
    def __init__(self, z_dim=5):
        self.z_dim = z_dim

    def sample(self, n):
        z = np.random.randn(n, self.z_dim)
        samples = z
        return samples

# # Gaussian Random Path
class GaussianRandomPath:
    def __init__(self, gain, length, sigma_w):
        self.gain = gain
        self.length = length
        self.sigma_w = sigma_w

    def kernel(self, test_time, anchor_time):
        """
        Squared Exponential (SE) kernel function
        :param test_time: (T, )
        :param anchor_time: (M, )
        """
        test_time_ = test_time.reshape(-1, 1)
        anchor_time_ = anchor_time.reshape(1, -1)

        return self.gain * np.exp((-1 / (2. * self.length)) * np.power(test_time_ - anchor_time_, 2))

    def path_mean(self, test_time, anchor_time, anchor_point):
        kernel_t_a = self.kernel(test_time, anchor_time)
        kernel_a_a = self.kernel(anchor_time, anchor_time)

        mean = kernel_t_a @ np.linalg.pinv(kernel_a_a + self.sigma_w * np.identity(len(anchor_time))) @ anchor_point
        return mean

    def path_covariance(self, test_time, anchor_time):
        kernel_t_a = self.kernel(test_time, anchor_time)
        kernel_a_a = self.kernel(anchor_time, anchor_time)
        kernel_t_t = self.kernel(test_time, test_time)

        covariance = kernel_t_t - kernel_t_a @ np.linalg.pinv(
            kernel_a_a + self.sigma_w * np.identity(len(anchor_time))) @ kernel_t_a.T
        return covariance


# Data loader
def batch_generator(trajectory_len=100, anchors_per_batch=2, batch_size=128, batches_per_epoch=200):
    for i in range(batches_per_epoch):
        anchor_time = np.array([0., 1.])
        gen_time = np.linspace(0., 1., trajectory_len)

        # 생성되는 Trajectory의 Start Point와 End Point 설정
        start_points = np.clip(np.random.randn(anchors_per_batch, 3) * 0.2, -1., 1.)
        end_points = np.clip(np.random.randn(anchors_per_batch, 3) * 0.2, -1., 1.)

        batch_trajectory = []
        batch_anchor = []

        for start_point, end_point in zip(start_points, end_points):
            if sum(start_point - end_point) == 0.:
                continue

            # anchor point가 2개인 경우이므로,
            # 더 많은 포인트를 사용하고자하면 추가적으로 입력해주면됨.
            anchor_point = np.array([start_point, end_point])

            # GRP 의 parameter 조정
            down_scale = 3.4641 / np.linalg.norm(anchor_point[0] - anchor_point[1])
            scale_factor = down_scale ** 2
            # Trajectory 생성
            grp = GaussianRandomPath(gain=0.7 * scale_factor, length=0.2 * scale_factor, sigma_w=0.)

            mean = grp.path_mean(gen_time, anchor_time, anchor_point)
            covariance = grp.path_covariance(gen_time, anchor_time)

            try:
                trajectory_x = np.random.multivariate_normal(mean[:, 0], covariance, (batch_size // anchors_per_batch,))
                trajectory_y = np.random.multivariate_normal(mean[:, 1], covariance, (batch_size // anchors_per_batch,))
                trajectory_z = np.random.multivariate_normal(mean[:, 2], covariance, (batch_size // anchors_per_batch,))
            except np.linalg.LinAlgError:
                continue

            # X, Y, Z 축 좌표를 Concatenate하여 배열로 만듬 => Trajectory
            trajectory = np.concatenate((trajectory_x, trajectory_y, trajectory_z), axis=1)

            out_of_range_idx = np.union1d(np.where(trajectory > 1.)[0], np.where(trajectory < -1.)[0])
            trajectory = np.delete(trajectory, out_of_range_idx, axis=0)

            anchor = np.concatenate((start_point, end_point), axis=None)
            anchor = np.repeat(anchor.reshape(1, -1), trajectory.shape[0], axis=0)

            batch_trajectory.append(trajectory)
            batch_anchor.append(anchor)

        batch_trajectory = np.concatenate(batch_trajectory, axis=0)
        batch_anchor = np.concatenate(batch_anchor, axis=0)

        yield torch.FloatTensor(batch_trajectory), torch.FloatTensor(batch_anchor)


#     for start_point, end_point in zip(start_points, end_points):
#         anchor_time = np.array([0., 1.])
#         anchor_point = np.array([start_point, end_point])
#         gen_time = np.linspace(0., 1., trajectory_len)

#         down_scale = 3.4641 / np.linalg.norm(anchor_point[0] - anchor_point[1])
#         scale_factor = down_scale ** 2
#         grp = GaussianRandomPath(gain=0.7*scale_factor, length=0.2*scale_factor, sigma_w=0.)

#         mean = grp.path_mean(gen_time, anchor_time, anchor_point)
#         covariance = grp.path_covariance(gen_time, anchor_time)

#         trajectory_x = np.random.multivariate_normal(mean[:, 0], covariance, (batch_size, ))
#         trajectory_y = np.random.multivariate_normal(mean[:, 1], covariance, (batch_size, ))
#         trajectory_z = np.random.multivariate_normal(mean[:, 2], covariance, (batch_size, ))
#         trajectory = np.concatenate((trajectory_x, trajectory_y, trajectory_z), axis=1)

#         out_of_range_idx = np.union1d(np.where(trajectory > 1.)[0], np.where(trajectory < -1.)[0])
#         trajectory = np.delete(trajectory, out_of_range_idx, axis=0)

#         anchor = np.concatenate((start_point, end_point), axis=None)
#         anchor = np.repeat(anchor.reshape(1, -1), trajectory.shape[0], axis=0)

#         yield trajectory, anchor


# Define models for semi AAE
class AAEEncoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, y_dim, z_dim):
        super(AAEEncoder, self).__init__()
        self.input_block = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU()
        )

        self.out_y_block = nn.Sequential(
            nn.Linear(hidden_dim, y_dim),
            nn.Hardtanh()
        )

        self.out_z_block = nn.Sequential(
            nn.Linear(hidden_dim, z_dim)
            #             nn.Tanh()
        )

    def forward(self, x):
        out = self.input_block(x)
        y = self.out_y_block(out)
        #         z = self.out_z_block(out) * 3  # mapping to (-3, 3)
        z = self.out_z_block(out)
        return y, z


class AAEDecoder(nn.Module):
    def __init__(self, y_dim, z_dim, hidden_dim, x_hat_dim):
        super(AAEDecoder, self).__init__()
        self.input_y_block = nn.Sequential(
            nn.Linear(y_dim, 16),
            nn.LeakyReLU(0.2)
        )

        self.input_z_block = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2)
        )

        self.out_block = nn.Sequential(
            nn.Linear(16 + hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, x_hat_dim),
            nn.Tanh()
        )

    def forward(self, y, z):
        out_y = self.input_y_block(y)
        out_z = self.input_z_block(z)
        x_hat = self.out_block(torch.cat((out_y, out_z), dim=1))
        return x_hat

class AAEDiscriminator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(AAEDiscriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.block(z)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='./runs/semi_aae')

traj_len = 100
x_dim = traj_len * 3
hidden_dim = 512
y_dim = 6
z_dim = 100
epsilon = 1e-7

# exp_id = time.monotonic()
exp_id = 'lr-scheduling-hyperspherev2-0.5'
RESULT_PATH = f'./gen_paths/{exp_id}'
SAVE_MODEL_PATH = f'./save_model/{exp_id}'

z_sampler = HyperSphereInsideSampler(r=3, z_dim=z_dim)
# z_sampler = GaussianSampler(z_dim=z_dim)

def train(epochs=201, lr=0.001, betas=(0.5, 0.999)):
    os.mkdir(RESULT_PATH)
    os.mkdir(SAVE_MODEL_PATH)

    Q = AAEEncoder(x_dim, hidden_dim, y_dim, z_dim).to(device)
    P = AAEDecoder(y_dim, z_dim, hidden_dim, x_dim).to(device)
    D = AAEDiscriminator(z_dim, hidden_dim).to(device)

    criterion = nn.MSELoss()
    #     criterion_semi = nn.L1Loss()

    # optimizer for reconstruction loss
    q_enc_optim = optim.Adam(Q.parameters(), lr=lr, betas=betas)
    p_optim = optim.Adam(P.parameters(), lr=lr, betas=betas)

    # optimizer for semi-supervised learning
    q_semi_optim = optim.Adam(Q.parameters(), lr=lr, betas=betas)

    # optimizer for discriminator
    d_optim = optim.Adam(D.parameters(), lr=lr, betas=betas)

    # optimizer for z generating
    q_gen_optim = optim.Adam(Q.parameters(), lr=lr, betas=betas)

    # lr schedulers
    milestones = [1000, 2000, 4000]
    gamma = 0.5

    q_enc_optim_scheduler = optim.lr_scheduler.MultiStepLR(q_enc_optim, milestones=milestones, gamma=gamma)
    p_optim_scheduler = optim.lr_scheduler.MultiStepLR(p_optim, milestones=milestones, gamma=gamma)
    q_semi_optim_scheduler = optim.lr_scheduler.MultiStepLR(q_semi_optim, milestones=milestones, gamma=gamma)
    d_optim_scheduler = optim.lr_scheduler.MultiStepLR(d_optim, milestones=milestones, gamma=gamma)
    q_gen_optim_scheduler = optim.lr_scheduler.MultiStepLR(q_gen_optim, milestones=milestones, gamma=gamma)

    for epoch in range(epochs):
        recon_losses = []
        semi_losses = []
        disc_losses = []
        gen_losses = []

        for i, (trajectory, anchor_point) in enumerate(batch_generator(trajectory_len=100, batch_size=256)):
            real_traj = trajectory.to(device)
            anchor = anchor_point.to(device)

            if torch.isnan(real_traj).any().item() or torch.isinf(real_traj).any().item():
                continue

            # reconstruction loss
            recon_traj = P(*Q(real_traj))
            recon_loss = criterion(recon_traj, real_traj)

            q_enc_optim.zero_grad()
            p_optim.zero_grad()
            recon_loss.backward()
            q_enc_optim.step()
            p_optim.step()

            # semi-supervised loss
            anchor_pred, _ = Q(real_traj)
            semi_loss = criterion(anchor_pred, anchor)

            q_semi_optim.zero_grad()
            semi_loss.backward()
            q_semi_optim.step()

            # discriminator loss
            Q.eval()
            z_prior = torch.FloatTensor(z_sampler.sample(real_traj.shape[0])).to(device)
            _, z_gen = Q(real_traj)

            d_z_real = D(z_prior)
            d_z_fake = D(z_gen)
            d_loss = -torch.mean(torch.log(d_z_real + epsilon) + torch.log(1 - d_z_fake + epsilon))

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # z generator loss
            Q.train()
            _, z_gen = Q(real_traj)
            d_z_fake = D(z_gen)
            gen_loss = -torch.mean(torch.log(d_z_fake + epsilon))

            q_gen_optim.zero_grad()
            gen_loss.backward()
            q_gen_optim.step()

            # log history
            recon_losses.append(recon_loss.item())
            semi_losses.append(semi_loss.item())
            disc_losses.append(d_loss.item())
            gen_losses.append(gen_loss.item())

        q_enc_optim_scheduler.step()
        p_optim_scheduler.step()
        q_semi_optim_scheduler.step()
        d_optim_scheduler.step()
        q_gen_optim_scheduler.step()

        writer.add_scalar('Reconstruction loss', np.mean(recon_losses), epoch + 1)
        writer.add_scalar('Semi-supervised loss', np.mean(semi_losses), epoch + 1)
        writer.add_scalar('Discriminator loss', np.mean(disc_losses), epoch + 1)
        writer.add_scalar('Generator loss', np.mean(gen_losses), epoch + 1)

        print(
            f'{epoch + 1} recon_loss: {np.mean(recon_losses)}, semi_loss: {np.mean(semi_losses)}, d_loss: {np.mean(disc_losses)}, g_loss: {np.mean(gen_losses)}')

        if epoch % 10 == 0:
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure(figsize=(12, 10))
            ax = fig.gca(projection='3d')

            recon_traj = recon_traj.cpu().detach().numpy()[128:180]

            for traj in recon_traj:
                ax.plot3D(traj[:100], traj[100:200], traj[200:])
            plt.savefig(f'{RESULT_PATH}/{epoch}.png')

        # save moodel
        if epoch % 100 == 0:
            torch.save(P.state_dict(), f'{SAVE_MODEL_PATH}/aae-decoder-{epoch}.pt')

    writer.close()

def validate(model_path, start=[-0.423, 0.098, -0.041], end=[-0.291, -0.312, -0.002], num_of_traj=100):
    # load pretrained Decoder
    pre_P = AAEDecoder(y_dim, z_dim, hidden_dim, x_dim)
    pre_P.load_state_dict(torch.load(model_path))
    pre_P.to(device)

    y = np.repeat(np.concatenate((start, end), axis=None).reshape(1, -1), num_of_traj, axis=0)
    # y = np.repeat(np.array([-0.6, 0.1, 0.4, -0.4, 0.15, 0.395]).reshape(1, -1), num_of_traj, axis=0)
    y = torch.FloatTensor(y).to(device)
    prior_z = torch.FloatTensor(z_sampler.sample(num_of_traj)).to(device)

    trajectories = pre_P(y, prior_z)

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.gca(projection='3d')

    trajectories = trajectories.cpu().detach().numpy()

    for traj in trajectories:
        ax.plot3D(traj[:100], traj[100:200], traj[200:])
    plt.show()


# In[ ]:


train(epochs=6001, lr=0.0002)
# validate('./save_model/sota/aae-decoder-2000.pt', start=[-0.423, 0.098, -0.041], end=[-0.291, -0.312, -0.002], num_of_traj=100)
