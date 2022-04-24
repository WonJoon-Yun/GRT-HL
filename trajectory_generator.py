import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy.matlib as npm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HyperSphereInsideSampler(object):
    """
    Sampler for prior z
    """
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


class TrajectoryGenerator:
    def __init__(self, z_dim=100, y_dim=6, hidden_dim=512, x_dim=300, lr=2e-4, pretrained_path='./save_model/aae-decoder-6000.pt'):
        self.x_dim = x_dim
        self.lr = lr

        self.z_sampler = HyperSphereInsideSampler(r=3, z_dim=z_dim)
        self.decoder = AAEDecoder(y_dim, z_dim, hidden_dim, x_dim)
        self.decoder.load_state_dict(torch.load(pretrained_path))
        self.decoder.to(device)

        self.optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.criterion = nn.L1Loss()

        self.trajectories = None
        self.start = None
        self.end = None

    def generate(self, start, end, num_of_traj=128):
        y = np.repeat(np.concatenate((start, end), axis=None).reshape(1, -1), num_of_traj, axis=0)
        y = torch.FloatTensor(y).to(device)
        prior_z = torch.FloatTensor(self.z_sampler.sample(num_of_traj)).to(device)
        self.trajectories = self.decoder(y, prior_z)
        self.start = start
        self.end = end

        return self.trajectories.cpu().detach().numpy()

    def update(self):
        # calculate contrastive loss
        trajectories = self.trajectories.reshape(-1, 3, 100).transpose(1, 2)
        ideal_traj = self.start + (self.end - self.start) * np.linspace(0, 1, 100)[:, None]
        ideal_traj = torch.FloatTensor(ideal_traj).to(device)

        rewards = torch.sum(1. / torch.norm(ideal_traj - trajectories, dim=2), dim=1)
        high_reward = torch.max(rewards.detach())
        gamma = 0.1
        margin = 0.05 * high_reward

        good_traj_mask = rewards >= ((1. - gamma) * high_reward)
        bad_traj_mask = rewards < ((1. - gamma) * high_reward)

        diff_loss = torch.mean(good_traj_mask * torch.clamp_min(margin - torch.abs(high_reward - rewards), 0.))
        sim_loss = torch.mean(bad_traj_mask * torch.abs(high_reward - rewards))

        loss = diff_loss + sim_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.trajectories = None
        self.start = None
        self.end = None


if __name__ == '__main__':
    traj_generator = TrajectoryGenerator(pretrained_path='./save_model/aae-decoder-6000.pt', lr=4e-5)

    start = np.array([-0.27500741, -0.17016127, 0.32826351])
    end = np.array([0.4, 0., 0.04996991])

    for i in range(201):
        trajectories = traj_generator.generate(start, end, num_of_traj=180)
        traj_generator.update()

        if i % 10 == 0:
            print(i)

            plt.style.use('seaborn-whitegrid')
            fig = plt.figure(figsize=(12, 10))
            ax = fig.gca(projection='3d')
            ax.set_xlim(-0.4, 0.5)
            ax.set_ylim(-0.4, 0.3)
            ax.set_zlim(-0.2, 0.6)

            for traj in trajectories:
                ax.plot3D(traj[:100], traj[100:200], traj[200:])
            plt.savefig(f'./imgs/{i}.png')
