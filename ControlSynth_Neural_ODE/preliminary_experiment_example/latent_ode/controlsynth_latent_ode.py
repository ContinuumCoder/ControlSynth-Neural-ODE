import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import importlib.util
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.0025)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint





current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)  
solver_path = current_directory + '/../../src/solver.py'
spec = importlib.util.spec_from_file_location("solver", solver_path)
solver_module = importlib.util.module_from_spec(spec)
sys.modules["solver"] = solver_module
spec.loader.exec_module(solver_module)







def generate_spiral2d(nspiral=1,
                      ntotal=300,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.0001,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        # plt.savefig('./ground_truth.png', dpi=500)
        # print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        # cc = bool(npr.rand() > .5)  # uniformly select rotation
        cc = True
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


# class ControlODEfunc(nn.Module):
#     def __init__(self, latent_dim=4):
#         super(ControlODEfunc, self).__init__()
#         self.fc1 = nn.Linear(latent_dim, latent_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = self.fc1(x)
#         return out

class ControlODEfunc(nn.Module):
    def __init__(self, latent_dim=4):
        super(ControlODEfunc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=1)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = out.permute(0, 2, 1).squeeze()
        return out

class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == '__main__':
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = 0.
    stop = 6 * np.pi
    noise_std = .03
    a = 0.
    b = .3
    ntotal = 1000
    nsample = 100
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    orig_ts = torch.from_numpy(orig_ts).float().to(device)

    # train models with different dynamics
    models = ['ControlSynth', 'standard']
    for DYNAMICS in models:
        func = LatentODEfunc(latent_dim, nhidden).to(device)
        g_func = ControlODEfunc(latent_dim).to(device)
        rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
        dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
        print(f"Training with {DYNAMICS} dynamics...")
        params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
        # optimizer = optim.Adam(params, lr=args.lr)
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)

        loss_meter = RunningAverageMeter()

        try:
            for itr in range(1, args.niters + 1):
                optimizer.zero_grad()
                # backward in time to infer q(z_0)
                h = rec.initHidden().to(device)
                for t in reversed(range(samp_trajs.size(1))):
                    obs = samp_trajs[:, t, :]
                    out, h = rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                # forward in time and solve ode for reconstructions   
                pred_z = z0
                if DYNAMICS == "ControlSynth":       
                    solver = solver_module.DynamicODESolver(func, z0, ufunc=g_func, u0=z0)
                    pred_z = solver.integrate(samp_ts).permute(1, 0, 2)
                else:
                    pred_z = odeint(func, z0, samp_ts, method='euler').permute(1, 0, 2)

                pred_x = dec(pred_z)
                # loss = torch.mean(torch.abs(samp_trajs - pred_x))
                loss = torch.mean((samp_trajs - pred_x) ** 2) 

                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item())

                print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

        except KeyboardInterrupt:
            pass

        print('Training complete after {} iters.'.format(itr))

        with torch.no_grad():
            # sample from trajectorys' approx. posterior
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0s = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

           

            ts_pos = np.linspace(0., 2.5 * np.pi, num=2500)
            ts_pos = torch.from_numpy(ts_pos).float().to(device)

            for traj_id in range(5):

                # take first trajectory for visualization
                z0 = z0s[traj_id]

                zs_pos = z0
                if DYNAMICS == "ControlSynth":  
                    solver = solver_module.DynamicODESolver(func, z0, ufunc=g_func, u0=z0)
                    zs_pos = solver.integrate(ts_pos)
                else:
                    zs_pos = odeint(func, z0, ts_pos, method='euler')

                xs_pos = dec(zs_pos)

                xs_pos = xs_pos.cpu().numpy()
                orig_traj = orig_trajs[traj_id].cpu().numpy()
                samp_traj = samp_trajs[traj_id].cpu().numpy()

                plt.figure()
                plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                        'g', label='true trajectory')
                plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                        label=f'learned trajectory ({DYNAMICS} dynamics)')
                plt.scatter(samp_traj[:, 0], samp_traj[:, 1], 
                            label='sampled data', s=3)
                plt.legend()
                plt.title(f"{DYNAMICS} Dynamics")
                plt.savefig(f'./{DYNAMICS}_vis_{str(traj_id)}.png', dpi=500)
                print('Saved visualization figure at {}'.format(f'./{DYNAMICS}_vis_{str(traj_id)}.png'))