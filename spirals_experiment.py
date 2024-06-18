import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import importlib.util
import sys
from thop import profile


# let READONLY = False to retrain models
READONLY = True


DATASET_VIS = False

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)  


def generate_spiral2d(nspiral=512,
                      ntotal=300,
                      nsample=110,
                      start=0.,
                      stop=6 * np.pi,  # approximately equal to 6pi
                    #   noise_std=2e-1,
                      noise_std=1.5e-1,
                      a=0.,
                      b=1.):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    initial_idx = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        orig_traj = orig_traj_cc 
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        # samp_traj_unnoised = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        # samp_trajs_unnoised.append(samp_traj_unnoised)
        initial_idx.append(t0_idx)
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    # samp_trajs_unnoised = np.stack(samp_trajs_unnoised, axis=0)

    return orig_trajs, samp_trajs, initial_idx, orig_ts, samp_ts



def euler_ode_solver(func, y0, t, g=None):
    dt = t[1] - t[0]
    y = y0
    ys = [y0]

    for i in range(len(t) - 1):
        u = torch.zeros_like(y)
        if g is not None:
            u = g(y)
        t_start, t_end = t[i], t[i+1]
        y = y + (func(t_start, y) + u) * dt
        t_start += dt
        ys.append(y)
    return torch.stack(ys) 



class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        y = self.net(y)
        return y


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()  
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, input_dim))     
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        y = self.net(y)
        return y


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = [nn.Conv1d(input_dim, hidden_dim, kernel_size=1), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
            layers.append(nn.Softplus())
        layers.append(nn.Conv1d(hidden_dim, input_dim, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, y):
        y = y.unsqueeze(-1)
        y = self.net(y)
        y = y.squeeze(-1)
        return y
    
class CSNeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, g_hidden_dim, g_num_layers):
        super(CSNeuralODE, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.g_func = CNN(input_dim, g_hidden_dim, g_num_layers)
        
    def forward(self, y0, t):
        out = euler_ode_solver(self.func, y0, t, self.g_func)
        out = out
        return out


class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)

    def forward(self, y0, t):
        out = euler_ode_solver(self.func, y0, t)
        out = out
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
orig_trajs, samp_trajs, initial_idx, orig_ts, samp_ts = generate_spiral2d()


if DATASET_VIS:
    orig_traj0 = orig_trajs[0]
    samp_traj0 = samp_trajs[0]

    plt.figure(figsize=(8, 8))
    plt.plot(orig_traj0[:, 0], orig_traj0[:, 1], 'b-', label='True Trajectory')
    plt.scatter(samp_traj0[:, 0], samp_traj0[:, 1], color='r', s=30, label='Noisy Observations')
    plt.ylabel('y') 
    plt.title('First Spiral')
    plt.legend()
    plt.axis('equal')
    plt.show()


input_dim = 2
hidden_dim = 64
num_layers = 2
cs_hidden_dim = 32
cs_num_layers = 2
g_hidden_dim = 52
g_num_layers = 2
neural_ode = NeuralODE(input_dim, hidden_dim, num_layers).to(device)
cs_neural_ode = CSNeuralODE(input_dim, cs_hidden_dim, cs_num_layers, g_hidden_dim, g_num_layers).to(device)
epochs = 2000
lr = 1e-3

neural_ode_path = current_directory + '/neural_ode_noised1.pth'
cs_neural_ode_path = current_directory + '/cs_neural_ode_noised1.pth'

def train_model(model):
    x0 = samp_trajs[:, 0, :]
    x0_tensor = torch.tensor(x0, dtype=torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    samp_ts_tensor = torch.tensor(samp_ts, dtype=torch.float32).to(device)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
        predicted_traj = model(x0_tensor, samp_ts_tensor).permute(1, 0, 2)
        target = torch.tensor(samp_trajs, dtype=torch.float32).to(device)
        loss = criterion(predicted_traj, target)
        if (epoch + 1) % 50 == 0:
            print(f"Loss at epoch {epoch + 1}: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_model(model, model_name):
    idx = npr.randint(orig_ts.shape[0]//3, orig_ts.shape[0]//2)
    ts = orig_ts[idx:]
    ts = ts - ts[0]
    ts_tensor = torch.tensor(ts, dtype=torch.float32).to(device)
    x0 =orig_trajs[:, idx, :]

    x0_tensor = torch.tensor(x0, dtype=torch.float32).to(device)
    predicted_traj = model(x0_tensor, ts_tensor).permute(1, 0, 2)
    predicted_traj = predicted_traj.cpu().detach().numpy()
    orig_traj0 = orig_trajs[0]
    gt_traj0 = orig_trajs[0][idx:]
    predicted_traj0 = predicted_traj[0]
    mse_loss = np.mean((gt_traj0 - predicted_traj0) ** 2)
    l1_loss = np.mean(np.abs(gt_traj0 - predicted_traj0))
    print(model_name + f": MSE Loss: {mse_loss}, L1 Loss: {l1_loss}")
    print(orig_traj0.shape, predicted_traj0.shape)

    plt.figure(figsize=(8, 8))
    plt.plot(orig_traj0[:, 0], orig_traj0[:, 1], 'b-', label='True Trajectory')
    plt.plot(predicted_traj0[:, 0], predicted_traj0[:, 1], 'r-', label='Predicted Trajectory')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.title('Spiral predicted by ' + model_name)
    plt.legend()
    plt.axis('equal')
    plt.show()



def eval_model_1(model, model_name):
    test_spiral_idx = 0
    idx = initial_idx[test_spiral_idx]
    ts = orig_ts[idx:]
    ts = ts - ts[0]
    ts_tensor = torch.tensor(ts, dtype=torch.float32).to(device)
    x0 =orig_trajs[:, idx, :]
    x0_tensor = torch.tensor(x0, dtype=torch.float32).to(device)
    predicted_traj = model(x0_tensor, ts_tensor).permute(1, 0, 2)
    predicted_traj = predicted_traj.cpu().detach().numpy()
    gt_traj0 = orig_trajs[test_spiral_idx][idx:]
    samp_traj0 = samp_trajs[test_spiral_idx]
    predicted_traj0 = predicted_traj[test_spiral_idx]
    orig_traj0 = orig_trajs[test_spiral_idx]
    mse_loss = np.mean((gt_traj0 - predicted_traj0) ** 2)
    l1_loss = np.mean(np.abs(gt_traj0 - predicted_traj0))
    print("\n" + model_name + f": MSE Loss: {mse_loss}")
    print(f"L1 Loss: {l1_loss}")

    plt.figure(figsize=(8, 8))
    plt.plot(orig_traj0[:, 0], orig_traj0[:, 1], 'g-', label='True Trajectory')
    plt.plot(gt_traj0[:, 0], gt_traj0[:, 1], 'b-', label='True Trajectory')
    plt.plot(predicted_traj0[:, 0], predicted_traj0[:, 1], 'r-', label='Predicted Trajectory')
    plt.scatter(samp_traj0[:, 0], samp_traj0[:, 1], color='orange', s=30, label='Noisy Observations')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.title('Spiral predicted by ' + model_name)
    plt.legend()
    plt.axis('equal')
    plt.show()



def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model, model_name):
    total_params = count_parameters(model)
    print(model_name + f" - total number of trainable parameters: {total_params}")

# def print_model_parameters(model, model_name):
#     x0 = torch.randn(1, 1, input_dim).to(device)
#     ts = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device) 
#     flops, params = profile(model, inputs=(x0, ts))
#     print(model_name + f" - FLOPs: {flops}, Parameters: {params}")


if not READONLY:
    train_model(neural_ode)
    save_model(neural_ode, neural_ode_path)
    train_model(cs_neural_ode)
    save_model(cs_neural_ode, cs_neural_ode_path)

neural_ode = load_model(neural_ode, neural_ode_path)
cs_neural_ode = load_model(cs_neural_ode, cs_neural_ode_path)

print()
print_model_parameters(neural_ode, "Neural ODE")
print_model_parameters(cs_neural_ode, "ControlSynth Neural ODE")

eval_model_1(neural_ode, "Neural ODE")
eval_model_1(cs_neural_ode, "ControlSynth Neural ODE")
