import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from thop import profile

from torch.utils.data import Dataset, DataLoader
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from skimage.metrics import structural_similarity as ssim

step_size = 0.1
num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 1024
num_layers = 2
num_steps_per_interval = 1
lr = 5e-4
augment_dim = 10

USE_BASIC_SOLVER = False
# if USE_BASIC_SOLVER = False, the code uses ode solver in torchdiffeq


def normalize_array(arr):
    min_vals = arr.min(axis=0)
    range_vals = arr.max(axis=0) - min_vals
    range_vals[range_vals == 0] = 1 
    norm_arr = (arr - min_vals) / range_vals
    return norm_arr, min_vals, range_vals

def unnormalize_array(norm_arr, min_vals, range_vals):
    original_arr = norm_arr * range_vals + min_vals
    return original_arr


def hindmarsh_rose(state, t, a, b, c, d, r, s, x0, I):
    x, y, z = state
    dx = y - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y
    dz = r * (s * (x - x0) - z)
    return [dx, dy, dz]


a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.5
s = 1
x0 = -0.5
I = 3.0



init_state = [2, -8, 0.5]
# T = 15
T = 15
train_len = 50
test_len = train_len
t = np.linspace(0, T, train_len + test_len)



states = odeint(hindmarsh_rose, init_state, t, args=(a, b, c, d, r, s, x0, I))

x = states[:, 0]
y = states[:, 1]
z = states[:, 2]

data = np.stack([x, y, z], axis=1)


data, min_vals, range_vals = normalize_array(data)

data = torch.tensor(data, dtype=torch.float)
split_index = train_len
train_data = data[:split_index] 
test_data = data[split_index:]  
# test_data = test_data[:55]

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
                       mode='markers', name='Predicted',
                       marker=dict(size=3, color='rgb(255, 178, 0)', opacity=0.8)))
fig.update_layout(title='3D Scatter Plot',
              scene=dict(xaxis_title='X-axis',
                         yaxis_title='Y-axis',
                         zaxis_title='Z-axis'))



np_test_true_y = test_data.numpy()
np_test_true_y = unnormalize_array(np_test_true_y, min_vals, range_vals)

true_y = train_data.to(device)
test_y0 = test_data[0].clone().detach().to(device)
test_true_y = test_data.to(device)
t = torch.linspace(0, 1, train_data.shape[0]).to(device)
y0 = train_data[0].clone().detach().to(device)







import os
import importlib.util
import sys

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


solver_path = current_directory + '/solver.py'
spec = importlib.util.spec_from_file_location("solver", solver_path)
solver_module = importlib.util.module_from_spec(spec)
sys.modules["solver"] = solver_module
spec.loader.exec_module(solver_module)





    
m = 1
seq_len = train_len
batch_size = 1


class SlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len, step=m):

        self.data = data
        self.seq_len = seq_len
        self.step = step
        self.windows = [self.data[i:i + seq_len] for i in range(0, len(data) - seq_len + 1, step)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        input = window[0] 
        target = window   
        return input, target

train_dataset = SlidingWindowDataset(train_data, seq_len)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)






input_dim = data.shape[1] + 1



from torchdiffeq import odeint, odeint_adjoint


def basic_euler_ode_solver(func, y0, t, gu_update_func=None, gu0=None):
    dt = t[1] - t[0]
    y = y0
    ys = [y0]
    if gu0 is None:
        gu = torch.zeros_like(y0).to(device)
    else:
        gu = gu0
    for i in range(len(t) - 1):
        t_start, t_end = t[i], t[i+1]
        if gu_update_func is not None: 
            gu = gu_update_func(t_start, gu)
        else:
            gu = torch.zeros_like(y0).to(device)

        y = y + (func(t_start, y) + gu) * dt
        t_start += dt
        ys.append(y)
    return torch.stack(ys) 



class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers, input_dim=input_dim):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
            # layers.append(nn.ELU())
            # layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y =  self.net(t_and_y)[:, :self.input_dim-1]
        return y

class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc()
        
    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t, method='euler')
        out = out.view(-1, t.shape[0], input_dim-1)
        return out


class CSNeuralODE(nn.Module):
    def __init__(self):
        super(CSNeuralODE, self).__init__()
        # self.func = ODEFunc(hidden_dim=int(hidden_dim * 0.57))
        # self.gfunc = ODEFunc(hidden_dim=int(hidden_dim * 0.57))

        # self.func = ODEFunc(hidden_dim=int(hidden_dim * 0.63))
        # self.gfunc = simple_fc_layer()

        self.func = ODEFunc(hidden_dim=int(hidden_dim))
        self.gfunc = ODEFunc(hidden_dim=int(hidden_dim * 0.05))
    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t, self.gfunc, gu0=y0)
        else:
            solver = solver_module.DynamicODESolver(self.func, y0, ufunc=self.gfunc, u0=y0)
            out = solver.integrate(t)
        out = out.view(-1, t.shape[0], input_dim-1)
        return out



class CSNeuralODE_CNN(nn.Module):
    def __init__(self):
        super(CSNeuralODE_CNN, self).__init__()
        self.func = ODEFunc(hidden_dim=int(hidden_dim))
        self.gfunc = TwoConvLayer()
    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t, self.gfunc, gu0=y0)
        else:
            solver = solver_module.DynamicODESolver(self.func, y0, ufunc=self.gfunc, u0=y0)
            out = solver.integrate(t)
        out = out.view(-1, t.shape[0], input_dim-1)
        return out


class simple_fc_layer(nn.Module):
    def __init__(self, dim=input_dim):
        super(simple_fc_layer, self).__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, t, y):

        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y =  self.linear(t_and_y)[:, :input_dim-1]
        return y
    
    

class TwoConvLayer(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(TwoConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size, stride, padding)
        # self.tanh = nn.Tanh()
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size, stride, padding)

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y = self.conv1(t_and_y)
        # x = self.tanh(x) 
        y = self.conv2(y)
        y = y.view(y.size(0), -1)
        y = y[:, :input_dim-1]
        return y



class AugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers):
        super(AugmentedODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim + augment_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim + augment_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y = self.net(t_and_y)[:, :input_dim-1]

        return y


class AugmentedNeuralODE(nn.Module):
    def __init__(self, augment_dim=augment_dim, use_second_order=False):
        super(AugmentedNeuralODE, self).__init__()
        self.use_second_order = use_second_order
        
        if use_second_order:
            self.func = SecondOrderAugmentedODEFunc(hidden_dim=int(hidden_dim))
        else:
            self.func = ODEFunc(hidden_dim=int(hidden_dim), input_dim=augment_dim+input_dim)
        
        self.augment_dim = augment_dim

    def forward(self, y0, t):
        y_aug = torch.cat([y0, torch.zeros(y0.shape[0], self.augment_dim).to(y0)], dim=1)
        
        if self.use_second_order:
            v_aug = torch.zeros_like(y_aug)
            z0 = torch.cat((y_aug, v_aug), dim=1)
        else:
            z0 = y_aug
        
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, z0, t)
        else:
            out = odeint(self.func, z0, t, method='euler')
        
        if self.use_second_order:
            out = out[:, :, :input_dim-1]
        
        out = out.view(-1, t.shape[0], input_dim-1)
        return out


class SecondOrderAugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers):
        super(SecondOrderAugmentedODEFunc, self).__init__()
        
        self.fc1 = nn.Linear(2*(input_dim + augment_dim) + 1, hidden_dim)  # +1 for time input
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim + augment_dim)
        self.tanh = nn.Tanh()

    def forward(self, t, z):
        cutoff = int(len(z)/2)
        y = z[:cutoff]
        v = z[cutoff:]
        t_vec = torch.ones(y.shape[0], 1).to(y) * t
        into = torch.cat((y, v, t_vec), dim=1)  # concatenate t_vec with y and v
        out = self.fc1(into)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return torch.cat((v, out[:, :input_dim-1]))




class MLP(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers, input_dim=input_dim-1):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
    def forward(self, y0, t):
        ys = []
        y = y0
        ys.append(y)
        for _ in range(t.shape[0] - 1):
            y = self.net(y)
            ys.append(y)
        out = torch.stack(ys)
        return out.view(-1, t.shape[0], self.input_dim)
    





def train(model, model_name):
    optimizer = optim.Adam(model.parameters(), lr=lr)  
    for epoch in range(num_epochs):
        model.train()  
        inputs, targets = y0,  true_y
        inputs = inputs.reshape(-1, input_dim-1)
        targets = targets.reshape(-1, input_dim-1)
        t = torch.linspace(0, 1, train_len).to(device)
        pred = model(inputs, t)
        loss = criterion(pred[0], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(model_name + f": Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            model.eval()
            with torch.no_grad():
                inputs, targets = test_y0,  test_true_y
                inputs = inputs.reshape(-1, input_dim-1)
                t = torch.linspace(0, 1, train_len).to(device)
                pred = model(inputs, t)
                Z = pred.cpu().detach().numpy().reshape(-1, t.shape[0], input_dim-1)
                Z = unnormalize_array(Z, min_vals, range_vals)
                # print(Z.shape, test_true_y.shape)
                mse = mean_squared_error(Z[0], np_test_true_y)
                print(model_name + " Test MSE: ", mse, "\n")








criterion = nn.MSELoss()
ode = NeuralODE().to(device)
csode = CSNeuralODE().to(device)
mlp = MLP().to(device)
augode = AugmentedNeuralODE().to(device)
csodecnn = CSNeuralODE_CNN().to(device)

print("\nTrain CSODE:")
train(csode, "ControlSynth_Neural_ODE")
torch.save(csode, "hr_csode.pth")
csode = torch.load("hr_csode.pth").to(device)

print("\nTrain CSODE CNN:")
train(csodecnn, "ControlSynth_Neural_ODE_CNN")
torch.save(csodecnn, "hr_csode_cnn.pth")
csodecnn = torch.load("hr_csode_cnn.pth").to(device)


print("\nTrain AUGODE:")
train(augode, "Augmented_Neural_ODE")
torch.save(augode, "hr_augode.pth")
augode = torch.load("hr_augode.pth").to(device)

print("\nTrain ODE:")
train(ode, "Neural_ODE")
torch.save(ode, "hr_ode.pth")
ode = torch.load("hr_ode.pth").to(device)

print("\nTrain MLP:")
train(mlp, "MLPs")
torch.save(mlp, "hr_mlp.pth")
mlp = torch.load("hr_mlp.pth").to(device)



with torch.no_grad():
    test_t = torch.linspace(0, 1 * test_len / train_len, test_data.shape[0]).to(device)
    test_y0 = test_data[0].clone().detach().to(device)
    test_y0 = test_y0.reshape(-1, input_dim-1)

    test_pred_y_ode = ode(test_y0, test_t)
    test_pred_y_csode = csode(test_y0, test_t)
    test_pred_y_augode = augode(test_y0, test_t)
    test_pred_y_csodecnn = csodecnn(test_y0, test_t)
    test_pred_y_mlp = mlp(test_y0, test_t)
    
    flops, params = profile(mlp, inputs=(test_y0, test_t), verbose=False)
    print("mlp flops and params:", flops, params)
    flops, params = profile(ode, inputs=(test_y0, test_t), verbose=False)
    print("ode flops and params:", flops, params)
    flops, params = profile(augode, inputs=(test_y0, test_t), verbose=False)
    print("augode flops and params:", flops, params)
    flops, params = profile(csode, inputs=(test_y0, test_t), verbose=False)
    print("csode flops and params:", flops, params)
    flops, params = profile(csodecnn, inputs=(test_y0, test_t), verbose=False)
    print("csodecnn flops and params:", flops, params)
    
    print()
    
    test_pred_y_ode = test_pred_y_ode.cpu().numpy()[0] 
    test_pred_y_csode = test_pred_y_csode.cpu().numpy()[0]
    test_pred_y_augode = test_pred_y_augode.cpu().numpy()[0] 
    test_pred_y_csodecnn = test_pred_y_csodecnn.cpu().numpy()[0]
    test_pred_y_mlp = test_pred_y_mlp.cpu().numpy()[0]
    
    test_pred_y_ode = unnormalize_array(test_pred_y_ode, min_vals, range_vals)
    test_pred_y_csode = unnormalize_array(test_pred_y_csode, min_vals, range_vals)
    test_pred_y_csodecnn = unnormalize_array(test_pred_y_csodecnn, min_vals, range_vals)
    test_pred_y_augode = unnormalize_array(test_pred_y_augode, min_vals, range_vals)
    test_pred_y_mlp = unnormalize_array(test_pred_y_mlp, min_vals, range_vals)


    
#     test_data = test_data.cpu().numpy()

    mae_mlp = np.mean(np.abs(test_pred_y_mlp - np_test_true_y))
    mse_mlp = np.mean((test_pred_y_mlp - np_test_true_y) ** 2)
    print("mlp mse:", mse_mlp, "; mlp mae:", mae_mlp)

    mae_ode = np.mean(np.abs(test_pred_y_ode - np_test_true_y))
    mse_ode = np.mean((test_pred_y_ode - np_test_true_y) ** 2)
    print("ode mse:", mse_ode, "; ode mae:", mae_ode)

    mae_csode = np.mean(np.abs(test_pred_y_csode - np_test_true_y))
    mse_csode = np.mean((test_pred_y_csode - np_test_true_y) ** 2)
    print("csode mse:", mse_csode, "; csode mae:", mae_csode)
    
    mae_augode = np.mean(np.abs(test_pred_y_augode - np_test_true_y))
    mse_augode = np.mean((test_pred_y_augode - np_test_true_y) ** 2)
    print("augode mse:", mse_augode, "; augode mae:", mae_augode)

    mae_csodecnn = np.mean(np.abs(test_pred_y_csodecnn - np_test_true_y))
    mse_csodecnn = np.mean((test_pred_y_csodecnn - np_test_true_y) ** 2)
    print("csodecnn mse:", mse_csodecnn, "; csodecnn mae:", mae_csodecnn)
    print()



    dtw_distance_mlp, _ = fastdtw(test_pred_y_mlp, np_test_true_y, dist=euclidean)
    print("mlp dtw:", dtw_distance_mlp)

    dtw_distance_ode, _ = fastdtw(test_pred_y_ode, np_test_true_y, dist=euclidean)
    print("ode dtw:", dtw_distance_ode)

    dtw_distance_augode, _ = fastdtw(test_pred_y_augode, np_test_true_y, dist=euclidean)
    print("augode dtw:", dtw_distance_augode)

    dtw_distance_csode, _ = fastdtw(test_pred_y_csode, np_test_true_y, dist=euclidean)
    print("csode dtw:", dtw_distance_csode)

    dtw_distance_csodecnn, _ = fastdtw(test_pred_y_csodecnn, np_test_true_y, dist=euclidean)
    print("csodecnn dtw:", dtw_distance_csodecnn)
    print()



    
    r2_mlp = r2_score(np_test_true_y, test_pred_y_mlp)
    print("mlp r2:", r2_mlp)

    r2_ode = r2_score(np_test_true_y, test_pred_y_ode)
    print("ode r2:", r2_ode)

    # Calculate R² score for the CSODE model
    r2_csode = r2_score(np_test_true_y, test_pred_y_csode)
    print("csode r2:", r2_csode)

    # Calculate R² score for the AUGODE model
    r2_augode = r2_score(np_test_true_y, test_pred_y_augode)
    print("augode r2:", r2_augode)

    # Calculate R² score for the CSODECNN model
    r2_csodecnn = r2_score(np_test_true_y, test_pred_y_csodecnn)
    print("csodecnn r2:", r2_csodecnn)


    marker_size = 7



    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=test_pred_y_ode[:, 0], y=test_pred_y_ode[:, 1], z=test_pred_y_ode[:, 2],
                            mode='lines+markers', name='Predicted by Neural ODE',
                            marker=dict(size=marker_size, color='rgb(255, 178, 0)', opacity=0.8)))

    fig.add_trace(go.Scatter3d(x=test_pred_y_augode[:, 0], y=test_pred_y_augode[:, 1], z=test_pred_y_augode[:, 2],
                            mode='lines+markers', name='Predicted by Augmented Neural ODE',
                            marker=dict(size=marker_size, color='rgb(95, 166, 250)', opacity=0.8)))

    fig.add_trace(go.Scatter3d(x=test_pred_y_csode[:, 0], y=test_pred_y_csode[:, 1], z=test_pred_y_csode[:, 2],
                            mode='lines+markers', name='Predicted by ControlSynth Neural ODE',
                            marker=dict(size=marker_size, color='rgb(31, 208, 178)', opacity=0.8)))
    
        
    fig.add_trace(go.Scatter3d(x=test_pred_y_csodecnn[:, 0], y=test_pred_y_csodecnn[:, 1], z=test_pred_y_csodecnn[:, 2],
                            mode='lines+markers', name='Predicted by ControlSynth Neural ODE Adapt',
                            marker=dict(size=marker_size, color='rgb(208, 73, 178)', opacity=0.8)))

    fig.add_trace(go.Scatter3d(x=np_test_true_y[:, 0], y=np_test_true_y[:, 1], z=np_test_true_y[:, 2],
                            mode='lines+markers', name='Ground Truth',
                            marker=dict(size=marker_size, color='rgb(255, 0, 0)', opacity=0.8)))


    fig.update_layout(title='3D Scatter Plot',
                    scene=dict(
                        xaxis_title='X-axis',
                        yaxis_title='Y-axis',
                        zaxis_title='Z-axis',
                        xaxis=dict(tickfont=dict(size=13)),
                        yaxis=dict(tickfont=dict(size=13)),
                        zaxis=dict(tickfont=dict(size=13))
                    ),
                    font=dict(
                        family="Times New Roman, Times, serif",
                        size=18.5,
                        color="black"
                    )           
                    )
    fig.show()

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_pred_y_ode[:, 0], test_pred_y_ode[:, 1], test_pred_y_ode[:, 2],
               c='orange', marker='o', label='Predicted by Neural ODE')
    ax.scatter(test_pred_y_augode[:, 0], test_pred_y_augode[:, 1], test_pred_y_augode[:, 2],
               c='green', marker='o', label='Predicted by Augmented Neural ODE')
    ax.scatter(test_pred_y_csode[:, 0], test_pred_y_csode[:, 1], test_pred_y_csode[:, 2],
               c='blue', marker='o', label='Predicted by CSODE')
    ax.scatter(test_true_y[:, 0], test_true_y[:, 1], test_true_y[:, 2],
               c='red', marker='o', label='Ground Truth')

    ax.legend()
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()
