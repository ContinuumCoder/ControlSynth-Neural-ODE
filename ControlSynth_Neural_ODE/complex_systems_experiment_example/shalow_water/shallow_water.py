import numpy as np
from scipy.integrate import odeint as scipy_odeint
import plotly.graph_objects as go
from scipy.interpolate import interp2d
import torch
import torch.nn as nn
import torch.optim as optim
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from thop import profile
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from skimage.metrics import structural_similarity as ssim
from torchdiffeq import odeint, odeint_adjoint
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


USE_BASIC_SOLVER = False
# if USE_BASIC_SOLVER = False, the code uses ode solver in torchdiffeq

train_mode = True
initial_mode = True


def normalize_array(arr):
    min_vals = arr.min(axis=0)
    range_vals = arr.max(axis=0) - min_vals
    range_vals[range_vals == 0] = 1  # avoid deviding by 0
    norm_arr = (arr - min_vals) / range_vals
    return norm_arr, min_vals, range_vals

def unnormalize_array(norm_arr, min_vals, range_vals):
    original_arr = norm_arr * range_vals + min_vals
    return original_arr


n_vars = 1 
global_min_vals = None
global_range_vals = None
N_fine = 50
# n_seq = 1000
# batch_size = 128
n_seq = 1
batch_size = 1
n_test = 0


train_len = 25
test_len = int(train_len * 0.5)
num_time_points = test_len + train_len

# lr = 3e-4
lr = 1e-3
input_dim = N_fine * N_fine * n_vars + 1
hidden_dim = 2048
num_layers = 2
num_epochs = 1000
whole_seq = None
augment_dim = 10


g = 5  
H = 1.0  
L = 100  
N = 64   
T = 70 

L = 15
N = 50
dx = L / (N - 1)  



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




class shallow_water_dataset(Dataset):
    def __init__(self, n_seq, L=L, T=T, N_fine=N_fine, num_time_points=num_time_points):
        self.inputs = []
        self.targets = []
        self.h = L / (N - 1) 

        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X, Y = np.meshgrid(x, y)

        h0 = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (0.1*L)**2)
        vx0 = np.zeros((N, N))
        vy0 = np.zeros((N, N))

        u0 = np.zeros((N, N, 3))
        u0[:, :, 0] = h0
        u0[:, :, 1] = vx0
        u0[:, :, 2] = vy0
        u0 = u0.flatten()

        t = np.linspace(0, T, int(num_time_points * 10))

        sol = scipy_odeint(self.shallow_water_2d, u0, t, args=(g, H, N))


        sol = sol.reshape(-1, N, N, 3)[-num_time_points:]
        sol_fine = np.zeros((num_time_points, N_fine, N_fine, 3))
        x_fine = np.linspace(0, L, N_fine)
        y_fine = np.linspace(0, L, N_fine)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)

        global whole_seq

        for i in range(num_time_points):
            interp_h = interp2d(x, y, sol[i, :, :, 0], kind='cubic')
            sol_fine[i, :, :, 0] = interp_h(x_fine, y_fine)
        # we take h now

        fig = go.Figure(data=[go.Surface(x=np.linspace(0, L, N_fine), y=np.linspace(0, L, N_fine), z=sol_fine[-1][:, :, 0])])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='U'),
            title="Shallow water (Target)",
            width=800,
            height=600
        )

        fig.show() 

        fig = go.Figure(data=[go.Surface(x=np.linspace(0, L, N_fine), y=np.linspace(0, L, N_fine), z=sol_fine[test_len][:, :, 0])])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='U'),
            title="Shallow water (Input)",
            width=800,
            height=600
        )

        fig.show() 

        sol_fine = sol_fine[:, :, :, [0]]


        normalized_sol_fine = np.zeros_like(sol_fine)
        for i in range(sol_fine.shape[0]):
            min_val = np.min(sol_fine[i]) 
            max_val = np.max(sol_fine[i])  
            normalized_sol_fine[i] = (sol_fine[i] - min_val) / (max_val - min_val)


        # be careful
        sol_fine = normalized_sol_fine
        
        whole_seq = sol_fine
        np.save("sol.npy", sol_fine)
        



    def shallow_water_2d(self, u, t, g, H, N):
        u = u.reshape((N, N, 3))
        h = u[:, :, 0]
        vx = u[:, :, 1]
        vy = u[:, :, 2]

        h = np.pad(h, ((1, 1), (1, 1)), mode='wrap')
        vx = np.pad(vx, ((1, 1), (1, 1)), mode='wrap')
        vy = np.pad(vy, ((1, 1), (1, 1)), mode='wrap')

        dhdt = -H * (np.gradient(vx[1:-1, 1:-1], axis=0) + np.gradient(vy[1:-1, 1:-1], axis=1))
        dvxdt = -g * np.gradient(h[1:-1, 1:-1], axis=0)
        dvydt = -g * np.gradient(h[1:-1, 1:-1], axis=1)

        dudt = np.zeros((N, N, 3))
        dudt[:, :, 0] = dhdt
        dudt[:, :, 1] = dvxdt
        dudt[:, :, 2] = dvydt

        return dudt.flatten()



    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.inputs[idx]
        target = self.targets[idx]
        return input, target



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
        # self.func = ODEFunc(hidden_dim=int(hidden_dim * 0.6))
        # self.gfunc = ODEFunc(hidden_dim=int(hidden_dim * 0.6))

        self.func = ODEFunc(hidden_dim=int(hidden_dim * 0.23))
        self.gfunc = TwoConvLayer()
    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t, self.gfunc, gu0=y0)
        else:
            solver = solver_module.DynamicODESolver(self.func, y0, ufunc=self.gfunc, u0=y0)
            out = solver.integrate(t)
        out = out.view(-1, t.shape[0], input_dim-1)
        return out

class TwoConvLayer(nn.Module):
    def __init__(self, in_channels=1, mid_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(TwoConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding)


    def forward(self, t, x):
        BS, dim = x.shape
        x = x.view(BS, 1, N_fine, N_fine)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(BS, input_dim-1)
        return x


    

class simple_fc_layer(nn.Module):
    def __init__(self, dim=input_dim):
        super(simple_fc_layer, self).__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, t, y):

        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y =  self.linear(t_and_y)[:, :input_dim-1]
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


class SecondOrderAugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers):
        super(SecondOrderAugmentedODEFunc, self).__init__()
        
        layers = [nn.Linear(2*(input_dim + augment_dim), hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim + augment_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        cutoff = int(len(z)/2)
        y = z[:cutoff]
        v = z[cutoff:]
        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y_and_v = torch.cat([t_vec, y, v], 1)
        out = self.net(t_and_y_and_v)
        return torch.cat((v, out[:, :input_dim-1]))

    

class AugmentedNeuralODE(nn.Module):
    def __init__(self, augment_dim=augment_dim, use_second_order=False):
        super(AugmentedNeuralODE, self).__init__()
        self.use_second_order = use_second_order
        
        if use_second_order:
            self.func = SecondOrderAugmentedODEFunc(hidden_dim=int(hidden_dim))
        else:
            self.func = AugmentedODEFunc(hidden_dim=int(hidden_dim), input_dim=augment_dim+input_dim)
        
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
    # optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.train()  
        inputs, targets = whole_seq[0].to(device),  whole_seq[:train_len].to(device)
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
                inputs, targets = whole_seq[train_len].to(device),  whole_seq[-1].to(device)
                inputs = inputs.reshape(-1, input_dim-1)
                targets = targets.reshape(-1, input_dim-1)
                t = torch.linspace(0, 1, train_len).to(device)
                pred = model(inputs, t)
                Z = pred.cpu().detach().numpy().reshape(-1, t.shape[0], N_fine, N_fine, n_vars)
                Z = unnormalize_array(Z, global_min_vals, global_range_vals)[0]
                mse = mean_squared_error(Z[test_len-1][:, :, 0], sol_fine[-1][:, :, 0])
                print(model_name + " Test MSE: ", mse, "\n")

def eval_model(model, model_name):

    inputs, targets = whole_seq[train_len].to(device),  whole_seq[-1].to(device)
    inputs = inputs.reshape(-1, input_dim-1)
    targets = targets.reshape(-1, input_dim-1)
    # t = torch.linspace(0, 1 * (test_len / train_len), test_len).to(device)
    t = torch.linspace(0, 1/2, test_len).to(device)
    pred = model(inputs, t)
    Z = pred.cpu().detach().numpy().reshape(-1, t.shape[0], N_fine, N_fine, n_vars)
    Z = unnormalize_array(Z, global_min_vals, global_range_vals)[0]
    flops, params = profile(model, inputs=(inputs, t), verbose=False)
    print(model_name + " flops and params:", flops, params)
    return Z[test_len-1]

if initial_mode:
    train_mode = True
    dataset = shallow_water_dataset(n_seq=n_seq)
    ode = NeuralODE().to(device)
    csode = CSNeuralODE().to(device)
    augode = AugmentedNeuralODE().to(device)
    mlp = MLP().to(device)
    torch.save(mlp, current_directory+"/mlp.pth")
    torch.save(ode, current_directory+"/ode.pth")
    torch.save(augode, current_directory+"/augode.pth")
    torch.save(csode, current_directory+"/csode.pth")



sol_fine = np.load("sol.npy")
print(sol_fine.shape)
whole_seq, global_min_vals, global_range_vals = normalize_array(sol_fine)
whole_seq = torch.tensor(whole_seq, dtype=torch.float32)   


class CombinedLoss(nn.Module):
    def __init__(self, weight_mse=0.5, weight_l1=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.weight_mse = weight_mse
        self.weight_l1 = weight_l1

    def forward(self, output, target):
        loss_mse = self.mse_loss(output, target)
        loss_l1 = self.l1_loss(output, target)
        combined_loss = self.weight_mse * loss_mse + self.weight_l1 * loss_l1
        return combined_loss


criterion = CombinedLoss(weight_mse=0, weight_l1=1).to(device)

if train_mode:
    csode = torch.load(current_directory+"/csode.pth").to(device)
    train(csode, "csode")
    torch.save(csode, current_directory+"/csode.pth")

csode = torch.load(current_directory+"/csode.pth").to(device)
Z_csode = eval_model(csode, "csode")

print()

if train_mode:
    ode = torch.load(current_directory+"/ode.pth").to(device)
    train(ode, "ode")
    torch.save(ode, current_directory+"/ode.pth")

ode = torch.load(current_directory+"/ode.pth").to(device)
Z_ode = eval_model(ode, "ode")

print()




if train_mode:
    augode = torch.load(current_directory+"/augode.pth").to(device)
    train(augode, "augode")
    torch.save(augode, current_directory+"/augode.pth")

augode = torch.load(current_directory+"/augode.pth").to(device)
Z_augode = eval_model(augode, "augode")



if train_mode:
    mlp = torch.load(current_directory+"/mlp.pth").to(device)
    train(mlp, "mlp")
    torch.save(mlp, current_directory+"/mlp.pth")

mlp = torch.load(current_directory+"/mlp.pth").to(device)
Z_mlp = eval_model(mlp, "mlp")


def chamfer_distance(set1, set2):
    # Compute all pairwise distances between set1 and set2
    dist1 = np.sqrt(((set1[:, np.newaxis, :] - set2[np.newaxis, :, :]) ** 2).sum(axis=2))
    dist2 = np.sqrt(((set2[:, np.newaxis, :] - set1[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # For each point in set1, find the closest point in set2 and vice versa
    nearest_dist1 = np.min(dist1, axis=1)
    nearest_dist2 = np.min(dist2, axis=0)
    
    # Return the average of the nearest distances
    return np.mean(nearest_dist1) + np.mean(nearest_dist2)


X = np.linspace(0, L, N_fine)
Y = np.linspace(0, L, N_fine)
Z1 = Z_ode[:, :, 0] 
Z2 = Z_augode[:, :, 0]  
Z3 = Z_csode[:, :, 0] 
Z4 = sol_fine[-1][:, :, 0] 
Z5 = Z_mlp[:, :, 0] 
Z_initial = sol_fine[-test_len][:, :, 0] 


Z1 = gaussian_filter(Z1, sigma=1) 
Z2 = gaussian_filter(Z2, sigma=1)
Z3 = gaussian_filter(Z3, sigma=1)

all_data = np.concatenate([Z1.flatten(), Z3.flatten(), Z4.flatten(), Z_initial.flatten()])
mean_val = np.mean(all_data)
std_val = np.std(all_data)


vmin = mean_val - std_val * 1.75
vmax = mean_val + std_val * 1.75


fig = make_subplots(rows=2, cols=3,
                    column_widths=[0.33, 0.33, 0.33],
                    specs=[[{'type': 'surface', 'colspan': 2}, {'type': 'surface', 'colspan': 2}, None],
                           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
                    subplot_titles=('Initial Condition', 'Prediction Ground Truth',
                                    'Neural ODE', 'Augmented Neural ODE', 'ControlSynth Neural ODE'))

fig.add_trace(go.Surface(x=X, y=Y, z=Z_initial, colorscale='RdYlBu', cmin=vmin, cmax=vmax), row=1, col=1)
fig.add_trace(go.Surface(x=X, y=Y, z=Z4, colorscale='RdYlBu', cmin=vmin, cmax=vmax), row=1, col=2)
fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale='RdYlBu', cmin=vmin, cmax=vmax), row=2, col=1)
fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale='RdYlBu', cmin=vmin, cmax=vmax), row=2, col=2)
fig.add_trace(go.Surface(x=X, y=Y, z=Z3, colorscale='RdYlBu', cmin=vmin, cmax=vmax), row=2, col=3)





fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  scene3=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  scene4=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  scene5=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  scene6=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                  )

dist = 1.6
x_eye = 0.8 * dist
y_eye = 0.4 * dist
z_eye = 0.85 * dist
fig.update_scenes(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)), row=1, col=1)
fig.update_scenes(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)), row=1, col=2)
fig.update_scenes(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)), row=2, col=1)
fig.update_scenes(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)), row=2, col=2)
fig.update_scenes(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)), row=2, col=3)
fig.update_scenes(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye)), row=1, col=3)



fig.update_layout(    title={
                        'text': "2D Shallow Water Equations, Water Depth h (Meters)",
                        # 'text': r'$\text{2D Shallow Water Equations, Water Depth } h \text{ (Meters)}$',
                        'y':0.97,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    
                    font=dict(
                        family="Times New Roman, Times, serif", 
                        size=17.5,
                        color="black"
                    )          
                )
axes_dict = dict(showbackground=True, showticklabels=False, title='')
fig.update_scenes(xaxis=axes_dict, yaxis=axes_dict, zaxis=axes_dict)


axes_dict_visible = dict(showbackground=True, showticklabels=False, title='')
axes_dict_hidden = dict(showbackground=False, showticklabels=False, title='')


fig.update_scenes(xaxis=axes_dict_visible, yaxis=axes_dict_visible, zaxis=axes_dict_visible)

fig.update_scenes(xaxis=axes_dict_hidden, yaxis=axes_dict_hidden, zaxis=axes_dict_hidden, row=1, col=3)
fig.update_scenes(xaxis=dict(showbackground=False, showticklabels=False, title=''),
                  yaxis=dict(showbackground=False, showticklabels=False, title=''),
                  zaxis=dict(showbackground=False, showticklabels=False, title=''),
                  row=1, col=3)

for i in fig['layout']['annotations']:
    i['font'] = dict(size=25, family="Times New Roman, Times, serif")


fig.show()


mse_Z1_Z4 = mean_squared_error(Z1, Z4)
mse_Z2_Z4 = mean_squared_error(Z2, Z4)
mse_Z3_Z4 = mean_squared_error(Z3, Z4)
mse_Z5_Z4 = mean_squared_error(Z5, Z4)


mae_Z1_Z4 = mean_absolute_error(Z1, Z4)
mae_Z2_Z4 = mean_absolute_error(Z2, Z4)
mae_Z3_Z4 = mean_absolute_error(Z3, Z4)
mae_Z5_Z4 = mean_absolute_error(Z5, Z4)


max_err_Z1_Z4 = max_error(Z1.ravel(), Z4.ravel())
max_err_Z2_Z4 = max_error(Z2.ravel(), Z4.ravel())
max_err_Z3_Z4 = max_error(Z3.ravel(), Z4.ravel())
max_err_Z5_Z4 = max_error(Z5.ravel(), Z4.ravel())

ssim_Z1_Z4 = ssim(Z1, Z4, data_range=Z4.max() - Z4.min())
ssim_Z2_Z4 = ssim(Z2, Z4, data_range=Z4.max() - Z4.min())
ssim_Z3_Z4 = ssim(Z3, Z4, data_range=Z4.max() - Z4.min())
ssim_Z5_Z4 = ssim(Z5, Z4, data_range=Z4.max() - Z4.min())

correlation_coefficient_Z1_Z4 = np.corrcoef(Z1.flatten(), Z4.flatten())[0, 1]
correlation_coefficient_Z2_Z4 = np.corrcoef(Z2.flatten(), Z4.flatten())[0, 1]
correlation_coefficient_Z3_Z4 = np.corrcoef(Z3.flatten(), Z4.flatten())[0, 1]
correlation_coefficient_Z5_Z4 = np.corrcoef(Z5.flatten(), Z4.flatten())[0, 1]






# Calculate Chamfer Distance
chamfer_Z1_Z4 = chamfer_distance(Z1, Z4)
chamfer_Z2_Z4 = chamfer_distance(Z2, Z4)
chamfer_Z3_Z4 = chamfer_distance(Z3, Z4)
chamfer_Z5_Z4 = chamfer_distance(Z5, Z4)



print("MSE between Neural ODE Prediction and GT:", mse_Z1_Z4)
print("MSE between Augmented Neural ODE Prediction and GT:", mse_Z2_Z4)
print("MSE between ControlSynth Neural ODE Prediction and GT:", mse_Z3_Z4)
print("MSE between MLP Prediction and GT:", mse_Z5_Z4)
print()
print("MAE between Neural ODE Prediction and GT:", mae_Z1_Z4)
print("MAE between Augmented Neural ODE Prediction and GT:", mae_Z2_Z4)
print("MAE between ControlSynth Neural ODE Prediction and GT:", mae_Z3_Z4)
print("MAE between MLP Prediction and GT:", mae_Z5_Z4)
print()
print("Max Error between Neural ODE Prediction and GT:", max_err_Z1_Z4)
print("Max Error between Augmented Neural ODE Prediction and GT:", max_err_Z2_Z4)
print("Max Error between ControlSynth Neural ODE Prediction and GT:", max_err_Z3_Z4)
print("Max Error between MLP Prediction and GT:", max_err_Z5_Z4)
print()
print("SSIM between Neural ODE Prediction and GT:", ssim_Z1_Z4)
print("SSIM between Augmented Neural ODE Prediction and GT:", ssim_Z2_Z4)
print("SSIM between ControlSynth Neural ODE Prediction and GT:", ssim_Z3_Z4)
print("SSIM between MLP Prediction and GT:", ssim_Z5_Z4)
print()
print("Correlation Coefficient between Neural ODE Prediction and GT:", correlation_coefficient_Z1_Z4)
print("Correlation Coefficient between Augmented Neural ODE Prediction and GT:", correlation_coefficient_Z2_Z4)
print("Correlation Coefficient between ControlSynth Neural ODE Prediction and GT:", correlation_coefficient_Z3_Z4)
print("Correlation Coefficient between MLP Prediction and GT:", correlation_coefficient_Z5_Z4)
print()
print("Chamfer Distance between Neural ODE Prediction and GT:", chamfer_Z1_Z4)
print("Chamfer Distance between Augmented Neural ODE Prediction and GT:", chamfer_Z2_Z4)
print("Chamfer Distance between ControlSynth Neural ODE Prediction and GT:", chamfer_Z3_Z4)
print("Chamfer Distance between MLP Prediction and GT:", chamfer_Z5_Z4)