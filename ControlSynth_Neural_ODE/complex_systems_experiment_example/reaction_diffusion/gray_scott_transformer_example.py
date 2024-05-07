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


solver_path = current_directory + '/../../src/solver.py'
spec = importlib.util.spec_from_file_location("solver", solver_path)
solver_module = importlib.util.module_from_spec(spec)
sys.modules["solver"] = solver_module
spec.loader.exec_module(solver_module)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


USE_BASIC_SOLVER = True
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
n_seq = 1
batch_size = 1
n_test = 0
L = 2.5
train_len = 25
test_len = int(train_len * 0.5)
num_time_points = test_len + train_len
T = 150
lr = 5e-4
input_dim = N_fine * N_fine * n_vars + 1
hidden_dim = 2048
num_layers = 2
num_epochs = 1000
whole_seq = None
augment_dim = 10

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



def generate_nonuniform_field(N, min_val, max_val):
    field = np.random.uniform(min_val, max_val, (N, N))
    return field
class GrayScottDataset(Dataset):
    def __init__(self, n_seq, Du=0.16, Dv=0.08, f=0.035, k=0.065, L=L, N=50, T=T, N_fine=N_fine, num_time_points=num_time_points):
        self.inputs = []
        self.targets = []
        self.h = L / (N - 1) 
        global global_min_vals, global_range_vals

        U0 = 1 - 0.5 * np.random.rand(N, N)
        V0 = 0.25 * np.random.rand(N, N)
        u0 = np.zeros((N, N, 2))
        u0[:, :, 0] = U0
        u0[:, :, 1] = V0
        u0 = u0.flatten()

        t = np.linspace(0, T, num_time_points)


        Du_field = generate_nonuniform_field(N, 0.15, 0.17)
        Dv_field = generate_nonuniform_field(N, 0.05, 0.10)

        sol = scipy_odeint(self.gray_scott_2d_periodic, u0, t, args=(Du_field, Dv_field, f, k, N))


        sol = sol.reshape(-1, N, N, 2)
        
        sol_fine = np.zeros((num_time_points, N_fine, N_fine, 2))
        x_fine = np.linspace(0, L, N_fine)
        y_fine = np.linspace(0, L, N_fine)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)

        global whole_seq

        for i in range(num_time_points):
            interp_U = interp2d(x, y, sol[i, :, :, 0], kind='cubic')
            interp_V = interp2d(x, y, sol[i, :, :, 1], kind='cubic')
            sol_fine[i, :, :, 0] = interp_U(x_fine, y_fine)
            sol_fine[i, :, :, 1] = interp_V(x_fine, y_fine)

        fig = go.Figure(data=[go.Surface(x=np.linspace(0, L, N_fine), y=np.linspace(0, L, N_fine), z=sol_fine[-1][:, :, 0])])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='U'),
            title="Gray Scott Equation (Target)",
            width=800,
            height=600
        )

        fig.show() 

        fig = go.Figure(data=[go.Surface(x=np.linspace(0, L, N_fine), y=np.linspace(0, L, N_fine), z=sol_fine[test_len][:, :, 0])])
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='U'),
            title="Gray Scott Equation (Input)",
            width=800,
            height=600
        )

        fig.show() 

        sol_fine = sol_fine[:, :, :, [0]]
        whole_seq = sol_fine
        np.save(current_directory+"/sol.npy", sol_fine)
        



    def gray_scott_2d_periodic(self, u, t, Du, Dv, f, k, N):
        u = u.reshape((N, N, 2))
        U = u[:, :, 0]
        V = u[:, :, 1]
        
        laplacian_U = np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0) + np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1) - 4 * U
        laplacian_V = np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) + np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1) - 4 * V
        
        dudt = Du * laplacian_U - U * V**2 + f * (1 - U)
        dvdt = Dv * laplacian_V + U * V**2 - (f + k) * V
        du = np.zeros((N, N, 2))
        du[:, :, 0] = dudt
        du[:, :, 1] = dvdt
        return du.flatten()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.inputs[idx]
        target = self.targets[idx]
        return input, target





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
        self.func = ODEFunc(hidden_dim=int(hidden_dim * 0.57))
        self.gfunc = ODEFunc(hidden_dim=int(hidden_dim * 0.57))

        # self.func = ODEFunc(hidden_dim=int(hidden_dim * 0.63))
        # self.gfunc = TwoConvLayer()
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
    def __init__(self, in_channels=1, mid_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(TwoConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding)


    def forward(self, t, x):
        BS, dim = x.shape
        x = x.view(BS, 1, N_fine, N_fine)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(BS, input_dim-1)
        return x
    


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, 
                 num_layers=num_layers, input_dim=input_dim,
                 d_model=512, nhead=4, transformer_num_layers=2):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.transformer_layer = \
            TransformerLayer(input_dim, d_model, nhead, transformer_num_layers)

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y1 = self.net(t_and_y)[:, :self.input_dim-1]
        y2 = self.transformer_layer(t_and_y)
        y2 = y2.squeeze(1)[:, :self.input_dim-1]
        return y1+y2





class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=2):
        super(Transformer, self).__init__()
        self.transformer_layer = \
            TransformerLayer(input_dim, d_model, nhead, num_layers)
    def forward(self, y0, t):
        pred_seq = y0.unsqueeze(0)
        pred_seq = torch.cat([pred_seq, t[0].view(1, -1, 1)], dim=2)
        for i in range(t.shape[0] - 1):
            output = self.transformer_layer(pred_seq)
            output = output[:, :, :-1]
            next_t = t[:i + 1].view(-1, 1, 1)
            output = torch.cat([output, next_t], dim=2)
            pred_seq = torch.cat([pred_seq, output[-1:]], dim=0)
        pred_seq = pred_seq[:,:,:input_dim-1].squeeze(1)
        return pred_seq.view(-1, t.shape[0], input_dim-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(TransformerLayer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output




def train(model, model_name):
    optimizer = optim.Adam(model.parameters(), lr=lr)  
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
    t = torch.linspace(0, 1, train_len).to(device)
    pred = model(inputs, t)
    Z = pred.cpu().detach().numpy().reshape(-1, t.shape[0], N_fine, N_fine, n_vars)
    Z = unnormalize_array(Z, global_min_vals, global_range_vals)[0]
    flops, params = profile(model, inputs=(inputs, t), verbose=False)
    print(model_name + " flops and params:", flops, params)
    return Z[test_len-1]

if initial_mode:
    train_mode = True
    dataset = GrayScottDataset(n_seq=n_seq)
    transformer = Transformer().to(device)
    transformer_ode = NeuralODE().to(device)
    torch.save(transformer, current_directory+"/transformer.pth")
    torch.save(transformer_ode, current_directory+"/transformer_ode.pth")

sol_fine = np.load(current_directory + "/sol.npy")
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
    transformer = torch.load(current_directory+"/transformer.pth").to(device)
    train(transformer, "transformer")
    torch.save(transformer, current_directory+"/transformer.pth")
transformer = torch.load(current_directory+"/transformer.pth").to(device)
Z_transformer = eval_model(transformer, "transformer")



if train_mode:
    transformer_ode = torch.load(current_directory+"/transformer_ode.pth").to(device)
    train(transformer_ode, "transformer_ode")
    torch.save(transformer_ode, current_directory+"/transformer_ode.pth")
transformer_ode = torch.load(current_directory+"/transformer_ode.pth").to(device)
Z_transformer_ode = eval_model(transformer_ode, "transformer_ode")




def chamfer_distance(set1, set2):
    dist1 = np.sqrt(((set1[:, np.newaxis, :] - set2[np.newaxis, :, :]) ** 2).sum(axis=2))
    dist2 = np.sqrt(((set2[:, np.newaxis, :] - set1[np.newaxis, :, :]) ** 2).sum(axis=2))  
    nearest_dist1 = np.min(dist1, axis=1)
    nearest_dist2 = np.min(dist2, axis=0)
    return np.mean(nearest_dist1) + np.mean(nearest_dist2)


X = np.linspace(0, L, N_fine)
Y = np.linspace(0, L, N_fine)
Z1 = Z_transformer_ode[:, :, 0] 
# Z2 = Z_augode[:, :, 0]  
# Z3 = Z_csode[:, :, 0] 
Z4 = sol_fine[-1][:, :, 0] 
Z5 = Z_transformer[:, :, 0] 
Z_initial = sol_fine[-test_len][:, :, 0] 




mse_Z1_Z4 = mean_squared_error(Z1, Z4)
# mse_Z2_Z4 = mean_squared_error(Z2, Z4)
# mse_Z3_Z4 = mean_squared_error(Z3, Z4)
mse_Z5_Z4 = mean_squared_error(Z5, Z4)


mae_Z1_Z4 = mean_absolute_error(Z1, Z4)
# mae_Z2_Z4 = mean_absolute_error(Z2, Z4)
# mae_Z3_Z4 = mean_absolute_error(Z3, Z4)
mae_Z5_Z4 = mean_absolute_error(Z5, Z4)


max_err_Z1_Z4 = max_error(Z1.ravel(), Z4.ravel())
# max_err_Z2_Z4 = max_error(Z2.ravel(), Z4.ravel())
# max_err_Z3_Z4 = max_error(Z3.ravel(), Z4.ravel())
max_err_Z5_Z4 = max_error(Z5.ravel(), Z4.ravel())

ssim_Z1_Z4 = ssim(Z1, Z4, data_range=Z4.max() - Z4.min())
# ssim_Z2_Z4 = ssim(Z2, Z4, data_range=Z4.max() - Z4.min())
# ssim_Z3_Z4 = ssim(Z3, Z4, data_range=Z4.max() - Z4.min())
ssim_Z5_Z4 = ssim(Z5, Z4, data_range=Z4.max() - Z4.min())

correlation_coefficient_Z1_Z4 = np.corrcoef(Z1.flatten(), Z4.flatten())[0, 1]
# correlation_coefficient_Z2_Z4 = np.corrcoef(Z2.flatten(), Z4.flatten())[0, 1]
# correlation_coefficient_Z3_Z4 = np.corrcoef(Z3.flatten(), Z4.flatten())[0, 1]
correlation_coefficient_Z5_Z4 = np.corrcoef(Z5.flatten(), Z4.flatten())[0, 1]


chamfer_Z1_Z4 = chamfer_distance(Z1, Z4)
# chamfer_Z2_Z4 = chamfer_distance(Z2, Z4)
# chamfer_Z3_Z4 = chamfer_distance(Z3, Z4)
chamfer_Z5_Z4 = chamfer_distance(Z5, Z4)





print("MSE between Neural ODE Prediction and GT:", mse_Z1_Z4)
# print("MSE between Augmented Neural ODE Prediction and GT:", mse_Z2_Z4)
# print("MSE between ControlSynth Neural ODE Prediction and GT:", mse_Z3_Z4)
print("MSE between MLP Prediction and GT:", mse_Z5_Z4)
print()
print("MAE between Neural ODE Prediction and GT:", mae_Z1_Z4)
# print("MAE between Augmented Neural ODE Prediction and GT:", mae_Z2_Z4)
# print("MAE between ControlSynth Neural ODE Prediction and GT:", mae_Z3_Z4)
print("MAE between MLP Prediction and GT:", mae_Z5_Z4)
print()
print("Max Error between Neural ODE Prediction and GT:", max_err_Z1_Z4)
# print("Max Error between Augmented Neural ODE Prediction and GT:", max_err_Z2_Z4)
# print("Max Error between ControlSynth Neural ODE Prediction and GT:", max_err_Z3_Z4)
print("Max Error between MLP Prediction and GT:", max_err_Z5_Z4)
print()
print("SSIM between Neural ODE Prediction and GT:", ssim_Z1_Z4)
# print("SSIM between Augmented Neural ODE Prediction and GT:", ssim_Z2_Z4)
# print("SSIM between ControlSynth Neural ODE Prediction and GT:", ssim_Z3_Z4)
print("SSIM between MLP Prediction and GT:", ssim_Z5_Z4)
print()
print("Correlation Coefficient between Neural ODE Prediction and GT:", correlation_coefficient_Z1_Z4)
# print("Correlation Coefficient between Augmented Neural ODE Prediction and GT:", correlation_coefficient_Z2_Z4)
# print("Correlation Coefficient between ControlSynth Neural ODE Prediction and GT:", correlation_coefficient_Z3_Z4)
print("Correlation Coefficient between MLP Prediction and GT:", correlation_coefficient_Z5_Z4)
print()
print("Chamfer Distance between Neural ODE Prediction and GT:", chamfer_Z1_Z4)
# print("Chamfer Distance between Augmented Neural ODE Prediction and GT:", chamfer_Z2_Z4)
# print("Chamfer Distance between ControlSynth Neural ODE Prediction and GT:", chamfer_Z3_Z4)
print("Chamfer Distance between MLP Prediction and GT:", chamfer_Z5_Z4)