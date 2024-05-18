
from torchdiffeq import odeint, odeint_adjoint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint

class DynamicODESolver:
    def __init__(self, func, y0, gfunc=None, ufunc=None, u0=None, step_size=None, interp="linear", atol=1e-6, norm=None):
        self.func = func
        self.y0 = y0
        self.gfunc = gfunc
        self.ufunc = ufunc
        self.u0 = u0 if u0 is not None else y0
        self.step_size = step_size
        self.interp = interp
        self.atol = atol
        self.norm = norm

    def _before_integrate(self, t):
        pass

    def _advance(self, next_t):
        t0 = self.t
        y0 = self.y
        u0 = self.u
        dt = next_t - t0
        if self.ufunc is None:
            u1 = u0
            udot = u1
        else:
            udot = self.ufunc(t0, u0)
            u1 = udot * dt + u0
        if self.gfunc is None:
            gu1 = udot
        else:
            gu1 = self.gfunc(t0, udot)
        dy, f0 = self._step_func(t0, dt, next_t, y0, gu1)
        y1 = y0 + dy
        if self.interp == "linear":
            y_next = self._linear_interp(t0, next_t, y0, y1, next_t)
            u_next = self._linear_interp(t0, next_t, u0, u1, next_t)
        elif self.interp == "cubic":
            f1 = self.func(next_t, y1) + self.gfunc(next_t, self.gu)
            y_next = self._cubic_hermite_interp(t0, y0, f0, next_t, y1, f1, next_t)
        else:
            y_next = y1
            u_next = u1
        self.t = next_t
        self.y = y_next
        self.u = u_next
        return y_next

    def integrate(self, t):
        if self.step_size is None:
            self.step_size = t[1] - t[0]
        self.t = t[0]
        self.y = self.y0
        self.u = self.u0
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.y0.device, self.y0.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution

    def _step_func(self, t0, dt, t1, y0, gu1):
        f0 = self.func(t0, y0) + gu1
        dy = f0 * dt 
        return dy, f0

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)



class NeuralODE(nn.Module):
    def __init__(self, func, input_dim):
        super(NeuralODE, self).__init__()
        self.func = func
        self.input_dim = input_dim-1
        
    def forward(self, y0, t):
        out = odeint(self.func, y0, t, method='euler').permute(1, 0, 2)
        # out = out.view(-1, t.shape[0], self.input_dim)
        return out


class CSNeuralODE(nn.Module):
    def __init__(self, func, gfunc, input_dim):
        super(CSNeuralODE, self).__init__()
        self.func = func
        self.gfunc = gfunc
        self.input_dim = input_dim-1

    def forward(self, y0, t):
        solver = DynamicODESolver(self.func, y0, ufunc=self.gfunc, u0=y0)
        out = solver.integrate(t).permute(1, 0, 2)
        # out = out.view(-1, t.shape[0], self.input_dim)
        return out
    

    
# class Transformer(nn.Module):
#     def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, batch_size=15):
#         super(Transformer, self).__init__()
#         self.input_dim = input_dim + 1
#         self.transformer_layer = \
#             TransformerLayer(input_dim + 1, d_model, nhead, num_layers)
#         self.batch_size = batch_size
        
#     def forward(self, y0, t):
#         pred_seq_list = []
#         num_batches = (t.shape[0] - 1) // self.batch_size + 1
        
#         for i in range(num_batches):
#             start_idx = i * self.batch_size
#             end_idx = min((i + 1) * self.batch_size, t.shape[0])
#             batch_t = t[start_idx:end_idx]
            
#             pred_seq = y0.unsqueeze(0)
#             t0 = torch.ones(1, y0.shape[0], 1) * batch_t[0]
#             pred_seq = torch.cat([pred_seq, t0], dim=2)
            
#             for j in range(batch_t.shape[0] - 1):
#                 output = self.transformer_layer(pred_seq)
#                 output = output[:, :, :-1]
#                 next_t = batch_t[:j + 1]
#                 next_t = torch.ones(1, y0.shape[0], 1) * next_t
#                 next_t = next_t.permute(2, 1, 0)
#                 output = torch.cat([output, next_t], dim=2)
#                 pred_seq = torch.cat([pred_seq, output[-1:]], dim=0)
#             pred_seq = pred_seq[:,:,:self.input_dim].squeeze(1)
#             pred_seq_list.append(pred_seq)
#             print(len(pred_seq_list))
        
#         pred_seq = torch.cat(pred_seq_list, dim=0)
#         return pred_seq.view(-1, t.shape[0], self.input_dim)

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1, batch_size=5):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.transformer_layer = \
            TransformerLayer(input_dim + 1, d_model, nhead, num_layers)
        self.batch_size = batch_size
        
    def forward(self, y0, t):
        pred_seq_list = []
        num_batches = (t.shape[0] - 1) // self.batch_size + 1
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, t.shape[0])
            batch_t = t[start_idx:end_idx]
            
            pred_seq = y0.unsqueeze(0)
            t0 = torch.ones(1, y0.shape[0], 1) * batch_t[0]
            pred_seq = torch.cat([pred_seq, t0], dim=2)
            
            for j in range(batch_t.shape[0] - 1):
                output = self.transformer_layer(pred_seq)
                output = output[:, :, :-1]
                next_t = batch_t[:j + 1]
                next_t = torch.ones(1, y0.shape[0], 1) * next_t
                next_t = next_t.permute(2, 1, 0)
                output = torch.cat([output, next_t], dim=2)
                pred_seq = torch.cat([pred_seq, output[-1:]], dim=0)
            pred_seq = pred_seq[:,:,:self.input_dim].squeeze(1)
            pred_seq_list.append(pred_seq)
            # print(len(pred_seq_list))
        
        pred_seq = torch.cat(pred_seq_list, dim=0)
        return pred_seq.view(-1, t.shape[0], self.input_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(0)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return x + pe


    
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




class MLP(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim):
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
        out = torch.stack(ys).permute(1, 0, 2)
        return out

class RNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim
        
    def forward(self, y0, t):
        h0 = torch.zeros(self.num_layers, y0.size(0), self.hidden_dim).to(y0.device)
        outputs = [y0]
        for _ in range(1, t.shape[0]):
            input = outputs[-1].unsqueeze(1)
            _, h0 = self.rnn(input, h0)
            output = self.fc(h0[-1])
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
class TransformerODEFunc(nn.Module):
    def __init__(self, hidden_dim, 
                 num_layers, input_dim, device,
                 d_model=32, nhead=2, transformer_num_layers=1):
        super(TransformerODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, input_dim + 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.device = device
        self.transformer_layer = \
            TransformerLayer(input_dim + 1, d_model, nhead, transformer_num_layers)

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(self.device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y1 = self.net(t_and_y)[:, :self.input_dim]
        y2 = self.transformer_layer(t_and_y.unsqueeze(0))
        y2 = y2.squeeze(0)[:, :self.input_dim]
        out = y1 + y2
        return out
    


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim, device):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # layers.append(nn.Softplus())
            # layers.append(nn.ELU())
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim + 1))
        
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.device = device

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(self.device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y = self.net(t_and_y)
        y = y[:, :self.input_dim]
        return y


class ODEFuncG(nn.Module):
    def __init__(self, input_dim, device):
        super(ODEFuncG, self).__init__()
        self.linear = nn.Linear(input_dim + 1, input_dim + 1)
        self.device = device
        self.input_dim = input_dim
        
    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(self.device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y =  self.linear(t_and_y)[:, :self.input_dim]
        return y


class AdaptODEFuncG_2D(nn.Module):
    def __init__(self, input_dim, mid_channels, N, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(AdaptODEFuncG_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding)
        # self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding)
        self.input_dim = input_dim
        self.N = N

    def forward(self, t, x):
        BS, dim = x.shape
        x = x.view(BS, 1, self.N, self.N)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(BS, self.input_dim)
        return x

class AdaptODEFuncG_1D(nn.Module):
    def __init__(self, input_dim, mid_channels, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(AdaptODEFuncG_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size, stride, padding)
        # self.tanh = nn.Tanh()
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size, stride, padding)
        self.input_dim = input_dim

    def forward(self, t, x):
        BS, dim = x.shape
        x = x.view(BS, 1, dim)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(BS, dim)


class AugmentedNeuralODE(nn.Module):
    def __init__(self, augment_dim, input_dim, func, use_second_order=False):
        super(AugmentedNeuralODE, self).__init__()
        self.use_second_order = use_second_order
        self.func = func    
        self.augment_dim = augment_dim
        self.input_dim = input_dim

    def forward(self, y0, t):
        y_aug = torch.cat([y0, torch.zeros(y0.shape[0], self.augment_dim).to(y0)], dim=1)
        # if self.use_second_order:
        #     v_aug = torch.zeros_like(y_aug)
        #     z0 = torch.cat((y_aug, v_aug), dim=1)
        # else:
        z0 = y_aug      
        out = odeint(self.func, z0, t, method='euler').permute(1, 0, 2)
        out = out[:, :, :self.input_dim]
        # out = out.view(-1, t.shape[0], self.input_dim)
        return out
    

class AugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim, augment_dim, device):
        super(AugmentedODEFunc, self).__init__()
        layers = [nn.Linear(input_dim + augment_dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim + augment_dim + 1))
        self.net = nn.Sequential(*layers)
        self.device = device
        self.output_dim = input_dim + augment_dim

    def forward(self, t, z):
        y = z
        t_vec = torch.ones(y.shape[0], 1).to(self.device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        out = self.net(t_and_y)
        out = out[:, :self.output_dim]
        return out


class SecondOrderAugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim, augment_dim, device):
        super(SecondOrderAugmentedODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim + augment_dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim + augment_dim + 1))
        
        self.net = nn.Sequential(*layers)
        self.device = device
        self.out_dim = input_dim

    def forward(self, t, z):
        cutoff = int(z.shape[1]/2)
        y = z[:, :cutoff]
        v = z[:, cutoff:]
        t_vec = torch.ones(y.shape[0], 1).to(self.device) * t
        t_and_y_and_v = torch.cat([t_vec, y, v], 1)    
        out = self.net(t_and_y_and_v)[:, :self.out_dim]
        out = torch.cat((v, out), dim = 1)
        return out
    

class SeqDataset(Dataset):
    def __init__(self, data):
        """
        data: A tensor of shape (n_seq, n_point, feature_dim)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx, 0, :]
        target = self.data[idx]
        return input, target


import torch
import torch.optim as optim

def train_and_evaluate(train_loader, eval_loader, model, t, criterion, device, num_epochs=1000, lr=1e-3):
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train() 
        total_train_loss = 0.0
        batch_idx = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, t)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Train with batch {batch_idx}.")
            batch_idx += 1
            total_train_loss += loss.item() * inputs.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        model.eval() 
        total_eval_loss = 0.0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, t)
                loss = criterion(outputs, targets)
                total_eval_loss += loss.item() * inputs.size(0)
        eval_loss = total_eval_loss / len(eval_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Eval Loss: {eval_loss:.4f}")
    print("Training complete.")


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


