import os
current_directory = os.path.dirname(os.path.abspath(__file__))
utils_path = current_directory + '/../src/'
import sys
sys.path.append(utils_path)
import numpy as np
import csode_utils
import torch
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = current_directory + '/../dataset/shallow_water_dataset.npy'
dataset = np.load(dataset_path)
dataset = dataset.astype(np.float32)
n_seq = dataset.shape[0]
n_point = dataset.shape[1]
dataset.reshape(n_seq, n_point, -1)
input_dim = dataset.shape[2]
batch_size = 16

dataset = csode_utils.SeqDataset(dataset)
n_train = int(len(dataset) * 0.8)
n_test = len(dataset) - n_train
train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_layers = 2
hidden_dim = 2048
epochs = 100
lr = 1e-3
criterion = csode_utils.CombinedLoss(weight_mse=0, weight_l1=1)
t = torch.linspace(0, 1, n_point).to(device)

csode_adapt_f = csode_utils.ODEFunc(hidden_dim, num_layers, input_dim, device)
csode_adapt_g = csode_utils.AdaptODEFuncG_2D(input_dim, 128, 50)
csode_adapt = csode_utils.CSNeuralODE(csode_adapt_f, csode_adapt_g, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, csode_adapt, t, criterion, device, epochs, lr)

csode_f = csode_utils.ODEFunc(hidden_dim, num_layers, input_dim, device)
csode_g = csode_utils.ODEFuncG(input_dim, device)
csode = csode_utils.CSNeuralODE(csode_f, csode_g, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, csode, t, criterion, device, epochs, lr)

node_f = csode_utils.ODEFunc(hidden_dim, num_layers, input_dim, device)
node = csode_utils.NeuralODE(node_f, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, node, t, criterion, device, epochs, lr)

anode_f = csode_utils.AugmentedODEFunc(hidden_dim, num_layers, input_dim, input_dim, device)
anode = csode_utils.AugmentedNeuralODE(input_dim, input_dim, anode_f, False)
csode_utils.train_and_evaluate(train_loader, test_loader, anode, t, criterion, device, epochs, lr)

sonode_f = csode_utils.SecondOrderAugmentedODEFunc(hidden_dim, num_layers, input_dim, input_dim, device)
sonode = csode_utils.AugmentedNeuralODE(input_dim, input_dim, sonode_f, True)
csode_utils.train_and_evaluate(train_loader, test_loader, sonode, t, criterion, device, epochs, lr)

mlp = csode_utils.MLP(hidden_dim, num_layers, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, mlp, t, criterion, device, epochs, lr)

rnn = csode_utils.RNN(hidden_dim, num_layers, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, rnn, t, criterion, device, epochs, lr)

transformer = csode_utils.Transformer(input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, transformer, t, criterion, device, epochs, lr)

tode_f = csode_utils.TransformerODEFunc(hidden_dim, num_layers, input_dim, device)
tode = csode_utils.NeuralODE(tode_f, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, tode, t, criterion, device, epochs, lr)

cstode_f = csode_utils.TransformerODEFunc(hidden_dim, num_layers, input_dim, device)
cstode_g = csode_utils.AdaptODEFuncG_2D(input_dim, 128, 50)
cstode = csode_utils.CSNeuralODE(cstode_f, cstode_g, input_dim)
csode_utils.train_and_evaluate(train_loader, test_loader, cstode, t, criterion, device, epochs, lr)
