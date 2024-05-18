import numpy as np
from scipy.integrate import odeint as scipy_odeint
import os
from scipy.interpolate import interp2d

# Directory setup
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Parameters
n_seq = 1000
num_time_points = 2500
L = 15
N_fine = 50
N = 50
T = 150

def generate_nonuniform_field(N, min_val, max_val):
    field = np.random.uniform(min_val, max_val, (N, N))
    return field

def gray_scott_2d_periodic(u, t, Du, Dv, f, k, N):
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

# Initialize the dataset container
datasets = []

# Simulation parameters
Du = 0.16
Dv = 0.08
f = 0.035
k = 0.065
t = np.linspace(0, T, num_time_points)

# Generate multiple sequences
for i in range(n_seq):
    print(f"Generating Sequence {i}...")
    U0 = 1 - 0.5 * np.random.rand(N, N)
    V0 = 0.25 * np.random.rand(N, N)
    u0 = np.zeros((N, N, 2))
    u0[:, :, 0] = U0
    u0[:, :, 1] = V0
    u0 = u0.flatten()

    Du_field = generate_nonuniform_field(N, 0.15, 0.17)
    Dv_field = generate_nonuniform_field(N, 0.05, 0.10)

    sol = scipy_odeint(gray_scott_2d_periodic, u0, t, args=(Du_field, Dv_field, f, k, N))
    sol = sol.reshape(-1, N, N, 2)

    # Interpolation to fine grid
    sol_fine = np.zeros((num_time_points, N_fine, N_fine, 2))
    x_fine = np.linspace(0, L, N_fine)
    y_fine = np.linspace(0, L, N_fine)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)

    for j in range(num_time_points):
        interp_U = interp2d(x, y, sol[j, :, :, 0], kind='cubic')
        interp_V = interp2d(x, y, sol[j, :, :, 1], kind='cubic')
        sol_fine[i, :, :, 0] = interp_U(x_fine, y_fine)
        sol_fine[i, :, :, 1] = interp_V(x_fine, y_fine)

    datasets.append(sol_fine[:, :, :, [0]])

# Stack and save the dataset
all_sequences = np.stack(datasets)
print(all_sequences.shape)
np.save(os.path.join(current_directory, 'gray_scott_dataset.npy'), all_sequences)
