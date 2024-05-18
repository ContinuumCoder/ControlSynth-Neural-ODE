import numpy as np
from scipy.integrate import odeint as scipy_odeint
import os
from scipy.interpolate import interp2d

# Directory setup
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Parameters
n_seq = 1000
num_time_points = 1500
g = 9.81  # Gravity
H = 1.0   # Water depth
L = 15    # Domain length
N_fine = 70  # Number of fine grid points
N = 50       # Number of coarse grid points
T = 7       # Simulation time
dx = L / (N - 1)

class shallow_water_dataset:
    def __init__(self, n_seq, L, T, N_fine, num_time_points):
        self.n_seq = n_seq
        self.sequences = []

        for i in range(n_seq):
            print(f"Generating Sequence {i}...")
            self.sequences.append(self.generate_sequence(L, T, N_fine, num_time_points))

    def generate_sequence(self, L, T, N_fine, num_time_points):
        # Simulation setup
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X, Y = np.meshgrid(x, y)

        # Initial conditions with random noise
        noise_intensity = 0.05  # Adjust noise intensity to your preference
        noise = np.random.normal(loc=0, scale=noise_intensity, size=(N, N))
        h0 = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (0.1*L)**2) + noise
        vx0 = np.zeros((N, N))
        vy0 = np.zeros((N, N))

        u0 = np.zeros((N, N, 3))
        u0[:, :, 0] = h0
        u0[:, :, 1] = vx0
        u0[:, :, 2] = vy0
        u0 = u0.flatten()

        t = np.linspace(0, T, num_time_points)

        # Integrate the shallow water equations over time
        sol = scipy_odeint(self.shallow_water_2d, u0, t, args=(g, H, N))
        sol = sol.reshape(-1, N, N, 3)[-num_time_points:]
        sol_fine = np.zeros((num_time_points, N_fine, N_fine, 3))
        x_fine = np.linspace(0, L, N_fine)
        y_fine = np.linspace(0, L, N_fine)

        for i in range(num_time_points):
            interp_h = interp2d(x, y, sol[i, :, :, 0], kind='cubic')
            sol_fine[i, :, :, 0] = interp_h(x_fine, y_fine)

        return sol_fine[:, :, :, [0]]  # Return only the height field

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

# Generate the dataset
dataset = shallow_water_dataset(n_seq=n_seq, L=L, T=T, N_fine=N_fine, num_time_points=num_time_points)

# Stack sequences and save the dataset
all_sequences = np.stack(dataset.sequences)

np.save(os.path.join(current_directory, 'shallow_water_dataset.npy'), all_sequences)
